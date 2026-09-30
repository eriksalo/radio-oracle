"""Music catalog — scan directory, extract tags, store in SQLite."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config.settings import settings

_MUSIC_EXTS = {".mp3", ".flac", ".ogg", ".opus", ".m4a", ".wav", ".aac", ".wma"}


@dataclass
class Track:
    id: str
    title: str
    artist: str
    album: str
    genre: str
    duration: float  # seconds
    path: str


# On-disk schema column names differ from Track fields; alias them here.
_TRACK_SELECT = (
    "SELECT track_id AS id, "
    "COALESCE(title, filename) AS title, "
    "COALESCE(artist, '') AS artist, "
    "COALESCE(album, '') AS album, "
    "COALESCE(genre, '') AS genre, "
    "COALESCE(duration_sec, 0) AS duration, "
    "filepath_rel AS path "
    "FROM tracks"
)


def _row_to_track(row: sqlite3.Row) -> Track:
    d = dict(row)
    d["path"] = str(_resolve_track_path(d["path"]))
    return Track(**d)


def _resolve_track_path(stored: str) -> Path:
    """Resolve a stored filepath_rel to an absolute path.

    Relative paths may be relative to the music directory (new indexer) or to
    the process working directory (legacy Jetson DB); prefer whichever exists.
    """
    p = Path(stored)
    if p.is_absolute():
        return p
    under_music = (settings.music_path / p).resolve()
    if under_music.exists():
        return under_music
    return p.resolve()


_QUOTES_RE = re.compile("[\u2018\u2019\u201a\u201b\u2032']")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")


def norm(text: str | None) -> str:
    """Fold a tag or a request so they compare the way people say them:
    ASCII-folded, lowercase, ``&`` ↔ ``and``, apostrophes dropped (the
    tag "Maybe I’m Amazed" carries a curly one), other punctuation to
    spaces. Registered as the SQLite function ``norm`` so LIKE sees it."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("&", " and ").replace("+", " and ")
    t = _QUOTES_RE.sub("", t)
    return " ".join(_NONWORD_RE.sub(" ", t).split())


def _word_match(needle: str, hay: str) -> bool:
    """*needle* appears in *hay* as whole words ("ryan adams" is not in
    "bryan adams"; "dylan" is in "bob dylan")."""
    return bool(needle) and bool(re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay))


_BY_RE = re.compile(r"^(.*?)\s+by\s+(.+)$", re.IGNORECASE)

# Nicknames people actually say. Normalised request → normalised artist.
ALIASES: dict[str, str] = {
    "the boss": "bruce springsteen",
    "springsteen": "bruce springsteen",
    "the king": "elvis presley",
    "elvis": "elvis presley",
    "the fab four": "the beatles",
    "beatles": "the beatles",
    "the stones": "the rolling stones",
    "zeppelin": "led zeppelin",
    "zep": "led zeppelin",
    "floyd": "pink floyd",
    "the dead": "grateful dead",
    "the who": "the who",
    "bob marley": "bob marley and the wailers",
    "ccr": "creedence clearwater revival",
    "creedence": "creedence clearwater revival",
    "csny": "crosby stills nash and young",
    "elo": "electric light orchestra",
    "the boss man": "bruce springsteen",
    "cash": "johnny cash",
    "dylan": "bob dylan",
    "emmylou": "emmylou harris",
    "knopfler": "mark knopfler",
    "cohen": "leonard cohen",
    "the piano man": "billy joel",
}
_MUSIC_SUFFIX_RE = re.compile(r"\s+(?:music|songs?|tracks?|tunes?|stuff)$")


class Catalog:
    """SQLite-backed music catalog with tag extraction."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.music_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function("norm", 1, norm, deterministic=True)
        self._init_schema()

    def _init_schema(self) -> None:
        # Must match the column names _TRACK_SELECT reads; the deployed Jetson
        # DB predates this code and already has this shape (plus extra columns).
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                title TEXT,
                artist TEXT DEFAULT '',
                album TEXT DEFAULT '',
                genre TEXT DEFAULT '',
                duration_sec REAL DEFAULT 0,
                filename TEXT NOT NULL,
                filepath_rel TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_tracks_title ON tracks(title);
            CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
        """)
        self._conn.commit()
        self._check_schema()

    def _check_schema(self) -> None:
        """Fail loudly if the tracks table lacks the columns the queries read."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(tracks)").fetchall()}
        required = {
            "track_id",
            "title",
            "artist",
            "album",
            "genre",
            "duration_sec",
            "filename",
            "filepath_rel",
        }
        missing = required - cols
        if missing:
            raise RuntimeError(
                f"music DB {self._db_path} has incompatible tracks schema; "
                f"missing columns: {sorted(missing)}. Re-index with "
                "scripts/index_music.py into a fresh DB or migrate the table."
            )

    # ---------------------------------------------------------------- query

    def list_tracks(self) -> list[Track]:
        rows = self._conn.execute(f"{_TRACK_SELECT} ORDER BY artist, album, title").fetchall()
        return [_row_to_track(r) for r in rows]

    def get_track(self, track_id: str) -> Track | None:
        row = self._conn.execute(f"{_TRACK_SELECT} WHERE track_id = ?", (track_id,)).fetchone()
        return _row_to_track(row) if row else None

    def search(self, query: str) -> list[Track]:
        """Loose search: the normalised request as a substring of the
        normalised title, artist, album or genre. ``&``/``and``, curly
        quotes and accents never get in the way. Ordered by artist, title."""
        pattern = f"%{norm(query)}%"
        if pattern == "%%":
            return []
        rows = self._conn.execute(
            f"{_TRACK_SELECT} WHERE norm(title) LIKE ? OR norm(artist) LIKE ? "
            "OR norm(album) LIKE ? OR norm(genre) LIKE ? ORDER BY artist, title",
            (pattern, pattern, pattern, pattern),
        ).fetchall()
        return [_row_to_track(r) for r in rows]

    def search_ranked(self, query: str, hint: str | None = None) -> tuple[list[Track], str]:
        """What "play <query>" should tune to: the best *tier* of matches
        and its name.

        "Song by Artist" is split and both halves must match. Then, in
        order: the request is a genre word (whole word in some genre);
        whole-word artist match ("dylan" → Bob Dylan and Dylan, not
        "Bryan Adams" for "Ryan Adams"); exact album; exact title;
        whole-word album; whole-word title; finally the loose substring
        search. *hint* ("album", "song") from the spoken sentence moves
        that field's tiers to the front ("play the album Harvest" must
        not tune to Barclay James Harvest). Substring collisions had the
        wrong artist playing a good share of the time (eval 2026-09-30).
        """
        q = norm(query)
        if not q:
            return [], "none"
        m = _BY_RE.match(query.strip())
        if m:
            what, who = norm(m.group(1)), norm(m.group(2))
            who = ALIASES.get(who, who)
            if what and who:
                by_artist = [t for t in self.search(who) if _word_match(who, norm(t.artist))]
                both = [
                    t
                    for t in by_artist
                    if _word_match(what, norm(t.title)) or _word_match(what, norm(t.album))
                ]
                if both:
                    return both, "title+artist"
                if by_artist:
                    # The song isn't here but the artist is: tune to them
                    # rather than answer "couldn't find anything".
                    return by_artist, "artist"
        loose = self.search(q)
        if not loose and q in ALIASES:
            q = ALIASES[q]
            loose = self.search(q)
        if not loose and _MUSIC_SUFFIX_RE.search(q):
            # "folk music", "country songs": the noun is the request's, not the tag's.
            q = _MUSIC_SUFFIX_RE.sub("", q)
            loose = self.search(q)
        if not loose:
            return [], "none"
        # A nickname that also happens to be a substring somewhere ("the
        # boss" inside a title) still means the artist.
        alias = ALIASES.get(q)
        if alias and not any(_word_match(q, norm(t.artist)) for t in loose):
            aliased = [t for t in self.search(alias) if _word_match(alias, norm(t.artist))]
            if aliased:
                return aliased, "artist"
        tiers: list[tuple[str, list[Track]]] = [
            ("genre", [t for t in loose if _word_match(q, norm(t.genre))]),
            ("artist", [t for t in loose if _word_match(q, norm(t.artist))]),
            ("album", [t for t in loose if norm(t.album) == q]),
            ("title", [t for t in loose if norm(t.title) == q]),
            ("album-word", [t for t in loose if _word_match(q, norm(t.album))]),
            ("title-word", [t for t in loose if _word_match(q, norm(t.title))]),
        ]
        if hint in ("album", "song", "track", "title"):
            field = "album" if hint == "album" else "title"
            front = [tier for tier in tiers if tier[0].startswith(field)]
            tiers = front + [tier for tier in tiers if not tier[0].startswith(field)]
        for name, hits in tiers:
            if hits:
                return hits, name
        return loose, "substring"

    def random_track(self) -> Track | None:
        row = self._conn.execute(f"{_TRACK_SELECT} ORDER BY RANDOM() LIMIT 1").fetchone()
        return _row_to_track(row) if row else None

    def album_tracks(self, album: str) -> list[Track]:
        """All tracks of an album in track order (filenames carry numbers)."""
        rows = self._conn.execute(
            f"{_TRACK_SELECT} WHERE album = ? ORDER BY filename", (album,)
        ).fetchall()
        return [_row_to_track(r) for r in rows]

    def random_album_tracks(self) -> list[Track]:
        """Pick a random album and return all its tracks in order."""
        row = self._conn.execute(
            "SELECT DISTINCT COALESCE(album, '') AS album FROM tracks "
            "WHERE album IS NOT NULL AND album != '' "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        if row is None:
            track = self.random_track()
            return [track] if track else []
        album_name = row["album"]
        # Filenames usually carry the track number ("01 - ..."); title doesn't.
        rows = self._conn.execute(
            f"{_TRACK_SELECT} WHERE album = ? ORDER BY filename",
            (album_name,),
        ).fetchall()
        return [_row_to_track(r) for r in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM tracks").fetchone()
        return row["cnt"]

    def stats(self) -> dict:
        """Archive summary for spoken exploration answers."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS tracks, "
            "COUNT(DISTINCT NULLIF(artist, '')) AS artists, "
            "COUNT(DISTINCT NULLIF(album, '')) AS albums, "
            "COALESCE(SUM(duration_sec), 0) AS seconds FROM tracks"
        ).fetchone()
        return {
            "tracks": row["tracks"],
            "artists": row["artists"],
            "albums": row["albums"],
            "hours": row["seconds"] / 3600.0,
        }

    def sample_artists(self, n: int = 6) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT artist FROM tracks WHERE artist != '' ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
        return [r["artist"] for r in rows]

    # ---------------------------------------------------------------- ingest

    def index_directory(self, music_dir: Path | None = None) -> int:
        """Scan a directory for music files and index them. Returns count added."""
        d = music_dir or settings.music_path
        if not d.is_dir():
            logger.warning(f"Music directory not found: {d}")
            return 0

        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in _MUSIC_EXTS)
        added = 0
        for f in files:
            if self._already_indexed(str(f)):
                continue
            try:
                self._index_file(f)
                added += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to index {f.name}: {e}")

        logger.info(f"Indexed {added} new tracks from {d} ({len(files)} total files)")
        return added

    def _already_indexed(self, path: str) -> bool:
        row = self._conn.execute(
            "SELECT track_id FROM tracks WHERE filepath_rel = ?",
            (_stored_path(Path(path)),),
        ).fetchone()
        return row is not None

    def _index_file(self, path: Path) -> None:
        """Extract tags and insert into the database."""
        title, artist, album, genre, duration = _extract_tags(path)
        stored = _stored_path(path)
        track_id = hashlib.sha1(stored.encode()).hexdigest()[:16]
        self._conn.execute(
            "INSERT INTO tracks (track_id, title, artist, album, genre, "
            "duration_sec, filename, filepath_rel) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (track_id, title, artist, album, genre, duration, path.name, stored),
        )
        self._conn.commit()
        logger.debug(f"Indexed: {artist} — {title} ({duration:.0f}s)")

    def close(self) -> None:
        self._conn.close()


def _stored_path(path: Path) -> str:
    """Path form written to filepath_rel: relative to the music dir when
    possible (portable across machines), absolute otherwise."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(settings.music_path.resolve()))
    except ValueError:
        return str(resolved)


def _extract_tags(path: Path) -> tuple[str, str, str, str, float]:
    """Extract title, artist, album, genre, duration from a music file."""
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    artist = ""
    album = ""
    genre = ""
    duration = 0.0

    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(path, easy=True)
        if audio is None:
            return title, artist, album, genre, duration

        if audio.info:
            duration = audio.info.length or 0.0

        tags = audio.tags
        if tags:
            title = _first_tag(tags, "title") or title
            artist = _first_tag(tags, "artist") or artist
            album = _first_tag(tags, "album") or album
            genre = _first_tag(tags, "genre") or genre
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Tag extraction failed for {path.name}: {e}")

    return title, artist, album, genre, duration


def _first_tag(tags: dict, key: str) -> str:
    """Get first value for a tag key, or empty string."""
    val = tags.get(key)
    if val and isinstance(val, list):
        return str(val[0]).strip()
    if val:
        return str(val).strip()
    return ""
