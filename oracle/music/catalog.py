"""Music catalog — scan directory, extract tags, store in SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config.settings import settings

_MUSIC_EXTS = {".mp3", ".flac", ".ogg", ".opus", ".m4a", ".wav", ".aac", ".wma"}


@dataclass
class Track:
    id: int
    title: str
    artist: str
    album: str
    genre: str
    duration: float  # seconds
    path: str


class Catalog:
    """SQLite-backed music catalog with tag extraction."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.music_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                artist TEXT NOT NULL DEFAULT '',
                album TEXT NOT NULL DEFAULT '',
                genre TEXT NOT NULL DEFAULT '',
                duration REAL NOT NULL DEFAULT 0,
                path TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_tracks_title ON tracks(title);
            CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
        """)
        self._conn.commit()

    # ---------------------------------------------------------------- query

    def list_tracks(self) -> list[Track]:
        rows = self._conn.execute(
            "SELECT * FROM tracks ORDER BY artist, album, title"
        ).fetchall()
        return [Track(**dict(r)) for r in rows]

    def get_track(self, track_id: int) -> Track | None:
        row = self._conn.execute(
            "SELECT * FROM tracks WHERE id = ?", (track_id,)
        ).fetchone()
        return Track(**dict(row)) if row else None

    def search(self, query: str) -> list[Track]:
        """Case-insensitive search across title, artist, album, genre."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM tracks WHERE title LIKE ? OR artist LIKE ? "
            "OR album LIKE ? OR genre LIKE ? ORDER BY artist, title",
            (pattern, pattern, pattern, pattern),
        ).fetchall()
        return [Track(**dict(r)) for r in rows]

    def random_track(self) -> Track | None:
        row = self._conn.execute(
            "SELECT * FROM tracks ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return Track(**dict(row)) if row else None

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM tracks").fetchone()
        return row["cnt"]

    # ---------------------------------------------------------------- ingest

    def index_directory(self, music_dir: Path | None = None) -> int:
        """Scan a directory for music files and index them. Returns count added."""
        d = music_dir or settings.music_path
        if not d.is_dir():
            logger.warning(f"Music directory not found: {d}")
            return 0

        files = sorted(
            f for f in d.rglob("*") if f.suffix.lower() in _MUSIC_EXTS
        )
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
            "SELECT id FROM tracks WHERE path = ?", (path,)
        ).fetchone()
        return row is not None

    def _index_file(self, path: Path) -> None:
        """Extract tags and insert into the database."""
        title, artist, album, genre, duration = _extract_tags(path)
        self._conn.execute(
            "INSERT INTO tracks (title, artist, album, genre, duration, path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (title, artist, album, genre, duration, str(path)),
        )
        self._conn.commit()
        logger.debug(f"Indexed: {artist} — {title} ({duration:.0f}s)")

    def close(self) -> None:
        self._conn.close()


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
