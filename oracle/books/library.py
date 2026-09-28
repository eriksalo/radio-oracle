"""Book library — scan directory, parse texts, store in SQLite."""

from __future__ import annotations

import functools
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config.settings import settings

_GUTENBERG_HEADER_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE,
)
_GUTENBERG_FOOTER_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE,
)
_CHAPTER_RE = re.compile(
    r"^(?:chapter|book|part|act|section|canto)\s+[\dIVXLCDMivxlcdm]+",
    re.IGNORECASE | re.MULTILINE,
)
# The number a heading carries ("CHAPTER 12", "Chapter XII.", "PART I").
_HEADING_NUM_RE = re.compile(
    r"^(?:chapter|book|part|act|section|canto)\s+([\d]+|[IVXLCDMivxlcdm]+)\b", re.IGNORECASE
)
_PREAMBLE_TITLES = {"preamble", "full text"}

_WORD_NUMBERS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_WORD_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}  # fmt: skip
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}  # fmt: skip
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(ch not in _ROMAN for ch in s):
        return None
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN[ch]
        if i + 1 < len(s) and _ROMAN[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def parse_chapter_number(spec: str) -> int | None:
    """ "12", "twelve", "twelfth", "twenty one", "XII" → 12; None if not a number."""
    words = re.findall(r"[a-z0-9]+", spec.lower())
    if not words:
        return None
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    if len(words) == 1 and words[0] in _ORDINALS:
        return _ORDINALS[words[0]]
    total = 0
    matched = False
    for w in words:
        if w in _WORD_TENS:
            total += _WORD_TENS[w]
            matched = True
        elif w in _WORD_NUMBERS:
            total += _WORD_NUMBERS[w]
            matched = True
        elif w in _ORDINALS:
            total += _ORDINALS[w]
            matched = True
        elif w in ("and", "number", "chapter"):
            continue
        else:
            matched = False
            break
    if matched:
        return total
    if len(words) == 1:
        return _roman_to_int(words[0])
    return None


def _clean_subtitle(text: str | None) -> str:
    """ ". Loomings." → "Loomings"; None/long/sentence-like → ""."""
    if not text:
        return ""
    t = text.strip().strip(".:;-— ").strip()
    if not t or len(t) > 60 or t.count(" ") > 8:
        return ""
    return t


def heading_number(title: str) -> int | None:
    """The number in a chapter heading, arabic or roman ("CHAPTER XII" → 12)."""
    m = _HEADING_NUM_RE.match(title.strip())
    if not m:
        return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else _roman_to_int(tok)


def is_content_heading(title: str) -> bool:
    return bool(_CHAPTER_RE.match(title.strip()))


def _synchronized(method):
    """Serialize access to the shared sqlite connection: the reader loop
    runs in a worker thread (asyncio.to_thread) while buttons/voice act
    from the event-loop thread. Reentrant, so methods may call each other."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Book:
    id: int
    title: str
    author: str
    path: str
    total_chapters: int
    total_paragraphs: int


@dataclass
class Chapter:
    book_id: int
    chapter_idx: int
    title: str
    text: str


class Library:
    """SQLite-backed book index with full paragraph text."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.books_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + _lock: created on the event-loop thread,
        # used from the reader worker thread (2026-09-27 crash in book mode).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @_synchronized
    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL DEFAULT '',
                path TEXT NOT NULL UNIQUE,
                total_chapters INTEGER NOT NULL DEFAULT 0,
                total_paragraphs INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                chapter_idx INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (book_id) REFERENCES books(id),
                UNIQUE (book_id, chapter_idx)
            );
            CREATE TABLE IF NOT EXISTS paragraphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                chapter_idx INTEGER NOT NULL,
                para_idx INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (book_id) REFERENCES books(id),
                UNIQUE (book_id, chapter_idx, para_idx)
            );
            CREATE INDEX IF NOT EXISTS idx_paragraphs_book
                ON paragraphs(book_id, chapter_idx, para_idx);
        """)
        self._conn.commit()

    # ---------------------------------------------------------------- query

    @_synchronized
    def list_books(self) -> list[Book]:
        rows = self._conn.execute("SELECT * FROM books ORDER BY title").fetchall()
        return [Book(**dict(r)) for r in rows]

    @_synchronized
    def get_book(self, book_id: int) -> Book | None:
        row = self._conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        return Book(**dict(row)) if row else None

    @_synchronized
    def search(self, query: str) -> list[Book]:
        """Title/author search — FTS5 (voice-friendly, word-based) with a
        LIKE fallback for substrings and for SQLite builds without FTS5."""
        if not query.strip():
            return []
        fts_hits = self._search_fts(query)
        if fts_hits:
            return fts_hits
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM books WHERE title LIKE ? OR author LIKE ? ORDER BY title",
            (pattern, pattern),
        ).fetchall()
        return [Book(**dict(r)) for r in rows]

    @_synchronized
    def _search_fts(self, query: str) -> list[Book]:
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        try:
            self._ensure_fts()
            # Quoted terms → no FTS syntax surprises; implicit AND ranks
            # "moby dick" straight to Moby-Dick across 60k titles.
            match = " ".join(f'"{t}"' for t in terms)
            rows = self._conn.execute(
                "SELECT b.* FROM books_fts f JOIN books b ON b.id = f.rowid "
                "WHERE books_fts MATCH ? ORDER BY rank LIMIT 20",
                (match,),
            ).fetchall()
            return [Book(**dict(r)) for r in rows]
        except sqlite3.OperationalError as e:
            logger.debug(f"FTS search unavailable ({e}); falling back to LIKE")
            return []

    @_synchronized
    def _ensure_fts(self) -> None:
        """Create and populate the FTS index on first use (one-time cost)."""
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS books_fts "
            "USING fts5(title, author, content='books', content_rowid='id')"
        )
        n_books = self._conn.execute("SELECT COUNT(*) c FROM books").fetchone()["c"]
        # COUNT(*) on an external-content FTS table reads the *content*
        # table, not the index — count the docsize shadow table instead.
        n_fts = self._conn.execute("SELECT COUNT(*) c FROM books_fts_docsize").fetchone()["c"]
        if n_fts < n_books:
            logger.info(f"Building books FTS index ({n_books} books)...")
            self._conn.execute("INSERT INTO books_fts(books_fts) VALUES('rebuild')")
            self._conn.commit()

    @_synchronized
    def get_chapter(self, book_id: int, chapter_idx: int) -> Chapter | None:
        row = self._conn.execute(
            "SELECT * FROM chapters WHERE book_id = ? AND chapter_idx = ?",
            (book_id, chapter_idx),
        ).fetchone()
        if not row:
            return None
        text_rows = self._conn.execute(
            "SELECT text FROM paragraphs WHERE book_id = ? AND chapter_idx = ? ORDER BY para_idx",
            (book_id, chapter_idx),
        ).fetchall()
        full_text = "\n\n".join(r["text"] for r in text_rows)
        return Chapter(
            book_id=row["book_id"],
            chapter_idx=row["chapter_idx"],
            title=row["title"],
            text=full_text,
        )

    @_synchronized
    def get_chapter_title(self, book_id: int, chapter_idx: int) -> str | None:
        row = self._conn.execute(
            "SELECT title FROM chapters WHERE book_id = ? AND chapter_idx = ?",
            (book_id, chapter_idx),
        ).fetchone()
        return row["title"] if row else None

    @_synchronized
    def list_chapter_titles(self, book_id: int) -> list[tuple[int, str]]:
        rows = self._conn.execute(
            "SELECT chapter_idx, title FROM chapters WHERE book_id = ? ORDER BY chapter_idx",
            (book_id,),
        ).fetchall()
        return [(r["chapter_idx"], r["title"]) for r in rows]

    @_synchronized
    def list_chapter_headings(self, book_id: int) -> list[tuple[int, str, str]]:
        """(chapter_idx, title, subtitle). The indexer stores only the
        matched heading ("CHAPTER 1"); a short first paragraph (". Loomings.")
        is the chapter's descriptive title and is returned as subtitle."""
        rows = self._conn.execute(
            """SELECT c.chapter_idx, c.title,
                      (SELECT p.text FROM paragraphs p
                        WHERE p.book_id = c.book_id AND p.chapter_idx = c.chapter_idx
                          AND p.para_idx = 0 AND length(p.text) <= 80) AS sub
                 FROM chapters c WHERE c.book_id = ? ORDER BY c.chapter_idx""",
            (book_id,),
        ).fetchall()
        # Only real chapter headings get a subtitle; the preamble's first
        # line ("[Transcriber's notes]") is not a name.
        return [
            (
                r["chapter_idx"],
                r["title"],
                _clean_subtitle(r["sub"]) if is_content_heading(r["title"]) else "",
            )
            for r in rows
        ]

    def chapter_label(self, book_id: int, chapter_idx: int) -> str:
        """Spoken name of a chapter: "Chapter 1: Loomings" / "Preamble"."""
        for idx, title, sub in self.list_chapter_headings(book_id):
            if idx == chapter_idx:
                t = title.strip().rstrip(".")
                return f"{t}: {sub}" if sub else t
        return ""

    def first_content_chapter(self, book_id: int) -> int:
        """Where a fresh read should start: the first real chapter heading
        (Chapter/Part/Book N), skipping the Gutenberg preamble (transcriber's
        notes, contents, dedications) that the indexer files as chapter 0.
        Falls back to 1 if chapter 0 is a preamble, else 0."""
        titles = self.list_chapter_titles(book_id)
        for idx, title in titles:
            if is_content_heading(title):
                return idx
        if titles and titles[0][1].strip().lower() in _PREAMBLE_TITLES and len(titles) > 1:
            return 1
        return 0

    def resolve_chapter(self, book_id: int, spec: str) -> int | None:
        """Map a spoken chapter reference to a chapter_idx, or None.

        "one"/"1"/"first"/"XII" → the heading carrying that number, else the
        Nth content chapter; "the beginning"/"start" → first content chapter;
        "front matter"/"preface"/"preamble" → chapter 0; "last"/"final"/"end"
        → the last chapter; anything else → a title fragment ("loomings").
        """
        titles = self.list_chapter_titles(book_id)
        if not titles:
            return None
        norm = re.sub(r"[^a-z0-9 ]+", " ", spec.lower()).strip()
        norm = re.sub(r"^(?:the|chapter|to|at|with|from|of)\s+", "", norm).strip()
        if norm in ("front matter", "preface", "preamble", "introduction", "notes", "credits"):
            return titles[0][0]
        if norm in ("beginning", "start", "top", "beginning of the book"):
            return self.first_content_chapter(book_id)
        if norm in ("last", "final", "end", "last chapter", "final chapter", "end of the book"):
            return titles[-1][0]
        n = parse_chapter_number(norm)
        content = [(idx, t) for idx, t in titles if is_content_heading(t)]
        if n is not None:
            for idx, t in content:
                if heading_number(t) == n:
                    return idx
            pool = content or titles
            if 1 <= n <= len(pool):
                return pool[n - 1][0]
            return None
        if len(norm) >= 3:
            for idx, t, sub in self.list_chapter_headings(book_id):
                if norm in t.lower() or (sub and norm in sub.lower()):
                    return idx
        return None

    @_synchronized
    def get_paragraph(self, book_id: int, chapter_idx: int, para_idx: int) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM paragraphs WHERE book_id = ? AND chapter_idx = ? AND para_idx = ?",
            (book_id, chapter_idx, para_idx),
        ).fetchone()
        return row["text"] if row else None

    @_synchronized
    def count_books(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM books").fetchone()
        return row["cnt"]

    @_synchronized
    def sample_authors(self, n: int = 5) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT author FROM books WHERE author != '' ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
        return [r["author"] for r in rows]

    @_synchronized
    def get_paragraph_count(self, book_id: int, chapter_idx: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM paragraphs WHERE book_id = ? AND chapter_idx = ?",
            (book_id, chapter_idx),
        ).fetchone()
        return row["cnt"]

    # ---------------------------------------------------------------- ingest

    @_synchronized
    def index_directory(self, books_dir: Path | None = None) -> int:
        """Scan a directory for .txt files and index them. Returns count added."""
        d = books_dir or settings.books_path
        if not d.is_dir():
            logger.warning(f"Books directory not found: {d}")
            return 0

        txt_files = sorted(d.rglob("*.txt"))
        added = 0
        for f in txt_files:
            if self._already_indexed(str(f)):
                logger.debug(f"Already indexed: {f.name}")
                continue
            try:
                self._index_txt(f)
                added += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to index {f.name}: {e}")
        logger.info(f"Indexed {added} new books from {d} ({len(txt_files)} total files)")
        return added

    @_synchronized
    def _already_indexed(self, path: str) -> bool:
        row = self._conn.execute("SELECT id FROM books WHERE path = ?", (path,)).fetchone()
        return row is not None

    @_synchronized
    def _index_txt(self, path: Path) -> None:
        """Parse a plain-text book and insert into the database."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = _strip_gutenberg_boilerplate(raw)

        title, author = _extract_title_author(raw, path)
        chapters = _split_chapters(text)

        # Insert book
        cur = self._conn.execute(
            "INSERT INTO books (title, author, path, total_chapters, total_paragraphs) "
            "VALUES (?, ?, ?, 0, 0)",
            (title, author, str(path)),
        )
        book_id = cur.lastrowid

        total_paras = 0
        for ch_idx, (ch_title, ch_text) in enumerate(chapters):
            self._conn.execute(
                "INSERT INTO chapters (book_id, chapter_idx, title) VALUES (?, ?, ?)",
                (book_id, ch_idx, ch_title),
            )
            paras = _split_paragraphs(ch_text)
            for p_idx, para in enumerate(paras):
                self._conn.execute(
                    "INSERT INTO paragraphs (book_id, chapter_idx, para_idx, text) "
                    "VALUES (?, ?, ?, ?)",
                    (book_id, ch_idx, p_idx, para),
                )
            total_paras += len(paras)

        self._conn.execute(
            "UPDATE books SET total_chapters = ?, total_paragraphs = ? WHERE id = ?",
            (len(chapters), total_paras, book_id),
        )
        try:
            self._conn.execute(
                "INSERT INTO books_fts(rowid, title, author) VALUES (?, ?, ?)",
                (book_id, title, author),
            )
        except sqlite3.OperationalError:
            pass  # FTS table not built yet; _ensure_fts rebuilds lazily
        self._conn.commit()
        logger.info(f"Indexed: {title} — {len(chapters)} chapters, {total_paras} paragraphs")

    @_synchronized
    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------- text parsing


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start = _GUTENBERG_HEADER_RE.search(text)
    end = _GUTENBERG_FOOTER_RE.search(text)
    begin = start.end() if start else 0
    finish = end.start() if end else len(text)
    return text[begin:finish].strip()


def _extract_title_author(raw: str, path: Path) -> tuple[str, str]:
    """Best-effort title and author extraction from Gutenberg header or filename."""
    title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    author = ""

    # Try Gutenberg metadata lines
    for line in raw[:3000].splitlines():
        line = line.strip()
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("author:"):
            author = line.split(":", 1)[1].strip()
        if title and author:
            break

    return title, author


def _split_chapters(text: str) -> list[tuple[str, str]]:
    """Split text into (chapter_title, chapter_text) pairs.

    Falls back to a single chapter if no chapter headings are found.
    """
    splits = list(_CHAPTER_RE.finditer(text))
    if not splits:
        return [("Full Text", text)]

    chapters: list[tuple[str, str]] = []

    # Text before first chapter heading
    preamble = text[: splits[0].start()].strip()
    if preamble and len(preamble) > 200:
        chapters.append(("Preamble", preamble))

    for i, match in enumerate(splits):
        ch_title = match.group().strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        ch_text = text[start:end].strip()
        if ch_text:
            chapters.append((ch_title, ch_text))

    return chapters if chapters else [("Full Text", text)]


def _split_paragraphs(text: str) -> list[str]:
    """Split chapter text into non-empty paragraphs."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p and len(p) > 1]
