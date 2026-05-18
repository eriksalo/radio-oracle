"""Book library — scan directory, parse texts, store in SQLite."""

from __future__ import annotations

import re
import sqlite3
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
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

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

    def list_books(self) -> list[Book]:
        rows = self._conn.execute(
            "SELECT * FROM books ORDER BY title"
        ).fetchall()
        return [Book(**dict(r)) for r in rows]

    def get_book(self, book_id: int) -> Book | None:
        row = self._conn.execute(
            "SELECT * FROM books WHERE id = ?", (book_id,)
        ).fetchone()
        return Book(**dict(row)) if row else None

    def search(self, query: str) -> list[Book]:
        """Case-insensitive title/author search."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT * FROM books WHERE title LIKE ? OR author LIKE ? ORDER BY title",
            (pattern, pattern),
        ).fetchall()
        return [Book(**dict(r)) for r in rows]

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

    def get_paragraph(self, book_id: int, chapter_idx: int, para_idx: int) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM paragraphs WHERE book_id = ? AND chapter_idx = ? AND para_idx = ?",
            (book_id, chapter_idx, para_idx),
        ).fetchone()
        return row["text"] if row else None

    def get_paragraph_count(self, book_id: int, chapter_idx: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM paragraphs WHERE book_id = ? AND chapter_idx = ?",
            (book_id, chapter_idx),
        ).fetchone()
        return row["cnt"]

    # ---------------------------------------------------------------- ingest

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

    def _already_indexed(self, path: str) -> bool:
        row = self._conn.execute(
            "SELECT id FROM books WHERE path = ?", (path,)
        ).fetchone()
        return row is not None

    def _index_txt(self, path: Path) -> None:
        """Parse a plain-text book and insert into the database."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = _strip_gutenberg_boilerplate(raw)

        title, author = _extract_title_author(raw, path)
        chapters = _split_chapters(text)

        # Insert book
        cur = self._conn.execute(
            "INSERT INTO books (title, author, path, total_chapters, total_paragraphs) VALUES (?, ?, ?, 0, 0)",
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
                    "INSERT INTO paragraphs (book_id, chapter_idx, para_idx, text) VALUES (?, ?, ?, ?)",
                    (book_id, ch_idx, p_idx, para),
                )
            total_paras += len(paras)

        self._conn.execute(
            "UPDATE books SET total_chapters = ?, total_paragraphs = ? WHERE id = ?",
            (len(chapters), total_paras, book_id),
        )
        self._conn.commit()
        logger.info(f"Indexed: {title} — {len(chapters)} chapters, {total_paras} paragraphs")

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
    preamble = text[:splits[0].start()].strip()
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
