"""Bookmark persistence — tracks reading position per book."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings


@dataclass
class Bookmark:
    book_id: int
    chapter_idx: int
    para_idx: int
    updated_at: str


class BookmarkStore:
    """SQLite-backed reading position per book.

    Shares the same database file as Library (books.db) but manages
    its own table.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.books_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                book_id INTEGER PRIMARY KEY,
                chapter_idx INTEGER NOT NULL DEFAULT 0,
                para_idx INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (book_id) REFERENCES books(id)
            )
        """)
        self._conn.commit()

    def get(self, book_id: int) -> Bookmark | None:
        row = self._conn.execute(
            "SELECT * FROM bookmarks WHERE book_id = ?", (book_id,)
        ).fetchone()
        return Bookmark(**dict(row)) if row else None

    def save(self, book_id: int, chapter_idx: int, para_idx: int) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO bookmarks (book_id, chapter_idx, para_idx, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(book_id) DO UPDATE SET
                   chapter_idx = excluded.chapter_idx,
                   para_idx = excluded.para_idx,
                   updated_at = excluded.updated_at""",
            (book_id, chapter_idx, para_idx, now),
        )
        self._conn.commit()

    def delete(self, book_id: int) -> None:
        self._conn.execute("DELETE FROM bookmarks WHERE book_id = ?", (book_id,))
        self._conn.commit()

    def list_in_progress(self) -> list[Bookmark]:
        """Return all bookmarks (books that have been started)."""
        rows = self._conn.execute(
            "SELECT * FROM bookmarks ORDER BY updated_at DESC"
        ).fetchall()
        return [Bookmark(**dict(r)) for r in rows]

    def close(self) -> None:
        self._conn.close()
