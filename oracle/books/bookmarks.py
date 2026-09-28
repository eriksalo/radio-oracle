"""Bookmark persistence — tracks reading position per book."""

from __future__ import annotations

import functools
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings


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
        # check_same_thread=False + _lock: created on the event-loop thread,
        # used from the reader worker thread (2026-09-27 crash in book mode).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @_synchronized
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

    @_synchronized
    def get(self, book_id: int) -> Bookmark | None:
        row = self._conn.execute("SELECT * FROM bookmarks WHERE book_id = ?", (book_id,)).fetchone()
        return Bookmark(**dict(row)) if row else None

    @_synchronized
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

    @_synchronized
    def delete(self, book_id: int) -> None:
        self._conn.execute("DELETE FROM bookmarks WHERE book_id = ?", (book_id,))
        self._conn.commit()

    @_synchronized
    def list_in_progress(self) -> list[Bookmark]:
        """Return all bookmarks (books that have been started)."""
        rows = self._conn.execute("SELECT * FROM bookmarks ORDER BY updated_at DESC").fetchall()
        return [Bookmark(**dict(r)) for r in rows]

    @_synchronized
    def close(self) -> None:
        self._conn.close()
