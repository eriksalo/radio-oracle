"""Reader session — book selection and playback control for the app.

Bundles Library + BookmarkStore + Reader behind the small surface the
hardware state machine needs: pick a book (by voice query or by the most
recently read bookmark), start/resume it, and control playback while the
app polls buttons.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import Book, Library
from oracle.books.reader import Reader

if TYPE_CHECKING:
    from oracle.tts import KokoroTTS


class ReaderSession:
    """One long-lived reading session shared across reader-mode entries."""

    def __init__(self, tts: KokoroTTS | None = None) -> None:
        self._library = Library()
        self._bookmarks = BookmarkStore()
        self._reader = Reader(library=self._library, bookmarks=self._bookmarks, tts=tts)

    # ------------------------------------------------------------- selection

    def find_book(self, query: str) -> Book | None:
        hits = self._library.search(query)
        return hits[0] if hits else None

    def current_book(self) -> Book | None:
        """The most recently read book (freshest bookmark), if any."""
        for bm in self._bookmarks.list_in_progress():
            book = self._library.get_book(bm.book_id)
            if book:
                return book
        return None

    def has_bookmark(self, book_id: int) -> bool:
        bm = self._bookmarks.get(book_id)
        return bm is not None and (bm.chapter_idx, bm.para_idx) != (0, 0)

    def book_count(self) -> int:
        return len(self._library.list_books())

    # -------------------------------------------------------------- playback

    def start(self, book: Book) -> bool:
        pos = self._reader.start(book.id)
        if pos is None:
            logger.error(f"Could not start reading book {book.id}")
            return False
        return True

    def read_continuous(self, should_stop: Callable[[], bool] | None = None) -> None:
        """Blocking read loop — run via asyncio.to_thread from the app."""
        self._reader.read_continuous(should_stop=should_stop)

    @property
    def is_paused(self) -> bool:
        return self._reader.is_paused

    def toggle_pause(self) -> bool:
        """Toggle pause. Returns True if now paused."""
        if self.is_paused:
            self._reader.resume()
            return False
        self._reader.pause()
        return True

    def pause(self) -> None:
        """Pause (aborts the current paragraph mid-sentence; bookmark saved)."""
        self._reader.pause()

    def resume(self) -> None:
        self._reader.resume()

    def next_chapter(self) -> bool:
        return self._reader.next_chapter()

    def stop(self) -> None:
        """Stop reading and persist the bookmark."""
        self._reader.stop()

    def close(self) -> None:
        self._reader.close()
