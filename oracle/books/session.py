"""Reader session — book selection and playback control for the app.

Bundles Library + BookmarkStore + Reader behind the small surface the
hardware state machine needs: pick a book (by voice query or by the most
recently read bookmark), start/resume it, and control playback while the
app polls buttons.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import Book, Library
from oracle.books.reader import Reader

if TYPE_CHECKING:
    from oracle.tts import KokoroTTS


def _format_status(st: dict) -> str:
    title = st["chapter_title"].strip()
    where = (
        f"chapter {st['chapter_number']} of {st['chapter_total']}"
        if st["chapter_number"] > 0
        else "the front matter"
    )
    if title and st["chapter_number"] > 0 and title.lower().startswith(("chapter", "part", "book")):
        # "CHAPTER 3. Loomings." → keep the descriptive part if any
        rest = re.sub(r"^(?:chapter|part|book)\s+\S+[.:\s]*", "", title, flags=re.IGNORECASE)
        title = rest.strip(" .")
    tail = f": {title}" if title and st["chapter_number"] > 0 else ""
    return f"{st['book']}, {where}{tail}."


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

    @staticmethod
    def is_confident_match(query: str, book: Book) -> bool:
        """Every meaningful word of the request appears in the title or
        author — otherwise the app should confirm before reading aloud."""
        stop = {"the", "a", "an", "of", "by", "and", "book", "me", "to", "read", "please"}
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop and len(w) > 1]
        hay = f"{book.title} {book.author}".lower()
        return bool(words) and all(w in hay for w in words)

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

    def prev_chapter(self) -> str | None:
        return self._reader.prev_chapter()

    def goto_chapter(self, spec: str) -> str | None:
        """Jump to a spoken chapter reference ("three", "XII", "the last",
        "loomings", "the preface"). Returns the chapter title, or None when
        it doesn't resolve."""
        pos = self._reader.position
        if pos is None:
            return None
        idx = self._library.resolve_chapter(pos.book_id, spec)
        if idx is None:
            return None
        return self._reader.goto_chapter(idx)

    def restart(self) -> str | None:
        """Back to the first real chapter."""
        pos = self._reader.position
        if pos is None:
            return None
        return self._reader.goto_chapter(self._library.first_content_chapter(pos.book_id))

    @property
    def started_fresh(self) -> bool:
        return self._reader.started_fresh

    def set_user(self, name: str) -> None:
        """Bookmarks (and so "my book") belong to the identified user."""
        self._bookmarks.user = name

    def status_text(self) -> str | None:
        """Spoken summary of where we are, e.g. "Moby-Dick, chapter 3 of
        135: Loomings." None when nothing is open."""
        st = self._reader.status()
        if not st:
            return None
        return _format_status(st)

    def stop(self) -> None:
        """Stop reading and persist the bookmark."""
        self._reader.stop()

    def close(self) -> None:
        self._reader.close()
