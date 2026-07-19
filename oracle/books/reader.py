"""Book reader — paragraph-by-paragraph TTS playback with pause/resume."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import settings
from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import Library

if TYPE_CHECKING:
    from oracle.tts import KokoroTTS


@dataclass
class ReadingPosition:
    book_id: int
    chapter_idx: int
    para_idx: int
    total_chapters: int


class Reader:
    """Plays a book aloud paragraph-by-paragraph via TTS.

    Designed to run in the main async loop. Call ``read_paragraph`` to
    advance one step, or ``read_continuous`` for hands-free playback
    with a stop callback.
    """

    def __init__(
        self,
        library: Library | None = None,
        bookmarks: BookmarkStore | None = None,
        tts: KokoroTTS | None = None,
    ) -> None:
        self._library = library or Library()
        self._bookmarks = bookmarks or BookmarkStore()
        self._tts: KokoroTTS | None = tts
        self._position: ReadingPosition | None = None
        self._paused = threading.Event()
        self._paused.set()  # starts unpaused
        # Abort check consulted during playback so pause/stop takes effect
        # mid-paragraph instead of after it finishes (~30s later).
        self._should_stop: Callable[[], bool] | None = None

    @property
    def position(self) -> ReadingPosition | None:
        return self._position

    @property
    def is_reading(self) -> bool:
        return self._position is not None

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def _get_tts(self) -> KokoroTTS:
        if self._tts is None:
            from oracle.tts import KokoroTTS

            self._tts = KokoroTTS()
        return self._tts

    def start(
        self, book_id: int, chapter_idx: int = 0, para_idx: int = 0
    ) -> ReadingPosition | None:
        """Begin reading from a position; resumes from bookmark if none given."""
        book = self._library.get_book(book_id)
        if not book:
            logger.error(f"Book {book_id} not found")
            return None

        # Resume from bookmark if starting from the beginning
        if chapter_idx == 0 and para_idx == 0:
            bm = self._bookmarks.get(book_id)
            if bm:
                chapter_idx = bm.chapter_idx
                para_idx = bm.para_idx
                logger.info(f"Resuming '{book.title}' from ch {chapter_idx}, para {para_idx}")

        self._position = ReadingPosition(
            book_id=book_id,
            chapter_idx=chapter_idx,
            para_idx=para_idx,
            total_chapters=book.total_chapters,
        )
        self._paused.set()
        # Persist immediately so this book becomes the "current book"
        # (freshest bookmark) even before the first paragraph completes.
        self._save_bookmark()
        logger.info(f"Reading: '{book.title}' — ch {chapter_idx}, para {para_idx}")
        return self._position

    def stop(self) -> None:
        """Stop reading and save bookmark."""
        if self._position:
            self._save_bookmark()
            logger.info(f"Stopped reading book {self._position.book_id}")
        self._position = None

    def pause(self) -> None:
        self._paused.clear()
        if self._position:
            self._save_bookmark()
        logger.debug("Reader paused")

    def resume(self) -> None:
        self._paused.set()
        logger.debug("Reader resumed")

    def read_paragraph(self) -> str | None:
        """Read the next paragraph via TTS. Returns the text, or None if finished.

        Advances the position and saves the bookmark. Blocks while audio plays.
        """
        if not self._position:
            return None

        pos = self._position
        text = self._library.get_paragraph(pos.book_id, pos.chapter_idx, pos.para_idx)

        if text is None:
            # Try next chapter
            if not self._advance_chapter():
                self.stop()
                return None
            pos = self._position
            text = self._library.get_paragraph(pos.book_id, pos.chapter_idx, pos.para_idx)
            if text is None:
                self.stop()
                return None

        # Speak it
        self._speak(text)

        # If playback was interrupted (pause/stop) or the position was
        # jumped (next_chapter) while speaking, don't advance — resume
        # should re-read the interrupted paragraph.
        if self._interrupted() or self._position is not pos:
            self._save_bookmark()
            return text

        # Advance to next paragraph
        self._position = ReadingPosition(
            book_id=pos.book_id,
            chapter_idx=pos.chapter_idx,
            para_idx=pos.para_idx + 1,
            total_chapters=pos.total_chapters,
        )
        self._save_bookmark()
        return text

    def read_continuous(
        self,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Read paragraphs in a loop until stopped or book ends.

        Args:
            should_stop: callback returning True to interrupt reading
        """
        self._should_stop = should_stop
        try:
            while self._position is not None:
                # Check pause
                while not self._paused.is_set():
                    if should_stop and should_stop():
                        return
                    time.sleep(0.1)

                if should_stop and should_stop():
                    self._save_bookmark()
                    return

                text = self.read_paragraph()
                if text is None:
                    break

                # Pause between paragraphs
                time.sleep(settings.reading_paragraph_pause)
        finally:
            self._should_stop = None

    def _advance_chapter(self) -> bool:
        """Move to the first paragraph of the next chapter. Returns False if at end."""
        pos = self._position
        if not pos:
            return False

        next_ch = pos.chapter_idx + 1
        if next_ch >= pos.total_chapters:
            logger.info("Reached end of book")
            return False

        chapter = self._library.get_chapter(pos.book_id, next_ch)
        if not chapter:
            return False

        logger.info(f"Chapter {next_ch}: {chapter.title}")
        # Announce chapter
        self._speak(f"Chapter: {chapter.title}")
        time.sleep(settings.reading_chapter_pause)

        self._position = ReadingPosition(
            book_id=pos.book_id,
            chapter_idx=next_ch,
            para_idx=0,
            total_chapters=pos.total_chapters,
        )
        return True

    def next_chapter(self) -> bool:
        """Jump to the start of the next chapter. Returns False at book end."""
        pos = self._position
        if not pos:
            return False
        next_ch = pos.chapter_idx + 1
        if next_ch >= pos.total_chapters:
            return False
        self._position = ReadingPosition(
            book_id=pos.book_id,
            chapter_idx=next_ch,
            para_idx=0,
            total_chapters=pos.total_chapters,
        )
        self._save_bookmark()
        logger.info(f"Skipped to chapter {next_ch}")
        return True

    def _interrupted(self) -> bool:
        return not self._paused.is_set() or bool(self._should_stop and self._should_stop())

    def _speak(self, text: str) -> None:
        from oracle.audio import play_audio

        tts = self._get_tts()
        audio = tts.synthesize(text)
        play_audio(audio, tts.sample_rate, should_abort=self._interrupted)

    def _save_bookmark(self) -> None:
        if self._position:
            self._bookmarks.save(
                self._position.book_id,
                self._position.chapter_idx,
                self._position.para_idx,
            )

    def close(self) -> None:
        if self._position:
            self._save_bookmark()
        self._bookmarks.close()
        self._library.close()
