"""Book reader — paragraph-by-paragraph TTS playback with pause/resume."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import settings
from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import Library, heading_number

if TYPE_CHECKING:
    from oracle.tts import KokoroTTS


@dataclass
class ReadingPosition:
    book_id: int
    chapter_idx: int
    para_idx: int
    total_chapters: int


# Sentence boundaries that keep the terminal punctuation (prosody).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _group_units(sentences: list[str], max_words: int) -> list[str]:
    """Pack sentences into units of at most *max_words* (a long sentence
    stays whole)."""
    units: list[str] = []
    cur: list[str] = []
    n = 0
    for sent in sentences:
        w = len(sent.split())
        if cur and n + w > max_words:
            units.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        units.append(" ".join(cur))
    return units


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
        # True after start() opened a book with no bookmark (announce
        # "starting at chapter one" instead of "resuming").
        self.started_fresh = False
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
        self, book_id: int, chapter_idx: int | None = None, para_idx: int = 0
    ) -> ReadingPosition | None:
        """Begin reading. With no chapter given: resume the bookmark, or —
        for a fresh book — start at the first real chapter, past the
        Gutenberg front matter (transcriber's notes, contents). An explicit
        chapter_idx (0 included, for "read the preface") is honoured."""
        book = self._library.get_book(book_id)
        if not book:
            logger.error(f"Book {book_id} not found")
            return None

        self.started_fresh = False
        if chapter_idx is None:
            bm = self._bookmarks.get(book_id)
            if bm:
                chapter_idx, para_idx = bm.chapter_idx, bm.para_idx
                logger.info(f"Resuming '{book.title}' from ch {chapter_idx}, para {para_idx}")
            else:
                chapter_idx = self._library.first_content_chapter(book_id)
                self.started_fresh = True
                if chapter_idx:
                    logger.info(
                        f"'{book.title}': skipping front matter, starting at ch {chapter_idx}"
                    )

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
        from oracle.activity import emit

        emit("reading", book=book.title, chapter=chapter_idx, paragraph=para_idx)
        self._journal("started" if self.started_fresh else "resumed")
        return self._position

    def _journal(self, event: str) -> None:
        """Durable memory of where the user is in which book."""
        try:
            from oracle.books.session import _format_status
            from oracle.memory.journal import record

            st = self.status()
            if st:
                record(
                    "book",
                    event=event,
                    book=st["book"],
                    author=st["author"],
                    status=_format_status(st),
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"book journal failed: {e}")

    def stop(self, finished: bool = False) -> None:
        """Stop reading and save bookmark."""
        if self._position:
            self._save_bookmark()
            self._journal("finished" if finished else "stopped")
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
                self.stop(finished=True)
                return None
            pos = self._position
            text = self._library.get_paragraph(pos.book_id, pos.chapter_idx, pos.para_idx)
            if text is None:
                self.stop(finished=True)
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
        from oracle.activity import emit

        emit("reading", chapter=next_ch, chapter_title=chapter.title)
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
        return self.goto_chapter(pos.chapter_idx + 1) is not None

    def prev_chapter(self) -> str | None:
        """Jump to the start of the previous chapter (or restart this one
        when already at the first). Returns the chapter title."""
        pos = self._position
        if not pos:
            return None
        first = self._library.first_content_chapter(pos.book_id)
        return self.goto_chapter(max(first, pos.chapter_idx - 1))

    def goto_chapter(self, chapter_idx: int) -> str | None:
        """Jump to the start of *chapter_idx*. Returns its title, or None
        when out of range. The read loop picks the new position up on its
        next paragraph (a paragraph in flight is not advanced past)."""
        pos = self._position
        if not pos or chapter_idx < 0 or chapter_idx >= pos.total_chapters:
            return None
        chapter = self._library.chapter_label(pos.book_id, chapter_idx) or None
        if chapter is None:
            return None
        self._position = ReadingPosition(
            book_id=pos.book_id,
            chapter_idx=chapter_idx,
            para_idx=0,
            total_chapters=pos.total_chapters,
        )
        self._save_bookmark()
        logger.info(f"Jumped to chapter {chapter_idx}: {chapter!r}")
        self._journal("chapter")
        return chapter

    def status(self) -> dict | None:
        """Where we are: book title/author, chapter index, its number among
        the content chapters, total, and the chapter title."""
        pos = self._position
        if not pos:
            return None
        book = self._library.get_book(pos.book_id)
        if not book:
            return None
        headings = self._library.list_chapter_headings(pos.book_id)
        first = self._library.first_content_chapter(pos.book_id)
        title = next(
            (sub or t for i, t, sub in headings if i == pos.chapter_idx),
            "",
        )
        # Prefer the numbers the headings themselves carry ("CHAPTER XCIX"
        # is chapter 99 even when stray headings inflate the row count).
        own = next((heading_number(t) for i, t, _ in headings if i == pos.chapter_idx), None)
        numbers = [heading_number(t) for i, t, _ in headings if i >= first]
        numbers = [n for n in numbers if n]
        if own is not None and numbers:
            number, total = own, max(numbers)
        else:
            number = pos.chapter_idx - first + 1 if pos.chapter_idx >= first else 0
            total = max(len(headings) - first, 1)
        return {
            "book": book.title,
            "author": book.author,
            "chapter_idx": pos.chapter_idx,
            "chapter_number": number,
            "chapter_total": total,
            "chapter_title": title,
            "paragraph": pos.para_idx,
        }

    def _interrupted(self) -> bool:
        return not self._paused.is_set() or bool(self._should_stop and self._should_stop())

    def _speak(self, text: str) -> None:
        """Speak a paragraph as a pipeline of short units: one thread
        synthesizes unit N+1 while unit N plays. Whole paragraphs (a Moby
        Dick paragraph can be 3 minutes of speech) overflowed the GPU
        sidecar's arena and delayed the first word by the full synthesis."""
        import queue
        import threading

        from oracle.audio import play_audio

        tts = self._get_tts()
        sentences = [x.strip() for x in _SENTENCE_SPLIT_RE.split(text) if x.strip()]
        units = _group_units(sentences, settings.reading_unit_max_words)
        if not units:
            return
        q: queue.Queue = queue.Queue(maxsize=2)

        def synth() -> None:
            for u in units:
                if self._interrupted():
                    break
                try:
                    q.put(tts.synthesize(u))
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Reader TTS failed on a unit: {e}")
            q.put(None)

        worker = threading.Thread(target=synth, name="reader-tts", daemon=True)
        worker.start()
        try:
            while True:
                audio = q.get()
                if audio is None:
                    break
                if self._interrupted():
                    # Drain so the producer can finish.
                    while q.get() is not None:
                        pass
                    break
                play_audio(audio, tts.sample_rate, should_abort=self._interrupted)
        finally:
            worker.join(timeout=60)

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
