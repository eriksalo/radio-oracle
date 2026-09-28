"""Probe: the reader on the production books.db — fresh start, chapter
navigation, status — without a mic and without leaving bookmarks behind.

Any bookmark this probe creates is deleted (and a pre-existing one for the
same book restored) in a finally block: a stray bookmark once made the
radio "resume" a book nobody asked for.

    SIM_SCRIPT=scripts/probe_reader.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    from oracle import audio

    played: list[float] = []
    audio.play_audio = lambda a, sample_rate=None, should_abort=None: played.append(
        len(a) / (sample_rate or 24000)
    )
    from oracle.books.session import ReaderSession
    from oracle.tts import KokoroTTS

    tts = KokoroTTS()
    tts.load()
    s = ReaderSession(tts=tts)
    book = s.find_book("Moby Dick")
    assert book, "Moby Dick not found"
    prior = s._bookmarks.get(book.id)
    print(f"book: {book.title!r} by {book.author!r}; existing bookmark: {prior}")
    titles = s._library.list_chapter_titles(book.id)
    first = s._library.first_content_chapter(book.id)
    print(f"chapters: {len(titles)}; first content chapter idx {first}: {titles[first][1]!r}")
    print("first few headings:", [t[:30] for _, t in titles[:6]])
    try:
        s._bookmarks.delete(book.id)
        assert s.start(book)
        print("fresh start:", s.started_fresh, "|", s.status_text())
        for spec in ("one", "three", "XV", "the last", "loomings", "the preface", "ninety nine"):
            title = s.goto_chapter(spec)
            print(f"  goto {spec!r:14s} -> {title!r:45s} | {s.status_text()}")
        print("  prev ->", s.prev_chapter())
        print("  restart ->", s.restart(), "|", s.status_text())
        print(
            "confident('moby dick'):",
            s.is_confident_match("moby dick", book),
            " confident('rock and roll'):",
            s.is_confident_match("books on rock and roll", book),
        )

        async def read_two():
            await asyncio.to_thread(s.read_continuous, lambda: len(played) >= 2)

        asyncio.run(read_two())
        print(
            f"read {len(played)} paragraphs ({sum(played):.1f}s audio) from chapter 1 — "
            f"now at: {s.status_text()}"
        )
    finally:
        s.stop()
        s._bookmarks.delete(book.id)
        if prior is not None:
            s._bookmarks.save(prior.book_id, prior.chapter_idx, prior.para_idx)
        print("bookmark cleanup: restored" if prior else "bookmark cleanup: removed")
        s.close()


if __name__ == "__main__":
    main()
