"""The reader loop runs in a worker thread while Library/BookmarkStore are
created on the event-loop thread — sqlite must allow that (it crashed
the radio in book mode on 2026-09-27)."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import Library


def _in_thread(fn):
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except Exception as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out.get("value")


def test_library_usable_from_another_thread():
    with tempfile.TemporaryDirectory() as tmp:
        lib = Library(Path(tmp) / "books.db")  # created here (main thread)
        assert _in_thread(lib.count_books) == 0
        assert _in_thread(lambda: lib.get_paragraph(1, 0, 0)) is None
        assert _in_thread(lambda: lib.search("moby")) == []


def test_bookmarks_usable_from_another_thread():
    with tempfile.TemporaryDirectory() as tmp:
        bm = BookmarkStore(Path(tmp) / "bm.db")
        _in_thread(lambda: bm.save(37048, 2, 5))
        got = bm.get(37048)
        assert got is not None and (got.chapter_idx, got.para_idx) == (2, 5)
