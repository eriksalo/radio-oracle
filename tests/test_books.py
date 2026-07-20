"""Tests for the book library, reader, and reading session."""

from __future__ import annotations

import pytest

from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import Library
from oracle.books.reader import Reader
from oracle.books.session import ReaderSession

_BOOK = """Title: Moby-Dick; Or, The Whale
Author: Herman Melville

*** START OF THE PROJECT GUTENBERG EBOOK MOBY-DICK ***

CHAPTER I

Call me Ishmael.

Some years ago, never mind how long precisely.

CHAPTER II

I stuffed a shirt or two into my old carpet-bag.

*** END OF THE PROJECT GUTENBERG EBOOK MOBY-DICK ***
"""

_OTHER = """Title: The Adventures of Sherlock Holmes
Author: Arthur Conan Doyle

*** START OF THE PROJECT GUTENBERG EBOOK SHERLOCK ***

CHAPTER I

To Sherlock Holmes she is always the woman.

*** END OF THE PROJECT GUTENBERG EBOOK SHERLOCK ***
"""


@pytest.fixture()
def library(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    (books / "moby.txt").write_text(_BOOK)
    (books / "sherlock.txt").write_text(_OTHER)
    lib = Library(db_path=tmp_path / "books.db")
    lib.index_directory(books)
    yield lib
    lib.close()


class _FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text: str):
        import numpy as np

        self.spoken.append(text)
        return np.zeros(10, dtype=np.float32)


@pytest.fixture(autouse=True)
def _silent_audio(monkeypatch):
    import oracle.audio

    monkeypatch.setattr(oracle.audio, "play_audio", lambda *a, **kw: None)


# ---------------------------------------------------------------- library


def test_index_extracts_metadata_and_chapters(library):
    books = library.list_books()
    assert len(books) == 2
    moby = next(b for b in books if "Moby" in b.title)
    assert moby.author == "Herman Melville"
    assert moby.total_chapters == 2
    assert moby.total_paragraphs >= 3


def test_fts_search_word_order_and_case(library):
    assert library.search("moby dick")[0].author == "Herman Melville"
    assert library.search("sherlock")[0].author == "Arthur Conan Doyle"
    assert library.search("melville moby")[0].title.startswith("Moby")


def test_search_falls_back_to_like(library):
    # Substring of a word — FTS whole-word match misses, LIKE catches.
    hits = library.search("herlock")
    assert hits and hits[0].author == "Arthur Conan Doyle"


def test_search_no_results(library):
    assert library.search("war and peace") == []
    assert library.search("") == []


# ---------------------------------------------------------------- reader


@pytest.fixture()
def reading(library, tmp_path):
    bookmarks = BookmarkStore(db_path=tmp_path / "books.db")
    tts = _FakeTTS()
    rdr = Reader(library=library, bookmarks=bookmarks, tts=tts)
    book = library.search("moby")[0]
    return rdr, tts, book


def test_reader_reads_and_bookmarks(reading):
    rdr, tts, book = reading
    rdr.start(book.id)
    assert rdr.read_paragraph() == "Call me Ishmael."
    assert tts.spoken == ["Call me Ishmael."]
    assert rdr.position.para_idx == 1
    rdr.stop()

    # A fresh reader resumes from the saved bookmark.
    rdr2 = Reader(library=rdr._library, bookmarks=rdr._bookmarks, tts=tts)
    pos = rdr2.start(book.id)
    assert (pos.chapter_idx, pos.para_idx) == (0, 1)


def test_reader_advances_chapters_with_announcement(reading):
    rdr, tts, book = reading
    rdr.start(book.id, chapter_idx=0, para_idx=1)
    rdr.read_paragraph()  # last paragraph of ch 0
    text = rdr.read_paragraph()  # crosses into ch 1
    assert "carpet-bag" in text
    assert any(s.startswith("Chapter:") for s in tts.spoken)


def test_reader_finishes_book(reading):
    rdr, _, book = reading
    rdr.start(book.id, chapter_idx=1, para_idx=0)
    assert rdr.read_paragraph() is not None
    assert rdr.read_paragraph() is None
    assert not rdr.is_reading


def test_next_chapter_jump(reading):
    rdr, _, book = reading
    rdr.start(book.id)
    assert rdr.next_chapter() is True
    assert rdr.position.chapter_idx == 1
    assert rdr.next_chapter() is False  # already at last chapter


def test_paused_interrupt_does_not_advance(reading):
    rdr, _, book = reading
    rdr.start(book.id)
    rdr.pause()
    rdr.read_paragraph()
    assert rdr.position.para_idx == 0  # interrupted → re-read on resume


# ---------------------------------------------------------------- session


def test_session_find_current_and_controls(library, tmp_path, monkeypatch):
    monkeypatch.setattr("oracle.books.session.Library", lambda: library)
    monkeypatch.setattr(
        "oracle.books.session.BookmarkStore",
        lambda: BookmarkStore(db_path=tmp_path / "books.db"),
    )
    session = ReaderSession(tts=_FakeTTS())
    assert session.book_count() == 2
    assert session.current_book() is None

    book = session.find_book("moby dick")
    assert book is not None
    assert session.start(book)
    assert session.current_book().id == book.id

    assert session.toggle_pause() is True
    assert session.is_paused
    assert session.toggle_pause() is False
    assert session.next_chapter() is True
    assert session.has_bookmark(book.id)
    session.stop()


def test_count_and_sample_authors(library):
    assert library.count_books() == 2
    authors = library.sample_authors(10)
    assert set(authors) == {"Herman Melville", "Arthur Conan Doyle"}
