"""Phase A reader fixes: front matter is skipped on a fresh start, chapter
references resolve by number/word/roman/title, status text, and the
dispatcher's chapter actions in both channels."""

from __future__ import annotations

import pytest

from oracle import commands
from oracle.books.bookmarks import BookmarkStore
from oracle.books.library import (
    Library,
    heading_number,
    is_content_heading,
    parse_chapter_number,
)
from oracle.books.reader import Reader
from oracle.books.session import ReaderSession, _format_status

_GUTENBERG = """Title: Moby-Dick; or, The Whale
Author: Herman Melville

*** START OF THE PROJECT GUTENBERG EBOOK MOBY-DICK ***

[Transcriber's notes]

Thanks to the volunteers who proofread this text over many years, and to
the library that lent the 1851 first edition from which the plates were
scanned; obvious typographical errors have been corrected silently.

CONTENTS

ETYMOLOGY. EXTRACTS. CHAPTER 1. Loomings.

CHAPTER 1. Loomings.

Call me Ishmael. Some years ago, never mind how long precisely.

CHAPTER 2. The Carpet-Bag.

I stuffed a shirt or two into my old carpet-bag.

CHAPTER 3. The Spouter-Inn.

Entering that gable-ended Spouter-Inn, you found yourself in a wide, low room.

*** END OF THE PROJECT GUTENBERG EBOOK MOBY-DICK ***
"""


class _FakeTTS:
    sample_rate = 24000

    def synthesize(self, text):
        import numpy as np

        return np.zeros(10, dtype=np.float32)


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    from oracle import audio

    monkeypatch.setattr(audio, "play_audio", lambda *a, **k: None)
    books = tmp_path / "books"
    books.mkdir()
    (books / "moby.txt").write_text(_GUTENBERG)
    library = Library(db_path=tmp_path / "books.db")
    library.index_directory(books)
    return library


# ------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "spec,n",
    [
        ("1", 1),
        ("one", 1),
        ("first", 1),
        ("twelve", 12),
        ("twenty one", 21),
        ("XII", 12),
        ("iv", 4),
        ("the last", None),
        ("loomings", None),
    ],
)
def test_parse_chapter_number(spec, n):
    assert parse_chapter_number(spec) == n


def test_heading_number_and_content_detection():
    assert heading_number("CHAPTER 12. The Whiteness") == 12
    assert heading_number("Chapter XII") == 12
    assert heading_number("PART I") == 1
    assert heading_number("Preamble") is None
    assert is_content_heading("CHAPTER 1. Loomings.")
    assert not is_content_heading("Preamble")


# ------------------------------------------------------------ library


def test_index_files_front_matter_as_preamble(lib):
    book = lib.search("moby")[0]
    titles = lib.list_chapter_titles(book.id)
    assert titles[0] == (0, "Preamble")
    assert titles[1][1].startswith("CHAPTER 1")
    assert lib.first_content_chapter(book.id) == 1


def test_resolve_chapter_specs(lib):
    book = lib.search("moby")[0]
    r = lambda spec: lib.resolve_chapter(book.id, spec)  # noqa: E731
    assert r("one") == 1
    assert r("chapter one") == 1
    assert r("2") == 2
    assert r("the third") == 3
    assert r("III") == 3
    assert r("the last") == 3
    assert r("the beginning") == 1
    assert r("the preface") == 0
    assert r("front matter") == 0
    assert r("spouter") == 3  # title fragment
    assert r("ninety nine") is None
    assert r("xyzzy") is None


# ------------------------------------------------------------- reader


def test_fresh_start_skips_front_matter(lib, tmp_path):
    bm = BookmarkStore(db_path=tmp_path / "books.db")
    rdr = Reader(library=lib, bookmarks=bm, tts=_FakeTTS())
    book = lib.search("moby")[0]
    pos = rdr.start(book.id)
    assert pos.chapter_idx == 1 and rdr.started_fresh
    # The indexer keeps ". Loomings." as paragraph 0 (spoken as the
    # chapter's name), then the prose.
    assert rdr.read_paragraph().strip(". ") == "Loomings"
    assert rdr.read_paragraph().startswith("Call me Ishmael")
    # Resume comes back to the bookmark, not the front matter, and is not "fresh".
    rdr2 = Reader(library=lib, bookmarks=bm, tts=_FakeTTS())
    pos2 = rdr2.start(book.id)
    assert (pos2.chapter_idx, pos2.para_idx) == (1, 2) and not rdr2.started_fresh


def test_explicit_chapter_zero_reads_preface(lib, tmp_path):
    bm = BookmarkStore(db_path=tmp_path / "books.db")
    rdr = Reader(library=lib, bookmarks=bm, tts=_FakeTTS())
    book = lib.search("moby")[0]
    assert rdr.start(book.id, chapter_idx=0).chapter_idx == 0
    assert "Transcriber" in rdr.read_paragraph()


def test_goto_prev_status(lib, tmp_path):
    bm = BookmarkStore(db_path=tmp_path / "books.db")
    session = ReaderSession(tts=_FakeTTS())
    session._library, session._bookmarks = lib, bm
    session._reader = Reader(library=lib, bookmarks=bm, tts=_FakeTTS())
    book = lib.search("moby")[0]
    assert session.start(book)
    assert session.status_text() == "Moby-Dick; or, The Whale, chapter 1 of 3: Loomings."
    assert session.goto_chapter("three") == "CHAPTER 3: The Spouter-Inn"
    assert session.status_text() == "Moby-Dick; or, The Whale, chapter 3 of 3: The Spouter-Inn."
    assert session.prev_chapter() == "CHAPTER 2: The Carpet-Bag"
    assert session.goto_chapter("ninety") is None
    assert session.restart() == "CHAPTER 1: Loomings"
    assert session.goto_chapter("the preface") == "Preamble"
    assert session.status_text().endswith("the front matter.")
    assert bm.get(book.id).chapter_idx == 0


def test_format_status_strips_heading_prefix():
    st = {
        "book": "Moby-Dick",
        "author": "",
        "chapter_idx": 5,
        "chapter_number": 3,
        "chapter_total": 135,
        "chapter_title": "CHAPTER 3. The Spouter-Inn.",
        "paragraph": 0,
    }
    assert _format_status(st) == "Moby-Dick, chapter 3 of 135: The Spouter-Inn."


def test_is_confident_match():
    from oracle.books.library import Book

    moby = Book(
        id=1,
        title="Moby-Dick; or, The Whale",
        author="Herman Melville",
        path="",
        total_chapters=3,
        total_paragraphs=3,
    )
    assert ReaderSession.is_confident_match("moby dick", moby)
    assert ReaderSession.is_confident_match("read me the whale by melville", moby)
    assert not ReaderSession.is_confident_match("books on rock and roll", moby)


# --------------------------------------------------------- dispatcher


@pytest.mark.parametrize(
    "text,expected",
    [
        ("next slide", "next"),
        ("go to chapter three", "goto_chapter"),
        ("start with chapter one", "goto_chapter"),
        ("chapter twelve", "goto_chapter"),
        ("read me the preface", "goto_chapter"),
        ("go back a chapter", "prev_chapter"),
        ("previous chapter", "prev_chapter"),
        ("start the book over", "restart_book"),
        ("from the beginning", "restart_book"),
        ("what am I reading?", "book_status"),
        ("what chapter is this", "book_status"),
        ("next chapter", "next_chapter"),
    ],
)
def test_chapter_keywords(text, expected):
    assert commands._keyword_match(text) == expected


def test_extract_chapter_spec():
    assert commands._extract_chapter_spec("go to chapter twenty one") == "twenty one"
    assert commands._extract_chapter_spec("read the chapter called Loomings") == "Loomings"
    assert commands._extract_chapter_spec("read me the preface") == "preface"
    assert commands._extract_chapter_spec("next song") is None


class _FakeReader:
    def __init__(self):
        self.calls = []

    def goto_chapter(self, spec):
        self.calls.append(("goto", spec))
        return "CHAPTER 3. The Spouter-Inn." if spec != "ninety" else None

    def prev_chapter(self):
        self.calls.append(("prev",))
        return "CHAPTER 2. The Carpet-Bag."

    def restart(self):
        self.calls.append(("restart",))
        return "CHAPTER 1. Loomings."

    def status_text(self):
        return "Moby-Dick, chapter 3 of 135: The Spouter-Inn."

    def next_chapter(self):
        return True


class _VC:
    tts = _FakeTTS()


@pytest.fixture()
def spoken(monkeypatch):
    out = []
    monkeypatch.setattr(commands, "_speak", lambda vc, text, should_abort=None: out.append(text))
    return out


def test_book_context_chapter_actions(spoken):
    rdr = _FakeReader()
    kw = dict(player=None, catalog=None, vc=_VC(), should_abort=None, context="book", reader=rdr)
    assert commands._do_action("goto_chapter", "three", **kw).next_mode == "reader"
    assert rdr.calls[-1] == ("goto", "three")
    assert spoken[-1] == "CHAPTER 3. The Spouter-Inn"
    commands._do_action("goto_chapter", None, raw_text="go to chapter ninety", **kw)
    assert spoken[-1] == "I couldn't find chapter ninety."
    commands._do_action("prev_chapter", None, **kw)
    assert rdr.calls[-1] == ("prev",)
    commands._do_action("restart_book", None, **kw)
    assert spoken[-1].startswith("From the beginning.")
    commands._do_action("book_status", None, **kw)
    assert spoken[-1].startswith("Moby-Dick, chapter 3")


def test_music_context_chapter_words_without_a_book(spoken, monkeypatch):
    monkeypatch.setattr(commands, "_book_in_progress", lambda: False)

    class P:
        def __init__(self):
            self.nexts = 0

        def next(self):
            self.nexts += 1

    p = P()
    out = commands._do_action("next_chapter", None, p, None, _VC(), None)
    assert out.next_mode == "radio" and p.nexts == 1  # misheard "next song"
    out = commands._do_action("goto_chapter", "three", p, None, _VC(), None)
    assert out.next_mode == "radio" and "not reading" in spoken[-1]


def test_music_context_chapter_words_with_a_book(spoken, monkeypatch):
    monkeypatch.setattr(commands, "_book_in_progress", lambda: True)
    out = commands._do_action("goto_chapter", "three", None, None, _VC(), None)
    assert out.next_mode == "reader" and out.reader_chapter == "three"
    out = commands._do_action("next_chapter", None, None, None, _VC(), None)
    assert out.next_mode == "reader" and out.reader_chapter == "next"
