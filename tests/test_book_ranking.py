"""The famous work must beat its look-alikes (quality eval, 2026-09-30)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from oracle.books import ranking
from oracle.books.library import Library


@dataclass
class _B:
    title: str
    author: str = ""
    total_paragraphs: int = 1000
    id: int = 0


def _best(query: str, *books: _B) -> str:
    return ranking.rank(query, list(books))[0].title


_CASES = [
    ("Little Women", _B("Two Little Women"), _B("Little Women", "Louisa May Alcott")),
    ("Peter Pan", _B("The Peter Pan Alphabet"), _B("Peter Pan", "J. M. Barrie")),
    (
        "Sherlock Holmes",
        _B("A doctor enjoys Sherlock Holmes"),
        _B("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    ),
    (
        "Tom Sawyer",
        _B("Tom Sawyer, Detective", "Mark Twain"),
        _B("The Adventures of Tom Sawyer, Complete", "Mark Twain"),
    ),
    ("War and Peace", _B("Poems of Peace and War"), _B("War and Peace", "Leo Tolstoy")),
    (
        "Walden",
        _B("The Voodoo Gold Trail", "Walter Walden"),
        _B("Walden, and On The Duty Of Civil Disobedience", "Henry David Thoreau"),
    ),
    (
        "Emma by Jane Austen",
        _B("The Poems of Emma Lazarus, Volume 2", "Emma Lazarus"),
        _B("Emma", "Jane Austen"),
    ),
    (
        "The Prince by Machiavelli",
        _B("The Constant Prince"),
        _B("The Prince", "Nicolo Machiavelli"),
    ),
    ("The Odyssey", _B("A Martian Odyssey", "Stanley Weinbaum"), _B("The Odyssey", "Homer")),
    ("Moby Dick", _B("Moby Dick Junior"), _B("Moby-Dick; or, The Whale", "Herman Melville")),
    (
        "the whale by Melville",
        _B("The Whale Hunters"),
        _B("Moby-Dick; or, The Whale", "Herman Melville"),
    ),
]


@pytest.mark.parametrize("query,wrong,right", _CASES)
def test_famous_work_beats_lookalike(query, wrong, right):
    assert _best(query, wrong, right) == right.title
    assert _best(query, right, wrong) == right.title  # order-independent


def test_author_request_prefers_books_by_not_about():
    land = _B("Dickens-Land", "Robert Allbut")
    about = _B("Dickens", "John Morley")
    notes = _B("American Notes", "Charles Dickens", 800)
    novel = _B("A Tale of Two Cities", "Charles Dickens", 3000)
    assert _best("some Dickens", land, about, notes, novel) == novel.title
    speeches = _B("Mark Twain's Speeches", "Mark Twain", 900)
    sawyer = _B("The Adventures of Tom Sawyer", "Mark Twain", 2500)
    assert _best("something by Mark Twain", speeches, sawyer) == sawyer.title
    index = _B("Index for Works of Rudyard Kipling", "Rudyard Kipling", 50)
    jungle = _B("The Jungle Book", "Rudyard Kipling", 1500)
    assert _best("something by Kipling", index, jungle) == jungle.title
    # Flagship with a blank author field still counts as the author's.
    watsons = _B("The Watsons: By Jane Austen, Concluded by L. Oulton", "Jane Austen", 400)
    pride = _B("Pride and Prejudice", "", 3000)
    assert _best("Jane Austen", watsons, pride) == pride.title


def test_subtitles_and_editions_still_exact():
    assert ranking.main_title("Moby-Dick; or, The Whale") == "moby dick"
    assert ranking.main_title("Frankenstein; Or, The Modern Prometheus") == "frankenstein"
    assert ranking.main_title("Kidnapped (Illustrated)") == "kidnapped"
    assert ranking.main_title("Jane Eyre: An Autobiography") == "jane eyre"


@pytest.mark.parametrize(
    "query,book,ok",
    [
        ("Read me Little Women", _B("Two Little Women"), False),
        ("Read me Little Women", _B("Little Women", "Alcott"), True),
        ("Read Sherlock Holmes to me", _B("A doctor enjoys Sherlock Holmes"), False),
        ("Read Sherlock Holmes to me", _B("The Adventures of Sherlock Holmes", "Doyle"), True),
        ("Read me Peter Pan", _B("The Peter Pan Alphabet"), False),
        ("Read Kidnapped", _B("Kidnapped (Illustrated)"), True),
        ("Read Moby Dick", _B("Moby-Dick; or, The Whale", "Melville"), True),
        ("Read something by Mark Twain", _B("Roughing It", "Mark Twain"), True),
        ("Read Walden", _B("The Voodoo Gold Trail", "Walter Walden"), False),
        ("Read Frankenstein", _B("Frankenstein; Or, The Modern Prometheus"), True),
        ("read me the whale by melville", _B("Moby-Dick; or, The Whale", "Herman Melville"), True),
        (
            "Read Dr. Jekyll and Mr. Hyde",
            _B("The Strange Case of Dr. Jekyll and Mr. Hyde", "Stevenson"),
            True,
        ),
        ("Read Jekyll", _B("Jekyll-Hyde Planet", "Jack Lewis"), False),
        ("Read me a book by Jane Austen", _B("Pride and Prejudice", ""), True),
    ],
)
def test_confidence(query, book, ok):
    assert ranking.is_confident_match(query, book) is ok


_TXT = """Title: {title}
Author: {author}
*** START OF THE PROJECT GUTENBERG EBOOK X ***
CHAPTER I
Some text here.
*** END OF THE PROJECT GUTENBERG EBOOK X ***
"""


def test_library_search_uses_ranking_and_finds_canonical_author(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    for i, (t, a) in enumerate(
        [
            ("Two Little Women", ""),
            ("Little Women", "Louisa May Alcott"),
            ("The Voodoo Gold Trail", "Walter Walden"),
            ("Soil Culture", "J. H. Walden"),
            ("Walden, and On The Duty Of Civil Disobedience", "Henry David Thoreau"),
            ("A doctor enjoys Sherlock Holmes", ""),
            ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
        ]
    ):
        (books / f"b{i}.txt").write_text(_TXT.format(title=t, author=a))
    lib = Library(db_path=tmp_path / "books.db")
    lib.index_directory(books)
    try:
        assert lib.search("Little Women")[0].title == "Little Women"
        assert lib.search("Walden")[0].author == "Henry David Thoreau"
        assert lib.search("Sherlock Holmes")[0].title == "The Adventures of Sherlock Holmes"
        assert lib.search("nonexistent title") == []
    finally:
        lib.close()
