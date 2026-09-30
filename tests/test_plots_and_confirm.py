"""Plot summaries for canonical works, and the book-confirmation parser."""

from __future__ import annotations

import pytest

from oracle.books.session import parse_confirmation
from oracle.rag import plots


@pytest.mark.parametrize(
    "question,title",
    [
        ("What happens at the end of A Tale of Two Cities?", "A Tale of Two Cities"),
        ("Summarize the plot of Treasure Island.", "Treasure Island"),
        ("Who were the three musketeers?", "The Three Musketeers"),
        ("Who was the captain of the Pequod in Moby Dick?", "Moby-Dick"),
        ("What is Frankenstein's monster made of?", "Frankenstein"),
        ("Who is Sherlock Holmes?", "The Adventures of Sherlock Holmes"),
        ("How does The Hound of the Baskervilles end?", "The Hound of the Baskervilles"),
        ("What is the novel Emma about?", "Emma"),
    ],
)
def test_match_canonical(question, title):
    e = plots.match(question)
    assert e is not None and e["title"] == title


@pytest.mark.parametrize(
    "question",
    [
        "How do I treat a second-degree burn?",
        "Tell me about the blue whale.",
        "What causes metamorphosis in butterflies?",
        "How do I treat an Achilles tendon injury?",
        "What is the population of Rochester?",
        "Who was Nikola Tesla?",
    ],
)
def test_no_match_on_everyday_questions(question):
    assert plots.match(question) is None


def test_every_entry_is_well_formed():
    entries = plots._entries()
    assert len(entries) >= 60
    for e in entries:
        assert e["title"] and e["author"] and len(e["summary"].split()) >= 50
        r = plots.as_result(e)
        assert r["source"] == "summary" and r["text"] == e["summary"]


@pytest.mark.parametrize(
    "answer,verdict,rest",
    [
        ("Yes.", "yes", ""),
        ("yeah that's it", "yes", ""),
        ("No.", "no", ""),
        ("Nope", "no", ""),
        ("No, the one by Thoreau.", "correction", "by Thoreau"),
        ("no I meant the Homer one", "correction", "the Homer one"),
        ("I meant Little Women by Alcott", "correction", "Little Women by Alcott"),
        ("", "unclear", ""),
        ("what's the weather", "unclear", "what's the weather"),
    ],
)
def test_parse_confirmation(answer, verdict, rest):
    assert parse_confirmation(answer) == (verdict, rest)
