"""Hand-written plot summaries for the canonical works, injected ahead of
the retrieved chunks when a question names one of them.

The 4B model garbled plots even with the right Wikipedia article and the
book's own text retrieved ("both men survive" at the end of A Tale of Two
Cities; a volcano in Treasure Island — quality eval 2026-09-30). The
lead of a novel's article is reception and history, and chapter one of
the book is the opening, not the plot. A short, checked summary that
names the characters and the ending gives the model what it needs.

Matching is by title or character alias as whole words in the question
(or the rewritten follow-up). Aliases that are also ordinary words
("emma", "persuasion", "metamorphosis") only count when the question
reads as a literature question.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from oracle.books.ranking import normalize

_PATH = Path(__file__).with_name("plot_summaries.json")
_POSSESSIVE_RE = re.compile(r"['\u2019]s\b")


@lru_cache(maxsize=1)
def _entries() -> list[dict]:
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    for e in data:
        e["_aliases"] = sorted(
            {normalize(a) for a in e.get("aliases", [])} | {normalize(e["title"])},
            key=len,
            reverse=True,
        )
    return data


def match(question: str) -> dict | None:
    """The summary entry the question is about, or None. Longest alias wins."""
    from oracle.rag.router import question_type

    # "Frankenstein's monster" → "frankenstein monster", not "frankensteins".
    plain = _POSSESSIVE_RE.sub("", question)
    q = " " + normalize(plain) + " "
    if not q.strip():
        return None
    literary = question_type(question) == "literature"
    best: tuple[int, dict] | None = None
    for e in _entries():
        if e.get("ambiguous") and not literary:
            continue
        for alias in e["_aliases"]:
            if alias and f" {alias} " in q:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), e)
                break
    return best[1] if best else None


def as_result(entry: dict) -> dict:
    """The entry shaped like a retriever hit, for ``format_context``."""
    return {
        "text": entry["summary"],
        "source": "summary",
        "distance": 0.0,
        "metadata": {"title": f"{entry['title']}, by {entry['author']}"},
        "chunk_id": f"summary:{normalize(entry['title'])}",
    }
