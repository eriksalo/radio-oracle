"""Query → collection-priority routing.

Rule-based for now (predictable, zero-cost). Returns the ordered list of
collections to query, prioritized for the user's intent. Unmatched
collections are returned at the tail so we never silently drop a corpus
that might have a useful hit — the router prefers, it doesn't filter.

Used in Tier-1 to query the top-priority collections first and stop early
once we have enough hits; in Tier-2 every collection is queried and the
ordering just affects pre-rerank pool ordering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Keyword patterns per collection. Earlier rules win; collections are
# ordered by how strongly the keyword signals a likely match.
_RULES: dict[str, re.Pattern[str]] = {
    "wikimed": re.compile(
        r"\b("
        r"medic(?:al|ine|ation)?|treat(?:ing|ment)?|symptom|disease|infection|"
        r"first aid|wound|injur(?:y|ies)|bleeding|fever|pain|prescription|"
        r"dosage|antibiotic|virus|bacteria|fracture|burn|cpr|emergency|"
        r"poison(?:ing)?|allergy|allergic|surgery|diagnos"
        r")\b",
        flags=re.IGNORECASE,
    ),
    "ifixit": re.compile(
        r"\b("
        r"how (?:do|can) i (?:fix|repair)|repair|broken|disassembl|replac(?:e|ing)\s+"
        r"(?:the\s+)?(?:battery|screen|fan|motor)|troubleshoot|diy|"
        r"tools? (?:needed|required)"
        r")\b",
        flags=re.IGNORECASE,
    ),
    "crashcourse": re.compile(
        r"\b("
        r"explain (?:like|to me)|in simple terms|what is\b.*\bexactly|"
        r"crash course|introduction to|basics of"
        r")\b",
        flags=re.IGNORECASE,
    ),
    "gutenberg": re.compile(
        r"\b("
        r"novel|poem|poetry|literature|literary|quote|quotation|story|"
        r"chapter|verse|character in\b|writer|author|book(?:s)? about\b|"
        r"shakespeare|dickens|twain|austen|melville"
        r")\b",
        flags=re.IGNORECASE,
    ),
    "wikipedia": re.compile(
        r"\b("
        r"who (?:is|was|are|were)|when (?:did|was|were)|where (?:is|was|did)|"
        r"history of|biography|founded|invented|discovered|capital of|"
        r"population of"
        r")\b",
        flags=re.IGNORECASE,
    ),
    "wikibooks": re.compile(
        r"\b("
        r"textbook|tutorial on|learn (?:to|about)|exercises?|study guide|"
        r"course on|introduction (?:to|on)"
        r")\b",
        flags=re.IGNORECASE,
    ),
    "music": re.compile(
        r"\b("
        # "play [me] [some|a|the] [<adj>] music/songs/tracks/tunes/album"
        r"play (?:me |us )?(?:some |a |an |the )?(?:\w+ )?(?:music|songs?|tracks?|tunes?|albums?)|"
        r"(?:music|songs?|tracks?|albums?|artists?) (?:like|by|from|similar to)|"
        r"songs? about|"
        r"what (?:music|songs|tracks|albums?|artists?) do (?:i|you|we) have|"
        r"queue (?:up )?(?:music|a song|some songs|the music)|"
        # explicit-genre fallback when paired with a play/listen verb earlier in the sentence
        r"(?:acoustic|folk|rock|jazz|blues|country|metal|punk|reggae|hip[- ]?hop|classical|electronic) (?:music|songs?|tunes?)"
        r")\b",
        flags=re.IGNORECASE,
    ),
}

# A safe default ordering when nothing matches — encyclopedic first since
# it's the broadest signal for "who/when/what" questions, then practical
# corpora, then literary.
_DEFAULT_ORDER: tuple[str, ...] = (
    "wikipedia", "wikimed", "ifixit", "wikibooks", "gutenberg", "crashcourse", "music"
)


@dataclass(frozen=True)
class RoutingResult:
    """Result of routing — ordered collections plus matched-rule names for log/debug."""

    order: list[str]
    matched: list[str]

    def __iter__(self):
        return iter(self.order)


def route(query: str, available: list[str] | None = None) -> RoutingResult:
    """Order the available collections by likely relevance to `query`.

    Matched collections come first (in the order rules fire), then the
    rest of `available` in the default order. Collections not in `available`
    are dropped. If `available` is None, every collection with a rule plus
    the default fallbacks is returned.
    """
    matched: list[str] = []
    for name, pattern in _RULES.items():
        if pattern.search(query):
            matched.append(name)

    if available is None:
        available_set = set(_DEFAULT_ORDER) | set(_RULES.keys())
    else:
        available_set = set(available)

    order: list[str] = []
    seen: set[str] = set()
    for name in matched:
        if name in available_set and name not in seen:
            order.append(name)
            seen.add(name)
    for name in _DEFAULT_ORDER:
        if name in available_set and name not in seen:
            order.append(name)
            seen.add(name)
    # Unknown collections (e.g. user added a new one) come last in availability order.
    if available is not None:
        for name in available:
            if name not in seen:
                order.append(name)
                seen.add(name)

    return RoutingResult(order=order, matched=matched)
