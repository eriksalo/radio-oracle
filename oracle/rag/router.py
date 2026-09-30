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
        r"(?:acoustic|folk|rock|jazz|blues|country|metal|punk|reggae|"
        r"hip[- ]?hop|classical|electronic) (?:music|songs?|tunes?)"
        r")\b",
        flags=re.IGNORECASE,
    ),
}

# A safe default ordering when nothing matches — encyclopedic first since
# it's the broadest signal for "who/when/what" questions, then practical
# corpora, then literary.
_DEFAULT_ORDER: tuple[str, ...] = (
    "wikipedia",
    "wikimed",
    "ifixit",
    "wikibooks",
    "gutenberg",
    "crashcourse",
    "music",
)


# ---------------------------------------------------------------------------
# Question type → per-collection distance bias.
#
# In snappy mode every collection is queried and the hits are merged by
# raw distance. Gutenberg is 10 M chunks of century-old prose and always
# has *something* close, so it showed up in the top-5 of all 100 eval
# answers (2026-09-30) and supplied 1900s medicine and farming advice.
# The bias below is added to a hit's distance before the merge (the
# relevance gate still sees the raw distance): a positive number pushes a
# collection back, a negative one pulls it forward. Real hits sit at
# 0.10–0.17, junk at 0.38+, so 0.06 is "lose a close race", 0.12 is
# "only if nothing else applies".
# ---------------------------------------------------------------------------

QuestionType = str  # "medical" | "howto" | "literature" | "factual" | "general"

_MEDICAL_RE = re.compile(
    r"\b(?:medic(?:al|ine|ation)?|treat(?:ing|ment)?|symptom|disease|infect(?:ed|ion)|"
    r"first aid|wound|injur(?:y|ies|ed)|bleed(?:ing)?|fever|pain|prescription|dosage|"
    r"antibiotic|virus|bacteri(?:a|um)|fracture|broken (?:arm|leg|bone|wrist|ankle)|burn(?:s|ed)?|"
    r"cpr|poison(?:ing|ous)?|venom|allerg(?:y|ic)|surgery|diagnos|concussion|dehydrat|"
    r"hypothermia|frostbite|heat ?stroke|blister|splint|tourniquet|snake ?bite|sprain|"
    r"choking|seizure|stroke|diabet|asthma|pregnan|childbirth|tooth|dental|rash|vaccine|"
    r"immune|disinfect|sanitize|sterili[sz]e|dose|swelling|vomit|diarrh|nausea|dizz)"
    r"(?:s|es|e|ed|ing|ion|ions|is|y|ies|ea|ous)?\b",
    re.IGNORECASE,
)
_HOWTO_RE = re.compile(
    r"\b(?:how (?:do|can|should|would) (?:i|you|we)|how to|fix|repair|replace|build|make|"
    r"grow|plant|harvest|preserve|store|purify|filter|sharpen|start a fire|tune|"
    r"install|wire|solder|troubleshoot|maintain|clean|patch|mend|sew|cook|bake|brew|"
    r"distill|forge|weld|carve|knit|hunt|trap|fish|navigate|find north|tell (?:if|whether))\b",
    re.IGNORECASE,
)
_LITERATURE_RE = re.compile(
    r"\b(?:novel|poem|poetry|literature|literary|play|sonnet|quote|quotation|story|"
    r"chapter|verse|character|protagonist|plot|ending|ends|happens (?:in|at the end)|"
    r"who (?:wrote|is|was) .{0,40}\b(?:in|from) (?:the )?(?:book|novel|play|story)|"
    r"which (?:novel|book|play|poem)|opening line|first line|last line|"
    r"summari[sz]e|summary of|about the book|"
    r"shakespeare|dickens|twain|austen|melville|tolstoy|dostoevsky|homer|dumas|verne|"
    r"kipling|stevenson|thoreau|whitman|hawthorne|wilde|bronte|conrad|joyce|hemingway|"
    r"moby.?dick|sherlock|frankenstein|dracula|treasure island|tale of two cities|"
    r"pride and prejudice|jane eyre|huckleberry|tom sawyer|musketeers|monte cristo|"
    r"odyssey|iliad|hamlet|macbeth|walden|meditations|gatsby|dorian gray)\b",
    re.IGNORECASE,
)
_PLOT_RE = re.compile(
    r"\b(?:what happens|how does .{0,30} end|ending of|end of|plot|summari[sz]e|summary|"
    r"who (?:is|was|are|were) .{0,40}\b(?:in|from)\b|main characters?|protagonist|"
    r"what is .{0,40} about|whaling ship|the captain)\b",
    re.IGNORECASE,
)
_FACTUAL_RE = re.compile(
    r"^\s*(?:who|when|where|which|what year|how (?:many|much|far|tall|old|big|long))\b",
    re.IGNORECASE,
)

_BIAS: dict[QuestionType, dict[str, float]] = {
    "medical": {
        "wikimed": -0.04,
        "wikipedia": 0.0,
        "wikibooks": 0.02,
        "crashcourse": 0.02,
        "ifixit": 0.08,
        "gutenberg": 0.12,
    },
    "howto": {
        "ifixit": -0.03,
        "wikibooks": -0.03,
        "wikipedia": 0.01,
        "wikimed": 0.02,
        "crashcourse": 0.02,
        "gutenberg": 0.08,
    },
    "literature": {
        "gutenberg": -0.02,
        "wikipedia": -0.02,
        "wikibooks": 0.04,
        "crashcourse": 0.04,
        "wikimed": 0.08,
        "ifixit": 0.10,
    },
    "factual": {
        "wikipedia": -0.03,
        "wikibooks": 0.0,
        "crashcourse": 0.0,
        "wikimed": 0.02,
        "ifixit": 0.05,
        "gutenberg": 0.06,
    },
    "general": {
        "wikipedia": -0.02,
        "wikibooks": 0.0,
        "crashcourse": 0.0,
        "wikimed": 0.02,
        "ifixit": 0.04,
        "gutenberg": 0.06,
    },
}


def question_type(query: str) -> QuestionType:
    """Coarse type of the question, for collection weighting.

    Literature wins over how-to ("how does Moby Dick end"), medical over
    how-to ("how do I treat a burn"), how-to over factual ("how do I…").
    """
    if _LITERATURE_RE.search(query):
        return "literature"
    if _MEDICAL_RE.search(query):
        return "medical"
    if _HOWTO_RE.search(query):
        return "howto"
    if _FACTUAL_RE.search(query):
        return "factual"
    return "general"


def bias_for(query: str) -> dict[str, float]:
    """Per-collection distance bias for *query* (see module notes)."""
    return dict(_BIAS[question_type(query)])


def augment_query(query: str) -> str:
    """Steer the embedding toward the article on the work for plot
    questions: "What happens at the end of A Tale of Two Cities?" found a
    video-game article and a French play (2026-09-30). Appending the
    words a plot section would use pulls the novel's Wikipedia article
    and the book's own text forward. Other questions pass through."""
    if _LITERATURE_RE.search(query) and _PLOT_RE.search(query):
        return f"{query.rstrip()} (novel plot summary, characters, ending)"
    return query


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
