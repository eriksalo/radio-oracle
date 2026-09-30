"""Rank book search hits so the famous work wins over its look-alikes.

FTS5 ``rank`` favours short titles that merely contain the words, so a
request for *Little Women* opened "Two Little Women", *Peter Pan* opened
"The Peter Pan Alphabet", *Walden* opened Walter Walden's "Voodoo Gold
Trail" and *Sherlock Holmes* opened "A doctor enjoys Sherlock Holmes"
(quality eval, 2026-09-30: 11 of 79 famous requests opened the wrong
book). This module scores each candidate against the request:

* an exact normalised title, or the title up to its subtitle separator
  (``Moby-Dick; or, The Whale``), wins outright;
* words the title has *before* the first requested word cost a lot
  (``Two`` Little Women, ``A doctor enjoys`` Sherlock Holmes), words
  after it a little (Peter Pan ``Alphabet``), a subtitle after ``;`` ``:``
  ``(`` or ``, or`` nothing;
* when the request names an author, books *by* that author beat books
  *about* them (``Dickens-Land``, ``Mark Twain's Speeches``), and among
  an author's books a canonical work beats an index, a speeches
  collection or a volume fragment;
* a small hand-kept table of canonical works breaks ties in favour of
  the book people mean.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol


class _BookLike(Protocol):
    title: str
    author: str
    total_paragraphs: int


_STOP = {
    "the",
    "a",
    "an",
    "of",
    "by",
    "and",
    "book",
    "me",
    "to",
    "read",
    "please",
    "some",
    "something",
    "story",
    "novel",
    "us",
    "from",
}

# Words in a title that mark it as *about* a work or author rather than
# the work itself, or as a fragment / apparatus rather than a book to read.
_META_WORDS = {
    "index",
    "speeches",
    "letters",
    "correspondence",
    "notes",
    "essays",
    "criticism",
    "study",
    "studies",
    "companion",
    "guide",
    "reader",
    "selections",
    "extracts",
    "bibliography",
    "biography",
    "life",
    "lives",
    "land",
    "alphabet",
    "abridged",
    "juvenile",
    "primer",
    "enjoys",
    "volume",
    "vol",
    "part",
    "excerpt",
    "critical",
    "handbook",
}

# Canonical works: request words → (title words that must appear, author
# surname). Kept small and obvious; it only breaks ties among real hits.
CANONICAL: dict[str, tuple[str, str]] = {
    "moby dick": ("moby dick", "melville"),
    "sherlock holmes": ("adventures of sherlock holmes", "doyle"),
    "tom sawyer": ("adventures of tom sawyer", "twain"),
    "huckleberry finn": ("adventures of huckleberry finn", "twain"),
    "war and peace": ("war and peace", "tolstoy"),
    "walden": ("walden", "thoreau"),
    "little women": ("little women", "alcott"),
    "peter pan": ("peter pan", "barrie"),
    "emma": ("emma", "austen"),
    "prince": ("prince", "machiavelli"),
    "odyssey": ("odyssey", "homer"),
    "iliad": ("iliad", "homer"),
    "republic": ("republic", "plato"),
    "art of war": ("art of war", "sun"),
    "meditations": ("meditations", "aurelius"),
    "origin of species": ("origin of species", "darwin"),
    "jekyll and hyde": ("jekyll and mr hyde", "stevenson"),
    "jekyll": ("jekyll and mr hyde", "stevenson"),
    "dracula": ("dracula", "stoker"),
    "frankenstein": ("frankenstein", "shelley"),
    "treasure island": ("treasure island", "stevenson"),
    "kidnapped": ("kidnapped", "stevenson"),
    "pride and prejudice": ("pride and prejudice", "austen"),
    "jane eyre": ("jane eyre", "bronte"),
    "wuthering heights": ("wuthering heights", "bronte"),
    "great expectations": ("great expectations", "dickens"),
    "tale of two cities": ("tale of two cities", "dickens"),
    "oliver twist": ("oliver twist", "dickens"),
    "david copperfield": ("david copperfield", "dickens"),
    "bleak house": ("bleak house", "dickens"),
    "christmas carol": ("christmas carol", "dickens"),
    "don quixote": ("don quixote", "cervantes"),
    "les miserables": ("les miserables", "hugo"),
    "count of monte cristo": ("count of monte cristo", "dumas"),
    "three musketeers": ("three musketeers", "dumas"),
    "time machine": ("time machine", "wells"),
    "war of the worlds": ("war of the worlds", "wells"),
    "invisible man": ("invisible man", "wells"),
    "twenty thousand leagues": ("twenty thousand leagues", "verne"),
    "around the world in eighty days": ("around the world in eighty days", "verne"),
    "anne of green gables": ("anne of green gables", "montgomery"),
    "secret garden": ("secret garden", "burnett"),
    "wind in the willows": ("wind in the willows", "grahame"),
    "jungle book": ("jungle book", "kipling"),
    "call of the wild": ("call of the wild", "london"),
    "white fang": ("white fang", "london"),
    "heart of darkness": ("heart of darkness", "conrad"),
    "crime and punishment": ("crime and punishment", "dostoyevsky"),
    "brothers karamazov": ("brothers karamazov", "dostoyevsky"),
    "anna karenina": ("anna karenina", "tolstoy"),
    "paradise lost": ("paradise lost", "milton"),
    "robinson crusoe": ("robinson crusoe", "defoe"),
    "gulliver": ("gulliver", "swift"),
    "scarlet letter": ("scarlet letter", "hawthorne"),
    "leaves of grass": ("leaves of grass", "whitman"),
    "persuasion": ("persuasion", "austen"),
    "sense and sensibility": ("sense and sensibility", "austen"),
    "northanger abbey": ("northanger abbey", "austen"),
    "dorian gray": ("picture of dorian gray", "wilde"),
    "wizard of oz": ("wonderful wizard of oz", "baum"),
    "alice in wonderland": ("alice", "carroll"),
    "black beauty": ("black beauty", "sewell"),
    "hamlet": ("hamlet", "shakespeare"),
    "macbeth": ("macbeth", "shakespeare"),
    "romeo and juliet": ("romeo and juliet", "shakespeare"),
    "common sense": ("common sense", "paine"),
    "federalist": ("federalist", "hamilton"),
    "autobiography of benjamin franklin": ("autobiography of benjamin franklin", "franklin"),
    "frederick douglass": ("narrative of the life of frederick douglass", "douglass"),
    "benjamin franklin": ("autobiography of benjamin franklin", "franklin"),
    "great gatsby": ("great gatsby", "fitzgerald"),
    "ulysses": ("ulysses", "joyce"),
    "metamorphosis": ("metamorphosis", "kafka"),
    "divine comedy": ("divine comedy", "dante"),
    "canterbury tales": ("canterbury tales", "chaucer"),
    "beowulf": ("beowulf", ""),
    "aeneid": ("aeneid", "virgil"),
    "grimm": ("grimm", "grimm"),
    "aesop": ("aesop", "aesop"),
    "winnie the pooh": ("winnie the pooh", "milne"),
    "phantom of the opera": ("phantom of the opera", "leroux"),
    "hound of the baskervilles": ("hound of the baskervilles", "doyle"),
    "study in scarlet": ("study in scarlet", "doyle"),
    "scarlet pimpernel": ("scarlet pimpernel", "orczy"),
    "princess of mars": ("princess of mars", "burroughs"),
    "tarzan": ("tarzan of the apes", "burroughs"),
    "thirty nine steps": ("thirty nine steps", "buchan"),
    "siddhartha": ("siddhartha", "hesse"),
    "ethan frome": ("ethan frome", "wharton"),
    "utopia": ("utopia", "more"),
    "wealth of nations": ("wealth of nations", "smith"),
    "communist manifesto": ("communist", "marx"),
    "souls of black folk": ("souls of black folk", "du bois"),
    "up from slavery": ("up from slavery", "washington"),
    "swiss family robinson": ("swiss family robinson", "wyss"),
    "king james bible": ("king james", ""),
    "koran": ("koran", ""),
}

# Best-known novel per prolific author, for bare "read me some Dickens".
AUTHOR_FLAGSHIP: dict[str, str] = {
    "twain": "adventures of tom sawyer",
    "dickens": "tale of two cities",
    "austen": "pride and prejudice",
    "london": "call of the wild",
    "wells": "time machine",
    "verne": "twenty thousand leagues",
    "kipling": "jungle book",
    "stevenson": "treasure island",
    "doyle": "adventures of sherlock holmes",
    "tolstoy": "anna karenina",
    "dostoyevsky": "crime and punishment",
    "melville": "moby dick",
    "hawthorne": "scarlet letter",
    "whitman": "leaves of grass",
    "shakespeare": "hamlet",
    "poe": "raven",
    "hardy": "tess of the d urbervilles",
    "eliot": "middlemarch",
    "conrad": "heart of darkness",
    "dumas": "three musketeers",
    "hugo": "les miserables",
    "wilde": "picture of dorian gray",
    "bronte": "jane eyre",
    "shelley": "frankenstein",
    "stoker": "dracula",
    "swift": "gulliver",
    "defoe": "robinson crusoe",
    "alcott": "little women",
    "carroll": "alice",
    "baum": "wonderful wizard of oz",
    "burroughs": "princess of mars",
    "montgomery": "anne of green gables",
    "burnett": "secret garden",
    "homer": "odyssey",
    "plato": "republic",
    "thoreau": "walden",
    "darwin": "origin of species",
    "franklin": "autobiography of benjamin franklin",
}


def flagship_for_author(query: str, books: list) -> tuple[str, str] | None:
    """(surname, flagship title words) when *query* is an author's name
    that some candidate carries — the book people mean by "some Dickens"."""
    q_words = query_words(query)
    if not q_words or canonical_for(query) is not None:
        return None
    for b in books:
        words = normalize(b.author).split()
        if words and all(w in words for w in q_words):
            surname = words[-1]
            flagship = AUTHOR_FLAGSHIP.get(surname)
            return (surname, flagship) if flagship else (surname, "")
    return None


_SUBTITLE_RE = re.compile(r"\s*(?:;|:|\(|\[|,\s*or\b|\bor,).*$", re.IGNORECASE)
_SUBTITLE_SPLIT_RE = re.compile(r"\s*(?:;|:|\(|\[|,\s*or\b|\bor,)\s*", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that may stand in front of the requested title without making it
# a different book: "The Adventures of Sherlock Holmes", "The Strange
# Case of Dr Jekyll", "The Complete Works of…". "Two" Little Women and
# "A doctor enjoys" Sherlock Holmes are not in here.
_ALLOWED_LEADING = {
    "the", "a", "an", "of", "adventures", "adventure", "complete", "works", "collected",
    "selected", "tale", "tales", "story", "stories", "strange", "case", "return", "memoirs",
    "further", "new", "history", "life", "narrative", "book", "annotated", "illustrated",
}  # fmt: skip
_ARTICLES = {"the", "a", "an"}


def normalize(text: str) -> str:
    """Lowercase, ASCII-fold, punctuation to spaces, collapse whitespace."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("&", " and ").replace("'", "").replace("’", "")
    return " ".join(_WORD_RE.findall(t))


def query_words(query: str) -> list[str]:
    """The words of a request that carry meaning ("Read me Moby Dick" → moby dick)."""
    return [w for w in normalize(query).split() if w not in _STOP]


def main_title(title: str) -> str:
    """The title before its subtitle separator, normalised."""
    return normalize(_SUBTITLE_RE.sub("", title or ""))


def title_segments(title: str) -> list[list[str]]:
    """Main title and each subtitle as word lists ("Moby-Dick; or, The
    Whale" → [[moby, dick], [the, whale]]). A request may name either."""
    parts = [normalize(p) for p in _SUBTITLE_SPLIT_RE.split(title or "")]
    segs = [p.split() for p in parts if p]
    return segs or [[]]


def _surname(author: str) -> str:
    words = normalize(author).split()
    return words[-1] if words else ""


def _placement(q_words: list[str], seg: list[str]) -> tuple[int, int, int, int] | None:
    """(leading, trailing, hard_leading, soft_leading) of *q_words* inside
    *seg*, or None when a word is missing. Hard leading words are ones
    that make it another book ("Two", "A doctor enjoys"); soft ones are
    connectives ("The Adventures of") that only cost a little."""
    if not q_words or not all(w in seg for w in q_words):
        return None
    first = min(seg.index(w) for w in q_words)
    last = max(seg.index(w) for w in q_words)
    lead = seg[:first]
    hard = sum(1 for w in lead if w not in _ALLOWED_LEADING)
    soft = sum(1 for w in lead if w in _ALLOWED_LEADING and w not in _ARTICLES)
    return first, len(seg) - 1 - last, hard, soft


_ALLOWED_TRAILING = {"complete", "illustrated", "annotated", "unabridged", "abridged", "edition"}


# Keys as a request normalises ("War and Peace" → "war peace").
_CANONICAL_NORM: dict[str, tuple[str, str]] = {
    " ".join(query_words(k)): v for k, v in CANONICAL.items()
}


def canonical_for(query: str) -> tuple[str, str] | None:
    """The canonical (title words, author surname) the request means, if known."""
    q_words = query_words(query)
    q = " ".join(q_words)
    canon = _CANONICAL_NORM.get(q)
    if canon is None and len(q_words) >= 2:
        # "read the meditations of marcus aurelius" → key "meditations"
        for key, val in _CANONICAL_NORM.items():
            if all(w in q_words for w in key.split()) and val[1] and val[1] in q_words:
                return val
    return canon


@dataclass(frozen=True)
class Scored:
    score: float
    reasons: tuple[str, ...]


def score_book(
    query: str,
    book: _BookLike,
    *,
    fts_position: int = 0,
    author_mode: bool | None = None,
    flagship: tuple[str, str] | None = None,
) -> Scored:
    """Score one candidate for *query*. Higher is better.

    *author_mode* says the request is an author's name (some candidate's
    author matches every request word); ``rank`` works it out across the
    candidate set, a single book cannot.
    """
    q_words = query_words(query)
    title_n = normalize(book.title)
    main_n = main_title(book.title)
    author_n = normalize(book.author)
    author_words = author_n.split()
    reasons: list[str] = []
    score = -fts_position * 0.5  # keep FTS order as the faintest tie-break

    if not q_words:
        return Scored(score, ("empty query",))

    in_author = [w for w in q_words if w in author_words]
    q_title = [w for w in q_words if w not in author_words] or q_words
    all_author = len(in_author) == len(q_words)
    if author_mode is None:
        author_mode = all_author

    # --- title shape -----------------------------------------------------
    # In author mode the request *is* the author's name: a title that
    # repeats it ("Mark Twain's Speeches", "The Watsons: By Jane Austen")
    # earns nothing for that.
    segs = title_segments(book.title)
    best: tuple[int, int, int, int] | None = None
    if not (author_mode and all_author):
        for seg in segs:
            pl = _placement(q_title, seg)
            if pl is not None and (
                best is None or pl[2] * 2 + pl[3] + pl[1] < best[2] * 2 + best[3] + best[1]
            ):
                best = pl
        q_title_s = " ".join(q_title)
        if main_n == q_title_s or any(" ".join(seg) == q_title_s for seg in segs):
            score += 100
            reasons.append("exact title")
        elif best is not None:
            score += 55
            reasons.append("all words in title")
            _leading, trailing, hard, soft = best
            if hard:
                score -= 14 * hard
                reasons.append(f"{hard} leading extra word(s)")
            if soft:
                score -= 5 * soft
            if trailing:
                score -= 5 * trailing
                reasons.append(f"{trailing} trailing extra word(s)")
        else:
            hit = sum(1 for w in q_title if w in title_n.split()) / len(q_title)
            score += 25 * hit

    # --- author ------------------------------------------------------------
    if in_author:
        score += 15 * len(in_author)
        reasons.append("request names the author")
    if author_mode:
        # "read me some Dickens": books *by* them, not about them. A
        # flagship title with a blank author field still counts as theirs
        # (the library has "Pride and Prejudice" with no author).
        surname, flag_title = flagship or (_surname(book.author), "")
        is_flagship = (
            bool(flag_title) and flag_title in title_n and (not author_n or surname in author_words)
        )
        if all_author or is_flagship:
            score += 60
            if is_flagship:
                score += 80
                reasons.append("author's flagship work")
        else:
            score -= 60
            reasons.append("about the author, not by them")

    # --- meta / apparatus titles -------------------------------------------
    meta = [w for w in title_n.split() if w in _META_WORDS and w not in q_words]
    if meta:
        score -= 10 * min(len(meta), 3)
        reasons.append("apparatus title: " + ", ".join(meta[:3]))
    if re.search(r"\bvol(?:ume)?\b|\b\d+\s*\(of\s*\d+\)|\bpart\s+\d", title_n):
        score -= 8
        reasons.append("volume fragment")

    # --- canonical table ---------------------------------------------------
    canon = canonical_for(query)
    if canon is not None:
        want_title, want_author = canon
        if want_title in title_n and (not want_author or want_author in author_n or not author_n):
            score += 80
            reasons.append("canonical work")
        elif want_title in title_n:
            score += 20
            reasons.append("canonical title, other author")
        elif want_author and want_author in author_n:
            score += 10

    # A blank author is common in this library; a named one is a small plus.
    if author_n:
        score += 1
    # Prefer the fuller text when everything else ties (novel over story).
    score += min(getattr(book, "total_paragraphs", 0) or 0, 4000) / 4000.0

    return Scored(score, tuple(reasons))


def rank(query: str, books: list, limit: int = 20) -> list:
    """Order *books* (FTS/LIKE candidates in their original order) for *query*."""
    q_words = query_words(query)
    author_mode = (
        bool(q_words)
        and canonical_for(query) is None
        and any(all(w in normalize(b.author).split() for w in q_words) for b in books)
    )
    flagship = flagship_for_author(query, books) if author_mode else None
    scored = [
        (score_book(query, b, fts_position=i, author_mode=author_mode, flagship=flagship), b)
        for i, b in enumerate(books)
    ]
    scored.sort(key=lambda sb: -sb[0].score)
    return [b for _, b in scored[:limit]]


def is_confident_match(query: str, book: _BookLike) -> bool:
    """Would a listener agree this is the book they asked for, without
    being asked to confirm?

    Every meaningful word of the request is in the title or author; the
    title words the request names sit at the front of the title or of a
    subtitle, behind nothing but articles and connectives ("The
    Adventures of Sherlock Holmes" yes, "Two Little Women" and "A doctor
    enjoys Sherlock Holmes" no); at most one extra word follows them
    ("Kidnapped (Illustrated)" yes, "The Peter Pan Alphabet" no). A bare
    canonical title ("Walden") must be that work, not another author who
    happens to share the name.
    """
    q_words = query_words(query)
    if not q_words:
        return False
    title_n = normalize(book.title)
    author_words = set(normalize(book.author).split())
    canon = canonical_for(query)
    if canon is not None and canon[0] not in title_n:
        return False
    q_title = [w for w in q_words if w not in author_words]
    if not q_title:
        # Pure author request ("something by Twain"): any book of theirs.
        return True
    for seg in title_segments(book.title):
        pl = _placement(q_title, seg)
        if pl is None:
            continue
        _leading, trailing, hard, _soft = pl
        if hard == 0 and (trailing == 0 or (trailing == 1 and seg[-1] in _ALLOWED_TRAILING)):
            return True
    return False
