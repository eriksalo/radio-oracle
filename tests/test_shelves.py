"""Shelves: classification, source weighting, spoken text, voice phrases,
and the shelf label on conversation threads."""

from __future__ import annotations

import numpy as np
import pytest

from oracle import commands, core
from oracle.memory import threads
from oracle.rag import shelves


@pytest.mark.parametrize(
    ("question", "shelf"),
    [
        ("How do I treat a second-degree burn?", "health"),
        ("How do I start a fire without matches?", "survival"),
        ("How do I make a sourdough starter?", "food"),
        ("Why does my well pump keep turning on and off?", "fixing"),
        ("What should the air pressure in the pressure tank be?", "fixing"),
        ("How does photosynthesis work?", "science"),
        ("Who was Napoleon Bonaparte?", "history"),
        ("What happens at the end of Moby Dick?", "stories"),
    ],
)
def test_keywords_put_a_question_on_its_shelf(question: str, shelf: str) -> None:
    assert shelves.classify(question) == shelf


@pytest.mark.parametrize("text", ["Good morning.", "What time is it?", "Thank you."])
def test_small_talk_has_no_shelf(text: str) -> None:
    assert shelves.classify(text) is None


def _fake_embed(text: str) -> np.ndarray:
    """One axis per shelf, picked by a marker word; un-normalised on purpose."""
    vec = np.full(len(shelves.KEYS), 0.01, dtype=np.float32)
    lower = text.lower()
    for i, shelf in enumerate(shelves.SHELVES):
        if any(e.lower() in lower for e in shelf.examples) or f"<{shelf.key}>" in lower:
            vec[i] = 5.0
    return vec


def test_clear_embedding_match_beats_a_lone_keyword() -> None:
    # "joints"/"saw" style false friends: the keyword says fixing.
    question = "My hands hurt after I use the saw <health>"
    assert shelves.classify(question) in ("fixing", "health")
    assert shelves.classify(question, _fake_embed) == "health"


def test_unclear_embedding_falls_back_to_keywords() -> None:
    flat = lambda _t: np.ones(len(shelves.KEYS), dtype=np.float32)  # noqa: E731
    assert shelves.classify("How do I fix a leaking faucet?", flat) == "fixing"
    assert shelves.classify("Good morning.", flat) is None


def test_ready_only_after_warm() -> None:
    fn = lambda t: _fake_embed(t)  # noqa: E731
    assert not shelves.ready(fn)
    assert shelves.warm(fn)
    assert shelves.ready(fn)
    assert not shelves.warm(lambda _t: None)  # no embedder resident


def test_every_shelf_weights_every_source() -> None:
    sources = {"wikipedia", "wikimed", "wikibooks", "ifixit", "crashcourse", "gutenberg"}
    for shelf in shelves.SHELVES:
        assert set(shelf.bias) == sources, shelf.key
    assert shelves.bias(None) is None
    assert min(shelves.bias("health"), key=shelves.bias("health").get) == "wikimed"
    assert min(shelves.bias("fixing"), key=shelves.bias("fixing").get) == "ifixit"


def test_named_shelf() -> None:
    assert shelves.named("what's on the first aid shelf") == "health"
    assert shelves.named("the gardening shelf") == "food"
    assert shelves.named("the plumbing shelf") is None


def test_spoken_texts() -> None:
    assert shelves.list_text().count(",") >= 6
    text = shelves.describe_text("fixing", ["the well pump"])
    assert text.startswith("The fixing and building shelf.")
    assert "We have talked about the well pump there." in text
    said = shelves.sources_text(
        "health", [("wikimed", "Burn"), ("wikimed", "Burn"), ("wikipedia", "First aid")]
    )
    assert said == (
        "That came from the health and first aid shelf: the WikiMed medical article on "
        "Burn, and the Wikipedia article on First aid."
    )
    assert "what I already know" in shelves.sources_text("health", [])
    spoken = shelves.sources_text(
        "food",
        [
            ("wikibooks", "Cookbook:Sourdough Starter (recipe)"),
            ("wikibooks", "Horticulture/Tomato"),
        ],
    )
    assert spoken.endswith(
        "the cookbook page on Sourdough Starter, and the textbook Horticulture, Tomato."
    )


# ------------------------------------------------------------- commands


@pytest.mark.parametrize(
    ("text", "action", "query"),
    [
        ("What shelves do you have?", "shelf_list", None),
        ("What can I ask you about?", "shelf_list", None),
        ("What's on the health shelf?", "shelf_describe", "health"),
        ("Tell me about the survival shelf", "shelf_describe", "survival"),
        ("What do you have on first aid?", "shelf_describe", "health"),
        ("What's on the plumbing shelf?", "shelf_list", None),
        ("Where did that come from?", "source_last", None),
        ("What's your source?", "source_last", None),
        ("What have we talked about on the health shelf?", "thread_list", None),
        ("What have we talked about in medicine?", "thread_list", None),
        ("What did we say about the well pump?", "thread_resume", "well pump"),
    ],
)
@pytest.mark.asyncio
async def test_shelf_phrases(text: str, action: str, query: str | None) -> None:
    assert await commands.classify(None, text) == (action, query)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "What do you know about the Roman Empire?",
        "How do I build a set of shelves?",
        "What do you have on Napoleon?",
    ],
)
def test_real_questions_are_not_shelf_commands(text: str) -> None:
    assert commands._keyword_match(text) not in ("shelf_list", "source_last")
    if "shelves" not in text:
        assert commands._keyword_match(text) != "shelf_describe"


# -------------------------------------------------------------- threads


def test_thread_carries_its_shelf_and_lists_by_it(tmp_path) -> None:
    store = threads.ThreadStore(tmp_path / "t.db")
    tracker = threads.ThreadTracker(store, None)
    a = tracker.file_turn(
        "erik", "s1", "Why does the pump cycle?", "A stuck switch.", shelf="fixing"
    )
    tracker.file_turn("erik", "s1", "How do I check it?", "With a gauge.", followup=True)
    tracker.leave()
    b = tracker.file_turn("erik", "s1", "How do I treat a burn?", "Cool water.", shelf=None)
    tracker.file_turn("erik", "s1", "And after that?", "A clean dressing.", True, "health")
    assert store.get(a).shelf == "fixing"
    assert store.get(b).shelf == "health"  # labelled by a later turn
    assert [t.id for t in store.recent("erik", shelf="fixing")] == [a]
    said = threads.list_text(
        store.recent("erik", shelf="health"), shelf_name="health and first aid"
    )
    assert said.startswith("On the health and first aid shelf we have talked about ")
    assert "shelf yet" in threads.list_text([], shelf_name="food and growing")
    store.close()


def test_old_thread_database_gains_the_shelf_column(tmp_path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE threads (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT NOT NULL, "
        "title TEXT, summary TEXT, status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, "
        "last_ts TEXT NOT NULL, last_session TEXT, turns INTEGER NOT NULL DEFAULT 0, "
        "summarized_turns INTEGER NOT NULL DEFAULT 0, first_question TEXT NOT NULL DEFAULT '', "
        "words TEXT NOT NULL DEFAULT '', vec_sum BLOB)"
    )
    con.execute(
        "INSERT INTO threads (user, created_at, last_ts, turns) VALUES ('erik', 'x', 'x', 2)"
    )
    con.commit()
    con.close()
    store = threads.ThreadStore(path)
    assert store.recent("erik")[0].shelf is None
    store.close()


# ----------------------------------------------------------------- turn


def _topic_embed(text: str) -> np.ndarray:
    """Pump talk on one axis, everything else on another."""
    pump = any(w in text.lower() for w in ("pump", "pressure", "tank"))
    return np.array([4.0, 0.2] if pump else [0.2, 4.0], dtype=np.float32)


@pytest.mark.asyncio
async def test_turn_shelf_and_thread_hint(tmp_path, monkeypatch) -> None:
    from oracle.memory.context import ContextBuilder
    from oracle.memory.store import ConversationStore

    store = ConversationStore(tmp_path / "o.db")
    ctx = ContextBuilder(store, store.new_session(), embed_fn=_topic_embed)
    assert ctx.threads is not None
    opener = "Why does my well pump keep turning on and off?"
    assert await core._turn_shelf(ctx, opener) == ("fixing", None)  # no thread yet
    ctx.threads.file_turn(ctx.user, None, opener, "A stuck switch.", shelf="fixing")
    # A pronoun follow-up: thread's shelf, and the LLM rewrite resolves it.
    assert await core._turn_shelf(ctx, "How do I check it?") == ("fixing", None)
    # Carries the thread on without a pronoun: the opener rides along.
    assert await core._turn_shelf(ctx, "What should the tank pressure be?") == ("fixing", opener)
    # Same shape and close by embedding, but it names its own subject.
    assert await core._turn_shelf(ctx, "Why does my sump pump hum?") == ("fixing", None)
    # A new topic: its own shelf, no hint.
    assert await core._turn_shelf(ctx, "Who was Napoleon Bonaparte?") == ("history", None)
    assert await core._turn_shelf(ctx, "What next?", sticky=True) == ("fixing", None)
    monkeypatch.setattr(core.settings, "shelves_enabled", False)
    assert await core._turn_shelf(ctx, "Who was Napoleon Bonaparte?") == (None, None)
    store.close()


def test_continuity_embedding_is_reused_for_filing(tmp_path) -> None:
    calls: list[str] = []

    def embed(text: str) -> np.ndarray:
        calls.append(text)
        return _topic_embed(text)

    store = threads.ThreadStore(tmp_path / "t.db")
    tracker = threads.ThreadTracker(store, embed)
    a = tracker.file_turn("erik", "s", "Why does the pump cycle?", "A switch.")
    calls.clear()
    assert tracker.continues("erik", "What should the tank pressure be?").id == a
    assert tracker.file_turn("erik", "s", "What should the tank pressure be?", "30 psi.") == a
    assert len(calls) == 1
    assert tracker.continues("erik", "Who was Napoleon?") is None
    assert tracker.continues("someone else", "What about the pump?") is None
    store.close()


def test_hint_rides_along_in_the_search_text(monkeypatch) -> None:
    seen: list[str] = []

    class _R:
        def query(self, text, mode="snappy", shelf=None):
            seen.append(text)
            return []

        def format_context(self, results):
            return ""

    monkeypatch.setattr(core, "_get_retriever", lambda: _R())
    core._try_rag_query("How often should I feed the starter?", "food", "How do I make sourdough?")
    core._try_rag_query("How do I make sourdough?", "food", "How do I make sourdough?")
    assert seen == [
        "How often should I feed the starter? (How do I make sourdough?)",
        "How do I make sourdough?",
    ]


def test_last_lookup_feeds_the_source_answer(monkeypatch) -> None:
    class _R:
        def query(self, text, mode="snappy", shelf=None):
            return [{"text": "x", "source": "ifixit", "metadata": {"title": "Well Pump Repair"}}]

        def format_context(self, results):
            return "ctx"

    monkeypatch.setattr(core, "_get_retriever", lambda: _R())
    assert core._try_rag_query("Why does my well pump cycle?", "fixing") == "ctx"
    assert commands._describe_last_source() == (
        "That came from the fixing and building shelf: the iFixit repair guide, Well Pump Repair."
    )
    monkeypatch.setattr(core, "_get_retriever", lambda: None)
    core._try_rag_query("Hello", None)
    assert "what I already know" in commands._describe_last_source()
