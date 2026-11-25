"""Conversation threads: filing turns by topic, picking a thread back up
across sessions, the deterministic recap, and the voice commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from config.settings import settings
from oracle import commands
from oracle.memory import context as context_mod
from oracle.memory import threads
from oracle.memory.context import ContextBuilder, summarize_threads
from oracle.memory.store import ConversationStore
from oracle.memory.summarizer import parse_thread_label
from oracle.memory.threads import ThreadStore, ThreadTracker

# A toy embedder: one axis per topic, so same-topic questions are close
# and different topics are orthogonal.
_AXES = {"pump": 0, "tank": 0, "sourdough": 1, "starter": 1, "loaf": 1, "napoleon": 2}


def _embed(text: str) -> np.ndarray:
    v = np.full(4, 0.05, dtype=np.float32)
    for word, axis in _AXES.items():
        if word in text.lower():
            v[axis] += 1.0
    return v


def _tracker(tmp_path) -> ThreadTracker:
    return ThreadTracker(ThreadStore(tmp_path / "t.db"), _embed)


def _age(store: ThreadStore, thread_id: int, days: float) -> None:
    ts = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with store._lock:
        store._conn.execute("UPDATE threads SET last_ts = ? WHERE id = ?", (ts, thread_id))
        store._conn.commit()


def test_same_topic_joins_and_new_topic_opens(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "Often the tank.")
    b = t.file_turn("erik", "s1", "How do I make a sourdough starter?", "Flour and water.")
    c = t.file_turn("erik", "s2", "What pressure should the pump tank hold?", "Two below cut-in.")
    assert a == c and a != b
    assert t.store.get(a).turns == 2 and t.store.get(b).turns == 1
    assert [x["question"] for x in t.store.turns(a)] == [
        "Why does my well pump short cycle?",
        "What pressure should the pump tank hold?",
    ]


def test_followup_stays_on_the_current_thread(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Who was Napoleon?", "A French emperor.")
    # No topic word at all: only continuity can place it.
    b = t.file_turn("erik", "s1", "Where did he die?", "On Saint Helena.", followup=True)
    assert a == b and t.store.get(a).turns == 2


def test_threads_are_per_user(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    b = t.file_turn("anna", "s1", "How do I prime a well pump?", "Fill the casing.")
    assert a != b
    assert [x.id for x in t.store.recent("anna")] == [b]
    # A follow-up from another speaker does not land in erik's thread.
    c = t.file_turn("anna", "s1", "Why is that?", "Air in the line.", followup=True)
    assert c != a


def test_without_an_embedder_only_followups_join(tmp_path):
    t = ThreadTracker(ThreadStore(tmp_path / "t.db"), None)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    b = t.file_turn("erik", "s1", "And how do I fix it?", "Recharge it.", followup=True)
    c = t.file_turn("erik", "s1", "What pressure should the pump tank hold?", "Two below.")
    assert a == b and c != a


def test_resume_makes_the_next_question_stick(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    t.file_turn("erik", "s1", "How do I make a sourdough starter?", "Flour and water.")
    t.resume(a)
    # Names no topic and is not a pronoun follow-up — still the pump.
    assert t.file_turn("erik", "s2", "What should I check next?", "The air charge.") == a
    # After that one, topic decides again.
    assert t.file_turn("erik", "s2", "Tell me about Napoleon", "An emperor.") != a


def test_find_by_spoken_topic(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    b = t.file_turn("erik", "s1", "How do I make a sourdough starter?", "Flour and water.")
    assert t.find("erik", "the well pump") == a
    assert t.find("erik", "sourdough") == b
    assert t.find("erik", "beekeeping") is None
    assert t.find("anna", "the well pump") is None


def test_recap_is_built_from_the_stored_turns(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    t.file_turn("erik", "s1", "What pressure should the pump tank hold?", "Two below cut-in.")
    _age(t.store, a, 1)
    thread = t.store.get(a)
    text = threads.recap_text(thread, t.store.turns(a))
    assert text == (
        "Yesterday we were talking about your question, Why does my well pump short cycle. "
        "You last asked: What pressure should the pump tank hold?"
    )
    t.store.set_label(a, "the well pump", "Pump short-cycles; tank suspected.", 2)
    assert threads.recap_text(t.store.get(a), t.store.turns(a)).startswith(
        "Yesterday we were talking about the well pump. You last asked:"
    )


def test_list_text_and_done_status(tmp_path):
    t = _tracker(tmp_path)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    t.file_turn("erik", "s1", "And the pump tank pressure?", "Two below.")
    t.store.set_label(a, "the well pump", "s", 2)
    assert "the well pump, a little while ago" in threads.list_text(t.store.recent("erik"))
    assert threads.list_text([]) == "We haven't had a longer conversation yet."
    t.store.set_status(a, "done")
    assert t.store.recent("erik", status="open") == []
    # Asking about it again reopens it.
    t.file_turn("erik", "s2", "Is the pump tank waterlogged?", "Tap it.")
    assert t.store.get(a).status == "open"


def test_parse_thread_label():
    assert parse_thread_label("Title: The Well Pump\nSummary: Short cycling.\nTank next.") == (
        "the well pump",
        "Short cycling. Tank next.",
    )
    title, summary = parse_thread_label("Summary: only this")
    assert title is None and summary == "only this"
    # A title too long to say is refused; the summary is still kept.
    long_title = "Title: " + "word " * 12 + "\nSummary: ok"
    assert parse_thread_label(long_title) == (None, "ok")
    assert parse_thread_label("no structure at all") == (None, None)


@pytest.mark.asyncio
async def test_summarize_threads_labels_once_and_keeps_the_title(monkeypatch):
    store = threads.get_store()
    t = ThreadTracker(store, _embed)
    a = t.file_turn("erik", "s1", "Why does my well pump short cycle?", "The tank.")
    t.file_turn("erik", "s1", "What pressure should the pump tank hold?", "Two below.")
    single = t.file_turn("erik", "s1", "Who was Napoleon?", "An emperor.")
    replies = iter(["Title: the well pump\nSummary: first.", "Title: plumbing\nSummary: second."])
    seen: list[str] = []

    async def fake_chat(messages, **_kw):
        seen.append(messages[1]["content"])
        return next(replies)

    monkeypatch.setattr("oracle.memory.summarizer.chat", fake_chat)
    assert await summarize_threads() == 1  # the one-off question is left alone
    assert store.get(a).title == "the well pump" and store.get(single).title is None
    assert await summarize_threads() == 0  # nothing new
    t.file_turn("erik", "s2", "Is the pump tank waterlogged?", "Tap it.")
    assert await summarize_threads() == 1
    th = store.get(a)
    assert (th.title, th.summary, th.summarized_turns) == ("the well pump", "second.", 3)
    # The second pass saw the earlier summary and only the new exchange.
    assert "Summary so far: first." in seen[1] and "short cycle" not in seen[1]


@pytest.mark.asyncio
async def test_context_builder_resume_adds_block_before_history(monkeypatch):
    monkeypatch.setattr(context_mod, "_resident_embed", _embed)
    store = ConversationStore()
    first = ContextBuilder(store, store.new_session(), embed_fn=_embed)
    first.threads.file_turn("erik", "old", "Why does my well pump short cycle?", "The tank.")
    tid = first.threads.file_turn("erik", "old", "And the pump tank pressure?", "Two below.")

    sid = store.new_session()
    ctx = ContextBuilder(store, sid, embed_fn=_embed)
    before = await ctx.build("PERSONA", user_text="hello")
    thread, turns = ctx.resume_thread(tid)
    assert thread.id == tid and len(turns) == 2
    after = await ctx.build("PERSONA", user_text="hello")
    assert "picking an earlier conversation back up" not in before[0]["content"]
    system = after[0]["content"]
    assert system.startswith("PERSONA") and "They asked: And the pump tank pressure?" in system
    assert after[-1] == {"role": "user", "content": "hello"}
    # Stable from here on: the prefix cache sees the same system message.
    assert (await ctx.build("PERSONA", user_text="again"))[0]["content"] == system

    ctx.schedule_file_turn("What should I check next?", "The air charge.")
    await ctx.close()
    assert ctx.threads.store.get(tid).turns == 3
    # Someone else cannot resume it, and a user switch drops the block.
    ctx.set_user("anna")
    assert ctx.resume_thread(tid) is None
    assert "picking an earlier" not in (await ctx.build("PERSONA"))[0]["content"]
    store.close()


def test_threads_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "threads_enabled", False)
    store = ConversationStore()
    ctx = ContextBuilder(store, store.new_session())
    assert ctx.threads is None and ctx.resume_thread(1) is None
    store.close()


# ------------------------------------------------------------ commands


@pytest.mark.parametrize(
    ("text", "action", "query"),
    [
        ("Where did we leave off?", "thread_resume", None),
        ("Let's pick up where we left off", "thread_resume", None),
        ("What were we talking about?", "thread_resume", None),
        ("Go back to our conversation about the well pump", "thread_resume", "well pump"),
        ("Go back to the sourdough discussion", "thread_resume", "sourdough"),
        ("What did we say about the well pump?", "thread_resume", "well pump"),
        ("What have we talked about?", "thread_list", None),
        ("What did we talk about last week?", "thread_list", None),
        ("What were we talking about yesterday?", "thread_list", None),
        ("We're done with that", "thread_done", None),
        # Unchanged neighbours.
        ("Where were we?", "book_status", None),
        ("What am I reading?", "book_status", None),
        ("Back to the music", "music_on", None),
    ],
)
@pytest.mark.asyncio
async def test_recall_phrases_classify(text, action, query):
    assert await commands.classify(None, text) == (action, query)


class _VC:
    def __init__(self, ctx: ContextBuilder, session_id: str):
        self.ctx_builder = ctx
        self.session_id = session_id


@pytest.fixture
def spoken(monkeypatch) -> list[str]:
    said: list[str] = []
    monkeypatch.setattr(commands, "_speak", lambda vc, text, abort=None: said.append(text))

    async def heard(vc, onset_timeout):
        return "what should I check next"

    monkeypatch.setattr(commands, "_listen_once", heard)
    return said


@pytest.mark.asyncio
async def test_resume_command_recaps_then_listens(spoken):
    store = ConversationStore()
    old = ContextBuilder(store, "old-session", embed_fn=_embed)
    old.threads.file_turn("erik", "old-session", "Why does my well pump short cycle?", "Tank.")
    pump = old.threads.file_turn("erik", "old-session", "And the pump tank pressure?", "Two.")
    old.threads.file_turn("erik", "old-session", "Who was Napoleon?", "An emperor.")  # one-off
    _age(old.threads.store, pump, 3)

    sid = store.new_session()
    vc = _VC(ContextBuilder(store, sid, embed_fn=_embed), sid)
    # This session's own two-turn thread is not "where we left off".
    vc.ctx_builder.threads.file_turn("erik", sid, "How do I make a sourdough starter?", "Flour.")
    vc.ctx_builder.threads.file_turn("erik", sid, "How warm for the starter?", "Room temp.")

    assert await commands._resume_thread(vc, None, None) == "what should I check next"
    assert spoken == [
        "3 days ago we were talking about your question, Why does my well pump short cycle. "
        "You last asked: And the pump tank pressure? Go ahead."
    ]
    assert vc.ctx_builder.threads.current_id == pump

    spoken.clear()
    assert await commands._resume_thread(vc, "beekeeping", None) is None
    assert spoken == ["I couldn't find a conversation about beekeeping."]
    spoken.clear()
    assert await commands._resume_thread(vc, "sourdough", None) is not None
    assert "sourdough starter" in spoken[0]
    assert commands._close_thread(vc).startswith("All right, I'll set your question, How do I")
    assert commands._close_thread(vc) == "There's no conversation open right now."
    store.close()


@pytest.mark.asyncio
async def test_resume_with_nothing_to_go_back_to(spoken):
    store = ConversationStore()
    sid = store.new_session()
    vc = _VC(ContextBuilder(store, sid, embed_fn=_embed), sid)
    assert await commands._resume_thread(vc, None, None) is None
    assert spoken == ["I don't have an earlier conversation to go back to."]
    assert commands._describe_threads(vc) == "We haven't had a longer conversation yet."
    store.close()


def test_short_question_naming_a_topic_is_not_a_followup() -> None:
    from oracle.core import _is_followup

    for q in ("Who was Napoleon Bonaparte?", "What is photosynthesis?"):
        assert not _is_followup(q), q
    for q in ("Why is that?", "Where exactly?", "How long?", "How do I check it?"):
        assert _is_followup(q), q


def test_resumed_history_is_handed_out_once(tmp_path) -> None:
    from oracle.memory.context import ContextBuilder
    from oracle.memory.store import ConversationStore

    store = ConversationStore(tmp_path / "o.db")
    ctx = ContextBuilder(store, store.new_session(), embed_fn=_embed)
    assert ctx.threads is not None
    tid = ctx.threads.file_turn(ctx.user, None, "How do I feed a starter?", "Twice a day.")
    assert ctx.resume_thread(tid) is not None
    history = ctx.take_resumed_history()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "How do I feed a starter?"
    assert ctx.take_resumed_history() == []
    store.close()
