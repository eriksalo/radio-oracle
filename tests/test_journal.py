"""Activity journal: durable events, the deterministic recent-activity
block, the summarizer's activity log, and the profile v2 reset."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from oracle import activity
from oracle.memory import journal as journal_mod
from oracle.memory.context import ContextBuilder
from oracle.memory.journal import Journal, _ago
from oracle.memory.store import ConversationStore


def _stamp(j: Journal, days_ago: float, kind: str, session: str | None = None, **data):
    ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    with j._lock:
        j._conn.execute(
            "INSERT INTO events (ts, user, session_id, kind, data) VALUES (?, ?, ?, ?, ?)",
            (ts, j.user, session, kind, json.dumps(data)),
        )
        j._conn.commit()


def test_record_and_read(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_context("erik", "s1")
    j.record("asked", text="Who was Tesla?")
    j.record("playing", artist="Pink Floyd", title="Time")
    evs = j.events("erik")
    assert [e["kind"] for e in evs] == ["playing", "asked"]
    assert evs[1]["data"]["text"] == "Who was Tesla?" and evs[1]["session_id"] == "s1"
    assert j.events("someone-else") == []


def test_recent_summary_excludes_current_session_but_keeps_book(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_context("erik", "now")
    _stamp(
        j,
        1.0,
        "book",
        "old",
        event="stopped",
        book="Moby-Dick",
        author="Melville",
        status="Moby-Dick, chapter 3 of 135: The Spouter-Inn.",
    )
    _stamp(j, 1.1, "playing", "old", artist="Pink Floyd", title="Time")
    _stamp(j, 1.2, "playing", "old", artist="Pink Floyd", title="Money")
    _stamp(j, 2.0, "playing", "old", artist="Koko Taylor", title="I'm a Woman")
    _stamp(j, 1.3, "music_request", "old", query="pink floyd")
    _stamp(j, 1.4, "asked", "old", text="Who was Nikola Tesla?")
    _stamp(j, 1.5, "asked", "old", text="Where did he die?")
    _stamp(j, 0.0, "asked", "now", text="What causes the northern lights?")  # current session
    _stamp(j, 0.0, "playing", "now", artist="Miles Davis", title="So What")

    text = j.recent_summary("erik", exclude_session="now")
    assert (
        "Current book: Moby-Dick, chapter 3 of 135: The Spouter-Inn. (stopped at this yesterday)"
        in text
    )
    assert "Pink Floyd (2×), Koko Taylor" in text
    assert "Miles Davis" not in text  # current session: already in the history
    assert "Music asked for by name: pink floyd." in text
    assert "“Who was Nikola Tesla?”; “Where did he die?”" in text
    assert "northern lights" not in text


def test_recent_summary_empty_when_nothing_logged(tmp_path):
    assert Journal(tmp_path / "j.db").recent_summary("erik") == ""


def test_finished_book_line(tmp_path):
    j = Journal(tmp_path / "j.db")
    _stamp(j, 3.0, "book", "old", event="finished", book="Moby-Dick", status="x")
    assert j.recent_summary("erik").startswith("Finished reading Moby-Dick 3 days ago.")


def test_session_activity_text(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.set_context("erik", "s")
    j.record(
        "book", event="started", book="Moby-Dick", status="Moby-Dick, chapter 1 of 135: Loomings."
    )
    j.record("music_request", query="jazz")
    j.record("playing", artist="Miles Davis", title="So What")
    j.record("asked", text="Who was Tesla?")
    text = j.session_activity_text("s")
    assert text.splitlines() == [
        "book started: Moby-Dick, chapter 1 of 135: Loomings.",
        "asked for music: jazz",
        "asked: Who was Tesla?",
        "music played: Miles Davis — So What",
    ]


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (30, "moment"),
        (600, "10 minutes ago"),
        (86400, "yesterday"),
        (3 * 86400, "3 days ago"),
        (15 * 86400, "2 weeks ago"),
    ],
)
def test_ago(seconds, expected):
    now = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    assert _ago((now - timedelta(seconds=seconds)).isoformat(), now) == expected


def test_activity_emit_records_durable_kinds(tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_ACTIVITY_FILE", str(tmp_path / "act.jsonl"))
    activity.emit("asked", text="Why is the sky blue?")
    activity.emit("phase", phase="radio")  # transient only
    j = journal_mod.get_journal()
    kinds = [e["kind"] for e in j.events()]
    assert kinds == ["asked"]


@pytest.mark.asyncio
async def test_context_builder_inserts_recent_block_after_profile(tmp_path):
    store = ConversationStore(tmp_path / "oracle.db")
    store.update_profile("Music: Pink Floyd")
    s = store.new_session()
    store.add_message(s, "user", "hi")
    j = journal_mod.get_journal()
    _stamp(
        j,
        1.0,
        "book",
        "old",
        event="stopped",
        book="Moby-Dick",
        status="Moby-Dick, chapter 2 of 135: The Carpet-Bag.",
    )
    ctx = ContextBuilder(store, s)
    msgs = await ctx.build("PERSONA", "", user_text="hi")
    assert [m["role"] for m in msgs] == ["system", "system", "system", "user"]
    assert "Pink Floyd" in msgs[1]["content"]
    assert msgs[2]["content"].startswith("What you remember doing with Erik")
    assert "Current book: Moby-Dick, chapter 2 of 135" in msgs[2]["content"]
    assert j.session_id == s and j.user == settings.default_user
    store.close()


def test_profile_v2_reset_once(tmp_path):
    """A pre-v2 database: the drifted single-row profile is wiped once."""
    db = tmp_path / "oracle.db"
    store = ConversationStore(db)
    store._conn.execute(
        "INSERT INTO profile (id, content, updated_at) VALUES (1, ?, 'x')",
        ("Name: Unknown. Interests: Montenegro.",),
    )
    store._conn.execute("DELETE FROM meta")
    store._conn.commit()
    store.close()
    store2 = ConversationStore(db)
    assert store2.get_profile() is None  # wiped once
    store2.update_profile("Music: Pink Floyd")
    store2.close()
    store3 = ConversationStore(db)
    assert store3.get_profile() == "Music: Pink Floyd"  # not wiped again
    store3.close()


def test_legacy_v2_profile_row_moves_to_default_user(tmp_path):
    """A v2 single-row profile (written before per-user profiles) carries
    over into the default user's row."""
    db = tmp_path / "oracle.db"
    store = ConversationStore(db)
    store._conn.execute(
        "INSERT INTO profile (id, content, updated_at) VALUES (1, 'Music: jazz', 'x')"
    )
    store._conn.commit()
    store.close()
    store2 = ConversationStore(db)
    assert store2.get_profile(settings.default_user) == "Music: jazz"
    assert store2.get_profile("guest") is None
    assert store2._conn.execute("SELECT COUNT(*) FROM profile").fetchone()[0] == 0
    store2.close()


@pytest.mark.asyncio
async def test_finalize_session_passes_activity(tmp_path, monkeypatch):
    from oracle.memory import context as ctx_mod

    seen = {}

    async def fake_summarize(messages, activity=""):
        seen["activity"] = activity
        return "summary"

    async def fake_fold(existing, new):
        return "profile"

    monkeypatch.setattr(ctx_mod, "summarize_conversation", fake_summarize)
    monkeypatch.setattr(ctx_mod, "fold_into_profile", fake_fold)
    store = ConversationStore(tmp_path / "oracle.db")
    s = store.new_session()
    j = journal_mod.get_journal()
    j.set_context("erik", s)
    j.record(
        "book", event="started", book="Moby-Dick", status="Moby-Dick, chapter 1 of 135: Loomings."
    )
    # No conversation messages at all: a reading-only session still gets summarized.
    await ctx_mod.finalize_session(store, s)
    assert "book started: Moby-Dick" in seen["activity"]
    assert store.get_summary(s) == "summary"
    store.close()
