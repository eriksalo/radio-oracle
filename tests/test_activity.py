"""Tests for the cross-process activity feed."""

from __future__ import annotations

import pytest

from oracle import activity


@pytest.fixture(autouse=True)
def _tmp_activity(tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_ACTIVITY_FILE", str(tmp_path / "activity.jsonl"))
    monkeypatch.setattr(activity, "_next_id", None)


def test_emit_and_read_roundtrip():
    activity.emit("heard", text="play pink floyd")
    activity.emit("decided", action="play", query="pink floyd")
    events = activity.read_events()
    assert [e["kind"] for e in events] == ["heard", "decided"]
    assert events[0]["text"] == "play pink floyd"
    assert events[1]["id"] > events[0]["id"]


def test_read_after_id_is_incremental():
    for i in range(5):
        activity.emit("phase", phase=f"p{i}")
    all_events = activity.read_events()
    tail = activity.read_events(after=all_events[2]["id"])
    assert [e["phase"] for e in tail] == ["p3", "p4"]


def test_long_text_is_clipped():
    activity.emit("answered", text="x" * 1000)
    ev = activity.read_events()[-1]
    assert len(ev["text"]) <= activity._TEXT_LIMIT


def test_ids_continue_across_writer_restarts(monkeypatch):
    activity.emit("wake")
    first = activity.read_events()[-1]["id"]
    monkeypatch.setattr(activity, "_next_id", None)  # simulate process restart
    activity.emit("wake")
    assert activity.read_events()[-1]["id"] == first + 1


def test_truncation_keeps_tail(monkeypatch):
    monkeypatch.setattr(activity, "_MAX_BYTES", 2000)
    for i in range(200):
        activity.emit("phase", phase=f"p{i}")
    events = activity.read_events(limit=300)
    assert len(events) <= activity._KEEP_LINES
    assert events[-1]["phase"] == "p199"


def test_emit_never_raises(monkeypatch):
    monkeypatch.setenv("ORACLE_ACTIVITY_FILE", "/nonexistent-dir/x/y.jsonl")
    monkeypatch.setattr(activity, "_next_id", None)
    activity.emit("wake")  # must not raise
