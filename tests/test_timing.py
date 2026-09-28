"""Tests for the per-turn stage timer."""

from __future__ import annotations

import asyncio
import json

import pytest

from oracle import llm, timing
from oracle.llm import stream_chat


def test_marks_are_deltas_and_ttfa_is_from_speech_end(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])

    t = timing.TurnTimer("t")
    clock[0] += 2.0
    t.mark("record")  # speech ended at 102
    clock[0] += 0.5
    t.mark("stt")
    clock[0] += 3.0
    t.mark("first_audio")
    clock[0] += 1.0

    s = t.summary()
    assert s["record"] == pytest.approx(2.0)
    assert s["stt"] == pytest.approx(0.5)
    assert s["first_audio"] == pytest.approx(3.0)
    assert s["ttfa"] == pytest.approx(3.5)
    assert s["total"] == pytest.approx(6.5)


def test_mark_once_keeps_first(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    t = timing.TurnTimer()
    clock[0] = 1.0
    t.mark_once("first_audio")
    clock[0] = 5.0
    t.mark_once("first_audio")
    assert t.summary()["first_audio"] == pytest.approx(1.0)


def test_speech_ended_without_record(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    t = timing.TurnTimer()
    clock[0] = 1.0
    t.speech_ended()
    clock[0] = 4.0
    t.mark("first_audio")
    assert t.summary()["ttfa"] == pytest.approx(3.0)


def test_get_or_start_ignores_finished(monkeypatch):
    timing.clear()
    a = timing.start("a")
    assert timing.get_or_start("b") is a
    a.finish()
    b = timing.get_or_start("b")
    assert b is not a and b.label == "b"
    timing.clear()
    assert timing.current() is None


def test_finish_emits_activity(monkeypatch):
    seen: list[tuple[str, dict]] = []
    from oracle import activity

    monkeypatch.setattr(activity, "emit", lambda kind, **f: seen.append((kind, f)))
    t = timing.TurnTimer("q")
    t.mark("record")
    t.set("prefill", 1.234)
    t.note(prompt_tokens=42)
    t.finish()
    assert seen and seen[0][0] == "timing"
    fields = seen[0][1]
    assert fields["label"] == "q"
    assert fields["prefill"] == 1.234
    assert fields["prompt_tokens"] == 42
    assert "total" in fields


def test_context_survives_to_thread():
    async def main():
        timing.clear()
        t = timing.start("x")
        seen = await asyncio.to_thread(timing.current)
        timing.clear()
        return t, seen

    t, seen = asyncio.run(main())
    assert seen is t


class _Resp:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class _Client:
    def __init__(self, lines):
        self._lines = lines
        self.is_closed = False

    def stream(self, method, url, json=None):
        return _Resp(self._lines)


@pytest.mark.asyncio
async def test_stream_chat_records_prefill_and_first_token(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hi"}, "done": False}),
        json.dumps(
            {
                "done": True,
                "prompt_eval_count": 1500,
                "prompt_eval_duration": 2_500_000_000,
                "eval_count": 40,
                "eval_duration": 2_000_000_000,
            }
        ),
    ]
    monkeypatch.setattr(llm, "_get_client", lambda: _Client(lines))
    timing.clear()
    t = timing.start("llm")
    tokens = [tok async for tok in stream_chat([{"role": "user", "content": "hi"}])]
    assert tokens == ["Hi"]
    s = t.summary()
    assert "first_token" in s
    assert s["prefill"] == pytest.approx(2.5)
    assert s["prompt_tokens"] == 1500
    assert s["reply_tokens"] == 40
    assert s["decode_tps"] == pytest.approx(20.0)
    timing.clear()
