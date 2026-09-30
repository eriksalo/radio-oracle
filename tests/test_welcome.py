"""Power-on welcome flow (oracle/welcome.py) with injected speak / chime /
listen / dispatch, plus the new keywords and the device summary."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import settings
from oracle import commands, welcome


class _Harness:
    def __init__(self, answers: list[str]):
        self.answers = answers  # what listen returns per call ("" = silence)
        self.spoken: list[str] = []
        self.chimes = 0
        self.waits: list[float] = []
        self.dispatched: list[str] = []

    async def speak(self, text):
        self.spoken.append(text)

    async def chime(self):
        self.chimes += 1

    async def listen(self, wait):
        self.waits.append(wait)
        text = self.answers.pop(0) if self.answers else ""
        return np.ones(16000, dtype=np.float32), text

    async def dispatch(self, text, audio):
        self.dispatched.append(text)
        return commands.DispatchResult("radio")


@pytest.fixture(autouse=True)
def _waits(monkeypatch):
    monkeypatch.setattr(settings, "welcome_first_wait", 7.0)
    monkeypatch.setattr(settings, "welcome_second_wait", 5.0)
    monkeypatch.setattr(settings, "welcome_third_wait", 5.0)


@pytest.mark.asyncio
async def test_input_right_after_chime_dispatches_without_prompts():
    h = _Harness(["play some jazz"])
    out = await welcome.run_welcome(h.speak, h.chime, h.listen, h.dispatch)
    assert h.chimes == 1 and h.spoken == [] and h.waits == [7.0]
    assert h.dispatched == ["play some jazz"] and out.heard == "play some jazz"


@pytest.mark.asyncio
async def test_silence_then_greeting_then_options_then_music():
    h = _Harness(["", "", ""])
    out = await welcome.run_welcome(h.speak, h.chime, h.listen, h.dispatch)
    assert h.spoken == [welcome.GREETING, welcome.OPTIONS]  # then silence: blink and wait
    assert h.waits == [7.0, 5.0, 5.0] and h.chimes == 3
    assert out.heard is None and out.dispatched is None and out.steps == 2


@pytest.mark.asyncio
async def test_answer_after_greeting():
    h = _Harness(["", "tell me about this device"])
    out = await welcome.run_welcome(h.speak, h.chime, h.listen, h.dispatch)
    assert h.spoken == [welcome.GREETING]
    assert h.dispatched == ["tell me about this device"] and out.steps == 1


@pytest.mark.asyncio
async def test_power_off_aborts_quietly():
    h = _Harness(["", ""])
    flips = iter([False, False, True, True])
    out = await welcome.run_welcome(
        h.speak, h.chime, h.listen, h.dispatch, should_abort=lambda: next(flips)
    )
    assert h.spoken == [welcome.GREETING]  # second step started, then power went off
    assert out.dispatched is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I'd like to ask a question", "mode_librarian"),
        ("ask questions", "mode_librarian"),
        ("question mode", "mode_librarian"),
        ("explore the music library", "list_music"),
        ("play music", "music_on"),
        ("explore the books", "list_books"),
        ("read a book", "mode_reader"),
        ("tell me about this device", "about_device"),
        ("who made you?", "about_device"),
        ("what are you", "about_device"),
    ],
)
def test_menu_keywords(text, expected):
    assert commands._keyword_match(text) == expected


class _Catalog:
    def stats(self):
        return {"tracks": 4024, "artists": 312, "albums": 401, "hours": 250.0}


def test_describe_device_is_the_fixed_passage():
    text = commands.describe_device(_Catalog())
    assert text == commands.DEVICE_DESCRIPTION
    assert text.startswith("Hello. I'm glad you found me. I am the Librarian.")
    assert "built by Erik Salo, in Boulder, Colorado, in 2026" in text
    assert text.endswith("I'm listening.")
    # No live counts any more: nothing in it depends on the catalog.
    assert commands.describe_device(None) == text


@pytest.mark.asyncio
async def test_dispatch_accepts_pre_text(monkeypatch):
    """The welcome hands an already-transcribed utterance to the dispatcher:
    no recording, straight to classification and action."""
    spoken = []
    monkeypatch.setattr(commands, "_speak", lambda vc, text, should_abort=None: spoken.append(text))
    monkeypatch.setattr(
        commands, "listen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not record"))
    )

    async def no_observe(*a, **k):
        return "erik"

    monkeypatch.setattr("oracle.speaker.observe", no_observe)
    monkeypatch.setattr("oracle.speaker.maybe_ask", no_observe)
    monkeypatch.setattr(commands, "describe_device", lambda catalog: "I'm the Librarian.")

    class _VC:
        stt_fast = object()
        speaker = None
        speaker_id = None

    out = await commands.dispatch_radio_command(
        None, None, _VC(), pre_text="tell me about this device", pre_audio=None
    )
    assert out.next_mode == "radio" and spoken == ["I'm the Librarian."]


@pytest.mark.parametrize(
    "action,starts",
    [
        ("music_on", True),
        ("play", True),
        ("next", True),
        ("resume", True),
        ("pause", False),
        ("stop", False),
        ("none", False),
    ],
)
def test_dispatch_result_marks_explicit_music_requests(monkeypatch, action, starts):
    monkeypatch.setattr(commands, "_speak", lambda *a, **k: None)
    monkeypatch.setattr(commands, "_play_query", lambda p, c, q, raw="": "Pink Floyd")

    class P:
        def next(self):
            pass

        def next_album(self):
            pass

        def resume(self):
            pass

        def pause(self):
            pass

        def stop(self):
            pass

    class VC:
        tts = None

    out = commands._do_action(action, "pink floyd", P(), object(), VC(), None)
    assert out.starts_music is starts
