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
    assert h.spoken == [welcome.GREETING, welcome.OPTIONS, welcome.FALLBACK]
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
    assert out.dispatched is None and welcome.FALLBACK not in h.spoken


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


def test_describe_device_counts(monkeypatch):
    class _Lib:
        def count_books(self):
            return 60030

        def close(self):
            pass

    monkeypatch.setattr("oracle.books.library.Library", _Lib)

    class _Retriever:
        def collection_sizes(self):
            return {
                "wikipedia": 11_476_000,
                "gutenberg": 10_301_735,
                "ifixit": 181_502,
                "music": 4024,
            }

    monkeypatch.setattr("oracle.core._get_retriever", lambda: _Retriever())
    text = commands.describe_device(_Catalog())
    assert "built by Erik Salo" in text and "April 2026" in text
    assert "4,024 songs from 312 artists across 401 albums" in text
    assert "60,030 books" in text
    assert "about 11.5 million passages from Wikipedia" in text
    assert "about 10.3 million passages from the Project Gutenberg books" in text
    kb = text.split("The knowledge base")[1].split("Ask me anything")[0]
    assert "plus iFixit repair guides" in kb and "music" not in kb


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
