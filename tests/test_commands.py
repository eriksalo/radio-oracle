"""Tests for the radio-mode wake-word command dispatcher."""

from __future__ import annotations

import pytest

from oracle import commands


# ---------------------------------------------------------------- keywords

@pytest.mark.parametrize(
    "text,expected",
    [
        ("next song", "next"),
        ("Skip", "next"),
        ("skip this song", "next"),
        ("next track", "next"),
        ("next album", "next_album"),
        ("change album please", "next_album"),
        ("another album", "next_album"),
        ("pause music", "pause"),
        ("stop the music", "pause"),
        ("resume music", "resume"),
        ("I have a question", "mode_librarian"),
        ("I'd like to read a book", "mode_reader"),
        ("read a book", "mode_reader"),
    ],
)
def test_keyword_match_hits(text, expected):
    assert commands._keyword_match(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "play some jazz",
        "put on Pink Floyd",
        "what's the weather",
        "",
        "nonsense words here",
    ],
)
def test_keyword_match_misses(text):
    assert commands._keyword_match(text) is None


# ---------------------------------------------------------------- llm-json

@pytest.mark.asyncio
async def test_llm_intent_parses_clean_json(monkeypatch):
    async def fake_chat(messages, model=None):
        return '{"action":"play","query":"jazz"}'

    monkeypatch.setattr(commands, "chat", fake_chat)
    action, query = await commands._llm_intent("play some jazz")
    assert action == "play"
    assert query == "jazz"


@pytest.mark.asyncio
async def test_llm_intent_strips_code_fences(monkeypatch):
    async def fake_chat(messages, model=None):
        return '```json\n{"action":"next","query":null}\n```'

    monkeypatch.setattr(commands, "chat", fake_chat)
    action, query = await commands._llm_intent("skip this")
    assert action == "next"
    assert query is None


@pytest.mark.asyncio
async def test_llm_intent_finds_json_in_noisy_response(monkeypatch):
    async def fake_chat(messages, model=None):
        return 'Sure! {"action":"play","query":"Pink Floyd"} done.'

    monkeypatch.setattr(commands, "chat", fake_chat)
    action, query = await commands._llm_intent("play pink floyd")
    assert action == "play"
    assert query == "Pink Floyd"


@pytest.mark.asyncio
async def test_llm_intent_falls_back_to_none_on_garbage(monkeypatch):
    async def fake_chat(messages, model=None):
        return "I have no idea what you mean."

    monkeypatch.setattr(commands, "chat", fake_chat)
    action, query = await commands._llm_intent("blah")
    assert action == "none"
    assert query is None


@pytest.mark.asyncio
async def test_llm_intent_returns_none_on_chat_exception(monkeypatch):
    async def fake_chat(messages, model=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(commands, "chat", fake_chat)
    action, query = await commands._llm_intent("hi")
    assert action == "none"
    assert query is None


# ---------------------------------------------------------------- _do_action

class _FakePlayer:
    def __init__(self):
        self.calls = []

    def next(self):
        self.calls.append(("next",))

    def next_album(self):
        self.calls.append(("next_album",))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def stop(self):
        self.calls.append(("stop",))

    def play(self, track=None, continuous=True):
        self.calls.append(("play", track))


class _FakeCatalog:
    def __init__(self, results):
        self._results = results
        self.queries = []

    def search(self, q):
        self.queries.append(q)
        return self._results


class _FakeTrack:
    def __init__(self, artist="Pink Floyd", album="Wish You Were Here", title="Have a Cigar"):
        self.artist = artist
        self.album = album
        self.title = title


class _FakeTTS:
    sample_rate = 24000

    def synthesize(self, text):
        return None  # play_audio will receive None; we monkeypatch it.


class _FakeVC:
    def __init__(self):
        self.tts = _FakeTTS()


@pytest.fixture
def silent_speak(monkeypatch):
    """Replace play_audio so _speak() doesn't hit real audio devices."""
    spoken: list[str] = []

    def fake_play(audio, sample_rate=None, should_abort=None):
        spoken.append("<played>")

    monkeypatch.setattr(commands, "play_audio", fake_play)
    return spoken


def test_do_action_next(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("next", None, player, None, _FakeVC(), None)
    assert out == "radio"
    assert player.calls == [("next",)]


def test_do_action_next_album(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("next_album", None, player, None, _FakeVC(), None)
    assert out == "radio"
    assert player.calls == [("next_album",)]


def test_do_action_pause_resume_stop(silent_speak):
    player = _FakePlayer()
    for a in ("pause", "resume", "stop"):
        commands._do_action(a, None, player, None, _FakeVC(), None)
    assert player.calls == [("pause",), ("resume",), ("stop",)]


def test_do_action_play_hits_search_and_plays_first(silent_speak):
    player = _FakePlayer()
    track = _FakeTrack()
    catalog = _FakeCatalog(results=[track])
    out = commands._do_action("play", "Pink Floyd", player, catalog, _FakeVC(), None)
    assert out == "radio"
    assert catalog.queries == ["Pink Floyd"]
    # stop() called first (to clear current album), then play(track=track)
    assert player.calls == [("stop",), ("play", track)]
    assert silent_speak == ["<played>"]  # acked


def test_do_action_play_no_results_acks_and_stays(silent_speak):
    player = _FakePlayer()
    catalog = _FakeCatalog(results=[])
    out = commands._do_action("play", "nonexistent", player, catalog, _FakeVC(), None)
    assert out == "radio"
    assert player.calls == []
    assert silent_speak == ["<played>"]


def test_do_action_play_without_query_asks_back(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("play", None, player, _FakeCatalog([]), _FakeVC(), None)
    assert out == "radio"
    assert player.calls == []
    assert silent_speak == ["<played>"]


def test_do_action_mode_transitions_return_correct_state(silent_speak):
    player = _FakePlayer()
    assert commands._do_action("mode_librarian", None, player, None, _FakeVC(), None) == "librarian"
    assert commands._do_action("mode_reader", None, player, None, _FakeVC(), None) == "reader"
    # No player actions for mode transitions.
    assert player.calls == []


def test_do_action_none_is_silent_noop(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("none", None, player, None, _FakeVC(), None)
    assert out == "radio"
    assert player.calls == []
    assert silent_speak == []


def test_do_action_no_player_speaks_unavailable(silent_speak):
    out = commands._do_action("next", None, None, None, _FakeVC(), None)
    assert out == "radio"
    assert silent_speak == ["<played>"]
