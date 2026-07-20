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
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert player.calls == [("next",)]


def test_do_action_next_album(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("next_album", None, player, None, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert player.calls == [("next_album",)]


def test_do_action_pause_does_not_resume(silent_speak):
    # Pause: wake handler already paused; dispatcher must signal "stay paused".
    player = _FakePlayer()
    out = commands._do_action("pause", None, player, None, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is False
    # Dispatcher shouldn't re-pause an already-paused player.
    assert player.calls == []


def test_do_action_resume_signals_resume(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("resume", None, player, None, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert player.calls == [("resume",)]


def test_do_action_stop_does_not_resume(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("stop", None, player, None, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is False
    assert player.calls == [("stop",)]


def test_do_action_play_hits_search_and_plays_first(silent_speak):
    player = _FakePlayer()
    track = _FakeTrack()
    catalog = _FakeCatalog(results=[track])
    out = commands._do_action("play", "Pink Floyd", player, catalog, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert catalog.queries == ["Pink Floyd"]
    # stop() called first (to clear current album), then play(track=track)
    assert player.calls == [("stop",), ("play", track)]
    assert silent_speak == ["<played>"]  # acked


def test_do_action_play_no_results_acks_and_stays(silent_speak):
    player = _FakePlayer()
    catalog = _FakeCatalog(results=[])
    out = commands._do_action("play", "nonexistent", player, catalog, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert player.calls == []
    assert silent_speak == ["<played>"]


def test_do_action_play_without_query_asks_back(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("play", None, player, _FakeCatalog([]), _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert player.calls == []
    assert silent_speak == ["<played>"]


def test_do_action_mode_transitions_return_correct_state(silent_speak):
    # mode_librarian is intercepted in dispatch (question flow) and never
    # reaches _do_action; channel switches do.
    player = _FakePlayer()
    rdr = commands._do_action("mode_reader", None, player, None, _FakeVC(), None)
    assert rdr.next_mode == "reader"
    assert rdr.resume_channel is False

    back = commands._do_action("music_on", None, player, None, _FakeVC(), None, context="book")
    assert back.next_mode == "radio"
    assert back.resume_channel is False

    # No transport actions for pure switches.
    assert player.calls == []


def test_do_action_none_is_silent_noop(silent_speak):
    player = _FakePlayer()
    out = commands._do_action("none", None, player, None, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert player.calls == []
    assert silent_speak == []


def test_do_action_no_player_speaks_unavailable(silent_speak):
    out = commands._do_action("next", None, None, None, _FakeVC(), None)
    assert out.next_mode == "radio"
    assert out.resume_channel is True
    assert silent_speak == ["<played>"]


def test_do_action_read_book_carries_query(silent_speak):
    out = commands._do_action("read_book", "moby dick", None, None, _FakeVC(), None)
    assert out.next_mode == "reader"
    assert out.resume_channel is False
    assert out.reader_query == "moby dick"
    # Reader announces the title itself — no generic ack here.
    assert silent_speak == []


def test_do_action_read_book_without_query(silent_speak):
    out = commands._do_action("read_book", None, None, None, _FakeVC(), None)
    assert out.next_mode == "reader"
    assert out.reader_query is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("why is the sky blue", True),
        ("Why is the weather different in Spain compared to America?", True),
        ("how do I splint a broken arm", True),
        ("tell me about the roman empire", True),
        ("what's the capital of France", True),
        ("Place a big flight.", False),
        ("play pink floyd", False),
        ("umm never mind", False),
    ],
)
def test_looks_like_question(text, expected):
    assert commands._looks_like_question(text) is expected


def test_keywords_win_over_question_detection():
    # "what music do we have" starts interrogative but keyword rules run
    # first in dispatch; here just confirm the keyword table's own hits
    # aren't question-shaped regressions.
    assert commands._keyword_match("next song") == "next"
    assert commands._looks_like_question("next song") is False


@pytest.mark.asyncio
async def test_question_turns_followup_window(monkeypatch):
    import numpy as np

    import oracle.core as core_mod
    from config.settings import settings

    turns: list[str] = []

    async def fake_voice_turn(vc, leds=None, should_abort=None, pre_text=None):
        turns.append(pre_text)
        return True

    recordings = [
        np.ones(100, dtype=np.float32),  # follow-up spoken
        np.array([], dtype=np.float32),  # then silence → window closes
    ]

    def fake_record(**kwargs):
        assert kwargs.get("onset_timeout") == settings.followup_window_s
        return recordings.pop(0)

    class _FakeSTT:
        def load(self):
            pass

        def transcribe(self, audio, sample_rate=None):
            return "and what about his brother?"

    vc = _FakeVC()
    vc.stt_fast = _FakeSTT()
    monkeypatch.setattr(core_mod, "voice_turn", fake_voice_turn)
    monkeypatch.setattr(commands, "record_until_silence", fake_record)
    monkeypatch.setattr(commands, "_play_thinking_ack", lambda vc, should_abort=None: None)

    await commands._question_turns(vc, "who was tesla?", None, None)

    assert turns == ["who was tesla?", "and what about his brother?"]
    assert recordings == []  # both window recordings consumed


@pytest.mark.parametrize(
    "text,expected",
    [
        ("play music", "music_on"),
        ("back to the music", "music_on"),
        ("put on some music", "music_on"),
        ("continue my book", "mode_reader"),
        ("read my book", "mode_reader"),
        ("next chapter", "next_chapter"),
        ("pause", "pause"),
        ("stop.", "pause"),
        ("quiet", "pause"),
        ("resume", "resume"),
        ("what music do we have", "list_music"),
        ("what books do you have by mark twain", "list_books"),
    ],
)
def test_channel_and_exploration_keywords(text, expected):
    assert commands._keyword_match(text) == expected


def test_extract_qualifier():
    assert commands._extract_qualifier("what books do you have by Mark Twain?") == "Mark Twain"
    assert commands._extract_qualifier("what music is there like jazz") == "jazz"
    assert commands._extract_qualifier("what music do we have") is None


class _FakeReader:
    def __init__(self):
        self.calls = []

    def next_chapter(self):
        self.calls.append("next_chapter")
        return True


def test_book_context_transport(silent_speak):
    rdr = _FakeReader()
    out = commands._do_action("next", None, None, None, _FakeVC(), None, context="book", reader=rdr)
    assert out.next_mode == "reader" and out.resume_channel is True
    assert rdr.calls == ["next_chapter"]

    out = commands._do_action(
        "pause", None, None, None, _FakeVC(), None, context="book", reader=rdr
    )
    assert out.next_mode == "reader" and out.resume_channel is False


def test_play_mid_book_switches_channel(silent_speak):
    out = commands._do_action("play", "pink floyd", None, None, _FakeVC(), None, context="book")
    assert out.next_mode == "radio"
    assert out.resume_channel is False
    assert out.play_query == "pink floyd"


class _StatsCatalog:
    def stats(self):
        return {"tracks": 4024, "artists": 1797, "albums": 1681, "hours": 296.8}

    def sample_artists(self, n=6):
        return ["Pink Floyd", "Blue Oyster Cult"]

    def search(self, q):
        class T:
            artist = "Pink Floyd"
            title = "Money"

        return [T(), T()] if q == "pink floyd" else []


def test_describe_music_summary_and_filtered():
    text = commands._describe_music(_StatsCatalog(), None)
    assert "4024 tracks" in text and "Pink Floyd" in text
    text = commands._describe_music(_StatsCatalog(), "pink floyd")
    assert "2 tracks match" in text
    text = commands._describe_music(_StatsCatalog(), "zzz")
    assert "Nothing" in text
