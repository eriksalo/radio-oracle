"""What a button press means in each state (Erik's table, 2026-11-23):

waiting : short = wake word, long = "about me"
talking : short = pause ("Paused") / carry on, long = stop
music   : short = next song, two short = replay, long = next album
book    : handled in the reader loop (pause/resume, long = waiting)
"""

from __future__ import annotations

import threading
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from oracle import audio
from oracle.app import OracleApp
from oracle.hardware.button import ButtonEvent


class _Player:
    def __init__(self, playing: bool = True, paused: bool = False) -> None:
        self.is_playing, self.is_paused = playing, paused
        self.calls: list[str] = []

    def next(self) -> None:
        self.calls.append("next")

    def next_album(self) -> None:
        self.calls.append("next_album")

    def replay(self) -> None:
        self.calls.append("replay")


class _App:
    _handle_buttons = OracleApp._handle_buttons
    _drain_events = OracleApp._drain_events
    _idle = OracleApp._idle
    _next_track = OracleApp._next_track
    _next_album = OracleApp._next_album
    _replay_track = OracleApp._replay_track
    _make_turn_abort = OracleApp._make_turn_abort
    _begin_turn = OracleApp._begin_turn
    _end_turn = OracleApp._end_turn
    _clear_speech_pause = OracleApp._clear_speech_pause

    def __init__(self, player: _Player | None, held: bool = False) -> None:
        self._state = "radio"
        self._music_held = held
        self._player = player
        self._pending_short_press = None
        self._double_press_window = 0.05
        self._about_pending = False
        self._led_before_pause = None
        self._in_turn = False
        self._barge_armed = False
        self._barge = threading.Event()
        self._wakeword = None
        self._wake_event = SimpleNamespace(
            fired=False, set=lambda: setattr(self._wake_event, "fired", True)
        )
        self.power = SimpleNamespace(is_on=True)
        self.button = SimpleNamespace(events=Queue(), pressed=False)
        self._state_writer = SimpleNamespace(record_button=lambda *a: None)
        self.leds = SimpleNamespace(mode="music_play")
        self.leds.set_mode = lambda m: setattr(self.leds, "mode", m)

    def _get_player(self):
        return self._player

    def press(self, kind: str = "short") -> None:
        self.button.events.put(ButtonEvent(kind=kind, duration=1.2 if kind == "long" else 0.2))


@pytest.fixture(autouse=True)
def _no_stale_pause():
    audio.resume_speech()
    yield
    audio.resume_speech()


# ------------------------------------------------------------------ waiting


@pytest.mark.parametrize(
    "player",
    [None, _Player(playing=False), _Player(playing=True, paused=True)],
    ids=["no-player", "stopped", "paused"],
)
def test_waiting_short_press_is_the_wake_word(player):
    app = _App(player)
    app.press()
    app._handle_buttons()
    assert app._wake_event.fired and not app._about_pending
    assert app.leds.mode == "q_listen"
    assert player is None or player.calls == []


def test_waiting_long_press_asks_about_the_device():
    app = _App(_Player(), held=True)
    app.press("long")
    app._handle_buttons()
    assert app._about_pending and not app._wake_event.fired
    assert app._player.calls == []


def test_about_request_is_matched_without_the_llm():
    from oracle.app import _ABOUT_REQUEST
    from oracle.commands import _keyword_match

    assert _keyword_match(_ABOUT_REQUEST) == "about_device"


# -------------------------------------------------------------------- music


def test_music_one_short_press_is_next_song_after_the_double_press_window():
    app = _App(_Player())
    app.press()
    app._handle_buttons()
    assert app._player.calls == []  # not yet: a second press may follow
    time.sleep(0.06)
    app._handle_buttons()
    assert app._player.calls == ["next"]
    app._handle_buttons()
    assert app._player.calls == ["next"]


def test_music_two_short_presses_replay_the_song():
    app = _App(_Player())
    app.press()
    app._handle_buttons()
    app.press()
    app._handle_buttons()
    time.sleep(0.06)
    app._handle_buttons()
    assert app._player.calls == ["replay"]


def test_music_second_press_still_held_down_is_waited_for():
    app = _App(_Player())
    app.press()
    app._handle_buttons()
    app.button.pressed = True  # finger is down again; no event until release
    time.sleep(0.06)
    app._handle_buttons()
    assert app._player.calls == []
    app.button.pressed = False
    app.press()
    app._handle_buttons()
    assert app._player.calls == ["replay"]


def test_music_long_press_is_next_album():
    app = _App(_Player())
    app.press("long")
    app._handle_buttons()
    assert app._player.calls == ["next_album"]
    assert not app._about_pending


# ------------------------------------------------------------------ talking


def test_short_press_while_talking_pauses_and_the_next_one_carries_on(monkeypatch):
    app = _App(_Player(), held=True)
    app.leds.mode = "q_speak"
    app._begin_turn()
    check = app._make_turn_abort()
    monkeypatch.setattr(audio, "speech_playing", lambda grace=0.0: True)

    app.press()
    assert check() is False  # a pause is not an abort
    assert audio.speech_paused() and app.leds.mode == "q_paused"
    app.press()
    assert check() is False
    assert not audio.speech_paused() and app.leds.mode == "q_speak"
    app._end_turn()


def test_long_press_while_talking_stops_the_turn(monkeypatch):
    app = _App(_Player(), held=True)
    app._begin_turn()
    check = app._make_turn_abort()
    monkeypatch.setattr(audio, "speech_playing", lambda grace=0.0: True)
    app.press("long")
    assert check() and check()
    assert not audio.speech_paused()


def test_press_while_the_mic_is_open_cancels_the_turn(monkeypatch):
    app = _App(_Player(), held=True)
    app._begin_turn()
    check = app._make_turn_abort()
    monkeypatch.setattr(audio, "speech_playing", lambda grace=0.0: False)
    app.press()
    assert check()
    assert not audio.speech_paused()


def test_a_pause_never_outlives_its_turn(monkeypatch):
    app = _App(_Player(), held=True)
    app._begin_turn()
    audio.pause_speech()
    app._end_turn()
    assert not audio.speech_paused()


# ------------------------------------------------------- pausable playback


def _fake_stream(monkeypatch):
    """_stream_play that 'plays' until its abort check fires or 3 polls."""
    played: list[int] = []

    def fake(out, sr, should_abort):
        played.append(len(out))
        for _ in range(3):
            if should_abort and should_abort():
                return
            time.sleep(0.01)

    monkeypatch.setattr(audio, "_stream_play", fake)
    monkeypatch.setattr(audio, "_resample_to_playback", lambda a, sr: (a, sr))
    return played


def test_paused_clip_says_paused_then_plays_again_from_its_start(monkeypatch):
    played = _fake_stream(monkeypatch)
    audio.set_pause_cue(np.zeros(7, dtype=np.float32), 16000)
    polls = {"n": 0}

    def button() -> bool:  # stands in for the app's abort check
        polls["n"] += 1
        if polls["n"] == 2:
            audio.pause_speech()
        if polls["n"] == 12:
            audio.resume_speech()
        return False

    audio.play_audio(np.zeros(100, dtype=np.float32), 16000, should_abort=button, text="hello")
    assert played == [100, 7, 100]  # clip, "Paused.", clip again
    assert not audio.speech_playing()
    audio.set_pause_cue(None)


def test_abort_while_paused_ends_the_clip(monkeypatch):
    played = _fake_stream(monkeypatch)
    audio.set_pause_cue(None)
    audio.pause_speech()
    polls = {"n": 0}

    def stop() -> bool:
        polls["n"] += 1
        return polls["n"] > 5

    audio.play_audio(np.zeros(100, dtype=np.float32), 16000, should_abort=stop)
    assert played == []  # paused before it began, then stopped


def test_clip_nobody_reads_the_button_for_ignores_a_pause(monkeypatch):
    played = _fake_stream(monkeypatch)
    audio.pause_speech()
    audio.play_audio(np.zeros(50, dtype=np.float32), 16000)  # a chime
    assert played == [50]


# ------------------------------------------------------------------- player


def test_player_replay_plays_the_same_track_again(monkeypatch):
    from oracle.music.player import Player

    tracks = [
        SimpleNamespace(path=f"/m/{i}.mp3", title=f"t{i}", artist="a", album="al") for i in (1, 2)
    ]
    catalog = SimpleNamespace(random_album_tracks=lambda: tracks)
    player = Player(catalog=catalog)
    player._continuous = False
    order: list[str] = []

    def fake_play_file(track) -> None:
        order.append(track.title)
        if order == ["t1"]:
            player.replay()  # pressed twice during the first song

    monkeypatch.setattr(player, "_play_file", fake_play_file)
    monkeypatch.setattr(player, "_play_intro", lambda: None)
    monkeypatch.setattr("oracle.music.player.time.sleep", lambda s: None)
    player._play_thread(first_track=None, play_intro=False)
    assert order == ["t1", "t1", "t2"]
