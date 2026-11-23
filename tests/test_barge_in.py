"""Voice barge-in: the wake word cuts the radio off only while it is
speaking inside a voice turn, and never on its own voice saying the word
(the mic hears the speaker; measured 2026-11-23)."""

from __future__ import annotations

import threading

import numpy as np

from oracle import audio


def _hold_playback(monkeypatch):
    """Make _stream_play block until released, like a clip in progress."""
    started, release = threading.Event(), threading.Event()

    def fake_stream_play(out, sr, should_abort):
        started.set()
        release.wait(2)

    monkeypatch.setattr(audio, "_stream_play", fake_stream_play)
    return started, release


def _play_in_thread(text: str) -> threading.Thread:
    t = threading.Thread(
        target=audio.play_audio,
        args=(np.zeros(160, dtype=np.float32), audio.settings.audio_playback_sample_rate),
        kwargs={"text": text},
    )
    t.start()
    return t


def test_speech_playing_tracks_the_clip(monkeypatch):
    started, release = _hold_playback(monkeypatch)
    assert not audio.speech_playing()
    t = _play_in_thread("The river ran slow.")
    assert started.wait(2)
    assert audio.speech_playing()
    assert not audio.speaker_said("librarian")
    release.set()
    t.join(2)
    assert not audio.speech_playing()


def test_speaker_said_covers_the_clip_and_its_echo_tail(monkeypatch):
    started, release = _hold_playback(monkeypatch)
    t = _play_in_thread("I am the Librarian.")
    assert started.wait(2)
    assert audio.speaker_said("librarian")
    release.set()
    t.join(2)
    # The detector fires a beat after the word: still ours just after the clip.
    assert audio.speaker_said("librarian")
    monkeypatch.setattr(audio, "_last_text_end", audio._last_text_end - 5)
    assert not audio.speaker_said("librarian")


class _App:
    """The slice of OracleApp the barge-in logic touches."""

    from oracle.app import OracleApp as _O

    _begin_turn = _O._begin_turn
    _end_turn = _O._end_turn
    _barged_in = _O._barged_in
    _make_turn_abort = _O._make_turn_abort

    def __init__(self):
        from queue import Queue
        from types import SimpleNamespace

        self._in_turn = False
        self._barge_armed = False
        self._barge = threading.Event()
        self._wakeword = SimpleNamespace(
            muted=False,
            mute=lambda: setattr(self._wakeword, "muted", True),
            unmute=lambda: setattr(self._wakeword, "muted", False),
        )
        self.power = SimpleNamespace(is_on=True)
        self.button = SimpleNamespace(events=Queue())
        self._state_writer = SimpleNamespace(record_button=lambda *a: None)


def test_wake_word_latches_the_turn_abort_and_asks_for_a_relisten():
    app = _App()
    app._begin_turn()
    assert not app._wakeword.muted  # detector stays live for barge-in
    check = app._make_turn_abort()
    assert app._barge_armed
    assert not check()
    app._barge.set()  # what on_wake does when it hears the word over speech
    assert check() and check()
    assert app._barged_in()
    app._end_turn()
    assert not app._barged_in() and not app._in_turn and not app._barge_armed


def test_barge_in_off_mutes_the_detector_for_the_turn(monkeypatch):
    from oracle import app as app_mod

    monkeypatch.setattr(app_mod.settings, "voice_barge_in", False)
    app = _App()
    app._begin_turn()
    assert app._wakeword.muted
    app._end_turn()
    assert not app._wakeword.muted


def test_abort_made_outside_a_turn_does_not_arm_barge_in():
    app = _App()
    app._make_turn_abort()  # the power-on welcome does this, detector muted
    assert not app._barge_armed
