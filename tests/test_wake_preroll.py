"""'Librarian, play the Beatles' in one breath: the words that follow the
wake word are kept and start the recording; room-noise blips are dropped."""

from __future__ import annotations

import sys
import types

import numpy as np

from oracle import audio, commands, wakeword
from tests.test_endpoint import _blocks, _FakeStream


def _fake_sd(monkeypatch, blocks):
    fake = types.ModuleType("sounddevice")
    fake.InputStream = lambda **kw: _FakeStream(blocks)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def _energy(monkeypatch):
    monkeypatch.setattr(audio.settings, "vad_backend", "energy")
    monkeypatch.setattr(audio, "_get_input_device", lambda: None)


def test_detector_keeps_only_what_follows_the_wake_word():
    det = wakeword.WakeWordDetector(model_path="x.onnx")
    assert len(det.audio_since_wake()) == 0  # no wake yet
    for v in (1, 2, 3):
        det._remember(np.full(1280, v, dtype=np.int16))
    det._wake_seq = det._seq  # detection on the third chunk
    import time

    t_mark = time.monotonic()
    for v in (4, 5):
        det._remember(np.full(1280, v * 3276, dtype=np.int16))
    got = det.audio_since_wake()
    assert got.dtype == np.float32 and len(got) == 2 * 1280
    assert abs(got[0] - 4 * 3276 / 32768) < 1e-4 and abs(got[-1] - 5 * 3276 / 32768) < 1e-4
    assert len(det.audio_since_time(t_mark)) == 2 * 1280
    assert len(det.audio_since_time(time.monotonic())) == 0


def test_preroll_becomes_the_start_of_the_recording(monkeypatch):
    _energy(monkeypatch)
    _fake_sd(monkeypatch, _blocks("SS" + "." * 12))
    pre = np.full(4800, 0.3, dtype=np.float32)  # 0.3 s said before the mic opened
    out = audio.record_until_silence(silence_duration=0.5, preroll_fn=lambda: pre)
    # Pre-roll (0.3 s) + two speech blocks + trailing silence, in that order.
    assert len(out) >= 4800 + 3200
    assert np.allclose(out[:4800] / out[0], 1.0)
    assert out[0] > out[4800] > 0  # 0.3-level pre-roll, then the 0.2-level blocks


def test_preroll_means_speech_already_started(monkeypatch):
    """With pre-roll and then silence the recording ends at the endpoint —
    it must not sit waiting for an onset that already happened."""
    _energy(monkeypatch)
    _fake_sd(monkeypatch, _blocks("." * 20))
    pre = np.full(8000, 0.3, dtype=np.float32)
    out = audio.record_until_silence(silence_duration=0.5, preroll_fn=lambda: pre)
    assert 8000 <= len(out) <= 8000 + 16000


def test_short_quiet_blip_is_not_a_request(monkeypatch):
    _energy(monkeypatch)
    quiet = [np.full(1600, 0.03, dtype=np.float32)] * 6 + _blocks("." * 12)
    _fake_sd(monkeypatch, quiet)
    assert len(audio.record_until_silence(silence_duration=0.5)) == 0


def test_quiet_but_long_speech_is_kept(monkeypatch):
    _energy(monkeypatch)
    quiet = [np.full(1600, 0.03, dtype=np.float32)] * 25 + _blocks("." * 12)
    _fake_sd(monkeypatch, quiet)
    assert len(audio.record_until_silence(silence_duration=0.5)) > 16000


def test_wake_word_tail_is_stripped_from_a_preroll_transcript():
    strip = lambda t: commands._WAKE_TAIL_RE.sub("", t, count=1)  # noqa: E731
    assert strip("Librarian, do you have any mystery books?") == "do you have any mystery books?"
    assert strip("brarian play the Beatles") == "play the Beatles"
    assert strip("Do you have books about librarians?") == "Do you have books about librarians?"
    assert strip("Library hours") == "Library hours"


def test_first_word_said_while_the_mic_was_opening_is_kept(monkeypatch):
    """'keep going' arrived as 'going.': the answer began before the
    recording stream was open. The detector's buffer fills the gap."""
    _energy(monkeypatch)
    _fake_sd(monkeypatch, _blocks("SS" + "." * 12))
    gap = np.concatenate([np.zeros(1600, np.float32), np.full(3200, 0.3, np.float32)])
    monkeypatch.setattr(audio, "_preroll_source", lambda t: gap)
    out = audio.record_until_silence(silence_duration=0.5)
    loud = np.flatnonzero(np.isclose(np.abs(out), np.abs(out).max()))  # output is level-normalised
    assert len(loud) == 3200  # the two blocks spoken before the stream opened


def test_quiet_gap_is_not_treated_as_speech(monkeypatch):
    _energy(monkeypatch)
    _fake_sd(monkeypatch, _blocks("." * 10))
    monkeypatch.setattr(audio, "_preroll_source", lambda t: np.zeros(4800, np.float32))
    assert len(audio.record_until_silence(silence_duration=0.5, onset_timeout=0.5)) == 0


def test_keep_going_is_understood_even_clipped():
    for said in ("Keep going.", "going.", "Go on", "please continue", "Carry on."):
        assert commands._CONTINUE_RE.match(said), said
    for said in ("Going to the store", "Continue reading Moby Dick", "What is going on?"):
        assert not commands._CONTINUE_RE.match(said), said
