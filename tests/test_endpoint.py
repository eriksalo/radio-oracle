"""Tests for oracle.endpoint: the numpy Whisper feature port, the
endpointer decision logic, and record_until_silence's loop with a fake
sound device."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from config.settings import settings
from oracle import endpoint
from oracle.endpoint import (
    EnergyEndpointer,
    VadEndpointer,
    whisper_log_mel,
    whisper_mel_filters,
)

DATA = Path(__file__).parent / "data"
MODELS = Path(__file__).parent.parent / "models"

# ------------------------------------------------------------- features


def test_mel_filterbank_matches_transformers():
    ref = (
        np.load(MODELS / "whisper_mel_80.npy") if (MODELS / "whisper_mel_80.npy").exists() else None
    )
    if ref is None:
        pytest.skip("reference filterbank not present")
    fb = whisper_mel_filters()
    assert fb.shape == (201, 80)
    np.testing.assert_allclose(fb, ref, atol=1e-6)


def test_log_mel_matches_transformers_reference():
    ref = np.load(DATA / "smart_turn_ref.npz")
    feats = whisper_log_mel(ref["signal"], whisper_mel_filters())
    assert feats.shape == (80, 800)
    np.testing.assert_allclose(feats, ref["features"], atol=2e-3)


def test_log_mel_keeps_last_eight_seconds():
    fb = whisper_mel_filters()
    long = np.random.default_rng(1).standard_normal(16000 * 12).astype(np.float32)
    assert np.allclose(whisper_log_mel(long, fb), whisper_log_mel(long[-16000 * 8 :], fb))


# ----------------------------------------------------------- endpointers


def test_energy_endpointer_is_the_legacy_rule():
    ep = EnergyEndpointer(threshold=0.01, max_silence=0.9)
    assert ep.is_speech(np.full(1600, 0.1, dtype=np.float32))
    assert not ep.is_speech(np.zeros(1600, dtype=np.float32))
    assert not ep.turn_complete(np.zeros(1), 0.8)
    assert ep.turn_complete(np.zeros(1), 0.9)


def test_vad_endpointer_without_smart_turn_stops_at_min_silence():
    ep = VadEndpointer(lambda b: False, None, min_silence=0.25, max_silence=2.0)
    assert not ep.turn_complete(np.zeros(1), 0.2)
    assert ep.turn_complete(np.zeros(1), 0.3)
    assert ep.last_decision == "vad"


def test_vad_endpointer_consults_smart_turn_and_caps():
    calls: list[float] = []
    verdicts = iter([False, False, True])

    def smart(audio):
        calls.append(len(audio))
        return next(verdicts)

    ep = VadEndpointer(lambda b: False, smart, min_silence=0.25, max_silence=2.0, interval=0.2)
    ep.start()
    # Below min silence: never asked.
    assert not ep.turn_complete(np.zeros(10), 0.1)
    assert calls == []
    # First checkpoint at 0.3 s: incomplete → keep listening.
    assert not ep.turn_complete(np.zeros(10), 0.3)
    assert ep.last_decision == "incomplete"
    # Not yet at the next checkpoint (0.3 + 0.2).
    assert not ep.turn_complete(np.zeros(10), 0.4)
    assert len(calls) == 1
    assert not ep.turn_complete(np.zeros(10), 0.5)
    assert len(calls) == 2
    assert ep.turn_complete(np.zeros(10), 0.7)
    assert ep.last_decision == "complete"


def test_vad_endpointer_hard_cap_without_asking():
    ep = VadEndpointer(lambda b: False, lambda a: False, min_silence=0.25, max_silence=1.0)
    ep.start()
    assert not ep.turn_complete(np.zeros(1), 0.3)
    assert ep.turn_complete(np.zeros(1), 1.0)
    assert ep.last_decision == "cap"


def test_build_endpointer_energy_default(monkeypatch):
    monkeypatch.setattr(settings, "vad_backend", "energy")
    assert isinstance(endpoint.build_endpointer(0.004, 0.9), EnergyEndpointer)


# ------------------------------------------------- record_until_silence


class _FakeStream:
    """A sounddevice.InputStream stand-in that serves pre-baked blocks."""

    def __init__(self, blocks):
        self._blocks = iter(blocks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        try:
            b = next(self._blocks)
        except StopIteration:
            b = np.zeros(n, dtype=np.float32)
        return b.reshape(-1, 1), False


def _install_fake_sd(monkeypatch, blocks):
    fake = types.ModuleType("sounddevice")
    fake.InputStream = lambda **kw: _FakeStream(blocks)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def _blocks(pattern: str, n: int = 1600):
    """'S' = loud block, '.' = silent block."""
    out = []
    for ch in pattern:
        if ch == "S":
            out.append(np.full(n, 0.2, dtype=np.float32))
        else:
            out.append(np.zeros(n, dtype=np.float32))
    return out


def test_record_until_silence_energy_backend(monkeypatch):
    from oracle import audio

    monkeypatch.setattr(settings, "vad_backend", "energy")
    monkeypatch.setattr(settings, "audio_capture_sample_rate", 16000)
    monkeypatch.setattr(settings, "audio_sample_rate", 16000)
    monkeypatch.setattr(audio, "_get_input_device", lambda: None)
    _install_fake_sd(monkeypatch, _blocks("..SSSS........SS"))
    out = audio.record_until_silence(silence_duration=0.3)
    # 4 speech blocks + 3 silence blocks (0.3 s) = 7 blocks of 1600.
    assert len(out) == 7 * 1600


def test_record_until_silence_smart_turn_keeps_listening(monkeypatch):
    """Silero says silence after the pause, Smart Turn says 'not done' at
    the first checkpoint and 'done' at the second."""
    from oracle import audio

    monkeypatch.setattr(settings, "vad_backend", "silero+smartturn")
    monkeypatch.setattr(settings, "audio_capture_sample_rate", 16000)
    monkeypatch.setattr(settings, "audio_sample_rate", 16000)
    monkeypatch.setattr(audio, "_get_input_device", lambda: None)

    verdicts = iter([False, True])
    fake_ep = VadEndpointer(
        lambda b: float(np.abs(b).max()) > 0.05,
        lambda a: next(verdicts),
        min_silence=0.2,
        max_silence=2.0,
        interval=0.2,
    )
    monkeypatch.setattr(endpoint, "build_endpointer", lambda *a, **k: fake_ep)
    _install_fake_sd(monkeypatch, _blocks("SSS.......SS...."))
    out = audio.record_until_silence()
    # Speech (3) + silence: checkpoint at 0.2 s → incomplete; next at 0.4 s
    # → complete. 3 + 4 = 7 blocks.
    assert len(out) == 7 * 1600
    assert fake_ep.last_decision == "complete"


def test_record_until_silence_onset_timeout(monkeypatch):
    from oracle import audio

    monkeypatch.setattr(settings, "vad_backend", "energy")
    monkeypatch.setattr(audio, "_get_input_device", lambda: None)
    _install_fake_sd(monkeypatch, _blocks("........"))
    assert len(audio.record_until_silence(onset_timeout=0.3)) == 0


# ------------------------------------------------------ smart turn smoke


@pytest.mark.skipif(
    not (MODELS / "smart-turn-v3.2-cpu.onnx").exists(), reason="Smart Turn model not present"
)
def test_smart_turn_runs_on_reference_signal():
    pytest.importorskip("onnxruntime")
    st = endpoint.SmartTurn(MODELS / "smart-turn-v3.2-cpu.onnx")
    ref = np.load(DATA / "smart_turn_ref.npz")
    p = st.probability(ref["signal"])
    assert 0.0 <= p <= 1.0
