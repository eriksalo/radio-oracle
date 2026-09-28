"""Tests for the streaming STT path: the shared listen() helper feeds a
streaming session block by block and flushes it at the endpoint; batch
backends keep their transcribe(audio) contract."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from config.settings import settings
from oracle import stt as stt_mod
from oracle import timing

MODELS = Path(__file__).parent.parent / "models"


class _FakeStream:
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


def _install_fake_sd(monkeypatch, pattern: str, n: int = 1600):
    blocks = [
        np.full(n, 0.2, dtype=np.float32) if ch == "S" else np.zeros(n, dtype=np.float32)
        for ch in pattern
    ]
    fake = types.ModuleType("sounddevice")
    fake.InputStream = lambda **kw: _FakeStream(blocks)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


class _StreamingBackend:
    def __init__(self):
        self.fed: list[int] = []
        self.finished = False
        self.transcribe_calls = 0

    def open_stream(self):
        return self

    def feed(self, block, sample_rate=None):
        self.fed.append(len(block))

    def finish(self):
        self.finished = True
        return "next song"

    def load(self):
        pass

    def transcribe(self, audio, sample_rate=None):
        self.transcribe_calls += 1
        return "batch"


class _BatchBackend:
    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, audio, sample_rate=None):
        return f"batch {len(audio)}"


@pytest.fixture(autouse=True)
def _energy_backend(monkeypatch):
    from oracle import audio

    monkeypatch.setattr(settings, "vad_backend", "energy")
    monkeypatch.setattr(settings, "audio_capture_sample_rate", 16000)
    monkeypatch.setattr(settings, "audio_sample_rate", 16000)
    monkeypatch.setattr(audio, "_get_input_device", lambda: None)
    timing.clear()


def test_listen_feeds_streaming_session_and_flushes(monkeypatch):
    _install_fake_sd(monkeypatch, "..SSS...")
    backend = _StreamingBackend()
    t = timing.start("t")
    audio, text = stt_mod.listen(backend, silence_duration=0.3)
    assert text == "next song"
    assert backend.finished and backend.transcribe_calls == 0
    # 3 speech blocks + 3 trailing-silence blocks, none before onset.
    assert backend.fed == [1600] * 6
    assert len(audio) == 6 * 1600
    s = t.summary()
    assert "record" in s and "stt" in s


def test_listen_batch_backend_transcribes_buffer(monkeypatch):
    _install_fake_sd(monkeypatch, "SS...")
    backend = _BatchBackend()
    audio, text = stt_mod.listen(backend, silence_duration=0.3)
    assert backend.loaded
    assert text == f"batch {len(audio)}"


def test_listen_silence_returns_empty_without_transcribing(monkeypatch):
    _install_fake_sd(monkeypatch, ".....")
    backend = _StreamingBackend()
    audio, text = stt_mod.listen(backend, onset_timeout=0.3)
    assert len(audio) == 0 and text == ""
    assert not backend.finished


def test_create_stt_selects_streaming_backend(monkeypatch):
    monkeypatch.setattr(settings, "stt_backend", "nemotron-streaming")
    from oracle.stt_streaming import StreamingSTT

    assert isinstance(stt_mod.create_stt(), StreamingSTT)


@pytest.mark.skipif(
    not (MODELS / settings.streaming_stt_model_dir.name).is_dir()
    or os.environ.get("ORACLE_SLOW_TESTS") != "1",
    reason="Nemotron bundle not present or ORACLE_SLOW_TESTS!=1",
)
def test_streaming_recognizer_transcribes_kokoro_speech(monkeypatch):
    """Real model, real (synthesized) speech — end-to-end sanity."""
    pytest.importorskip("sherpa_onnx")
    from scipy.signal import resample_poly

    from oracle.stt_streaming import StreamingSTT
    from oracle.tts import KokoroTTS

    monkeypatch.setattr(
        settings, "streaming_stt_model_dir", MODELS / settings.streaming_stt_model_dir.name
    )
    tts = KokoroTTS(MODELS / "kokoro-v1.0.onnx", MODELS / "voices-v1.0.bin")
    wav = resample_poly(tts.synthesize("Who was Nikola Tesla?"), 2, 3).astype(np.float32)
    stt = StreamingSTT()
    stt.load()
    session = stt.open_stream()
    for i in range(0, len(wav), 1600):
        session.feed(wav[i : i + 1600], sample_rate=16000)
    text = session.finish().lower()
    assert "tesla" in text
