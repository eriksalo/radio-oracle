"""Speech playback feeds PortAudio in blocking mode; the volume bridge only
calls pactl for real knob moves. Both regressions were audible: crackling
speech from a callback stream starved by the interpreter, and a `pactl`
fork ten times a second from a wobbling pot reading."""

from __future__ import annotations

import sys
import types

import numpy as np
from loguru import logger

from oracle import audio, volume_bridge


class _FakeStream:
    """Records the writes an OutputStream receives; flags one underflow."""

    instances: list[_FakeStream] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.write_available = 12000  # frames of ring buffer
        self.writes: list[int] = []
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.order: list[str] = []
        _FakeStream.instances.append(self)

    def write(self, data) -> bool:
        self.order.append("write")
        self.writes.append(len(data))
        return len(self.writes) == 3  # one underflow, on the third write

    def start(self) -> None:
        self.order.append("start")
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


def _install_fake_sounddevice(monkeypatch) -> None:
    fake = types.ModuleType("sounddevice")
    fake.OutputStream = _FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(audio, "_get_output_device", lambda: None)
    _FakeStream.instances.clear()


def test_stream_play_prefills_then_writes_chunks_in_blocking_mode(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    sr = 48000
    clip = np.zeros(sr, dtype=np.float32)  # 1 s mono
    try:
        audio._stream_play(clip, sr, None)
    finally:
        logger.remove(sink)
    (st,) = _FakeStream.instances
    assert "callback" not in st.kwargs, "no Python on the audio thread"
    assert st.kwargs["latency"] == audio._PLAYBACK_LATENCY_S
    assert st.kwargs["channels"] == 1
    assert st.writes[0] == 12000, "first write fills the ring buffer"
    assert st.order[:2] == ["start", "write"], "PortAudio rejects writes before start()"
    assert st.started and st.stopped and st.closed and not st.aborted
    assert sum(st.writes) == sr
    assert all(w <= int(sr * audio._PLAYBACK_CHUNK_S) for w in st.writes[1:])
    assert any("1 underflows in a 1.0s clip" in w for w in warnings)


def test_stream_play_abort_discards_buffer(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    calls = {"n": 0}

    def abort_after_two_checks() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    audio._stream_play(np.zeros(48000, dtype=np.float32), 48000, abort_after_two_checks)
    (st,) = _FakeStream.instances
    assert st.aborted and st.closed and not st.stopped
    assert sum(st.writes) < 48000


def test_stream_play_empty_clip_opens_nothing(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    audio._stream_play(np.zeros(0, dtype=np.float32), 48000, None)
    assert _FakeStream.instances == []


def test_knob_tracker_ignores_adc_wobble():
    t = volume_bridge._KnobTracker()
    assert t.update(0.660) == 66  # first reading always applies
    # ±1 % wobble around a resting knob: the old bridge fired on every one.
    for g in (0.664, 0.657, 0.668, 0.652, 0.663, 0.670, 0.655):
        assert t.update(g) is None
    # A real turn of the knob gets through.
    assert t.update(0.75) is not None
    # And smoothly reaches its target on the following polls.
    for _ in range(6):
        t.update(0.75)
    assert t.applied_pct in (74, 75)


def test_knob_tracker_lands_exactly_on_the_ends():
    t = volume_bridge._KnobTracker()
    t.update(0.02)
    for _ in range(12):
        t.update(0.0)
    assert t.applied_pct == 0
    for _ in range(12):
        t.update(1.0)
    assert t.applied_pct == 100
