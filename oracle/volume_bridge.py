"""Pot → PulseAudio sink-volume bridge (global, one per process).

The physical volume knob drives the *sink* volume, so every stream —
music (mpg123), TTS speech, the wake chime — is scaled once, identically,
and live. Previously the bridge lived inside the music Player (so the
knob was dead whenever music wasn't playing) while the TTS playback path
applied pot gain a second time in software: speech was quieter than
music by roughly the square of the knob position.
"""

from __future__ import annotations

import subprocess
import threading

from loguru import logger

from config.settings import settings

# Target Pulse's *default* sink rather than a hardcoded name — a USB
# card's profile suffix flips between .stereo-fallback and .analog-stereo
# depending on capture state, and @DEFAULT_SINK@ follows it.
_SPEAKER_SINK = "@DEFAULT_SINK@"
_POLL_S = 0.1  # well below human perception of knob lag
# The pot's ADC reading wanders by a few tens of mV at rest. With a 1 %
# gain deadband that fired `pactl` ~10×/s forever: a child process forked
# from the 1.8 GB app on every poll, 20 % of a core, and the interpreter
# stalls it caused made speech playback underrun (2026-09-30). Smooth the
# reading, apply only whole-percent moves of at least _DEADBAND_PCT, and
# never more often than _MIN_INTERVAL_S.
_SMOOTHING = 0.5  # EMA weight of the newest reading
_DEADBAND_PCT = 2
_MIN_INTERVAL_S = 0.25

_thread: threading.Thread | None = None
_stop = threading.Event()


def sink_percent(gain: float) -> int:
    """Knob position (0.0–1.0) → sink volume percent, under the ceiling."""
    ceiling = max(0, min(100, settings.speaker_max_volume_pct))
    return int(round(max(0.0, min(1.0, gain)) * ceiling))


def set_sink_volume(gain: float) -> None:
    """Set the speaker sink volume in Pulse. gain is the knob, 0.0–1.0;
    full knob lands on settings.speaker_max_volume_pct."""
    pct = sink_percent(gain)
    proc = subprocess.run(
        ["pactl", "set-sink-volume", _SPEAKER_SINK, f"{pct}%"],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(f"pactl set-sink-volume rc={proc.returncode}: {err[:160]}")


def start() -> None:
    """Start the bridge daemon (idempotent)."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="volume-bridge", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
    global _thread
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=0.5)
    _thread = None


# The ReSpeaker Lite's echo canceller (firmware 2.0.8 test build) loses
# its lock when the playback level jumps: 30 % -> 55 % in one step brought
# the radio's own voice back at full level on the mic until it re-learned,
# while the same change as 4-point steps every 0.25 s stayed cancelled, and
# so did a sudden drop (measured 2026-11-23). So the knob ramps UP at most
# this many knob-percent per pactl call; turning it down is immediate.
_RAISE_STEP_PCT = 5


class _Slew:
    """Walks the applied volume toward the knob's target."""

    def __init__(self) -> None:
        self.target: int | None = None
        self.applied: int | None = None

    def pending(self) -> bool:
        return self.target is not None and self.target != self.applied

    def step(self) -> int:
        assert self.target is not None
        if self.applied is None or self.target <= self.applied:
            self.applied = self.target  # first setting, or turning down
        else:
            self.applied = min(self.target, self.applied + _RAISE_STEP_PCT)
        return self.applied


class _KnobTracker:
    """Decides when a pot reading is a real knob move worth a `pactl` call."""

    def __init__(self) -> None:
        self.smoothed: float | None = None
        self.applied_pct: int | None = None

    def update(self, gain: float) -> int | None:
        """Feed one reading; return the percent to apply, or None to hold."""
        if self.smoothed is None:
            self.smoothed = gain
        else:
            self.smoothed += _SMOOTHING * (gain - self.smoothed)
        pct = int(round(self.smoothed * 100))
        if self.applied_pct is None or abs(pct - self.applied_pct) >= _DEADBAND_PCT:
            self.applied_pct = pct
            return pct
        # Let the extremes land exactly even inside the deadband.
        if pct in (0, 100) and pct != self.applied_pct:
            self.applied_pct = pct
            return pct
        return None


def _loop() -> None:
    import time

    try:
        from oracle.hardware.volume import get_volume_control

        ctl = get_volume_control()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Volume bridge unavailable: {e}")
        return
    logger.info(f"Volume bridge started (initial gain={ctl.gain:.2f})")
    tracker = _KnobTracker()
    slew = _Slew()
    last_applied = 0.0
    while not _stop.is_set():
        pct = tracker.update(ctl.gain)
        if pct is not None:
            slew.target = pct
        if slew.pending() and time.monotonic() - last_applied >= _MIN_INTERVAL_S:
            set_sink_volume(slew.step() / 100)
            last_applied = time.monotonic()
        _stop.wait(_POLL_S)
    logger.debug("Volume bridge stopped")
