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

# Target Pulse's *default* sink rather than a hardcoded name — the USB
# DAC's profile suffix flips between .stereo-fallback and .analog-stereo
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


def set_sink_volume(gain: float) -> None:
    """Set the speaker sink volume in Pulse. gain is 0.0–1.0."""
    pct = max(0, min(100, int(round(gain * 100))))
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
    last_applied = 0.0
    while not _stop.is_set():
        pct = tracker.update(ctl.gain)
        if pct is not None and time.monotonic() - last_applied >= _MIN_INTERVAL_S:
            set_sink_volume(pct / 100)
            last_applied = time.monotonic()
        _stop.wait(_POLL_S)
    logger.debug("Volume bridge stopped")
