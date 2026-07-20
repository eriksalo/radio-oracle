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
_DELTA = 0.01  # ignore sub-1% pot wiggle

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


def _loop() -> None:
    try:
        from oracle.hardware.volume import get_volume_control

        ctl = get_volume_control()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Volume bridge unavailable: {e}")
        return
    logger.info(f"Volume bridge started (initial gain={ctl.gain:.2f})")
    last = -1.0
    while not _stop.is_set():
        gain = ctl.gain
        if abs(gain - last) >= _DELTA:
            set_sink_volume(gain)
            last = gain
        _stop.wait(_POLL_S)
    logger.debug("Volume bridge stopped")
