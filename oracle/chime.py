"""Wake acknowledgment chirp — "ready to listen" cue.

A Star Trek communicator-style two-tone rising chirp, synthesized in
memory. The original wake chime (an MP3 via a spawned player process) was
removed because it cost ~3.4s before the mic opened, and played async its
echo tail tripped the VAD and truncated the user's speech. This one
avoids both failure modes:

- generated once as a numpy array (no file decode, no subprocess), played
  through the already-initialized playback path — ~0.5s wall clock total;
- played *synchronously* before recording opens, so nothing of it can
  leak into the capture window.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

_SAMPLE_RATE = 24000
_chirp: np.ndarray | None = None


def _tone(freq_start: float, freq_end: float, dur: float, amp: float) -> np.ndarray:
    """One tone with a slight upward glide and click-free edges."""
    n = int(_SAMPLE_RATE * dur)
    # Linear frequency glide via integrated phase.
    freq = np.linspace(freq_start, freq_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / _SAMPLE_RATE
    tone = np.sin(phase)
    # 8ms raised-cosine fade in/out so it chirps instead of clicks.
    fade = int(_SAMPLE_RATE * 0.008)
    env = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    env[:fade] = ramp
    env[-fade:] = ramp[::-1]
    return (amp * tone * env).astype(np.float32)


def wake_chirp() -> np.ndarray:
    """The cached two-tone 'listening' chirp (float32 @ 24 kHz, ~0.33s)."""
    global _chirp
    if _chirp is None:
        gap = np.zeros(int(_SAMPLE_RATE * 0.035), dtype=np.float32)
        # Two quick rising tones, second higher — reads as "go ahead".
        # Amplitude is modest so the mic-leak residue after AEC stays
        # far below the VAD energy threshold.
        _chirp = np.concatenate(
            [
                _tone(740, 830, 0.11, 0.30),
                gap,
                _tone(1100, 1320, 0.15, 0.30),
            ]
        )
    return _chirp


def play_wake_chirp() -> None:
    """Play the chirp, blocking until done (~0.35s of audio)."""
    from oracle.audio import play_audio

    try:
        play_audio(wake_chirp(), _SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Wake chirp playback failed: {e}")
