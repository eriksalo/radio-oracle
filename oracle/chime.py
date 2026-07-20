"""Wake acknowledgment chime — "ready to listen" cue.

Plays the chime-clean-short sound (decoded once from WAV into a cached
numpy array; silence-trimmed and level-normalized at load). The original
wake chime implementation was removed because it spawned an MP3 player
per wake (~3.4s before the mic opened — mostly this same file's padding
silence) and, when played async, its echo tail tripped the VAD and
truncated the user's speech. This version avoids both failure modes:

- loaded once, in memory, played through the already-initialized
  playback path (the audible chime is ~0.9s once silence is trimmed);
- played *synchronously* before recording opens, so nothing of it can
  leak into the capture window.

Falls back to a synthesized two-tone chirp if the WAV is missing.
"""

from __future__ import annotations

import wave

import numpy as np
from loguru import logger

from config.settings import settings

_SAMPLE_RATE = 24000
_TARGET_PEAK = 0.35  # audible but modest — echo residue stays below VAD
_TRIM_THRESHOLD = 0.01  # of peak; trims the file's padding silence

_chime: np.ndarray | None = None


def _load_wav(path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != _SAMPLE_RATE or w.getnchannels() != 1:
            raise ValueError(
                f"{path}: expected mono {_SAMPLE_RATE} Hz, got "
                f"{w.getnchannels()}ch {w.getframerate()} Hz"
            )
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = raw.astype(np.float32) / 32768.0

    # Trim leading/trailing silence — the source file is 3.4s but the
    # audible chime is ~0.9s in the middle.
    peak = float(abs(audio).max())
    if peak <= 0.0:
        raise ValueError(f"{path}: silent file")
    loud = np.flatnonzero(abs(audio) > peak * _TRIM_THRESHOLD)
    audio = audio[loud[0] : loud[-1] + 1]

    # Normalize: the source peaks at ~0.11, inaudibly quiet on the radio.
    return (audio * (_TARGET_PEAK / peak)).astype(np.float32)


def _tone(freq_start: float, freq_end: float, dur: float, amp: float) -> np.ndarray:
    """One tone with a slight upward glide and click-free edges."""
    n = int(_SAMPLE_RATE * dur)
    freq = np.linspace(freq_start, freq_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / _SAMPLE_RATE
    tone = np.sin(phase)
    fade = int(_SAMPLE_RATE * 0.008)
    env = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    env[:fade] = ramp
    env[-fade:] = ramp[::-1]
    return (amp * tone * env).astype(np.float32)


def _synth_chirp() -> np.ndarray:
    """Fallback: two quick rising tones — reads as 'go ahead'."""
    gap = np.zeros(int(_SAMPLE_RATE * 0.035), dtype=np.float32)
    return np.concatenate([_tone(740, 830, 0.11, 0.30), gap, _tone(1100, 1320, 0.15, 0.30)])


def wake_chime_audio() -> np.ndarray:
    """The cached wake chime (float32 @ 24 kHz)."""
    global _chime
    if _chime is None:
        try:
            _chime = _load_wav(settings.wake_chime_path)
            logger.debug(
                f"Wake chime loaded: {settings.wake_chime_path} "
                f"({len(_chime) / _SAMPLE_RATE:.2f}s after trim)"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Wake chime file unavailable ({e}); using synth chirp")
            _chime = _synth_chirp()
    return _chime


def play_wake_chime() -> None:
    """Play the chime, blocking until done."""
    from oracle.audio import play_audio

    try:
        play_audio(wake_chime_audio(), _SAMPLE_RATE)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Wake chime playback failed: {e}")
