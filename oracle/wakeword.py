"""Always-on wake word detection via energy VAD + lightweight STT.

Runs continuously on the AEC-cleaned mic source.  When speech energy
exceeds the threshold, buffers audio until silence returns, then runs
a tiny Whisper model to check for the configured wake word.

CPU cost: ~200 ms per short utterance (tiny.en int8 on Jetson CPU).
Memory: ~39 MB for the tiny.en model — separate from the main STT model.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
from loguru import logger

from config.settings import settings

_SAMPLE_RATE = 16000
_BLOCK_MS = 100
_BLOCK_SAMPLES = int(_SAMPLE_RATE * _BLOCK_MS / 1000)  # 1600
_MAX_WAKE_SECONDS = 3.0  # max buffered speech before forced transcription
_SILENCE_BLOCKS = 6  # 600 ms of silence ends the utterance
_MIN_SPEECH_BLOCKS = 4  # need ≥400 ms of speech to bother transcribing
_ENERGY_THRESHOLD = 0.10  # RMS — above AEC music residual (~0.06), below close speech (~0.3)
_COOLDOWN_S = 1.5  # min gap between transcription attempts


class WakeWordDetector:
    """Listens for a keyword via VAD + STT on the default input device.

    Call :meth:`start` to begin background detection.  When the keyword
    is detected, ``on_wake`` is called from the detector thread.
    """

    def __init__(
        self,
        wake_word: str | None = None,
        on_wake: Callable[[], None] | None = None,
        *,
        model_name: str | None = None,   # ignored, kept for compat
        threshold: float | None = None,   # ignored, kept for compat
    ) -> None:
        self._wake_word = (wake_word or settings.wake_word).lower()
        self.on_wake = on_wake
        self._stt = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._muted = threading.Event()
        self._muted.set()  # not muted by default (set = listening)

    def _load_stt(self) -> None:
        from faster_whisper import WhisperModel

        self._stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        logger.info(
            f"Wake word detector: keyword={self._wake_word!r} "
            f"(tiny.en int8, VAD+STT)"
        )

    def start(self) -> None:
        """Start background wake word detection."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="wakeword", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop detection and release resources."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._stt = None

    def mute(self) -> None:
        """Temporarily suspend detection (e.g. during own TTS playback)."""
        self._muted.clear()

    def unmute(self) -> None:
        """Resume detection after mute."""
        self._muted.set()

    def _loop(self) -> None:
        import sounddevice as sd

        try:
            self._load_stt()
        except Exception as e:
            logger.error(f"Wake word STT load failed: {e}")
            return

        max_buf_blocks = int(_MAX_WAKE_SECONDS / (_BLOCK_MS / 1000))
        last_transcribe = 0.0

        logger.debug("Wake word listener started")
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_BLOCK_SAMPLES,
            ) as stream:
                while not self._stop.is_set():
                    # If muted, drain audio but don't process
                    if not self._muted.is_set():
                        stream.read(_BLOCK_SAMPLES)
                        self._stop.wait(0.01)
                        continue

                    data, _ = stream.read(_BLOCK_SAMPLES)
                    chunk = data.flatten()
                    rms = float(np.sqrt(np.mean(chunk * chunk)))

                    if rms < _ENERGY_THRESHOLD:
                        continue

                    # Speech onset — buffer until silence
                    frames = [chunk]
                    speech_blocks = 1
                    silence_count = 0

                    while not self._stop.is_set() and len(frames) < max_buf_blocks:
                        data, _ = stream.read(_BLOCK_SAMPLES)
                        chunk = data.flatten()
                        frames.append(chunk)
                        rms = float(np.sqrt(np.mean(chunk * chunk)))

                        if rms < _ENERGY_THRESHOLD:
                            silence_count += 1
                            if silence_count >= _SILENCE_BLOCKS:
                                break
                        else:
                            silence_count = 0
                            speech_blocks += 1

                    # Skip if muted during buffering, too short, or in cooldown
                    if not self._muted.is_set():
                        continue
                    if speech_blocks < _MIN_SPEECH_BLOCKS:
                        continue
                    now = time.monotonic()
                    if now - last_transcribe < _COOLDOWN_S:
                        continue

                    audio = np.concatenate(frames)
                    duration = len(audio) / _SAMPLE_RATE
                    last_transcribe = now

                    # Transcribe with tiny model
                    segments, _ = self._stt.transcribe(
                        audio, beam_size=1, language="en",
                    )
                    text = " ".join(s.text.strip() for s in segments).strip().lower()

                    if self._wake_word in text:
                        logger.info(
                            f"Wake word detected: {text!r} ({duration:.1f}s)"
                        )
                        if self.on_wake is not None:
                            self.on_wake()
                    elif text and not _is_hallucination(text):
                        logger.debug(
                            f"Speech (no wake word): {text!r} ({duration:.1f}s)"
                        )

        except Exception as e:
            if not self._stop.is_set():
                logger.error(f"Wake word loop error: {e}")
        finally:
            logger.debug("Wake word listener stopped")


def _is_hallucination(text: str) -> bool:
    """Filter common Whisper hallucinations on silence/noise."""
    hallucinations = {
        "you", "thank you", "thanks for watching",
        "subscribe", "bye", "the end",
    }
    return text.strip(".!?, ").lower() in hallucinations
