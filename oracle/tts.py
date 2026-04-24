"""Text-to-speech via Piper TTS (CPU, ONNX)."""

from __future__ import annotations

import io
import re
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings

_SENTENCE_END = re.compile(r"[.!?\n]+\s*")


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for streaming TTS."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


class PiperTTS:
    """Wrapper around Piper for CPU-based TTS."""

    def __init__(self, model_path: Path | None = None):
        self._model_path = model_path or settings.piper_model_path
        self._voice = None

    def load(self) -> None:
        """Load Piper voice model."""
        if self._voice is not None:
            return
        try:
            from piper import PiperVoice

            logger.info(f"Loading Piper model from {self._model_path}")
            self._voice = PiperVoice.load(str(self._model_path))
            logger.info("Piper TTS loaded")
        except ImportError:
            logger.error("piper-tts not installed. Install with: pip install piper-tts")
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to float32 audio array."""
        if self._voice is None:
            self.load()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize(text, wf)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        return audio

    def synthesize_streaming(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks per sentence for low-latency playback."""
        sentences = split_sentences(text)
        for sentence in sentences:
            yield self.synthesize(sentence)

    @property
    def sample_rate(self) -> int:
        return settings.piper_sample_rate
