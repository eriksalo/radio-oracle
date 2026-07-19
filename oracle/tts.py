"""Text-to-speech via Kokoro TTS (ONNX, CPU)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings

_SENTENCE_END = re.compile(r"[.!?\n]+\s*")

SAMPLE_RATE = 24000


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for streaming TTS."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


class KokoroTTS:
    """Wrapper around kokoro-onnx for CPU-based TTS."""

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
    ):
        self._model_path = model_path or settings.tts_model_path
        self._voices_path = voices_path or settings.tts_voices_path
        self._kokoro = None

    def load(self) -> None:
        """Load Kokoro model and voice data."""
        if self._kokoro is not None:
            return
        try:
            from kokoro_onnx import Kokoro

            logger.info(
                f"Loading Kokoro model from {self._model_path}, voices from {self._voices_path}"
            )
            self._kokoro = Kokoro(
                str(self._model_path),
                str(self._voices_path),
            )
            logger.info("Kokoro TTS loaded")
        except ImportError:
            logger.error("kokoro-onnx not installed. Install with: pip install kokoro-onnx")
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to float32 audio array at 24 kHz."""
        if self._kokoro is None:
            self.load()

        samples, _sr = self._kokoro.create(
            text,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
        )
        return samples.astype(np.float32)

    def synthesize_streaming(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks per sentence for low-latency playback."""
        for sentence in split_sentences(text):
            yield self.synthesize(sentence)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE
