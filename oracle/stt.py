"""Speech-to-text via whisper.cpp (whispercpp Python bindings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings


class WhisperSTT:
    """Wrapper around whisper-cpp-python for GPU-accelerated STT."""

    def __init__(self, model_path: Path | None = None):
        self._model_path = model_path or settings.whisper_model_path
        self._model = None

    def load(self) -> None:
        """Load model into memory. Call before transcribing."""
        if self._model is not None:
            return
        try:
            from whispercpp import Whisper

            logger.info(f"Loading Whisper model from {self._model_path}")
            self._model = Whisper.from_pretrained(str(self._model_path))
            logger.info("Whisper model loaded")
        except ImportError:
            logger.error("whispercpp not installed. Install with: pip install whispercpp")
            raise

    def unload(self) -> None:
        """Free model memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("Whisper model unloaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> str:
        """Transcribe float32 audio to text.

        Args:
            audio: float32 numpy array, mono
            sample_rate: sample rate of audio (default from settings)

        Returns:
            Transcribed text string
        """
        sr = sample_rate or settings.audio_sample_rate

        if self._model is None:
            self.load()

        # Ensure correct format: float32, mono, 16kHz
        if sr != 16000:
            # Resample to 16kHz (whisper expects this)
            from scipy.signal import resample

            target_len = int(len(audio) * 16000 / sr)
            audio = resample(audio, target_len).astype(np.float32)

        logger.debug(f"Transcribing {len(audio) / 16000:.1f}s of audio")
        result = self._model.transcribe(audio)
        text = result.strip()
        logger.info(f"STT result: {text!r}")
        return text
