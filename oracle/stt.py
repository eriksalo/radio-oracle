"""Speech-to-text via a per-call whisper.cpp subprocess.

Whisper runs in a fresh Python subprocess for each `transcribe()` call so that
its CUDA context is torn down on exit and VRAM is fully released before the
LLM runner needs it. On a Jetson Orin Nano (8 GB unified) this is the only way
small.en + llama3.2:3b can coexist in one voice-loop turn.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings


class WhisperSTT:
    """Subprocess-backed wrapper for whisper.cpp inference."""

    def __init__(self, model_path: Path | None = None):
        self._model_path = model_path or settings.whisper_model_path

    def load(self) -> None:
        """No-op kept for API compatibility — the worker loads per call."""
        return

    def unload(self) -> None:
        """No-op kept for API compatibility — the worker exits per call."""
        return

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> str:
        """Transcribe float32 audio to text via a one-shot subprocess.

        Args:
            audio: float32 numpy array, mono
            sample_rate: sample rate of audio (default from settings)

        Returns:
            Transcribed text string
        """
        sr = sample_rate or settings.audio_sample_rate

        if sr != 16000:
            from scipy.signal import resample

            target_len = int(len(audio) * 16000 / sr)
            audio = resample(audio, target_len).astype(np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.ascontiguousarray(audio)

        logger.debug(f"Transcribing {len(audio) / 16000:.1f}s of audio")
        payload = struct.pack("<I", audio.shape[0]) + audio.tobytes()

        proc = subprocess.run(
            [sys.executable, "-m", "oracle.stt_worker"],
            input=payload,
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            logger.error(
                f"STT worker failed (rc={proc.returncode}): {proc.stderr.decode(errors='replace')}"
            )
            return ""

        text = proc.stdout.decode("utf-8", errors="replace").strip()
        logger.info(f"STT result: {text!r}")
        return text
