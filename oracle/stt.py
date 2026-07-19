"""Speech-to-text via faster-whisper (CTranslate2) or pywhispercpp.

When running on CPU (the default for faster-whisper int8), the model is kept
in-process to avoid ~6s subprocess startup overhead on each call.

When running on GPU, a subprocess is used so the CUDA context is torn down
on exit and VRAM is reclaimed before the LLM needs it.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings


class WhisperSTT:
    """Whisper STT with in-process (CPU) or subprocess (GPU) execution."""

    def __init__(
        self,
        model_path: Path | None = None,
        model_name: str | None = None,
    ):
        self._model_path = model_path or settings.whisper_model_path
        # If caller didn't override, fall back to the global setting.
        self._model_name = model_name or settings.faster_whisper_model
        self._model = None  # in-process model (CPU mode only)
        self._use_subprocess = settings.faster_whisper_device != "cpu"
        logger.info(
            f"STT backend: {settings.stt_backend} model={self._model_name} "
            f"({'subprocess' if self._use_subprocess else 'in-process'})"
        )

    def load(self) -> None:
        """Load the model into memory (in-process CPU mode only)."""
        if self._use_subprocess or settings.stt_backend != "faster-whisper":
            return
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_name,
            device=settings.faster_whisper_device,
            compute_type=settings.faster_whisper_compute,
        )
        logger.debug(f"faster-whisper model {self._model_name!r} loaded in-process")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.debug("faster-whisper model unloaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> str:
        """Transcribe float32 audio to text.

        Uses in-process inference for CPU mode, subprocess for GPU mode.
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

        if self._use_subprocess or settings.stt_backend != "faster-whisper":
            return self._transcribe_subprocess(audio)
        return self._transcribe_inprocess(audio)

    def _transcribe_inprocess(self, audio: np.ndarray) -> str:
        """Fast path: model stays loaded in-process (CPU only)."""
        if self._model is None:
            self.load()
        segments, _ = self._model.transcribe(
            audio,
            beam_size=1,
            language=settings.whisper_language,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        logger.info(f"STT result: {text!r}")
        return text

    def _transcribe_subprocess(self, audio: np.ndarray) -> str:
        """Subprocess path: tears down CUDA context on exit."""
        payload = struct.pack("<I", audio.shape[0]) + audio.tobytes()

        env = os.environ.copy()
        if settings.whisper_force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [sys.executable, "-m", "oracle.stt_worker"],
            input=payload,
            capture_output=True,
            timeout=60,
            env=env,
        )
        if proc.returncode != 0:
            logger.error(
                f"STT worker failed (rc={proc.returncode}): {proc.stderr.decode(errors='replace')}"
            )
            return ""

        text = proc.stdout.decode("utf-8", errors="replace").strip()
        logger.info(f"STT result: {text!r}")
        return text
