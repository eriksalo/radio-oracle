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
    """Kokoro TTS — in-process on the CPU, or via the GPU sidecar.

    With ``settings.tts_backend == "server"`` synthesis is delegated to
    ``oracle.tts_server`` (Kokoro on CUDA in a cp310 venv; see that
    module). If the sidecar is unreachable at load time the in-process CPU
    model is used instead, so the radio still talks.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
    ):
        self._model_path = model_path or settings.tts_model_path
        self._voices_path = voices_path or settings.tts_voices_path
        self._kokoro = None
        self._server: str | None = None

    def _try_server(self) -> bool:
        if settings.tts_backend != "server":
            return False
        import httpx

        url = settings.tts_server_url.rstrip("/")
        try:
            r = httpx.get(f"{url}/health", timeout=settings.tts_server_connect_timeout)
            r.raise_for_status()
            self._server = url
            logger.info(f"Kokoro TTS via sidecar {url} ({r.text.strip()})")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"TTS sidecar {url} unavailable ({e}); using in-process CPU Kokoro")
            return False

    def load(self) -> None:
        """Load Kokoro model and voice data (or connect to the sidecar)."""
        if self._kokoro is not None or self._server is not None:
            return
        if self._try_server():
            return
        self._load_local()

    def _load_local(self) -> None:
        """Load the in-process CPU model (idempotent)."""
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

    def _synth_remote(self, text: str) -> np.ndarray | None:
        import httpx

        try:
            r = httpx.post(
                f"{self._server}/synth",
                params={"voice": settings.tts_voice, "speed": settings.tts_speed},
                content=text.encode("utf-8"),
                timeout=settings.tts_server_timeout,
            )
            r.raise_for_status()
            return np.frombuffer(r.content, dtype="<f4").astype(np.float32)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"TTS sidecar failed ({e}); falling back to in-process Kokoro")
            self._server = None
            return None

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to float32 audio array at 24 kHz."""
        if self._kokoro is None and self._server is None:
            self.load()

        audio = self._synth_remote(text) if self._server is not None else None
        if audio is None:
            # Remote failed (or not configured): the in-process model must
            # actually be loaded here — load() would happily re-attach to
            # a sidecar that just failed us and leave _kokoro None.
            self._load_local()
            samples, _sr = self._kokoro.create(
                text,
                voice=settings.tts_voice,
                speed=settings.tts_speed,
            )
            audio = samples.astype(np.float32)
        # Peak-normalize: Kokoro synthesizes well below full scale, so
        # speech sounded quiet next to loudness-mastered music on the
        # same sink. 0 disables.
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if settings.tts_peak and peak > 1e-4:
            audio = audio * (settings.tts_peak / peak)
        return audio

    def synthesize_streaming(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks per sentence for low-latency playback."""
        for sentence in split_sentences(text):
            yield self.synthesize(sentence)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE
