"""Standalone Whisper transcription worker.

Reads a float32 mono 16 kHz waveform from stdin (length-prefixed: 4-byte
little-endian uint32 sample count, then raw float32 samples), runs whisper
inference, writes the transcript to stdout, and exits.

Running as a subprocess ensures the CUDA context (if any) is torn down on
exit so VRAM is reclaimed before the next pipeline stage.

Supports two backends via ORACLE_STT_BACKEND:
  - "faster-whisper" (default): CTranslate2, int8 on CPU, ~3-4s for 3s audio
  - "pywhispercpp" (legacy): whisper.cpp, ~38s for 3s audio on this Jetson
"""

from __future__ import annotations

import struct
import sys

import numpy as np

from config.settings import settings


def _transcribe_faster_whisper(audio: np.ndarray) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.faster_whisper_model,
        device=settings.faster_whisper_device,
        compute_type=settings.faster_whisper_compute,
    )
    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        language=settings.whisper_language,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def _transcribe_pywhispercpp(audio: np.ndarray) -> str:
    from pywhispercpp.model import Model  # type: ignore[import-not-found]
    from pathlib import Path

    model_path = settings.whisper_model_path
    if not model_path.is_absolute():
        model_path = (Path(__file__).resolve().parent.parent / model_path).resolve()

    model = Model(model=str(model_path), redirect_whispercpp_logs_to=False)
    segments = model.transcribe(audio)
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> int:
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) != 4:
        return 2
    n = struct.unpack("<I", raw_len)[0]
    payload = sys.stdin.buffer.read(n * 4)
    if len(payload) != n * 4:
        return 3

    audio = np.frombuffer(payload, dtype=np.float32)

    if settings.stt_backend == "faster-whisper":
        text = _transcribe_faster_whisper(audio)
    else:
        text = _transcribe_pywhispercpp(audio)

    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
