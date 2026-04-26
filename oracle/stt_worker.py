"""Standalone Whisper transcription worker.

Reads a float32 mono 16 kHz waveform from stdin (length-prefixed: 4-byte
little-endian uint32 sample count, then raw float32 samples), runs whisper.cpp
on GPU, writes the transcript to stdout, and exits. Running as a subprocess
ensures the CUDA context is torn down on exit so VRAM is reclaimed before the
next call.
"""

from __future__ import annotations

import struct
import sys

import numpy as np

from config.settings import settings


def main() -> int:
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) != 4:
        return 2
    n = struct.unpack("<I", raw_len)[0]
    payload = sys.stdin.buffer.read(n * 4)
    if len(payload) != n * 4:
        return 3

    audio = np.frombuffer(payload, dtype=np.float32)

    from pywhispercpp.model import Model

    model_path = settings.whisper_model_path
    if not model_path.is_absolute():
        # Relative paths in settings are resolved against the project root
        # (one level up from this file's package), since this worker may run
        # from a different CWD than the parent voice loop.
        from pathlib import Path
        model_path = (Path(__file__).resolve().parent.parent / model_path).resolve()

    model = Model(model=str(model_path), redirect_whispercpp_logs_to=False)
    segments = model.transcribe(audio)
    text = " ".join(s.text.strip() for s in segments).strip()
    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
