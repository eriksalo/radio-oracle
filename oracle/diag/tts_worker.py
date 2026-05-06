"""Standalone Piper TTS worker.

Reads UTF-8 text from stdin, writes raw WAV bytes to stdout, and exits.
Running as a subprocess means the Piper ONNX session and its tensors are
reclaimed on exit, freeing RSS in the parent diag server. On the Jetson's
unified-memory pool, RSS held by any process competes with Ollama's CUDA
allocations, so per-call eviction matters.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radio-filter", action="store_true")
    args = parser.parse_args()

    text = sys.stdin.read()
    if not text.strip():
        return 2

    from oracle.audio import apply_radio_filter, audio_to_wav_bytes
    from oracle.tts import PiperTTS

    tts = PiperTTS()
    tts.load()
    audio = tts.synthesize(text)
    if args.radio_filter:
        audio = apply_radio_filter(audio, tts.sample_rate)
    wav = audio_to_wav_bytes(audio, tts.sample_rate)
    sys.stdout.buffer.write(wav)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
