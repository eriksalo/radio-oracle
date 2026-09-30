"""Probe: Nemotron streaming STT on this box — accuracy and post-endpoint latency.

Kokoro synthesizes a set of radio phrases (real speech, no mic), each is
resampled to 16 kHz, scaled to a quiet-mic peak and fed in 100 ms blocks
through StreamingSession.feed() (timing the per-block cost, i.e. can it
keep up with capture), then finish() is timed — that is the only STT
latency left after the endpoint. Compares against the offline Parakeet
backend on the same audio.

Run via: SIM_SCRIPT=scripts/probe_stt_streaming.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scipy.signal import resample_poly  # noqa: E402

from config.settings import settings  # noqa: E402
from oracle.tts import KokoroTTS  # noqa: E402

PHRASES = [
    "Who was Nikola Tesla?",
    "Next song.",
    "Play some Pink Floyd.",
    "How do I treat a second degree burn?",
    "Read me Moby Dick.",
    "What is the boiling point of water at high altitude?",
    "Librarian, what causes the northern lights?",
    "Tell me more about that.",
]


def main() -> None:
    from oracle.stt_parakeet import ParakeetSTT
    from oracle.stt_streaming import StreamingSTT

    tts = KokoroTTS()
    tts.load()
    clips = []
    for p in PHRASES:
        a = resample_poly(tts.synthesize(p), 2, 3).astype(np.float32)
        a = a / (np.abs(a).max() + 1e-9) * 0.05  # quiet mic
        clips.append(a)

    t = time.monotonic()
    stt = StreamingSTT()
    stt.load()
    print(f"nemotron streaming load: {time.monotonic() - t:.1f}s", flush=True)

    block_ms: list[float] = []
    finish_ms: list[float] = []
    ok = 0
    for p, a in zip(PHRASES, clips, strict=True):
        s = stt.open_stream()
        for i in range(0, len(a), 1600):
            t = time.monotonic()
            s.feed(a[i : i + 1600], sample_rate=16000)
            block_ms.append((time.monotonic() - t) * 1000)
        partial = s.partial()
        t = time.monotonic()
        text = s.finish()
        finish_ms.append((time.monotonic() - t) * 1000)
        hit = p.lower().rstrip("?.").split()[-1] in text.lower()
        ok += hit
        print(
            f"  {'ok ' if hit else 'MISS'} {p!r:55s} -> {text!r}  (partial before flush: {partial!r})",
            flush=True,
        )
    print(
        f"streaming: {ok}/{len(PHRASES)} keyword hits; per-100ms-block feed "
        f"p50 {statistics.median(block_ms):.0f} ms, max {max(block_ms):.0f} ms; "
        f"finish() p50 {statistics.median(finish_ms):.0f} ms, max {max(finish_ms):.0f} ms",
        flush=True,
    )
    stt.release()

    if settings.parakeet_model_dir.is_dir():
        pk = ParakeetSTT()
        pk.load()
        lat: list[float] = []
        ok = 0
        for p, a in zip(PHRASES, clips, strict=True):
            boosted = a / (np.abs(a).max() + 1e-9) * 0.5
            t = time.monotonic()
            text = pk.transcribe(boosted, sample_rate=16000)
            lat.append((time.monotonic() - t) * 1000)
            ok += p.lower().rstrip("?.").split()[-1] in text.lower()
        print(
            f"parakeet offline: {ok}/{len(PHRASES)} keyword hits; transcribe p50 "
            f"{statistics.median(lat):.0f} ms, max {max(lat):.0f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
