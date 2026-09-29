"""Probe: does speech playback underrun on this box right now?

Synthesizes a few units through the running GPU sidecar and plays them via
the app's own path (oracle.audio.play_audio → PortAudio → pulse), counting
PortAudio output-underflow flags per clip and the process's major page
faults, with the radio left running. Quiet (peak 0.15). Prints one line
per clip.

    sudo -u oracle .venv/bin/python scripts/probe_playback.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def majflt() -> int:
    return int(Path("/proc/self/stat").read_text().split(")")[1].split()[9])


def main() -> None:
    import sounddevice as sd

    from config.settings import settings
    from oracle import audio
    from oracle.tts import KokoroTTS, speech_units

    settings.tts_peak = 0.15
    flags = {"n": 0}
    orig = sd.OutputStream

    class CountingStream(orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **k):
            cb = k.get("callback")

            def wrapped(outdata, frames, t, status):
                if status.output_underflow:
                    flags["n"] += 1
                return cb(outdata, frames, t, status)

            if cb is not None:
                k["callback"] = wrapped
            super().__init__(*a, **k)

    sd.OutputStream = CountingStream
    tts = KokoroTTS()
    tts.load()
    text = (
        "The book collection has sixty thousand books, from Project Gutenberg. "
        "The knowledge base has about eleven million passages from Wikipedia, "
        "plus iFixit repair guides, Wikibooks and Crash Course. "
        "Ask me anything, or say read a book, or play some music."
    )
    for i, unit in enumerate(speech_units(text)):
        t0 = time.monotonic()
        clip = tts.synthesize(unit)
        t1 = time.monotonic()
        flags["n"] = 0
        mf0 = majflt()
        audio.play_audio(clip, tts.sample_rate)
        t2 = time.monotonic()
        dur = len(clip) / tts.sample_rate
        print(
            f"unit {i}: {len(unit.split()):2d} words synth {t1 - t0:.2f}s audio {dur:.2f}s "
            f"played in {t2 - t1:.2f}s underflows={flags['n']} majflt+{majflt() - mf0}",
            flush=True,
        )


if __name__ == "__main__":
    main()
