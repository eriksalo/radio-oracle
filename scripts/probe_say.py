"""Probe: does a long spoken line make it through the GPU TTS sidecar in
units? Runs tts.say() on the device description with playback stubbed
(no sound) and reports unit count, synthesis time and any X-Error.

    sudo -u oracle .venv/bin/python scripts/probe_say.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    from loguru import logger

    from oracle import audio
    from oracle.commands import describe_device
    from oracle.tts import KokoroTTS, say, speech_units

    warnings: list[str] = []
    logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    played: list[float] = []
    audio.play_audio = lambda a, sr=None, should_abort=None: played.append(len(a) / (sr or 24000))
    tts = KokoroTTS()
    tts.load()
    text = describe_device(None)
    units = speech_units(text)
    print(f"{len(text)} chars → {len(units)} units, max {max(len(u.split()) for u in units)} words")
    for rnd in range(3):
        played.clear()
        t = time.monotonic()
        say(tts, text)
        silent = [p for p in played if p <= 0.35]
        print(
            f"round {rnd + 1}: spoke {len(played)} units, {sum(played):.1f}s of audio, "
            f"in {time.monotonic() - t:.1f}s wall; failed units: {len(silent)}"
        )
    print(f"warnings: {len(warnings)}")
    for w in warnings[:3]:
        print("  ", w.strip()[:160])


if __name__ == "__main__":
    main()
