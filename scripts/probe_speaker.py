"""Probe: speaker identification on the Jetson with Kokoro voices as
stand-in speakers — timing, thresholds, and the enroll/identify round
trip against a scratch users database (never the production one).

    ORACLE_DB_PATH=/tmp/probe_users.db sudo -E -u oracle .venv/bin/python scripts/probe_speaker.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scipy.signal import resample_poly  # noqa: E402

from config.settings import settings  # noqa: E402
from oracle.memory.users import UserStore  # noqa: E402
from oracle.speaker import SpeakerId  # noqa: E402
from oracle.tts import KokoroTTS  # noqa: E402

PHRASES = {
    "enroll": "Librarian, who was Nikola Tesla, and where was he born?",
    "test1": "Next song please, and then read me a chapter of Moby Dick.",
    "test2": "What is the boiling point of water at altitude?",
}


def main() -> None:
    assert "probe" in str(settings.db_path), "point ORACLE_DB_PATH at a scratch db"
    tts = KokoroTTS()
    tts.load()
    sid = SpeakerId(users=UserStore(settings.db_path))
    t = time.monotonic()
    sid.load()
    print(f"speaker model load {time.monotonic() - t:.2f}s")
    voices = ("am_michael", "am_adam", "af_sarah", "bm_george")
    clips = {}
    for v in voices:
        settings.tts_voice = v
        clips[v] = {
            k: resample_poly(tts.synthesize(p), 2, 3).astype(np.float32) * 0.3
            for k, p in PHRASES.items()
        }
    t = time.monotonic()
    sid.embed(clips["am_michael"]["enroll"])
    print(f"embed {(time.monotonic() - t) * 1000:.0f} ms for a {len(clips['am_michael']['enroll']) / 16000:.1f}s clip")
    for v in voices:
        sid.enroll(v, clips[v]["enroll"], "probe")
    ok = 0
    for v in voices:
        for k in ("test1", "test2"):
            name, score = sid.identify(clips[v][k])
            hit = name == v
            ok += hit
            print(f"  {'ok ' if hit else 'MISS'} {v:10s} {k}: -> {name} ({score:.2f})")
    print(f"{ok}/8 identified; threshold {settings.speaker_threshold}")


if __name__ == "__main__":
    main()
