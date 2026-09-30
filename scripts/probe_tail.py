"""Probe: does the end of each played clip reach the speaker?

Plays 0.6 s of 440 Hz then a 0.3 s 1 kHz marker at the very end of the
clip through the app's playback path (oracle.audio.play_audio → PortAudio
→ pulse), recording the speaker sink's *monitor* with parec (exactly what
goes to the DAC; no mic, no room). Reports milliseconds of each tone
heard, three trials. Stop the radio first — its own speech lands on the
same monitor — and run with the app's environment:

    sudo systemctl stop radio-oracle
    sudo -u oracle bash -c 'set -a; . ./.env; set +a; \
      XDG_RUNTIME_DIR=/run/user/999 .venv/bin/python scripts/probe_tail.py'

Before the silence tail in oracle.audio._stream_play: marker 0 ms of 300
(every clip lost its last ~270 ms). After: 300 of 300. A (0, 0) row is
parec missing the clip, not a playback fault. Optional arg: extra
trailing silence in seconds, to bracket the loss.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SR = 24000
MON = "alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00.analog-stereo.monitor"


def _band_ms(x: np.ndarray, f0: float, thr: float) -> int:
    n = 480  # 10 ms at 48 kHz
    hits = 0
    for i in range(0, len(x) - n, n):
        s = np.abs(np.fft.rfft(x[i : i + n] * np.hanning(n)))
        k = int(round(f0 * n / 48000))
        hits += s[k - 1 : k + 2].max() > thr
    return hits * 10


def main() -> None:
    from oracle import audio

    pad = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    t = np.arange(int(0.6 * SR)) / SR
    m = np.arange(int(0.3 * SR)) / SR
    clip = np.concatenate(
        [
            0.05 * np.sin(2 * np.pi * 440 * t),
            0.05 * np.sin(2 * np.pi * 1000 * m),
            np.zeros(int(pad * SR)),
        ]
    ).astype(np.float32)
    rows = []
    for _ in range(3):
        p = subprocess.Popen(
            [
                "parec",
                "-d",
                MON,
                "--rate=48000",
                "--channels=1",
                "--format=float32le",
                "--latency-msec=10",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1.0)
        audio.play_audio(clip, SR)
        time.sleep(0.8)
        p.terminate()
        x = np.frombuffer(p.stdout.read(), dtype="<f4")
        p.wait()
        peak = 0.0
        for f in (440, 1000):
            n = 480
            for i in range(0, len(x) - n, n):
                s = np.abs(np.fft.rfft(x[i : i + n] * np.hanning(n)))
                k = int(round(f * n / 48000))
                peak = max(peak, float(s[k - 1 : k + 2].max()))
        thr = peak * 0.25 if peak else 1.0
        rows.append((_band_ms(x, 440, thr), _band_ms(x, 1000, thr)))
    print(f"pad {pad:.2f}s — ms heard (440 Hz of 600, end marker of 300): {rows}", flush=True)


if __name__ == "__main__":
    main()
