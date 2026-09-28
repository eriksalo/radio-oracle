"""Probe: do Silero VAD and Smart Turn behave on *this* box's audio levels?

No mic needed: Kokoro synthesizes real speech, which is resampled to
16 kHz and scaled down to a quiet-mic peak, then fed through the same
endpointer the recorder uses, block by block, exactly as
record_until_silence would. Reports:

  * how many 100 ms blocks Silero flags as speech at each input level
  * Smart Turn's P(complete) for a finished sentence vs. the same sentence
    cut off mid-way vs. one ending in "and"
  * the endpoint decision timeline for "sentence, pause, more speech"

Run via: SIM_SCRIPT=scripts/probe_endpoint.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scipy.signal import resample_poly  # noqa: E402

from config.settings import settings  # noqa: E402
from oracle.endpoint import SileroVad, SmartTurn, VadEndpointer  # noqa: E402
from oracle.tts import KokoroTTS  # noqa: E402

DONE = "The northern lights are caused by charged particles from the sun hitting the atmosphere."
TRAILING_AND = "The northern lights are caused by charged particles from the sun and"


def speech(tts: KokoroTTS, text: str, peak: float) -> np.ndarray:
    a = tts.synthesize(text)
    a = resample_poly(a, 2, 3).astype(np.float32)  # 24 k → 16 k
    return (a / (np.abs(a).max() + 1e-9) * peak).astype(np.float32)


def blocks(a: np.ndarray, n: int = 1600):
    for i in range(0, len(a) - n + 1, n):
        yield a[i : i + n]


def main() -> None:
    tts = KokoroTTS()
    tts.load()
    vad = SileroVad()
    vad.load()
    st = SmartTurn()
    st.load()
    gain = settings.vad_input_gain
    print(f"vad_input_gain={gain} silero_threshold={settings.silero_vad_threshold}", flush=True)

    full = speech(tts, DONE, 0.5)
    for peak in (0.5, 0.1, 0.03, 0.01):
        a = full / 0.5 * peak
        vad.reset()
        flags = [vad.is_speech(np.clip(b * gain, -1, 1)) for b in blocks(a)]
        print(
            f"peak {peak:5.2f} (x{gain} → {min(peak * gain, 1):.2f}): "
            f"{sum(flags)}/{len(flags)} blocks speech; first speech block "
            f"{flags.index(True) if True in flags else None}",
            flush=True,
        )

    quiet = 0.03
    done = speech(tts, DONE, quiet) * gain
    cut = done[: int(len(done) * 0.55)]
    trailing = speech(tts, TRAILING_AND, quiet) * gain
    sil = np.zeros(16000 // 4, dtype=np.float32)
    for label, a in (
        ("finished sentence + 0.25 s", np.concatenate([done, sil])),
        ("cut mid-sentence + 0.25 s", np.concatenate([cut, sil])),
        ("trailing 'and' + 0.25 s", np.concatenate([trailing, sil])),
        ("finished sentence + 1.0 s", np.concatenate([done, sil, sil, sil, sil])),
    ):
        print(f"smart turn P(complete) {label:28s} = {st.probability(a):.2f}", flush=True)

    # Endpoint timeline: sentence, 0.6 s pause, more speech, 2 s silence.
    ep = VadEndpointer(
        vad,
        st,
        min_silence=settings.vad_silence_min,
        max_silence=settings.vad_silence_max,
        interval=settings.smart_turn_interval,
    )
    ep.start()
    more = speech(tts, "And they are best seen near the poles.", quiet) * gain
    stream = np.concatenate(
        [
            done,
            np.zeros(16000 * 6 // 10, dtype=np.float32),
            more,
            np.zeros(16000 * 2, dtype=np.float32),
        ]
    )
    started = False
    silence = 0.0
    frames = []
    for i, b in enumerate(blocks(stream)):
        b = np.clip(b, -1, 1)
        if ep.is_speech(b):
            started = True
            silence = 0.0
            frames.append(b)
        elif started:
            silence += 0.1
            frames.append(b)
            if ep.turn_complete(np.concatenate(frames), silence):
                print(
                    f"timeline: endpoint '{ep.last_decision}' at block {i} ({i / 10:.1f}s), after {silence:.1f}s silence",
                    flush=True,
                )
                break
    else:
        print("timeline: no endpoint (ran out of audio)")
    print(
        f"speech in stream ends at ~{(len(done) + 16000 * 6 // 10 + len(more)) / 16000:.1f}s; first sentence ends at {len(done) / 16000:.1f}s"
    )


if __name__ == "__main__":
    main()
