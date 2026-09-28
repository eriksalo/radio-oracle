"""Probe: smaller offline STT models vs Parakeet-TDT-0.6B on the Jetson —
accuracy on radio phrases (Kokoro voices, quiet-mic level), latency, and
anonymous memory per model. Parakeet 0.6B int8 costs ~1.1 GB resident.

    SIM_SCRIPT=scripts/probe_stt_small.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scipy.signal import resample_poly  # noqa: E402

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
    "Go to chapter three.",
    "What am I reading?",
    "Pause the music.",
    "Is this Erik? Yes it is.",
]
VOICES = ("am_michael", "af_sarah")

MODELS = {
    "parakeet-0.6b-int8 (current)": (
        "transducer",
        "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    ),
    "parakeet-110m-int8": (
        "transducer",
        "models/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8",
    ),
    "moonshine-base-v2-q": (
        "moonshine",
        "models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
    ),
}


def anon_mb() -> int:
    m = re.search(r"^RssAnon:\s+(\d+)", open("/proc/self/status").read(), re.M)
    return int(m.group(1)) // 1024 if m else -1


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def _wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref).split(), _norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d = d, [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def _pick(d: Path, stem: str) -> str:
    for name in (f"{stem}.int8.onnx", f"{stem}.onnx"):
        if (d / name).exists():
            return str(d / name)
    raise FileNotFoundError(stem)


def build(kind: str, d: Path):
    import sherpa_onnx

    if kind == "transducer":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_pick(d, "encoder"),
            decoder=_pick(d, "decoder"),
            joiner=_pick(d, "joiner"),
            tokens=str(d / "tokens.txt"),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            model_type="nemo_transducer",
        )
    if kind == "moonshine":
        files = {p.name: p for p in list(d.glob("*.onnx")) + list(d.glob("*.ort"))}

        def find(key):
            return str(next(p for n, p in files.items() if key in n))

        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=find("preprocess"),
            encoder=find("encode"),
            uncached_decoder=find("uncached_decode"),
            cached_decoder=find("cached_decode"),
            tokens=str(d / "tokens.txt"),
            num_threads=4,
        )
    raise ValueError(kind)


def main() -> None:
    tts = KokoroTTS()
    tts.load()
    clips = []
    for v in VOICES:
        from config.settings import settings

        settings.tts_voice = v
        for p in PHRASES:
            a = resample_poly(tts.synthesize(p), 2, 3).astype(np.float32)
            clips.append((p, a / (np.abs(a).max() + 1e-9) * 0.5))
    print(f"{len(clips)} clips", flush=True)
    for label, (kind, path) in MODELS.items():
        d = Path(path)
        if not d.is_dir():
            print(f"{label}: not present ({path})", flush=True)
            continue
        before = anon_mb()
        t = time.monotonic()
        try:
            rec = build(kind, d)
        except Exception as e:  # noqa: BLE001
            print(f"{label}: load failed: {e}", flush=True)
            continue
        load_s = time.monotonic() - t
        s = rec.create_stream()
        s.accept_waveform(16000, clips[0][1])
        rec.decode_stream(s)
        mem = anon_mb() - before
        wers, lat, misses = [], [], []
        for ref, a in clips:
            t = time.monotonic()
            s = rec.create_stream()
            s.accept_waveform(16000, a)
            rec.decode_stream(s)
            lat.append(time.monotonic() - t)
            hyp = s.result.text
            w = _wer(ref, hyp)
            wers.append(w)
            if w > 0.34:
                misses.append(f"{ref!r} -> {hyp!r}")
        print(
            f"{label:30s} load {load_s:4.1f}s  +{mem:5d} MB  WER {statistics.mean(wers):.2f}  "
            f"exact {sum(w == 0 for w in wers)}/{len(wers)}  latency p50 {statistics.median(lat) * 1000:.0f} ms",
            flush=True,
        )
        for m in misses[:6]:
            print(f"    miss: {m}")
        del rec


if __name__ == "__main__":
    main()
