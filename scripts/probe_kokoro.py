"""Probe: where do Kokoro's seconds go on this CPU, and does int8 help?

Times phonemization vs. ONNX inference for one sentence with the shipped
fp32 model, then again with explicit onnxruntime session options (all
cores, full graph optimisation), then with an int8 model if present at
models/kokoro-v1.0.int8.onnx. Reports audio seconds produced and RTF.

Run via:  SIM_SCRIPT=scripts/probe_kokoro.py sudo -E scripts/sim_turn.sh
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

import onnxruntime as ort  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402

SENTENCE = (
    "Nikola Tesla was a Serbian-American inventor and electrical engineer "
    "best known for his contributions to the design of the modern alternating "
    "current electricity supply system."
)
SHORT = "Checking the archives."


def bench(label: str, k: Kokoro, n: int = 2) -> None:
    for text in (SHORT, SENTENCE):
        best = 1e9
        secs = 0.0
        for _ in range(n):
            t = time.monotonic()
            samples, sr = k.create(text, voice="am_michael", speed=1.0)
            dt = time.monotonic() - t
            best = min(best, dt)
            secs = len(samples) / sr
        print(
            f"{label:34s} {len(text):4d} chars -> {secs:5.2f}s audio in {best:5.2f}s  RTF {best / secs:4.2f}",
            flush=True,
        )


def phonemize_time(k: Kokoro) -> None:
    t = time.monotonic()
    ph = k.tokenizer.phonemize(SENTENCE)
    print(f"phonemize: {time.monotonic() - t:.3f}s  ({len(ph)} phonemes)", flush=True)


def main() -> None:
    print("onnxruntime", ort.__version__, "providers", ort.get_available_providers(), flush=True)
    print("cpu count", os.cpu_count(), flush=True)

    k = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
    k.create("warm up.", voice="am_michael")
    try:
        phonemize_time(k)
    except Exception as e:  # noqa: BLE001
        print(f"phonemize probe failed: {e}")
    opts = k.sess.get_session_options() if hasattr(k.sess, "get_session_options") else None
    print("default session threads:", getattr(opts, "intra_op_num_threads", "?"), flush=True)
    bench("fp32 default session", k)

    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 6
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    k.sess = ort.InferenceSession("models/kokoro-v1.0.onnx", so, providers=["CPUExecutionProvider"])
    k.create("warm up.", voice="am_michael")
    bench("fp32 all-cores/opt-all", k)

    so2 = ort.SessionOptions()
    so2.intra_op_num_threads = os.cpu_count() or 6
    so2.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    so2.inter_op_num_threads = 2
    k.sess = ort.InferenceSession(
        "models/kokoro-v1.0.onnx", so2, providers=["CPUExecutionProvider"]
    )
    k.create("warm up.", voice="am_michael")
    bench("fp32 parallel-exec", k)

    int8 = Path("models/kokoro-v1.0.int8.onnx")
    if int8.exists():
        k.sess = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
        k.create("warm up.", voice="am_michael")
        bench("int8 all-cores", k)
        out = k.create(SHORT, voice="am_michael")[0]
        np.save("/tmp/kokoro_int8_sample.npy", out)
    else:
        print("no int8 model at", int8)


if __name__ == "__main__":
    main()
