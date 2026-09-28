"""Probe: Kokoro on CUDA via onnxruntime-gpu in a sidecar (cp310) venv.

The app venv is cp311 and the only JetPack 6 onnxruntime-gpu wheel is
cp310, so this runs from a separate interpreter. Reports RTF for short,
clause and long sentences and — the number that decides whether a GPU TTS
sidecar fits next to the LLM — system memory before/after each step.

Env knobs: KOKORO_MODEL (default models/kokoro-v1.0.onnx; try the fp16
export), GPU_MEM_LIMIT_MB (CUDA arena cap, default 0 = unlimited).

    uv venv --python 3.10 /opt/radio-oracle/.venv-tts
    uv pip install --python .venv-tts/bin/python \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        onnxruntime-gpu==1.24.0 && uv pip install ... kokoro-onnx --no-deps numpy ...
    ONNX_PROVIDER=CUDAExecutionProvider SIM_PYTHON=.venv-tts/bin/python \
        SIM_SCRIPT=scripts/probe_kokoro_gpu.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import onnxruntime as ort  # noqa: E402

SENTENCE = (
    "Nikola Tesla was a Serbian-American inventor and electrical engineer "
    "best known for his contributions to the design of the modern alternating "
    "current electricity supply system."
)
SHORT = "Checking the archives."
CLAUSE = "Nikola Tesla was a Serbian-American inventor and electrical engineer,"


def mem(tag: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1]
    used, avail = free.split()[2], free.split()[6]
    print(f"[mem] {tag:28s} maxRSS={rss:6.0f}MB  sys used={used}MB avail={avail}MB", flush=True)


def main() -> None:
    model = os.environ.get("KOKORO_MODEL", "models/kokoro-v1.0.onnx")
    limit_mb = int(os.environ.get("GPU_MEM_LIMIT_MB", "0"))
    print("onnxruntime", ort.__version__, "providers", ort.get_available_providers(), "model", model, flush=True)
    mem("start")
    from kokoro_onnx import Kokoro

    k = Kokoro(model, "models/voices-v1.0.bin")
    mem("after Kokoro() (cpu session)")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cuda_opts: dict[str, object] = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kSameAsRequested",
    }
    if limit_mb:
        cuda_opts["gpu_mem_limit"] = limit_mb * 1024 * 1024
    t = time.monotonic()
    k.sess = ort.InferenceSession(model, so, providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"])
    print("session providers:", k.sess.get_providers(), f"load {time.monotonic() - t:.1f}s", flush=True)
    mem("after CUDA session")
    k.create("warm up.", voice="am_michael")
    mem("after warm-up")
    for text in (SHORT, CLAUSE, SENTENCE):
        best = 1e9
        secs = 0.0
        try:
            for _ in range(3):
                t = time.monotonic()
                samples, sr = k.create(text, voice="am_michael", speed=1.0)
                best = min(best, time.monotonic() - t)
                secs = len(samples) / sr
            print(f"{len(text):4d} chars -> {secs:5.2f}s audio in {best:5.2f}s  RTF {best / secs:4.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{len(text):4d} chars -> FAILED: {e}", flush=True)
        mem(f"after {len(text)} chars")


if __name__ == "__main__":
    sys.exit(main())
