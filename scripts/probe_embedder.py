"""Probe: ONNX embedder vs sentence-transformers on the Jetson.

Loads both runtimes for nomic-embed-text-v1.5, embeds the golden
questions with the query prefix, and reports cosine similarity between
the two (index compatibility), per-query latency, and the RSS cost of
each runtime. Run via the harness wrapper (service stopped):

    SIM_SCRIPT=scripts/probe_embedder.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.settings import settings  # noqa: E402

MODEL = "nomic-ai/nomic-embed-text-v1.5"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    questions = [
        line.strip().lstrip("+ ").strip()
        for line in Path("docs/golden_questions.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    texts = [f"search_query: {q}" for q in questions]

    from oracle.rag.embedder import Embedder

    print(f"[mem] start maxRSS={rss_mb():.0f}MB", flush=True)
    settings.embedding_runtime = "onnx"
    results: dict[str, list] = {}
    for model_file in ("model_int8.onnx", "model.onnx"):
        if not (Path(settings.onnx_embedding_dirs[MODEL]) / model_file).exists():
            print(f"{model_file}: not present, skipped", flush=True)
            continue
        settings.onnx_embedding_file = model_file
        t = time.monotonic()
        onnx = Embedder(model_name=MODEL)
        onnx.load()
        print(f"{model_file} load {time.monotonic() - t:.2f}s  maxRSS={rss_mb():.0f}MB", flush=True)
        onnx.embed_single(texts[0])
        lat = []
        vecs = []
        for tx in texts:
            t = time.monotonic()
            vecs.append(onnx.embed_single(tx))
            lat.append(time.monotonic() - t)
        print(
            f"{model_file} per-query p50 {statistics.median(lat) * 1000:.0f} ms "
            f"max {max(lat) * 1000:.0f} ms",
            flush=True,
        )
        results[model_file] = vecs
        del onnx

    settings.embedding_runtime = "sentence-transformers"
    t = time.monotonic()
    st = Embedder(model_name=MODEL, device="cpu")
    st.load()
    print(
        f"sentence-transformers load {time.monotonic() - t:.2f}s  maxRSS={rss_mb():.0f}MB",
        flush=True,
    )
    st.embed_single(texts[0])
    lat = []
    vecs_st = []
    for tx in texts:
        t = time.monotonic()
        vecs_st.append(st.embed_single(tx))
        lat.append(time.monotonic() - t)
    print(
        f"sentence-transformers per-query p50 {statistics.median(lat) * 1000:.0f} ms "
        f"max {max(lat) * 1000:.0f} ms",
        flush=True,
    )

    b = np.asarray(vecs_st)
    for model_file, vecs in results.items():
        a = np.asarray(vecs)
        cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(f"cosine({model_file}, st): min {cos.min():.4f} mean {cos.mean():.4f}", flush=True)
        for q, c in zip(questions, cos, strict=True):
            print(f"  {c:.4f}  {q}")


if __name__ == "__main__":
    main()
