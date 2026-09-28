"""Probe: does the int8 ONNX embedder retrieve the same chunks as fp32?

The FAISS indices were built with fp32 vectors; fp32 ONNX reproduces
them exactly (cos 1.0000) while int8 is cos ≈0.97 — enough to reorder
neighbours. This measures what matters: for each golden question, the
overlap of the top-k chunk ids between the two runtimes, per collection
routing as the real retriever does it. Run via the harness wrapper:

    SIM_SCRIPT=scripts/probe_embedder_recall.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.settings import settings  # noqa: E402

K = 5


def top_ids(retriever, q: str) -> list[str]:
    hits = retriever.query(q, mode="snappy")
    return [h.get("chunk_id") or h.get("id") or h["text"][:60] for h in hits[:K]]


def main() -> None:
    questions = [
        line.strip().lstrip("+ ").strip()
        for line in Path("docs/golden_questions.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    settings.embedding_runtime = "onnx"
    settings.rag_max_distance = 9.0  # no gate: compare raw neighbours
    results: dict[str, dict[str, list[str]]] = {}
    for model_file in ("model.onnx", "model_int8.onnx"):
        settings.onnx_embedding_file = model_file
        # Fresh retriever + embedder cache per runtime.
        import oracle.rag.backends.faiss_ivfpq as fb
        from oracle.rag.retriever import Retriever

        fb._EMBEDDER_CACHE.clear()
        r = Retriever()
        t = time.monotonic()
        results[model_file] = {q: top_ids(r, q) for q in questions}
        print(f"{model_file}: {len(questions)} queries in {time.monotonic() - t:.1f}s", flush=True)

    a, b = results["model.onnx"], results["model_int8.onnx"]
    total = 0.0
    for q in questions:
        sa, sb = set(a[q]), set(b[q])
        j = len(sa & sb) / max(len(sa | sb), 1)
        top1 = a[q][:1] == b[q][:1]
        total += j
        print(f"  overlap@{K} {j:.2f}  top1 {'same' if top1 else 'DIFF'}  {q}")
    print(f"mean top-{K} Jaccard overlap int8 vs fp32: {total / len(questions):.2f}")


if __name__ == "__main__":
    main()
