#!/usr/bin/env python3
"""Retrieval eval harness — recall@k over a golden question set.

Measures whether the expected article/chunk surfaces in the top-k for
each question, so index/model changes (nprobe, re-embeds, rerankers)
can be compared with a number instead of vibes.

Golden set format (JSON list; see data/eval/golden.example.json):

    [
      {
        "question": "who was nikola tesla",
        "expect": ["Nikola Tesla"],          # substrings; a hit counts if ANY
        "in": "title_or_text",               # where to look: "title", "text",
                                             #   or "title_or_text" (default)
        "collections": ["wikipedia"]         # optional override
      },
      ...
    ]

Usage:
    # Current settings (whatever backends/env are configured)
    python scripts/eval_rag.py --golden data/eval/golden.json --k 5

    # Compare two FAISS index dirs (old vs re-embedded)
    ORACLE_FAISS_INDEX_DIR=data/faiss     python scripts/eval_rag.py --golden g.json
    ORACLE_FAISS_INDEX_DIR=data/faiss_v2  python scripts/eval_rag.py --golden g.json

Prints per-question hits and overall recall@k; exits non-zero if recall
falls below --min-recall (for use as a ship gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger


def _hit(result: dict, needles: list[str], where: str) -> bool:
    title = str((result.get("metadata") or {}).get("title") or "")
    text = result.get("text") or ""
    for needle in needles:
        n = needle.lower()
        if where in ("title", "title_or_text") and n in title.lower():
            return True
        if where in ("text", "title_or_text") and n in text.lower():
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="RAG recall@k eval")
    p.add_argument("--golden", required=True, help="Path to golden-set JSON")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode", default="snappy", choices=["snappy", "deep"])
    p.add_argument(
        "--min-recall", type=float, default=0.0,
        help="Exit 1 if overall recall@k is below this (ship gate)",
    )
    args = p.parse_args()

    cases = json.loads(Path(args.golden).read_text())
    if not cases:
        logger.error("Golden set is empty")
        sys.exit(2)

    from oracle.rag.retriever import Retriever

    retriever = Retriever()
    hits = 0
    latencies: list[float] = []

    for case in cases:
        q = case["question"]
        t0 = time.perf_counter()
        results = retriever.query(
            q,
            collection_names=case.get("collections"),
            top_k=args.k,
            mode=args.mode,
        )
        dt = time.perf_counter() - t0
        latencies.append(dt)
        ok = any(
            _hit(r, case["expect"], case.get("in", "title_or_text"))
            for r in results
        )
        hits += ok
        top = (results[0].get("metadata") or {}).get("title") or (
            results[0]["text"][:60] if results else "—"
        ) if results else "—"
        print(f"  [{'HIT ' if ok else 'MISS'}] {dt*1000:6.0f}ms  {q!r}  top: {top!r}")

    recall = hits / len(cases)
    lat = sorted(latencies)
    print(
        f"\nrecall@{args.k}: {recall:.2%} ({hits}/{len(cases)})   "
        f"latency p50 {lat[len(lat)//2]*1000:.0f}ms  "
        f"max {lat[-1]*1000:.0f}ms   mode={args.mode}"
    )
    if recall < args.min_recall:
        logger.error(f"recall {recall:.2%} below gate {args.min_recall:.2%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
