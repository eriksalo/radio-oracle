"""The practical guides (scripts/ingest_sources.py) on the device: for
each question, the shelf, what the lookup costs, the passages it returns
and how their source would be named aloud.

Retrieval only (no LLM, no speech, no database). Needs the service
stopped — a second embedder does not fit beside it:

    nohup sudo SIM_SCRIPT=scripts/probe_guides.py \
        /opt/radio-oracle/scripts/sim_turn.sh > /dev/null 2>&1 &
    tail -f /tmp/sim_turn.log
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# (question, should a guides_* passage be among the answers?)
QUESTIONS = [
    ("Can I take ibuprofen and paracetamol together?", True),
    ("What are the side effects of amoxicillin?", True),
    ("How much paracetamol can I give a child?", True),
    ("Can I drink alcohol while taking antibiotics?", True),
    ("How do I treat a second-degree burn?", False),
    ("How do I keep my feet dry on a long hike?", True),
    ("How do I hang a bear bag?", True),
    ("How do I make ketchup?", True),
    ("How do I make a sourdough starter?", False),
    ("Why does my well pump keep turning on and off?", False),
    ("Who was Napoleon Bonaparte?", False),
    ("Why is the sky blue?", False),
    ("What happens at the end of Moby Dick?", False),
]


def main() -> int:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from oracle import core
    from oracle.memory.context import _resident_embed
    from oracle.rag import shelves

    retriever = core._get_retriever()
    if retriever is None:
        print("FAIL: no retriever")
        return 1
    guides = sorted(n for n in retriever.list_collections() if n.startswith("guides_"))
    print(f"# guide collections: {guides or 'NONE'}")
    retriever.query("warm up the indices")
    shelves.warm(_resident_embed)
    print(f"# sizes: { {k: v for k, v in retriever.collection_sizes().items() if k in guides} }")

    as_expected = 0
    for question, wants_guide in QUESTIONS:
        shelf = shelves.classify(question, _resident_embed)
        t0 = time.monotonic()
        results = retriever.query(question, shelf=shelf)
        ms = (time.monotonic() - t0) * 1000
        has_guide = any(r["source"].startswith("guides_") for r in results)
        ok = has_guide or not wants_guide
        as_expected += ok
        note = "" if ok else "  <-- EXPECTED A GUIDE"
        if has_guide and not wants_guide:
            note = "  (a guide answered; look at it)"
        print(f"\n{question}\n  shelf={shelf}  lookup {ms:.0f} ms{note}")
        for r in results:
            meta = r.get("metadata") or {}
            print(f"  {r['distance']:.2f} {r['source']:16s} {(meta.get('title') or '?')[:60]}")
        sources = [
            (
                (r.get("metadata") or {}).get("source") or r["source"]
                if r["source"].startswith("guides_")
                else r["source"],
                (r.get("metadata") or {}).get("title") or "",
            )
            for r in results
        ]
        print(f"  said: {shelves.sources_text(shelf, sources)}")
        sys.stdout.flush()
    print(f"\n# as expected: {as_expected}/{len(QUESTIONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
