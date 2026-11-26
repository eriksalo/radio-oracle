"""Shelves on the device: which shelf each question lands on, what that
costs, and what the shelf changes in the passages retrieved.

Retrieval only (no LLM, no speech, no database). Needs the service
stopped — a second embedder does not fit beside it:

    nohup sudo SIM_SCRIPT=scripts/probe_shelves.py \
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

QUESTIONS = [
    ("Why does my well pump keep turning on and off?", "fixing"),
    ("What should the air pressure in the pressure tank be?", "fixing"),
    ("How do I check the air pressure in a well pressure tank?", "fixing"),
    ("How do I fix a leaking faucet?", "fixing"),
    ("How do I make a sourdough starter?", "food"),
    ("How often should I feed the starter?", "food"),
    ("When should I plant tomatoes?", "food"),
    ("How do I treat a second-degree burn?", "health"),
    ("My ankle is swollen after a fall, what do I do?", "health"),
    ("How do I make water safe to drink?", "survival"),
    ("How do I start a fire without matches?", "survival"),
    ("Why is the sky blue?", "science"),
    ("How do magnets work?", "science"),
    ("Who was Napoleon Bonaparte?", "history"),
    ("What caused the First World War?", "history"),
    ("What happens at the end of Moby Dick?", "stories"),
]

# (the question that opened the thread, the next question, carries it on?)
THREADS = [
    (
        "Why does my well pump keep turning on and off?",
        "What should the air pressure in the pressure tank be?",
        True,
    ),
    ("How do I make a sourdough starter?", "How often should I feed the starter?", True),
    ("When should I plant tomatoes?", "How far apart should the plants be?", True),
    ("How do I treat a second-degree burn?", "When should the dressing be changed?", True),
    ("How do I make a sourdough starter?", "Why does the starter smell like vinegar?", True),
    ("Who was Napoleon Bonaparte?", "Who was Julius Caesar?", False),
    ("How do I fix a leaking faucet?", "How do I fix a flat tire?", False),
    ("Who was Napoleon Bonaparte?", "What is the capital of Peru?", False),
    ("How do I make a sourdough starter?", "Why is the sky blue?", False),
    ("How do I make a sourdough starter?", "How do I make cheese from milk?", False),
    ("Why does my well pump keep turning on and off?", "Who was Napoleon Bonaparte?", False),
]


def _show(results: list[dict]) -> str:
    return " | ".join(
        f"{r['source']}:{((r.get('metadata') or {}).get('title') or '?')[:28]}@{r['distance']:.2f}"
        for r in results
    )


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
    retriever.query("warm up the indices")  # loads the embedder and the indices
    t0 = time.monotonic()
    ok = shelves.warm(_resident_embed)
    print(f"# example questions embedded: {ok} in {time.monotonic() - t0:.1f}s", flush=True)

    right = 0
    changed = 0
    for question, expected in QUESTIONS:
        by_words = shelves.classify(question)
        t0 = time.monotonic()
        shelf = shelves.classify(question, _resident_embed if ok else None)
        ms = (time.monotonic() - t0) * 1000
        right += shelf == expected
        plain = retriever.query(question)
        shelved = retriever.query(question, shelf=shelf)
        differs = _show(plain) != _show(shelved)
        changed += differs
        mark = "OK" if shelf == expected else f"EXPECTED {expected}"
        print(f"\n{question}\n  shelf={shelf} (keywords: {by_words}) {ms:.0f} ms  {mark}")
        print(f"  without: {_show(plain) or '(nothing)'}")
        if differs:
            print(f"  shelf:   {_show(shelved) or '(nothing)'}")
        else:
            print("  shelf:   (same passages)")
        sys.stdout.flush()
    print(f"\n# shelf right: {right}/{len(QUESTIONS)}; retrieval changed on {changed}")

    # A question that carries a thread on without a pronoun: is that
    # noticed, and what does the thread's opening question do to the lookup?
    import tempfile

    from oracle.memory import threads

    print("\n# thread hint")
    noticed = 0
    with tempfile.TemporaryDirectory(prefix="probe_shelves_") as tmp:
        store = threads.ThreadStore(Path(tmp) / "threads.db")
        for opener, nxt, expected in THREADS:
            tracker = threads.ThreadTracker(store, _resident_embed)
            tracker.file_turn("probe", None, opener, "An answer.", shelf=shelves.classify(opener))
            t0 = time.monotonic()
            thread = tracker.continues("probe", nxt) if core._refers_back(nxt) else None
            ms = (time.monotonic() - t0) * 1000
            carried = thread is not None
            noticed += carried == expected
            mark = "OK" if carried == expected else "UNEXPECTED"
            shelf = shelves.classify(nxt, _resident_embed) or (thread.shelf if thread else None)
            print(
                f"\n{opener}  ->  {nxt}\n  carries on: {carried} {ms:.0f} ms  {mark}  shelf={shelf}"
            )
            print(f"  without: {_show(retriever.query(nxt, shelf=shelf)) or '(nothing)'}")
            if carried:
                hinted = retriever.query(f"{nxt} ({opener})", shelf=shelf)
                print(f"  hinted:  {_show(hinted) or '(nothing)'}")
            sys.stdout.flush()
        store.close()
    print(f"\n# carry-on test right: {noticed}/{len(THREADS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
