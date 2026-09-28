"""Simulated librarian turns — the latency regression harness.

Drives ``oracle.core.voice_turn`` with pre-supplied text (no mic, no STT)
through the real pipeline: rewrite → retrieval → context build → Ollama
stream → Kokoro synthesis. Playback is *simulated* by default (sleeps for
the audio's duration) so the timings match the real device without needing
a speaker; ``--play`` uses the real output device.

Runs on the Jetson over ssh. The radio-oracle service must be stopped
first — two copies of the embedder + Kokoro + FAISS don't fit in 8 GB:

    sudo systemctl stop radio-oracle
    sudo -u oracle -H ORACLE_COLLECTION_BACKENDS=... \
        /opt/radio-oracle/.venv/bin/python scripts/sim_turn.py
    sudo systemctl start radio-oracle

(or: ``set -a; source /opt/radio-oracle/.env; set +a`` first.)

Conversation memory is read from a *copy* of the production DB so the
long-term profile is realistic but the simulated turns never pollute it.

Output: one row per question (ttfa, rewrite, retrieve, prefill, first_token,
decode tok/s, prompt tokens) plus medians — paste into the deploy doc.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _load_questions(path: Path) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("+"):
            out.append((line[1:].strip(), True))
        else:
            out.append((line, False))
    return out


async def _main(args: argparse.Namespace) -> int:
    # Point the store at a scratch copy BEFORE settings are imported.
    prod_db = Path(os.environ.get("ORACLE_DB_PATH", "data/oracle.db"))
    tmp_dir = Path(tempfile.mkdtemp(prefix="sim_turn_"))
    tmp_db = tmp_dir / "oracle.db"
    if prod_db.exists():
        shutil.copy2(prod_db, tmp_db)
    os.environ["ORACLE_DB_PATH"] = str(tmp_db)
    os.environ.setdefault("ORACLE_ACTIVITY_FILE", str(tmp_dir / "activity.jsonl"))

    from loguru import logger

    from oracle import audio, core, timing
    from oracle.memory.context import ContextBuilder
    from oracle.tts import KokoroTTS

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if not args.play:

        def fake_play(a, sample_rate=None, should_abort=None):
            time.sleep(len(a) / (sample_rate or 24000))

        audio.play_audio = fake_play  # voice_turn resolves this at call time

    system_prompt, store, session_id = await core._init_common()
    tts = KokoroTTS()
    t0 = time.monotonic()
    await asyncio.gather(
        asyncio.to_thread(tts.load),
        asyncio.to_thread(core._get_retriever),
    )
    print(f"# warm-up (Kokoro + retriever): {time.monotonic() - t0:.1f}s", flush=True)

    vc = core.VoiceContext(
        stt=None,  # type: ignore[arg-type]  # pre_text path never touches STT
        stt_fast=None,  # type: ignore[arg-type]
        tts=tts,
        store=store,
        ctx_builder=ContextBuilder(store, session_id),
        system_prompt=system_prompt,
        session_id=session_id,
    )

    rows: list[dict] = []
    orig_finish = timing.TurnTimer.finish

    def capture(self):
        s = orig_finish(self)
        rows.append(dict(s, question=current_q))
        return s

    timing.TurnTimer.finish = capture  # type: ignore[method-assign]

    questions = _load_questions(Path(args.questions))
    cols = ("ttfa", "rewrite", "retrieve", "build", "prefill", "first_token", "total")
    print(
        f"{'question':42s} "
        + " ".join(f"{c:>10s}" for c in cols)
        + f" {'tok/s':>6s} {'ptok':>5s} {'rag':>5s}",
        flush=True,
    )
    for rep in range(args.repeat):
        for q, _is_followup in questions:
            current_q = q
            n_before = len(rows)
            await core.voice_turn(vc, pre_text=q)
            if len(rows) == n_before:
                print(f"{q[:42]:42s} (no timing row)")
                continue
            r = rows[-1]
            print(
                f"{q[:42]:42s} "
                + " ".join(f"{r.get(c, float('nan')):10.2f}" for c in cols)
                + f" {r.get('decode_tps', 0):6.1f} {r.get('prompt_tokens', 0):5d} "
                f"{r.get('rag_chars', 0):5d}",
                flush=True,
            )

    if rows:
        print("\n# medians")
        for c in cols + ("decode_tps", "prompt_tokens"):
            vals = [r[c] for r in rows if isinstance(r.get(c), int | float)]
            if vals:
                print(f"  {c:12s} {statistics.median(vals):8.2f}")
    await core.voice_close(vc)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--questions", default="docs/golden_questions.txt")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--play", action="store_true", help="really play audio (default: simulate)")
    p.add_argument("--log-level", default="WARNING")
    sys.exit(asyncio.run(_main(p.parse_args())))


if __name__ == "__main__":
    main()
