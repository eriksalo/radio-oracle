"""Conversation threads end to end on the device, in a scratch database.

Session 1 asks questions on three topics through the real pipeline
(retrieval → LLM → Kokoro, playback simulated) and prints which thread
each one was filed into; closing the session runs the real summarizer,
which titles the threads. Session 2 says "where did we leave off", prints
the spoken recap, and asks a question that names no topic.

Needs the service stopped (two embedders + Kokoro do not fit in 8 GB):

    nohup sudo SIM_SCRIPT=scripts/probe_threads.py \
        /opt/radio-oracle/scripts/sim_turn.sh > /dev/null 2>&1 &
    tail -f /tmp/sim_turn.log

Nothing is written to the production oracle.db.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SESSION_1 = [
    "Why does my well pump keep turning on and off?",
    "What should the air pressure in the pressure tank be?",
    "How do I check it?",
    "How do I make a sourdough starter?",
    "How often should I feed the starter?",
    "Who was Napoleon Bonaparte?",
]
EXPECTED = [0, 0, 0, 1, 1, 2]  # thread index per question
SESSION_2_QUESTION = "What should I check next?"


async def _main() -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="probe_threads_"))
    os.environ["ORACLE_DB_PATH"] = str(tmp_dir / "oracle.db")
    os.environ.setdefault("ORACLE_ACTIVITY_FILE", str(tmp_dir / "activity.jsonl"))

    from loguru import logger

    from oracle import audio, commands, core
    from oracle.memory.context import ContextBuilder
    from oracle.tts import KokoroTTS

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    def fake_play(a, sample_rate=None, should_abort=None, text=""):
        time.sleep(min(0.05, len(a) / (sample_rate or 24000)))

    audio.play_audio = fake_play

    system_prompt, store, session_id = await core._init_common()
    tts = KokoroTTS()
    await asyncio.gather(asyncio.to_thread(tts.load), asyncio.to_thread(core._get_retriever))

    def make_vc(sid: str) -> core.VoiceContext:
        return core.VoiceContext(
            stt=None,  # type: ignore[arg-type]
            stt_fast=None,  # type: ignore[arg-type]
            tts=tts,
            store=store,
            ctx_builder=ContextBuilder(store, sid),
            system_prompt=system_prompt,
            session_id=sid,
        )

    async def ask(vc: core.VoiceContext, q: str) -> int | None:
        await core.voice_turn(vc, pre_text=q)
        ctx = vc.ctx_builder
        if ctx._filing:
            await asyncio.gather(*list(ctx._filing))
        return ctx.threads.current_id if ctx.threads else None

    vc = make_vc(session_id)
    if vc.ctx_builder.threads is None:
        print("FAIL: threads are not available")
        return 1
    print("# session 1")
    ids: list[int | None] = []
    for q in SESSION_1:
        tid = await ask(vc, q)
        ids.append(tid)
        print(f"  thread {tid}: {q}", flush=True)
    order = {tid: i for i, tid in enumerate(dict.fromkeys(ids))}
    got = [order[t] for t in ids]
    print(f"# filing: expected {EXPECTED} got {got} -> {'OK' if got == EXPECTED else 'DIFFERENT'}")

    t0 = time.monotonic()
    await vc.ctx_builder.close()  # session summary + thread titles (real LLM)
    print(f"# close + summaries: {time.monotonic() - t0:.1f}s")
    tstore = vc.ctx_builder.threads.store
    for th in tstore.recent(vc.ctx_builder.user):
        print(f"  thread {th.id} turns={th.turns} title={th.title!r}\n    summary={th.summary!r}")

    print("# session 2")
    sid2 = store.new_session()
    vc2 = make_vc(sid2)
    said: list[str] = []
    commands._speak = lambda _vc, text, _abort=None: said.append(text)  # type: ignore[assignment]

    async def heard(_vc, onset_timeout):
        return SESSION_2_QUESTION

    commands._listen_once = heard  # type: ignore[assignment]
    action, query = await commands.classify(vc2, "Where did we leave off?")
    opening = await commands._resume_thread(vc2, query, None)
    print(f"  classify -> {action}; radio says: {said[-1] if said else None!r}")
    resumed = vc2.ctx_builder.threads.current_id
    tid = await ask(vc2, opening or SESSION_2_QUESTION)
    answer = store.get_messages(sid2, limit=1)[-1]["content"]
    print(f"  {SESSION_2_QUESTION!r} filed into thread {tid} (resumed {resumed})")
    print(f"  answer: {answer!r}")
    found = vc2.ctx_builder.threads.find(vc2.ctx_builder.user, "sourdough")
    print(f"  find('sourdough') -> thread {found}")

    from oracle.llm import close_client

    await close_client()
    store.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    # Bounded: the radio is off while this runs.
    raise SystemExit(asyncio.run(asyncio.wait_for(_main(), timeout=900)))
