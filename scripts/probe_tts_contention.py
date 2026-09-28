"""Probe: how much does Ollama decoding slow Kokoro synthesis (and vice versa)?

Measures Kokoro synth time for a fixed sentence (a) on an idle box, (b)
while Ollama is streaming a reply with its default thread count, and (c)
while it streams with ``num_thread`` reduced. Also reports decode tok/s in
each case, so the trade-off is visible in one table.

Run through scripts/sim_turn.sh (service stopped, env sourced):

    SIM_SCRIPT=scripts/probe_tts_contention.py nohup sudo scripts/sim_turn.sh &
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.settings import settings  # noqa: E402
from oracle.tts import KokoroTTS  # noqa: E402

SENTENCE = (
    "Nikola Tesla was a Serbian-American inventor and electrical engineer "
    "best known for his contributions to the design of the modern alternating "
    "current electricity supply system."
)
PROMPT = "Write four paragraphs about the history of radio broadcasting in America."


async def _decode(num_thread: int | None, stop: asyncio.Event) -> float:
    """Stream one long reply; return decode tok/s. Sets *stop* when done."""
    options = {"num_ctx": 2048, "num_predict": 220, "temperature": 0.7}
    if num_thread:
        options["num_thread"] = num_thread
    payload = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "keep_alive": -1,
        "options": options,
    }
    tps = 0.0
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{settings.ollama_host}/api/chat", json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                ch = json.loads(line)
                if ch.get("done"):
                    tps = ch["eval_count"] / (ch["eval_duration"] / 1e9)
                    break
    stop.set()
    return tps


async def _synth_loop(tts: KokoroTTS, stop: asyncio.Event, n_max: int = 6) -> list[float]:
    times: list[float] = []
    while not stop.is_set() and len(times) < n_max:
        t = time.monotonic()
        await asyncio.to_thread(tts.synthesize, SENTENCE)
        times.append(time.monotonic() - t)
    return times


async def main() -> None:
    tts = KokoroTTS()
    tts.load()
    tts.synthesize("warm up.")

    idle = []
    for _ in range(3):
        t = time.monotonic()
        tts.synthesize(SENTENCE)
        idle.append(time.monotonic() - t)
    print(f"kokoro idle:            {statistics.median(idle):.2f}s per sentence", flush=True)

    # Warm the LLM so the first measured decode isn't paying a load.
    stop = asyncio.Event()
    await _decode(None, stop)

    for label, nt in (("ollama default threads", None), ("ollama num_thread=2", 2), ("ollama num_thread=1", 1)):
        stop = asyncio.Event()
        dec_task = asyncio.create_task(_decode(nt, stop))
        synth_times = await _synth_loop(tts, stop)
        tps = await dec_task
        print(
            f"kokoro during {label:22s}: {statistics.median(synth_times):.2f}s per sentence "
            f"(n={len(synth_times)}), decode {tps:.1f} tok/s",
            flush=True,
        )
        # decode alone with this thread count, no synth
        stop = asyncio.Event()
        tps_alone = await _decode(nt, stop)
        print(f"  decode alone, same threads: {tps_alone:.1f} tok/s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
