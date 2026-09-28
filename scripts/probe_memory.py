"""Probe: where does the app's anonymous memory go? Loads the runtime
components one at a time, the way voice_init does, and prints RssAnon
after each. Run via the harness wrapper (service stopped):

    SIM_SCRIPT=scripts/probe_memory.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def anon_mb() -> int:
    m = re.search(r"^RssAnon:\s+(\d+)", open("/proc/self/status").read(), re.M)
    return int(m.group(1)) // 1024 if m else -1


_last = anon_mb()


def step(label: str) -> None:
    global _last
    now = anon_mb()
    print(f"{label:48s} +{now - _last:5d} MB  (total {now} MB)", flush=True)
    _last = now


def main() -> None:
    step("python baseline")
    import numpy  # noqa: F401

    step("numpy")
    from config.settings import settings  # noqa: F401

    step("config (pydantic)")
    import oracle.core  # noqa: F401

    step("import oracle.core (+ llm, memory, persona)")
    import oracle.commands  # noqa: F401

    step("import oracle.commands")
    import oracle.rag.retriever  # noqa: F401

    step("import oracle.rag.retriever (+ reranker module)")
    try:
        import sys as _s

        print(f"   torch imported: {'torch' in _s.modules}; sentence_transformers: "
              f"{'sentence_transformers' in _s.modules}")
    except Exception:  # noqa: BLE001
        pass
    from oracle.stt import create_stt

    stt = create_stt()
    stt.load()
    step("STT loaded (parakeet)")
    from oracle.tts import KokoroTTS

    tts = KokoroTTS()
    tts.load()
    step("TTS client (sidecar or local)")
    from oracle.core import _get_retriever

    t = time.monotonic()
    r = _get_retriever()
    step(f"retriever + all FAISS + embedder ({time.monotonic() - t:.1f}s)")
    print(f"   torch imported: {'torch' in sys.modules}")
    if r:
        r.query("who was nikola tesla", mode="snappy")
        step("one RAG query (index pages touched)")
    from oracle.endpoint import warm

    warm()
    step("endpoint warm (vad backend)")
    from oracle.speaker import SpeakerId

    sid = SpeakerId()
    sid.load()
    step("speaker id (titanet)")
    from oracle.wakeword import WakeWordDetector

    try:
        w = WakeWordDetector(on_wake=lambda: None)
        w.start()
        time.sleep(2)
        step("wake word detector")
        w.stop()
    except Exception as e:  # noqa: BLE001
        print(f"   wakeword skipped: {e}")
    from oracle.commands import warm_thinking_acks

    class _VC:
        pass

    vc = _VC()
    vc.tts = tts
    warm_thinking_acks(vc)
    step("thinking acks synthesized")
    print(f"   torch imported at end: {'torch' in sys.modules}")


if __name__ == "__main__":
    main()
