"""Quality evaluation: pose ~100 questions per channel, record what the radio
would say and do, and auto-grade against expectations.

Three question sets in docs/eval/:
  librarian.json  — knowledge questions through the real oracle turn
                    (retrieval → context → LLM). Follow-ups (``followup``)
                    stay in the previous question's session so the rewrite
                    path is exercised; everything else gets a fresh session.
  music.json      — voice commands through ``commands.classify`` (keyword
                    table → question heuristic → LLM intent), then resolved
                    against the music catalog the way ``_play_query`` would.
  books.json      — the same for the book channel: title/author resolution
                    through ``Library.search`` + the confidence check, chapter
                    specs, exploration.

Nothing is played or spoken: TTS and playback are stubbed (the LLM and the
retriever are real), so a run is ~30 min for all three sets. Like sim_turn.py
it needs the radio-oracle service stopped (memory) and copies the production
oracle.db to a scratch file so the eval never pollutes long-term memory:

    sudo SIM_SCRIPT=scripts/eval_quality.py SIM_LOG=/tmp/eval_quality.log \\
        /opt/radio-oracle/scripts/sim_turn.sh --out /tmp/eval_results

Output: one JSONL per domain (every question with the answer / decision,
retrieved sources, timings and auto-grades) plus summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


# ----------------------------------------------------------------- helpers


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has(hay: str, needle: str) -> bool:
    return needle.lower() in hay.lower()


def _any(hay: str, needles: list[str] | None) -> bool | None:
    """True/False when *needles* given; None when the check doesn't apply."""
    if not needles:
        return None
    return any(_has(hay, n) for n in needles)


def _keyword_grade(answer: str, groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    """Every group must have at least one member in the answer."""
    missing = [g for g in groups if not any(_has(answer, k) for k in g)]
    return (not missing, missing)


class _StubTTS:
    """Stands in for KokoroTTS: the pipeline still splits and 'synthesizes'."""

    sample_rate = 24000

    def load(self) -> None:
        pass

    def synthesize(self, text: str):
        import numpy as np

        return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)


class _StubSTT:
    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


# ----------------------------------------------------------------- librarian


async def _run_librarian(items: list[dict], out: Path, limit: int | None) -> dict:
    from loguru import logger

    from oracle import activity, audio, core, timing
    from oracle.memory.context import ContextBuilder

    audio.play_audio = lambda a, sample_rate=None, should_abort=None: None  # type: ignore[assignment]

    events: list[dict] = []
    orig_emit = activity.emit

    def capture_emit(kind: str, **fields: Any) -> None:
        events.append({"kind": kind, **fields})
        orig_emit(kind, **fields)

    activity.emit = capture_emit  # type: ignore[assignment]

    rows: list[dict] = []
    orig_finish = timing.TurnTimer.finish

    def capture_finish(self):
        s = orig_finish(self)
        rows.append(dict(s))
        return s

    timing.TurnTimer.finish = capture_finish  # type: ignore[method-assign]

    system_prompt, store, session_id = await core._init_common()
    t0 = time.monotonic()
    retriever = await asyncio.to_thread(core._get_retriever)
    print(f"# retriever warm-up {time.monotonic() - t0:.1f}s", flush=True)

    retrieved: list[dict] = []
    if retriever is not None:
        orig_query = retriever.query

        def capture_query(*a, **k):
            res = orig_query(*a, **k)
            retrieved[:] = [
                {
                    "source": r.get("source"),
                    "title": (r.get("metadata") or {}).get("title"),
                    "distance": round(float(r.get("distance", 0.0)), 3),
                }
                for r in res
            ]
            return res

        retriever.query = capture_query  # type: ignore[method-assign]

    vc = core.VoiceContext(
        stt=None,  # type: ignore[arg-type]
        stt_fast=_StubSTT(),  # type: ignore[arg-type]
        tts=_StubTTS(),  # type: ignore[arg-type]
        store=store,
        ctx_builder=ContextBuilder(store, session_id),
        system_prompt=system_prompt,
        session_id=session_id,
    )

    path = out / "librarian.jsonl"
    summary = {"n": 0, "keyword_pass": 0, "errors": 0, "no_rag": 0}
    with path.open("w", encoding="utf-8") as f:
        for item in items[:limit]:
            if not item.get("followup"):
                vc.session_id = store.new_session()
                vc.ctx_builder = ContextBuilder(store, vc.session_id)
            events.clear()
            retrieved.clear()
            n_rows = len(rows)
            t_start = time.monotonic()
            err = None
            try:
                await asyncio.wait_for(core.voice_turn(vc, pre_text=item["text"]), timeout=240)
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                logger.warning(f"{item['id']}: {err}")
            answer = next((e.get("text", "") for e in events if e["kind"] == "answered"), "")
            timing_row = rows[-1] if len(rows) > n_rows else {}
            ok, missing = _keyword_grade(answer, item.get("expect", []))
            rec = {
                "id": item["id"],
                "text": item["text"],
                "followup": bool(item.get("followup")),
                "answer": answer,
                "words": len(answer.split()),
                "keyword_pass": ok,
                "missing": missing,
                "retrieved": list(retrieved),
                "rag_chars": timing_row.get("rag_chars"),
                "timing": {
                    k: timing_row.get(k)
                    for k in (
                        "retrieve",
                        "prefill",
                        "first_token",
                        "total",
                        "decode_tps",
                        "prompt_tokens",
                        "cached_tokens",
                        "done_reason",
                    )
                    if k in timing_row
                },
                "wall_s": round(time.monotonic() - t_start, 2),
                "error": err,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            summary["n"] += 1
            summary["keyword_pass"] += int(ok)
            summary["errors"] += int(err is not None)
            summary["no_rag"] += int(not rec["rag_chars"])
            mark = "ok " if ok else "MISS"
            print(
                f"{item['id']:5s} {mark} {rec['wall_s']:5.1f}s {rec['words']:3d}w "
                f"rag={rec['rag_chars'] or 0:5d} {item['text'][:48]}",
                flush=True,
            )
    await core.voice_close(vc)
    return summary


# ----------------------------------------------------------------- commands


def _action_ok(action: str, expect: dict) -> bool:
    if "action_any" in expect:
        return action in expect["action_any"]
    return action == expect.get("action")


async def _classify_all(
    items: list[dict], limit: int | None
) -> list[tuple[dict, str, str | None, float]]:
    from oracle import commands, core
    from oracle.memory.context import ContextBuilder

    system_prompt, store, session_id = await core._init_common()
    vc = core.VoiceContext(
        stt=None,  # type: ignore[arg-type]
        stt_fast=_StubSTT(),  # type: ignore[arg-type]
        tts=_StubTTS(),  # type: ignore[arg-type]
        store=store,
        ctx_builder=ContextBuilder(store, session_id),
        system_prompt=system_prompt,
        session_id=session_id,
    )
    out: list[tuple[dict, str, str | None, float]] = []
    for item in items[:limit]:
        t0 = time.monotonic()
        try:
            action, query = await asyncio.wait_for(commands.classify(vc, item["text"]), timeout=60)
        except Exception as e:  # noqa: BLE001
            action, query = f"error:{e!r}", None
        out.append((item, action, query, round(time.monotonic() - t0, 2)))
    store.close()
    return out


async def _run_music(items: list[dict], out: Path, limit: int | None) -> dict:
    from oracle import commands
    from oracle.music.catalog import Catalog

    catalog = Catalog()
    decided = await _classify_all(items, limit)
    path = out / "music.jsonl"
    summary = {"n": 0, "action_pass": 0, "resolve_pass": 0, "resolve_n": 0, "llm_intent": 0}
    with path.open("w", encoding="utf-8") as f:
        for item, action, query, secs in decided:
            expect = item["expect"]
            rec: dict[str, Any] = {
                "id": item["id"],
                "text": item["text"],
                "action": action,
                "query": query,
                "classify_s": secs,
                "action_pass": _action_ok(action, expect),
                "resolve_pass": None,
                "spoken": None,
            }
            if secs > 0.5:
                summary["llm_intent"] += 1
            if action == "play":
                if query:
                    hits = catalog.search(query)
                    artists = sorted({t.artist for t in hits if t.artist})
                    rec["hits"] = len(hits)
                    rec["hit_artists"] = artists[:8]
                    rec["hit_genres"] = sorted({t.genre for t in hits if t.genre})[:6]
                    if hits:
                        rec["spoken"] = (
                            f"Playing {hits[0].artist or hits[0].album or hits[0].title}."
                        )
                    else:
                        rec["spoken"] = f"I couldn't find anything for {query}."
                    checks = []
                    for key, attr in (
                        ("artist_any", "artist"),
                        ("genre_any", "genre"),
                        ("title_any", "title"),
                    ):
                        wanted = expect.get(key)
                        if not wanted:
                            continue
                        matched = [t for t in hits if _any(getattr(t, attr) or "", wanted)]
                        checks.append(bool(matched))
                        rec[f"{attr}_precision"] = (
                            round(len(matched) / len(hits), 2) if hits else 0.0
                        )
                    if checks:
                        rec["resolve_pass"] = all(checks)
                else:
                    rec["spoken"] = "What would you like to hear?"
                    if any(k in expect for k in ("artist_any", "genre_any", "title_any")):
                        rec["resolve_pass"] = False
            elif action == "list_music":
                q = query or commands._extract_qualifier(item["text"])
                rec["spoken"] = commands._describe_music(catalog, q)
                if expect.get("query_contains"):
                    rec["resolve_pass"] = _has(q or "", expect["query_contains"])
            elif action == "about_device":
                rec["spoken"] = "(device description)"
            if rec["resolve_pass"] is not None:
                summary["resolve_n"] += 1
                summary["resolve_pass"] += int(rec["resolve_pass"])
            summary["n"] += 1
            summary["action_pass"] += int(rec["action_pass"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = "ok " if rec["action_pass"] and rec["resolve_pass"] is not False else "MISS"
            print(
                f"{item['id']:5s} {mark} {secs:4.1f}s {action:12s} {str(query)[:24]:24s} "
                f"{item['text'][:40]}",
                flush=True,
            )
    catalog.close()
    return summary


async def _run_books(items: list[dict], out: Path, limit: int | None) -> dict:
    from oracle import commands
    from oracle.books.library import Library
    from oracle.books.session import ReaderSession

    lib = Library()
    decided = await _classify_all(items, limit)
    path = out / "books.jsonl"
    summary = {
        "n": 0,
        "action_pass": 0,
        "resolve_pass": 0,
        "resolve_n": 0,
        "confident": 0,
        "llm_intent": 0,
    }
    with path.open("w", encoding="utf-8") as f:
        for item, action, query, secs in decided:
            expect = item["expect"]
            rec: dict[str, Any] = {
                "id": item["id"],
                "text": item["text"],
                "context": item.get("context", "music"),
                "action": action,
                "query": query,
                "classify_s": secs,
                "action_pass": _action_ok(action, expect),
                "resolve_pass": None,
                "spoken": None,
            }
            if secs > 0.5:
                summary["llm_intent"] += 1
            if action == "read_book":
                q = query or ""
                hits = lib.search(q) if q else []
                rec["hits"] = len(hits)
                rec["top"] = [f"{b.title} — {b.author}" for b in hits[:5]]
                if hits:
                    first = hits[0]
                    rec["confident"] = ReaderSession.is_confident_match(q, first)
                    summary["confident"] += int(rec["confident"])
                    t_ok = _any(first.title, expect.get("title_any"))
                    a_ok = _any(first.author, expect.get("author_any"))
                    checks = [c for c in (t_ok, a_ok) if c is not None]
                    if checks:
                        # Title OR author is enough when both are given
                        # ("Read Walden" → Thoreau's Walden, either field).
                        rec["resolve_pass"] = any(checks)
                    rec["in_top5"] = any(
                        (_any(b.title, expect.get("title_any")) or False)
                        or (_any(b.author, expect.get("author_any")) or False)
                        for b in hits[:5]
                    )
                    rec["spoken"] = f"{first.title}" + (
                        f", by {first.author}" if first.author else ""
                    )
                else:
                    rec["spoken"] = f"I couldn't find {q}." if q else "Which book?"
                    if expect.get("title_any") or expect.get("author_any"):
                        rec["resolve_pass"] = False
            elif action == "list_books":
                q = query or commands._extract_qualifier(item["text"])
                rec["spoken"] = commands._describe_books(q)
                if expect.get("query_contains"):
                    rec["resolve_pass"] = _has(q or "", expect["query_contains"])
            elif action == "goto_chapter":
                spec = query or commands._extract_chapter_spec(item["text"])
                rec["spec"] = spec
                if expect.get("spec_any"):
                    rec["resolve_pass"] = _any(spec or "", expect["spec_any"])
            if rec["resolve_pass"] is not None:
                summary["resolve_n"] += 1
                summary["resolve_pass"] += int(rec["resolve_pass"])
            summary["n"] += 1
            summary["action_pass"] += int(rec["action_pass"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = "ok " if rec["action_pass"] and rec["resolve_pass"] is not False else "MISS"
            print(
                f"{item['id']:5s} {mark} {secs:4.1f}s {action:12s} {str(query)[:24]:24s} "
                f"{(rec['spoken'] or '')[:50]}",
                flush=True,
            )
    lib.close()
    return summary


# ----------------------------------------------------------------- main


async def _main(args: argparse.Namespace) -> int:
    prod_db = Path(os.environ.get("ORACLE_DB_PATH", "data/oracle.db"))
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_quality_"))
    tmp_db = tmp_dir / "oracle.db"
    if prod_db.exists():
        shutil.copy2(prod_db, tmp_db)
    os.environ["ORACLE_DB_PATH"] = str(tmp_db)
    os.environ.setdefault("ORACLE_ACTIVITY_FILE", str(tmp_dir / "activity.jsonl"))

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    summary: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "domains": {}}
    runners = {"librarian": _run_librarian, "music": _run_music, "books": _run_books}
    for d in domains:
        items = _load(ROOT / "docs" / "eval" / f"{d}.json")
        print(f"\n## {d}: {len(items)} items", flush=True)
        t0 = time.monotonic()
        summary["domains"][d] = await runners[d](items, out, args.limit)
        summary["domains"][d]["wall_s"] = round(time.monotonic() - t0, 1)
        print(f"# {d}: {json.dumps(summary['domains'][d])}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--domains", default="music,books,librarian")
    p.add_argument("--out", default="/tmp/eval_results")
    p.add_argument("--limit", type=int, default=None, help="first N items per domain")
    p.add_argument("--log-level", default="WARNING")
    sys.exit(asyncio.run(_main(p.parse_args())))


if __name__ == "__main__":
    main()
