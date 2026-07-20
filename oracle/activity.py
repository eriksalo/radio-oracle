"""Cross-process activity feed: what the oracle is doing, hearing, deciding.

The main app appends one JSON line per event to a small file in the
runtime dir (tmpfs); the diagnostics dashboard tails it. Same pattern as
oracle.state but for a rolling feed rather than a snapshot.

Events are best-effort: emit() must never raise into the voice pipeline.
Kinds in use: phase, wake, heard, decided, spoke, asked, answered,
consulted, playing, reading, error.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

_MAX_BYTES = 256 * 1024  # rewrite keeping the tail once the file grows past this
_KEEP_LINES = 200
_TEXT_LIMIT = 300  # clip long free text (answers) per event

_lock = threading.Lock()
_next_id: int | None = None


def activity_path() -> Path:
    env = os.environ.get("ORACLE_ACTIVITY_FILE")
    if env:
        return Path(env).expanduser()
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    return base / "radio-oracle-activity.jsonl"


def emit(kind: str, **fields: Any) -> None:
    """Append one event. Cheap, thread-safe, never raises."""
    global _next_id
    try:
        for k, v in fields.items():
            if isinstance(v, str) and len(v) > _TEXT_LIMIT:
                fields[k] = v[: _TEXT_LIMIT - 1] + "…"
        with _lock:
            p = activity_path()
            if _next_id is None:
                _next_id = _scan_last_id(p) + 1
            event = {"id": _next_id, "ts": time.time(), "kind": kind, **fields}
            _next_id += 1
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            if p.stat().st_size > _MAX_BYTES:
                _truncate(p)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"activity emit failed: {e}")


def read_events(after: int = 0, limit: int = 100) -> list[dict]:
    """Events with id > after, oldest first, at most limit (from the tail)."""
    try:
        lines = activity_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines[-max(limit * 3, _KEEP_LINES) :]:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("id", 0) > after:
            out.append(ev)
    return out[-limit:]


def _scan_last_id(p: Path) -> int:
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                return int(json.loads(line).get("id", 0))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    except OSError:
        pass
    return 0


def _truncate(p: Path) -> None:
    lines = p.read_text(encoding="utf-8").splitlines()[-_KEEP_LINES:]
    tmp = p.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
