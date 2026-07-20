"""Cross-process app state file so the diag UI can show what the running
radio-oracle.service is doing without sharing memory.

The main app (oracle/app.py) writes a small JSON snapshot on every state
transition; the diag server reads it on demand. Atomic write via tempfile +
rename so a partial read can never observe a torn document.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_STATE_PATH = Path(
    os.environ.get("ORACLE_STATE_FILE") or (os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
).expanduser()


def state_path() -> Path:
    """Resolve the state file location.

    `ORACLE_STATE_FILE` overrides everything if set to a full path. Otherwise
    the file lives at `$XDG_RUNTIME_DIR/radio-oracle-state.json` (typical
    systemd user runtime dir) or `/tmp/radio-oracle-state.json`.
    """
    env = os.environ.get("ORACLE_STATE_FILE")
    if env:
        return Path(env).expanduser()
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    return base / "radio-oracle-state.json"


def read_state() -> dict[str, Any] | None:
    """Read the current state snapshot. Returns None if absent or unreadable."""
    p = state_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"state read failed: {e}")
        return None


class StateWriter:
    """Atomically publish a snapshot of the app's state to disk.

    Cheap to construct; one writer per process.
    """

    def __init__(self, path: Path | None = None, pid: int | None = None) -> None:
        self._path = path or state_path()
        self._pid = pid if pid is not None else os.getpid()
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {
            "pid": self._pid,
            "started_at": time.time(),
            "mode": "starting",
            "power_on": False,
            "last_button": None,  # {"kind": "short"|"long", "ts": <unix>}
            "last_transition_ts": time.time(),
            "last_transcription": None,  # {"text": str, "ts": <unix>}
            "updated_at": time.time(),
        }
        self._write()

    def update(self, **fields: Any) -> None:
        self._snapshot.update(fields)
        self._snapshot["updated_at"] = time.time()
        self._write()

    def set_mode(self, mode: str) -> None:
        self._snapshot["mode"] = mode
        self._snapshot["last_transition_ts"] = time.time()
        self._write()

    def set_power(self, on: bool) -> None:
        self._snapshot["power_on"] = bool(on)
        self._write()

    def record_button(self, kind: str, duration: float) -> None:
        self._snapshot["last_button"] = {
            "kind": kind,
            "duration": round(duration, 3),
            "ts": time.time(),
        }
        self._write()

    def record_transcription(self, text: str) -> None:
        self._snapshot["last_transcription"] = {"text": text, "ts": time.time()}
        self._write()

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"state clear failed: {e}")

    def _write(self) -> None:
        with self._lock:
            self._snapshot["updated_at"] = time.time()
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "w",
                    dir=self._path.parent,
                    prefix=".radio-oracle-state-",
                    suffix=".json",
                    delete=False,
                ) as tmp:
                    json.dump(self._snapshot, tmp)
                    tmp_name = tmp.name
                os.replace(tmp_name, self._path)
            except OSError as e:
                logger.debug(f"state write failed: {e}")
