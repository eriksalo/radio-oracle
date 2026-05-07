"""Background `tegrastats` poller for Jetson GPU/RAM/temp metrics.

`tegrastats` is the Jetson-only utility that prints one line per interval
with GPU%, RAM, EMC, temps, etc. We launch it once on diag startup and
parse each line into a dict the /api/gpu endpoint can serve.

Falls back to "tegrastats unavailable" silently on dev machines where the
binary doesn't exist (e.g. a laptop running the diag for UI work).
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from typing import Any

from loguru import logger

# Match enough of the tegrastats output to surface useful numbers without
# being brittle. tegrastats' format varies across L4T releases; we extract
# best-effort and leave None when a field is missing.
_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_RE_SWAP = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
_RE_GPU_PCT = re.compile(r"GR3D_FREQ\s+(\d+)%")
_RE_GPU_FREQ = re.compile(r"GR3D_FREQ\s+\d+%@\[?(\d+)")
_RE_CPU_PCT = re.compile(r"CPU\s+\[([^\]]+)\]")
_RE_TEMP = re.compile(r"(\w+)@(-?\d+(?:\.\d+)?)C")


class TegrastatsState:
    """Latest parsed tegrastats line + when we got it."""

    def __init__(self) -> None:
        self.available: bool = False
        self.last_update: float = 0.0
        self.values: dict[str, Any] = {}
        self.raw_line: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "last_update": self.last_update,
            "stale_sec": (time.time() - self.last_update) if self.last_update else None,
            **self.values,
            "raw": self.raw_line,
        }


_state = TegrastatsState()
_task: asyncio.Task | None = None


def parse_line(line: str) -> dict[str, Any]:
    """Extract the structured fields we care about from one tegrastats line."""
    out: dict[str, Any] = {}

    if (m := _RE_RAM.search(line)):
        out["ram_used_mb"] = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))
    if (m := _RE_SWAP.search(line)):
        out["swap_used_mb"] = int(m.group(1))
        out["swap_total_mb"] = int(m.group(2))
    if (m := _RE_GPU_PCT.search(line)):
        out["gpu_pct"] = int(m.group(1))
    if (m := _RE_GPU_FREQ.search(line)):
        out["gpu_freq_mhz"] = int(m.group(1))
    if (m := _RE_CPU_PCT.search(line)):
        cpus: list[int] = []
        for part in m.group(1).split(","):
            pct = part.split("%")[0].strip()
            if pct.isdigit():
                cpus.append(int(pct))
            elif pct == "off":
                cpus.append(-1)
        if cpus:
            out["cpu_per_core_pct"] = cpus

    temps: dict[str, float] = {}
    for name, val in _RE_TEMP.findall(line):
        try:
            temps[name] = float(val)
        except ValueError:
            continue
    if temps:
        out["temps_c"] = temps

    return out


async def _run() -> None:
    """Spawn tegrastats and stream-parse its output until cancelled."""
    binary = shutil.which("tegrastats")
    if binary is None:
        logger.info("tegrastats not on PATH — GPU stats disabled (likely a dev host)")
        _state.available = False
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--interval",
            "1000",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as e:
        logger.warning(f"failed to launch tegrastats: {e}")
        _state.available = False
        return

    _state.available = True
    logger.info("tegrastats poller started")
    try:
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parsed = parse_line(line)
            _state.values = parsed
            _state.raw_line = line
            _state.last_update = time.time()
    except asyncio.CancelledError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()
        raise
    finally:
        _state.available = False


def start() -> None:
    """Kick off the poller in the running event loop. Idempotent."""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.create_task(_run(), name="tegrastats-poller")


async def stop() -> None:
    global _task
    if _task is None:
        return
    _task.cancel()
    try:
        await _task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
    _task = None


def snapshot() -> dict[str, Any]:
    return _state.snapshot()
