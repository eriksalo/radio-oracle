"""Logging setup with an in-process ring buffer for the diag /api/logs feed."""

from __future__ import annotations

import sys
from collections import deque
from threading import Lock

from loguru import logger

from config.settings import settings

# Bounded ring buffer of recent log records. Lives in-process — the diag
# server's /api/logs endpoint reads from this. To get logs from a *different*
# process (e.g. the running radio-oracle service) the diag uses journalctl.
_LOG_RING: deque[dict] = deque(maxlen=1000)
_LOG_LOCK = Lock()


def _ring_sink(message) -> None:
    """Loguru sink: append a structured record to the ring buffer."""
    record = message.record
    entry = {
        "ts": record["time"].isoformat(timespec="seconds"),
        "level": record["level"].name,
        "name": record["name"],
        "message": record["message"],
    }
    with _LOG_LOCK:
        _LOG_RING.append(entry)


def get_recent_logs(tail: int = 200, level: str | None = None) -> list[dict]:
    """Return the last `tail` log entries (newest last). Optional level filter."""
    with _LOG_LOCK:
        items = list(_LOG_RING)
    if level:
        levels_at_or_above = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        wanted = level.upper()
        if wanted in levels_at_or_above:
            order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            min_idx = order.index(wanted)
            items = [e for e in items if e["level"] in order[min_idx:]]
    return items[-tail:]


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
            "<cyan>{name}</cyan> - {message}"
        ),
    )
    logger.add(
        "data/oracle.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )
    # Ring buffer sink — always on, cheap.
    logger.add(_ring_sink, level=settings.log_level, format="{message}")
