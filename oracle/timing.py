"""Per-stage timing for one voice turn.

A ``TurnTimer`` is created when a turn starts (wake word / button) and
marks each stage as it completes: record, stt, rewrite, retrieve, build,
first_token, first_audio, total. The LLM client drops Ollama's own
``prompt_eval_duration`` into it, so prefill cost is measured by the
runtime rather than inferred from wall clock.

The current timer travels in a ``ContextVar`` so the dispatcher, the turn,
the LLM client and the TTS worker thread (``asyncio.to_thread`` copies the
context) all see the same object without threading it through every
signature.

The headline number is ``ttfa`` — end of the user's speech to first audio
out — since that is what the user perceives as "how long did it think".
"""

from __future__ import annotations

import contextvars
import threading
import time

from loguru import logger

_current: contextvars.ContextVar[TurnTimer | None] = contextvars.ContextVar(
    "turn_timer", default=None
)

# Stage order for the log line; anything else recorded is appended after.
_ORDER = (
    "record",
    "stt",
    "rewrite",
    "retrieve",
    "build",
    "prefill",
    "first_token",
    "first_audio",
    "ttfa",
    "total",
)


class TurnTimer:
    """Wall-clock stage marks for one turn. Thread-safe, never raises."""

    def __init__(self, label: str = "turn") -> None:
        self.label = label
        self._t0 = time.monotonic()
        self._last = self._t0
        self._speech_end: float | None = None
        self._stages: dict[str, float] = {}
        self._abs: dict[str, float] = {}  # absolute time of each mark()
        self._extra: dict[str, float | int | str] = {}
        self._lock = threading.Lock()
        self.finished = False

    # ------------------------------------------------------------- marks

    def mark(self, stage: str) -> float:
        """Record *stage* as the time since the previous mark. Returns it."""
        now = time.monotonic()
        with self._lock:
            dt = now - self._last
            self._last = now
            self._stages[stage] = dt
            self._abs[stage] = now
            if stage == "record":
                self._speech_end = now
        return dt

    def mark_once(self, stage: str) -> None:
        """Like mark(), but only the first call for *stage* counts —
        for events that recur (first sentence played) or race (threads)."""
        with self._lock:
            if stage in self._stages:
                return
        self.mark(stage)

    def set(self, stage: str, seconds: float) -> None:
        """Record an externally measured duration (e.g. Ollama's prefill)."""
        with self._lock:
            self._stages[stage] = seconds

    def note(self, **fields: float | int | str) -> None:
        """Attach non-duration facts (token counts, model name)."""
        with self._lock:
            self._extra.update(fields)

    def speech_ended(self) -> None:
        """Explicitly mark the end of user speech when no record stage ran
        (text pre-supplied by the dispatcher)."""
        with self._lock:
            if self._speech_end is None:
                self._speech_end = time.monotonic()

    # ----------------------------------------------------------- summary

    def summary(self) -> dict[str, float | int | str]:
        now = time.monotonic()
        with self._lock:
            out: dict[str, float | int | str] = dict(self._stages)
            out["total"] = now - self._t0
            # Since the end of speech: what the user actually waits.
            if self._speech_end is not None and "first_audio" in self._abs:
                out["ttfa"] = self._abs["first_audio"] - self._speech_end
            out.update(self._extra)
        return out

    def finish(self) -> dict[str, float | int | str]:
        """Log the stage line and publish it to the activity feed."""
        self.finished = True
        s = self.summary()
        ordered = [k for k in _ORDER if k in s] + [
            k for k in s if k not in _ORDER and isinstance(s[k], float)
        ]
        parts = [f"{k}={s[k]:.2f}s" for k in ordered]
        extras = [f"{k}={v}" for k, v in s.items() if not isinstance(v, float)]
        logger.info(f"TURN {self.label}: " + " ".join(parts + extras))
        try:
            from oracle.activity import emit

            rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}
            emit("timing", label=self.label, **rounded)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"timing emit failed: {e}")
        return s


# --------------------------------------------------------------- context


def current() -> TurnTimer | None:
    return _current.get()


def start(label: str = "turn") -> TurnTimer:
    """Create a timer and make it the current one for this context."""
    t = TurnTimer(label)
    _current.set(t)
    return t


def get_or_start(label: str = "turn") -> TurnTimer:
    """The current unfinished timer, or a fresh one."""
    t = _current.get()
    if t is not None and not t.finished:
        return t
    return start(label)


def clear() -> None:
    _current.set(None)


def mark(stage: str) -> None:
    """Mark on the current timer, if any. Safe to call anywhere."""
    t = _current.get()
    if t is not None:
        t.mark(stage)


def mark_once(stage: str) -> None:
    t = _current.get()
    if t is not None:
        t.mark_once(stage)
