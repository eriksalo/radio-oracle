"""Durable activity journal — what the user read, played and asked.

The dashboard feed (``oracle.activity.emit``) already sees every event
worth remembering but forgets it in a 256 KB tmpfs file. This module
keeps the durable kinds in an ``events`` table in oracle.db, tagged with
the user and session, and renders a deterministic "recent activity"
block for the prompt — no LLM in the loop, so "you were on chapter three
of Moby Dick yesterday" is always right.

Thread-safety: the reader loop records from its worker thread while
turns record from the event loop — same sqlite pattern as books/library.py
(check_same_thread=False + a lock). ``record`` never raises.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from config.settings import settings

# Kinds copied from the activity feed as-is (see oracle.activity.emit).
DURABLE_ACTIVITY_KINDS = frozenset({"playing", "asked", "answered", "music_request"})


class Journal:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.user: str = settings.default_user
        self.session_id: str | None = None
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    user TEXT NOT NULL,
                    session_id TEXT,
                    kind TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user, ts);
                CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            """)
            self._conn.commit()

    # ------------------------------------------------------------- write

    def set_context(self, user: str | None = None, session_id: str | None = None) -> None:
        with self._lock:
            if user:
                self.user = user
            self.session_id = session_id

    def record(self, kind: str, **fields) -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO events (ts, user, session_id, kind, data) VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now(UTC).isoformat(),
                        self.user,
                        self.session_id,
                        kind,
                        json.dumps(fields, ensure_ascii=False, default=str),
                    ),
                )
                self._conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"journal record failed: {e}")

    # -------------------------------------------------------------- read

    def events(
        self,
        user: str | None = None,
        kinds: tuple[str, ...] | None = None,
        session_id: str | None = None,
        exclude_session: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        sql = "SELECT ts, user, session_id, kind, data FROM events WHERE 1=1"
        args: list = []
        if user:
            sql += " AND user = ?"
            args.append(user)
        if kinds:
            sql += f" AND kind IN ({','.join('?' for _ in kinds)})"
            args.extend(kinds)
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        if exclude_session:
            sql += " AND (session_id IS NULL OR session_id != ?)"
            args.append(exclude_session)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"])
            out.append(d)
        return out

    def recent_summary(
        self,
        user: str | None = None,
        exclude_session: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """The prompt block. Only *previous* sessions' questions/music are
        listed (the current session is already in the history, and a block
        that changed every turn would defeat the prompt prefix cache); the
        current book is always included."""
        user = user or self.user
        now = now or datetime.now(UTC)
        lines: list[str] = []

        book = self.events(user, kinds=("book",), limit=1)
        if book:
            d = book[0]["data"]
            when = _ago(book[0]["ts"], now)
            status = d.get("status") or d.get("book", "")
            verb = {"started": "started", "stopped": "stopped at", "chapter": "moved to"}.get(
                d.get("event", ""), "was at"
            )
            if d.get("event") == "finished":
                lines.append(f"Finished reading {d.get('book', '')} {when}.")
            else:
                lines.append(f"Current book: {status} ({verb} this {when}).")

        plays = self.events(user, kinds=("playing",), exclude_session=exclude_session, limit=60)
        if plays:
            artists = Counter(p["data"].get("artist") for p in plays if p["data"].get("artist"))
            top = ", ".join(f"{a} ({n}×)" if n > 1 else a for a, n in artists.most_common(5))
            if top:
                lines.append(
                    f"Music played recently: {top}. Last played {_ago(plays[0]['ts'], now)}."
                )
        reqs = self.events(user, kinds=("music_request",), exclude_session=exclude_session, limit=5)
        if reqs:
            asked_for = ", ".join(
                dict.fromkeys(r["data"].get("query", "") for r in reqs if r["data"].get("query"))
            )
            if asked_for:
                lines.append(f"Music asked for by name: {asked_for}.")

        qs = self.events(user, kinds=("asked",), exclude_session=exclude_session, limit=30)
        seen: list[str] = []
        for q in qs:
            t = (q["data"].get("text") or "").strip()
            if t and t not in seen:
                seen.append(t)
            if len(seen) == 5:
                break
        if seen:
            lines.append(
                "Questions asked in earlier sessions (most recent first): "
                + "; ".join(f"“{s}”" for s in seen)
                + f". Most recent was {_ago(qs[0]['ts'], now)}."
            )
        return "\n".join(lines)

    def session_activity_text(self, session_id: str) -> str:
        """Compact log of a session's books / music / questions for the
        summarizer (chronological)."""
        evs = list(
            reversed(
                self.events(
                    session_id=session_id,
                    kinds=("book", "music_request", "playing", "asked"),
                    limit=200,
                )
            )
        )
        out: list[str] = []
        played: list[str] = []
        for e in evs:
            d = e["data"]
            if e["kind"] == "book":
                out.append(f"book {d.get('event', '')}: {d.get('status') or d.get('book', '')}")
            elif e["kind"] == "music_request":
                out.append(f"asked for music: {d.get('query', '')}")
            elif e["kind"] == "playing":
                label = f"{d.get('artist', '')} — {d.get('title', '')}".strip(" —")
                if label and label not in played:
                    played.append(label)
            elif e["kind"] == "asked":
                out.append(f"asked: {d.get('text', '')}")
        if played:
            out.append("music played: " + ", ".join(played[:12]))
        return "\n".join(out)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _ago(iso: str, now: datetime) -> str:
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return "recently"
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    delta = now - t
    s = delta.total_seconds()
    if s < 90:
        return "moment"
    if s < 3600:
        return f"{int(s // 60)} minutes ago"
    if s < 86400 and t.date() == now.date():
        return "today"
    days = (now.date() - t.date()).days
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return t.strftime("on %B %d")


# ------------------------------------------------------------- singleton

_journal: Journal | None = None
_journal_lock = threading.Lock()


def get_journal() -> Journal | None:
    global _journal
    with _journal_lock:
        if _journal is None:
            try:
                _journal = Journal()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Activity journal unavailable: {e}")
                return None
        return _journal


def record(kind: str, **fields) -> None:
    """Record an event on the shared journal. Never raises."""
    j = get_journal()
    if j is not None:
        j.record(kind, **fields)


def set_context(user: str | None = None, session_id: str | None = None) -> None:
    j = get_journal()
    if j is not None:
        j.set_context(user, session_id)
