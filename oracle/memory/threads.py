"""Conversation threads — one topic that can span many sessions.

A session is one power-on; a *thread* is "the well pump" or "sourdough
starter", picked up again days later. Every answered question is filed
into a thread after the answer has been spoken (never on the turn's
critical path):

* a follow-up that leans on the previous turn ("why is that?") stays in
  the thread the conversation is on;
* otherwise the question is embedded and compared with the user's recent
  threads — close enough joins one, anything else opens a new thread.

Titles and summaries are written later by the summarizer (session close,
or the catch-up at next boot). What the radio *says* when it picks a
thread back up is built from the stored turns, not from a model, so
"you last asked …" is always what was actually asked.

Thresholds (settings.thread_*) were set on 22 hand-written questions in
7 topics with nomic-v1.5 and the ``clustering:`` prefix (2026-11-25):
question-to-question cosine is 0.60–0.85 within a topic and up to 0.72
across topics, so similarity alone cannot separate them. A shared topic
word in the 0.65–0.75 band does: 13/15 same-topic questions joined,
2/129 cross-topic ones did. Not yet checked on real conversations.

Thread-safety: turns are filed from a worker thread while commands read
from the event loop — same sqlite pattern as journal.py.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings

EmbedFn = Callable[[str], "np.ndarray | None"]

# nomic-embed task prefix for "are these two texts about the same thing".
EMBED_PREFIX = "clustering: "

_STOP = frozenset(
    "a an the is are was were be been do does did how what why when where who which should "
    "could would can will my your our i we you it its of in on at to for from with and or not "
    "so that this there keep much many make need get got have has had about off into out up "
    "down as by if than then too very often tell me please more some any one thing things "
    "know mean say said".split()
)


def topic_words(text: str) -> set[str]:
    """Lowercase content words, crudely stemmed — the lexical half of the
    same-topic test ("pump", "starter", "napoleon")."""
    out: set[str] = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


@dataclass(frozen=True)
class Thread:
    id: int
    user: str
    title: str | None
    summary: str | None
    status: str  # "open" | "done"
    created_at: str
    last_ts: str
    last_session: str | None
    turns: int
    summarized_turns: int
    first_question: str
    shelf: str | None = None  # oracle/rag/shelves.py key

    @property
    def label(self) -> str:
        """What to call the thread aloud: its title, else the question
        that opened it."""
        if self.title:
            return self.title
        return f"your question, {self.first_question.rstrip(' ?.!')}"


class ThreadStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS threads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    title TEXT,
                    summary TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    last_ts TEXT NOT NULL,
                    last_session TEXT,
                    turns INTEGER NOT NULL DEFAULT 0,
                    summarized_turns INTEGER NOT NULL DEFAULT 0,
                    first_question TEXT NOT NULL DEFAULT '',
                    words TEXT NOT NULL DEFAULT '',
                    vec_sum BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_threads_user_ts ON threads(user, last_ts);
                CREATE TABLE IF NOT EXISTS thread_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id INTEGER NOT NULL,
                    session_id TEXT,
                    ts TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES threads(id)
                );
                CREATE INDEX IF NOT EXISTS idx_thread_turns_thread ON thread_turns(thread_id, id);
            """)
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(threads)")}
            if "shelf" not in cols:  # databases from before the shelves
                self._conn.execute("ALTER TABLE threads ADD COLUMN shelf TEXT")
            self._conn.commit()

    # ------------------------------------------------------------- write

    def create(self, user: str, first_question: str, shelf: str | None = None) -> int:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO threads (user, created_at, last_ts, first_question, shelf) "
                "VALUES (?, ?, ?, ?, ?)",
                (user, now, now, first_question.strip(), shelf),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_turn(
        self,
        thread_id: int,
        session_id: str | None,
        question: str,
        answer: str,
        vec: np.ndarray | None = None,
    ) -> None:
        """Append one exchange; a joined thread is open again."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT words, vec_sum FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if row is None:
                return
            words = set(row["words"].split()) | topic_words(question)
            vec_sum = _from_blob(row["vec_sum"])
            if vec is not None:
                vec_sum = vec if vec_sum is None or vec_sum.shape != vec.shape else vec_sum + vec
            self._conn.execute(
                "INSERT INTO thread_turns (thread_id, session_id, ts, question, answer) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, session_id, now, question.strip(), answer.strip()),
            )
            self._conn.execute(
                "UPDATE threads SET turns = turns + 1, last_ts = ?, last_session = ?, "
                "status = 'open', words = ?, vec_sum = ? WHERE id = ?",
                (
                    now,
                    session_id,
                    " ".join(sorted(words)),
                    None if vec_sum is None else vec_sum.astype(np.float32).tobytes(),
                    thread_id,
                ),
            )
            self._conn.commit()

    def set_label(self, thread_id: int, title: str | None, summary: str, upto_turns: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE threads SET title = COALESCE(?, title), summary = ?, "
                "summarized_turns = ? WHERE id = ?",
                (title, summary, upto_turns, thread_id),
            )
            self._conn.commit()

    def set_shelf(self, thread_id: int, shelf: str) -> None:
        """Label a thread that has no shelf yet (the first one sticks)."""
        with self._lock:
            self._conn.execute(
                "UPDATE threads SET shelf = ? WHERE id = ? AND shelf IS NULL", (shelf, thread_id)
            )
            self._conn.commit()

    def set_status(self, thread_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE threads SET status = ? WHERE id = ?", (status, thread_id))
            self._conn.commit()

    # -------------------------------------------------------------- read

    _COLS = (
        "id, user, title, summary, status, created_at, last_ts, last_session, turns, "
        "summarized_turns, first_question, shelf"
    )

    def get(self, thread_id: int) -> Thread | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {self._COLS} FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
        return Thread(**dict(row)) if row else None

    def recent(
        self,
        user: str,
        limit: int = 40,
        min_turns: int = 1,
        status: str | None = None,
        shelf: str | None = None,
    ) -> list[Thread]:
        """The user's threads, most recently touched first."""
        sql = f"SELECT {self._COLS} FROM threads WHERE user = ? AND turns >= ?"
        args: list = [user, min_turns]
        if shelf:
            sql += " AND shelf = ?"
            args.append(shelf)
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY last_ts DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [Thread(**dict(r)) for r in rows]

    def match_data(self, user: str, limit: int = 40) -> list[tuple[int, np.ndarray | None, set]]:
        """(id, unit centroid, topic words) of the user's recent threads."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, vec_sum, words, title FROM threads WHERE user = ? AND turns >= 1 "
                "ORDER BY last_ts DESC, id DESC LIMIT ?",
                (user, limit),
            ).fetchall()
        out = []
        for r in rows:
            words = set(r["words"].split()) | topic_words(r["title"] or "")
            out.append((int(r["id"]), _unit(_from_blob(r["vec_sum"])), words))
        return out

    def turns(self, thread_id: int, limit: int | None = None) -> list[dict]:
        """Exchanges of a thread, oldest first (the last *limit* of them)."""
        sql = "SELECT ts, session_id, question, answer FROM thread_turns WHERE thread_id = ?"
        with self._lock:
            if limit:
                rows = self._conn.execute(
                    sql + " ORDER BY id DESC LIMIT ?", (thread_id, limit)
                ).fetchall()
                rows.reverse()
            else:
                rows = self._conn.execute(sql + " ORDER BY id", (thread_id,)).fetchall()
        return [dict(r) for r in rows]

    def unsummarized(self, limit: int = 3, min_turns: int = 2) -> list[Thread]:
        """Threads with exchanges the summarizer has not seen yet."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._COLS} FROM threads WHERE turns >= ? AND summarized_turns < turns "
                "ORDER BY last_ts DESC LIMIT ?",
                (min_turns, limit),
            ).fetchall()
        return [Thread(**dict(r)) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _from_blob(blob: bytes | None) -> np.ndarray | None:
    return None if not blob else np.frombuffer(blob, dtype=np.float32).copy()


def _unit(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else None


class ThreadTracker:
    """Files a session's turns into threads and remembers which thread
    the conversation is on."""

    def __init__(self, store: ThreadStore, embed_fn: EmbedFn | None = None) -> None:
        self.store = store
        self._embed_fn = embed_fn
        self.current_id: int | None = None
        # After a resume the next question belongs to that thread whatever
        # its wording ("what should I check next?" names no topic).
        self._sticky = False
        self._last_vec: tuple[str, np.ndarray | None] | None = None
        self._lock = threading.Lock()

    def _embed(self, text: str) -> np.ndarray | None:
        if self._embed_fn is None:
            return None
        try:
            # A unit-length copy for cosine. The retriever's own query
            # vectors stay un-normalised; this never feeds FAISS.
            return _unit(np.asarray(self._embed_fn(EMBED_PREFIX + text), dtype=np.float32))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Thread embedding unavailable: {e}")
            return None

    def find(self, user: str, text: str) -> int | None:
        """The thread *text* names ("the well pump"), or None. Used for
        "go back to our conversation about …": a spoken title is short, so
        a shared topic word decides, and similarity breaks ties."""
        words = topic_words(text)
        vec = self._embed(text)
        best: tuple[float, int] | None = None
        for tid, centroid, twords in self.store.match_data(user):
            overlap = len(words & twords)
            sim = float(vec @ centroid) if vec is not None and centroid is not None else 0.0
            if overlap == 0 and sim < settings.thread_join_sim:
                continue
            score = overlap + sim
            if best is None or score > best[0]:
                best = (score, tid)
        return best[1] if best else None

    def continues(self, user: str, question: str) -> Thread | None:
        """The thread the conversation is on, if *question* carries it on
        (same test as filing: close to the thread, or fairly close and
        sharing a topic word). Asked *before* the answer, so the lookup
        can be told the topic: "How often should I feed the starter?"
        fetched infant feeding in the middle of a sourdough conversation
        (2026-11-25). The embedding is kept for :meth:`file_turn`."""
        with self._lock:
            current = self.store.get(self.current_id) if self.current_id else None
        if current is None or current.user != user:
            return None
        vec = self._embed(question)
        self._last_vec = (question, vec)
        if vec is None:
            return None
        words = topic_words(question)
        for tid, centroid, twords in self.store.match_data(user):
            if tid != current.id:
                continue
            if centroid is None or centroid.shape != vec.shape:
                return None
            sim = float(vec @ centroid)
            if sim >= settings.thread_continue_sim or (
                sim >= settings.thread_join_sim_shared and bool(words & twords)
            ):
                return current
            return None
        return None

    def _match(self, user: str, question: str, vec: np.ndarray | None) -> int | None:
        if vec is None:
            return None
        words = topic_words(question)
        best: tuple[float, int] | None = None
        for tid, centroid, twords in self.store.match_data(user):
            if centroid is None or centroid.shape != vec.shape:
                continue
            sim = float(vec @ centroid)
            joins = (
                sim >= settings.thread_join_sim
                or (sim >= settings.thread_join_sim_shared and bool(words & twords))
                # The thread the conversation is on right now needs less
                # evidence than one from last week.
                or (tid == self.current_id and sim >= settings.thread_continue_sim)
            )
            if joins and (best is None or sim > best[0]):
                best = (sim, tid)
        return best[1] if best else None

    def file_turn(
        self,
        user: str,
        session_id: str | None,
        question: str,
        answer: str,
        followup: bool = False,
        shelf: str | None = None,
    ) -> int | None:
        """File one answered question; returns the thread id. *shelf* is
        the subject shelf the question was put on, if any. Never raises."""
        if not question.strip() or not answer.strip():
            return None
        try:
            with self._lock:
                current = self.store.get(self.current_id) if self.current_id else None
                if current is not None and current.user != user:
                    current = None  # the speaker changed
                sticky, self._sticky = self._sticky, False
                if (followup or sticky) and current is not None:
                    self.store.add_turn(current.id, session_id, question, answer)
                    if shelf and current.shelf is None:
                        self.store.set_shelf(current.id, shelf)
                    return current.id
                cached = self._last_vec
                vec = cached[1] if cached and cached[0] == question else self._embed(question)
                self._last_vec = None
                tid = self._match(user, question, vec)
                if tid is None:
                    tid = self.store.create(user, question, shelf)
                    logger.info(f"Thread {tid} opened ({shelf or 'no shelf'}): {question!r}")
                else:
                    logger.info(f"Thread {tid} continues: {question!r}")
                    if shelf:
                        self.store.set_shelf(tid, shelf)
                self.store.add_turn(tid, session_id, question, answer, vec)
                self.current_id = tid
                return tid
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not file the turn into a thread: {e}")
            return None

    def resume(self, thread_id: int) -> None:
        with self._lock:
            self.current_id = thread_id
            self._sticky = True

    def leave(self) -> None:
        with self._lock:
            self.current_id = None
            self._sticky = False


# ------------------------------------------------------------ rendering


def when_phrase(iso: str, now: datetime | None = None) -> str:
    """ "Yesterday" / "Three days ago" / "Earlier today" — sentence-initial."""
    from oracle.memory.journal import _ago

    ago = _ago(iso, now or datetime.now(UTC))
    if ago == "moment" or ago.endswith("minutes ago"):
        return "A little while ago"
    if ago == "today":
        return "Earlier today"
    return ago[0].upper() + ago[1:]


def recap_text(thread: Thread, turns: list[dict], now: datetime | None = None) -> str:
    """What the radio says when it picks a thread back up. Built from the
    stored turns only."""
    text = f"{when_phrase(thread.last_ts, now)} we were talking about {thread.label}."
    if turns and (thread.title or len(turns) > 1):
        text += f" You last asked: {turns[-1]['question'].rstrip(' .')}"
        if text[-1] not in "?!":
            text += "."
    return text


def list_text(
    threads: list[Thread], now: datetime | None = None, shelf_name: str | None = None
) -> str:
    """The spoken list; *shelf_name* when it was narrowed to one shelf."""
    if not threads:
        if shelf_name:
            return f"We haven't had a longer conversation on the {shelf_name} shelf yet."
        return "We haven't had a longer conversation yet."
    parts = [f"{t.label}, {when_phrase(t.last_ts, now).lower()}" for t in threads]
    lead = f"On the {shelf_name} shelf we have talked about " if shelf_name else ""
    return (
        (lead or "We have talked about ")
        + "; ".join(parts)
        + ". Say 'go back to our conversation about', and the topic, to pick one up."
    )


def prompt_block(thread: Thread, turns: list[dict], char_limit: int = 1400) -> str:
    """The system-prompt block for a resumed thread: its summary plus the
    last exchanges verbatim, newest kept when over *char_limit*."""
    head = (
        f"You are picking an earlier conversation back up "
        f"({when_phrase(thread.last_ts).lower()}): {thread.label}."
    )
    if thread.summary:
        head += f"\nSummary so far: {thread.summary}"
    lines: list[str] = []
    used = len(head)
    for t in reversed(turns):
        answer = t["answer"] if len(t["answer"]) <= 400 else t["answer"][:400].rstrip() + "…"
        line = f"They asked: {t['question']}\nYou answered: {answer}"
        if used + len(line) > char_limit and lines:
            break
        lines.append(line)
        used += len(line)
    if lines:
        head += "\nThe last exchanges:\n" + "\n".join(reversed(lines))
    return head


# ------------------------------------------------------------- singleton

_store: ThreadStore | None = None
_store_lock = threading.Lock()


def get_store() -> ThreadStore | None:
    global _store
    with _store_lock:
        if _store is None:
            try:
                _store = ThreadStore()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Conversation threads unavailable: {e}")
                return None
        return _store
