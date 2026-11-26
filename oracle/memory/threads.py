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

Similarity alone cannot say whether a question carries a thread on.
Measured with nomic-v1.5 (``clustering:`` prefix) on 12 hand-written
topics, 28 continuations and 35 same-shaped questions about something
else (2026-11-26): a continuation scores 0.56-0.89 against its thread,
"Who was Julius Caesar?" scores 0.84 against a Napoleon thread, and a
plain change of topic reaches 0.68 a quarter of the time. So a question
joins a thread only on evidence in its wording (:meth:`ThreadTracker._joins`):

* it shares a topic word with the thread (and is not far off) - 1 wrong
  join in 440 cross-topic pairs;
* or it points back with "the ..." ("the loaf", "the dressing" -
  :func:`points_back`) and is near the thread the conversation is on;
* or it is very close (0.80) to the thread the conversation is on;

and never when it is an earlier question of the thread asked about
something else (:func:`is_sibling`). On scripted sessions with the real
embedder, before -> after: a same-shaped question straight after a
topic's first question merged 29 -> 0 of 35; an unrelated question after
a whole topic merged 74 -> 19 of 192 (those left say "the toilet", "the
oil"); continuations kept 24 -> 20 of 28 - the ones lost name something
new without pointing back ("What file size do I need?", "How much
paracetamol can I give?") and open a thread of their own. Known wrong
joins: "the battle of Hastings" into a thread that has "the battle of
Waterloo", car tyre pressure into the pressure-tank thread. Not yet
checked on real conversations.

Thread-safety: turns are filed from a worker thread while commands read
from the event loop — same sqlite pattern as journal.py.
"""

from __future__ import annotations

import difflib
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


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# Verbs that say what is being done, not what it is about: "plant" is
# shared by tomatoes and potatoes, "treat" by a burn and a sprained ankle.
# They never count as a topic word.
_GENERIC = frozenset(
    _stem(form)
    for form in (
        "make makes making made plant plants planting planted treat treats treating treated "
        "start starts starting started keeps keeping kept bring brings bringing brought "
        "feed feeds feeding fed work works working worked happen happens happening happened "
        "invade invades invading invaded end ends ending ended sharpen sharpens sharpening "
        "sharpened fix fixes fixing fixed repair repairs repairing repaired grow grows growing "
        "grew grown build builds building built clean cleans cleaning cleaned use uses using "
        "used check checks checking checked change changes changing changed remove removes "
        "removing removed replace replaces replacing replaced stop stops stopping stopped "
        "prevent prevents preventing prevented cook cooks cooking cooked bake bakes baking "
        "baked find finds finding found put puts putting take takes taking took give gives "
        "giving gave run runs running ran turn turns turning turned help helps helping helped "
        "cause causes causing caused call calls called show shows showing showed look looks "
        "looking looked try tries trying tried want wants wanted going went come comes coming "
        "came"
    ).split()
)


def topic_sequence(text: str) -> list[str]:
    """Lowercase content words in spoken order, crudely stemmed, without
    the generic verbs — the lexical half of the same-topic test ("pump",
    "starter", "napoleon")."""
    out: list[str] = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        w = _stem(w)
        if w in _GENERIC or w in out:
            continue
        out.append(w)
    return out


def topic_words(text: str) -> set[str]:
    return set(topic_sequence(text))


_DEFINITE_RE = re.compile(r"\bthe\s+([a-z]+)((?:\s+[a-z]+){0,2})", re.IGNORECASE)
_NOT_POINTING = frozenset({"best", "most", "same", "difference", "world", "sky", "sun", "moon"})


def points_back(text: str) -> bool:
    """True when *text* names something as already known: "the loaf",
    "the dressing", "the pressure tank". "The battle of Hastings" and
    "the capital of Australia" say what they mean themselves, and "the
    best ..." asks for something new, so they do not point back."""
    for m in _DEFINITE_RE.finditer(text):
        if m.group(1).lower() in _NOT_POINTING:
            continue
        if "of" in m.group(2).lower().split():
            continue
        return True
    return False


def is_sibling(question: str, earlier: str, thread_words: set[str]) -> bool:
    """True when *question* is *earlier* asked again about something else:
    "Who was Julius Caesar?" after "Who was Napoleon Bonaparte?", "When
    should I plant potatoes?" after "... tomatoes?". Such a pair is as
    close in embedding space as a real continuation (0.84 and 0.87;
    "Why did Napoleon invade Russia?" is 0.81), so wording decides: the
    two open alike, share at least half their words in order, and the
    first thing the new one names is new to the thread. A subject
    introduced with "the" points back ("How do I bake the loaf?") and is
    never a sibling."""
    q = re.findall(r"[a-z']+", question.lower())
    p = re.findall(r"[a-z']+", earlier.lower())
    if len(q) < 3 or len(p) < 3 or q[0] != p[0] or q == p:
        return False
    matcher = difflib.SequenceMatcher(a=p, b=q, autojunk=False)
    same = sum(block.size for block in matcher.get_matching_blocks())
    if same < 2 or same / min(len(q), len(p)) < 0.5:
        return False
    subject = next(iter(topic_sequence(question)), None)
    if subject is None or subject in thread_words:
        return False
    return not re.search(rf"\bthe\s+{re.escape(subject)}", question.lower())


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

    def match_data(
        self, user: str, limit: int = 40
    ) -> list[tuple[int, np.ndarray | None, set[str], list[str]]]:
        """(id, unit centroid, topic words, last questions) of the user's
        recent threads."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, vec_sum, words, title FROM threads WHERE user = ? AND turns >= 1 "
                "ORDER BY last_ts DESC, id DESC LIMIT ?",
                (user, limit),
            ).fetchall()
            ids = [int(r["id"]) for r in rows]
            asked: dict[int, list[str]] = {tid: [] for tid in ids}
            if ids:
                marks = ",".join("?" * len(ids))
                for t in self._conn.execute(
                    f"SELECT thread_id, question FROM thread_turns WHERE thread_id IN ({marks}) "
                    "ORDER BY id",
                    ids,
                ):
                    asked[int(t["thread_id"])].append(t["question"])
        out = []
        for r in rows:
            # Re-derived, so words stored before the generic verbs were
            # dropped do not count either.
            words = topic_words(r["words"]) | topic_words(r["title"] or "")
            tid = int(r["id"])
            out.append((tid, _unit(_from_blob(r["vec_sum"])), words, asked[tid][-12:]))
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
        for tid, centroid, twords, _asked in self.store.match_data(user):
            overlap = len(words & twords)
            sim = float(vec @ centroid) if vec is not None and centroid is not None else 0.0
            if overlap == 0 and sim < settings.thread_find_sim:
                continue
            score = overlap + sim
            if best is None or score > best[0]:
                best = (score, tid)
        return best[1] if best else None

    def continues(self, user: str, question: str) -> Thread | None:
        """The thread the conversation is on, if *question* carries it on
        (same test as filing, :meth:`_joins`). Asked *before* the answer, so the lookup
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
        for tid, centroid, twords, asked in self.store.match_data(user):
            if tid != current.id:
                continue
            if centroid is None or centroid.shape != vec.shape:
                return None
            joins = self._joins(question, float(vec @ centroid), twords, asked, True)
            return current if joins else None
        return None

    @staticmethod
    def _joins(
        question: str, sim: float, twords: set[str], asked: list[str], current: bool
    ) -> bool:
        """Does *question* belong to the thread with these topic words and
        questions? *current*: it is the thread the conversation is on.
        See the module docstring for why similarity is not enough."""
        if any(is_sibling(question, earlier, twords) for earlier in asked):
            return False  # the same question about something else
        if sim >= settings.thread_join_sim_shared and topic_words(question) & twords:
            return True
        if not current:
            return False
        if sim >= settings.thread_join_sim:
            return True
        return sim >= settings.thread_continue_sim and points_back(question)

    def _match(self, user: str, question: str, vec: np.ndarray | None) -> int | None:
        if vec is None:
            return None
        best: tuple[float, int] | None = None
        for tid, centroid, twords, asked in self.store.match_data(user):
            if centroid is None or centroid.shape != vec.shape:
                continue
            sim = float(vec @ centroid)
            if self._joins(question, sim, twords, asked, tid == self.current_id) and (
                best is None or sim > best[0]
            ):
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
