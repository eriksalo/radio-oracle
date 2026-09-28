"""Users and voiceprints — who the memory belongs to.

``users`` and ``voiceprints`` live in oracle.db next to the conversation
store and the activity journal. A voiceprint is a speaker embedding
(oracle.speaker) of one utterance; a user has several, and
``identify()`` scores a new embedding against the best of each user's
prints. Thread-safe like the other stores (the speaker check runs in a
worker thread).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings

_NAME_RE = re.compile(r"[^a-z0-9' -]+")


def normalize_name(spoken: str) -> str | None:
    """ "It's Erik." / "this is erik" / "my name is Erik Salo" → "erik salo"."""
    s = spoken.strip().lower()
    s = re.sub(
        r"^(?:it'?s|its|this is|i'?m|i am|my name is|the name is|call me|hi,? |hello,? )+\s*",
        "",
        s,
    )
    s = _NAME_RE.sub("", s).strip(" .'-")
    s = re.sub(r"\s+", " ", s)
    if not s or len(s) > 40 or s in ("no", "yes", "nobody", "none", "a guest", "guest"):
        return None
    return s


class UserStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voiceprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    source TEXT,
                    ts TEXT NOT NULL,
                    FOREIGN KEY (user) REFERENCES users(name)
                );
                CREATE INDEX IF NOT EXISTS idx_voiceprints_user ON voiceprints(user);
            """)
            self._conn.commit()
            self.ensure_user(settings.default_user)

    # ---------------------------------------------------------------- users

    def ensure_user(self, name: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO users (name, created_at) VALUES (?, ?)",
                (name, datetime.now(UTC).isoformat()),
            )
            self._conn.commit()

    def known_users(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT name FROM users ORDER BY created_at").fetchall()
        return [r["name"] for r in rows]

    # ----------------------------------------------------------- voiceprints

    def enroll(self, name: str, embedding: np.ndarray, source: str = "") -> int:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        self.ensure_user(name)
        with self._lock:
            self._conn.execute(
                "INSERT INTO voiceprints (user, dim, embedding, source, ts) VALUES (?, ?, ?, ?, ?)",
                (name, len(emb), emb.tobytes(), source, datetime.now(UTC).isoformat()),
            )
            self._conn.commit()
            n = self._conn.execute(
                "SELECT COUNT(*) FROM voiceprints WHERE user = ?", (name,)
            ).fetchone()[0]
        logger.info(f"Voiceprint enrolled for {name!r} ({n} total)")
        return n

    def voiceprint_count(self, name: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM voiceprints WHERE user = ?", (name,)
            ).fetchone()[0]

    def _prints(self) -> list[tuple[str, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute("SELECT user, dim, embedding FROM voiceprints").fetchall()
        return [
            (r["user"], np.frombuffer(r["embedding"], dtype=np.float32, count=r["dim"]))
            for r in rows
        ]

    def identify(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Best-matching user and the cosine score (best print of that user);
        (None, best_score) when nobody is enrolled."""
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        n = np.linalg.norm(emb)
        if n == 0:
            return None, 0.0
        emb = emb / n
        best_user, best = None, -1.0
        for user, p in self._prints():
            pn = np.linalg.norm(p)
            if pn == 0 or len(p) != len(emb):
                continue
            score = float(emb @ (p / pn))
            if score > best:
                best_user, best = user, score
        return best_user, max(best, 0.0)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
