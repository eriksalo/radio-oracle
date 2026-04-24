"""SQLite-based conversation storage."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from config.settings import settings


class ConversationStore:
    """Persistent conversation storage in SQLite."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or settings.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp);
        """)
        self._conn.commit()

    def new_session(self) -> str:
        """Create a new conversation session, return session_id."""
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO sessions (session_id, started_at) VALUES (?, ?)",
            (session_id, now),
        )
        self._conn.commit()
        logger.debug(f"New session: {session_id}")
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Store a message in the conversation."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        self._conn.commit()

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
        """Get messages for a session, optionally limited to most recent N."""
        query = "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp"
        if limit:
            query += f" DESC LIMIT {limit}"
            rows = self._conn.execute(query, (session_id,)).fetchall()
            rows.reverse()
        else:
            rows = self._conn.execute(query, (session_id,)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        """Get most recent sessions with their summaries."""
        rows = self._conn.execute(
            "SELECT session_id, started_at, summary FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_summary(self, session_id: str, summary: str) -> None:
        """Update the summary for a session."""
        self._conn.execute(
            "UPDATE sessions SET summary = ? WHERE session_id = ?",
            (summary, session_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
