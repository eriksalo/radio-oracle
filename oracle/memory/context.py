"""Context builder — assembles the messages array for the LLM."""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from config.settings import settings
from oracle.memory import journal
from oracle.memory.store import ConversationStore
from oracle.memory.summarizer import fold_into_profile, summarize_conversation


class ContextBuilder:
    """Builds the messages array: [system, long-term memory, summary, recent, rag, user].

    Order matters for latency: Ollama/llama.cpp reuse the KV cache for the
    longest byte-identical prefix of the previous prompt. Persona, memory
    and the summary are constant within a session and history only ever
    grows at the end, so everything up to the retrieved chunks is a cache
    hit on the next turn. The RAG block changes every turn and therefore
    goes *last*, right before the question. (Measured 2026-09-27: with the
    chunks inside the persona message the whole 2-3k-token prompt was
    re-prefilled every turn, ~5 s.)
    """

    def __init__(self, store: ConversationStore, session_id: str, user: str | None = None):
        self._store = store
        self._session_id = session_id
        self._user = user or settings.default_user
        # Survive a mid-session restart: reload whatever was persisted.
        self._summary: str | None = store.get_summary(session_id)
        self._long_term: str | None = self._load_long_term()
        self._bg_task: asyncio.Task | None = None
        journal.set_context(self._user, session_id)

    @property
    def user(self) -> str:
        return self._user

    def set_user(self, user: str) -> None:
        """Switch the session to *user*: reload their profile and last
        conversation, retag the session and the journal."""
        if user == self._user:
            return
        logger.info(f"Session user: {self._user!r} -> {user!r}")
        self._user = user
        self._store.set_session_user(self._session_id, user)
        self._long_term = self._load_long_term()
        journal.set_context(user, self._session_id)

    def _recent_activity(self) -> str | None:
        """Deterministic block from the activity journal (what the user
        read / played / asked before this session, plus the current book)."""
        j = journal.get_journal()
        if j is None:
            return None
        try:
            text = j.recent_summary(self._user, exclude_session=self._session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"recent activity unavailable: {e}")
            return None
        return (
            f"What you remember doing with {self._user.title()} (from the log, reliable):\n{text}"
            if text
            else None
        )

    def _load_long_term(self) -> str | None:
        """Compose the cross-session memory block injected into every turn."""
        parts: list[str] = []
        try:
            profile = self._store.get_profile(self._user)
            if profile:
                parts.append(f"What you remember about {self._user.title()}:\n{profile}")
            prior = self._store.latest_summarized_session(exclude=self._session_id, user=self._user)
            if prior:
                when = _humanize_date(prior["started_at"])
                parts.append(f"Your previous conversation ({when}):\n{prior['summary']}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not load long-term memory: {e}")
        return "\n\n".join(parts) if parts else None

    async def build(
        self,
        system_prompt: str,
        rag_context: str = "",
        user_text: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the full messages array for the LLM.

        Args:
            system_prompt: The system prompt (persona + instructions)
            rag_context: Formatted RAG retrieval context (goes last)
            user_text: The current question. Callers store it before
                building, so it is dropped from the history tail and
                appended once after the RAG block. None = legacy callers
                that append the user message themselves.

        Returns:
            List of message dicts ready for Ollama
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Cross-session memory (profile + last conversation)
        if self._long_term:
            messages.append({"role": "system", "content": self._long_term})

        # Recent books / music / questions — from the journal, not the LLM.
        recent = self._recent_activity()
        if recent:
            messages.append({"role": "system", "content": recent})

        # Session summary (if we've summarized older turns)
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {self._summary}",
                }
            )

        # Recent conversation turns — minus the current question, which
        # was already persisted and is re-added at the very end.
        recent = self._store.get_messages(self._session_id, limit=settings.max_context_turns)
        if (
            user_text is not None
            and recent
            and recent[-1]["role"] == "user"
            and recent[-1]["content"] == user_text
        ):
            recent = recent[:-1]
        messages.extend(recent)

        if rag_context:
            messages.append({"role": "system", "content": rag_context})
        if user_text is not None:
            messages.append({"role": "user", "content": user_text})

        return messages

    # ------------------------------------------------------------- summarize

    def schedule_summarize(self) -> None:
        """Run maybe_summarize in the background — never blocks the turn."""
        if self._bg_task is not None and not self._bg_task.done():
            return
        self._bg_task = asyncio.create_task(self._summarize_safe())

    async def _summarize_safe(self) -> None:
        try:
            await self.maybe_summarize()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Background summarization failed: {e}")

    async def maybe_summarize(self) -> None:
        """Summarize older turns if conversation exceeds threshold."""
        if self._store.count_messages(self._session_id) <= settings.summary_threshold:
            return

        all_messages = self._store.get_messages(self._session_id)
        # Summarize everything except the most recent turns
        older = all_messages[: -settings.max_context_turns]
        self._summary = await summarize_conversation(older)
        self._store.update_summary(self._session_id, self._summary)

    async def close(self) -> None:
        """Flush pending background work, then summarize this session if it
        has content but no summary yet (short sessions end below threshold)."""
        if self._bg_task is not None and not self._bg_task.done():
            try:
                await self._bg_task
            except Exception:  # noqa: BLE001
                pass
        try:
            await finalize_session(self._store, self._session_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Session finalize failed: {e}")


async def finalize_session(store: ConversationStore, session_id: str) -> None:
    """Summarize a finished session and fold it into the long-term profile."""
    # <2 messages means no real exchange happened (a lone misheard command,
    # a restart) — not worth an LLM call or a slot in long-term memory.
    j = journal.get_journal()
    activity = j.session_activity_text(session_id) if j is not None else ""
    if store.get_summary(session_id) or (store.count_messages(session_id) < 2 and not activity):
        return
    messages = store.get_messages(session_id)
    summary = await summarize_conversation(messages, activity=activity)
    store.update_summary(session_id, summary)
    user = store.get_session_user(session_id)
    profile = await fold_into_profile(store.get_profile(user), summary)
    store.update_profile(profile, user=user)
    logger.info(f"Session {session_id[:8]} summarized into long-term memory")


# Boot-time catch-up waits before touching the LLM: summarization calls were
# observed competing with the user's first voice commands right after
# power-on (STT stretched 7→20s under the contention).
CATCH_UP_DELAY_S = 180.0


async def catch_up_summaries(store: ConversationStore, current_session: str) -> None:
    """Summarize recent sessions that ended without a summary (power-off
    usually beats the in-session threshold). Called in the background at boot."""
    await asyncio.sleep(CATCH_UP_DELAY_S)
    for sess in store.unsummarized_sessions(exclude=current_session):
        try:
            await finalize_session(store, sess["session_id"])
            await asyncio.sleep(10)  # breathing room between LLM jobs
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Catch-up summarize failed for {sess['session_id'][:8]}: {e}")


def _humanize_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%B %d, %Y")
    except ValueError:
        return iso
