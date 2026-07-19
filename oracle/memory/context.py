"""Context builder — assembles the messages array for the LLM."""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from config.settings import settings
from oracle.memory.store import ConversationStore
from oracle.memory.summarizer import fold_into_profile, summarize_conversation


class ContextBuilder:
    """Builds the messages array: [system(+rag), long-term memory, summary, recent]."""

    def __init__(self, store: ConversationStore, session_id: str):
        self._store = store
        self._session_id = session_id
        # Survive a mid-session restart: reload whatever was persisted.
        self._summary: str | None = store.get_summary(session_id)
        self._long_term: str | None = self._load_long_term()
        self._bg_task: asyncio.Task | None = None

    def _load_long_term(self) -> str | None:
        """Compose the cross-session memory block injected into every turn."""
        parts: list[str] = []
        try:
            profile = self._store.get_profile()
            if profile:
                parts.append(f"What you remember about your user:\n{profile}")
            prior = self._store.latest_summarized_session(exclude=self._session_id)
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
    ) -> list[dict[str, str]]:
        """Build the full messages array for the LLM.

        Args:
            system_prompt: The system prompt (persona + instructions)
            rag_context: Formatted RAG retrieval context

        Returns:
            List of message dicts ready for Ollama
        """
        messages: list[dict[str, str]] = []

        # System prompt
        full_system = system_prompt
        if rag_context:
            full_system += f"\n\n{rag_context}"
        messages.append({"role": "system", "content": full_system})

        # Cross-session memory (profile + last conversation)
        if self._long_term:
            messages.append({"role": "system", "content": self._long_term})

        # Session summary (if we've summarized older turns)
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {self._summary}",
                }
            )

        # Recent conversation turns
        recent = self._store.get_messages(self._session_id, limit=settings.max_context_turns)
        messages.extend(recent)

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
    if store.get_summary(session_id) or store.count_messages(session_id) == 0:
        return
    messages = store.get_messages(session_id)
    summary = await summarize_conversation(messages)
    store.update_summary(session_id, summary)
    profile = await fold_into_profile(store.get_profile(), summary)
    store.update_profile(profile)
    logger.info(f"Session {session_id[:8]} summarized into long-term memory")


async def catch_up_summaries(store: ConversationStore, current_session: str) -> None:
    """Summarize recent sessions that ended without a summary (power-off
    usually beats the in-session threshold). Called in the background at boot."""
    for sess in store.unsummarized_sessions(exclude=current_session):
        try:
            await finalize_session(store, sess["session_id"])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Catch-up summarize failed for {sess['session_id'][:8]}: {e}")


def _humanize_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%B %d, %Y")
    except ValueError:
        return iso
