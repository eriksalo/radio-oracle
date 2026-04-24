"""Context builder — assembles the messages array for the LLM."""

from __future__ import annotations

from config.settings import settings
from oracle.memory.store import ConversationStore
from oracle.memory.summarizer import summarize_conversation


class ContextBuilder:
    """Builds the messages array: [system, summary, recent_turns, rag_context]."""

    def __init__(self, store: ConversationStore, session_id: str):
        self._store = store
        self._session_id = session_id
        self._summary: str | None = None

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

    async def maybe_summarize(self) -> None:
        """Summarize older turns if conversation exceeds threshold."""
        all_messages = self._store.get_messages(self._session_id)
        if len(all_messages) <= settings.summary_threshold:
            return

        # Summarize everything except the most recent turns
        older = all_messages[: -settings.max_context_turns]
        self._summary = await summarize_conversation(older)
        self._store.update_summary(self._session_id, self._summary)
