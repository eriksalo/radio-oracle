"""Context builder — assembles the messages array for the LLM."""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from config.settings import settings
from oracle.memory import journal, threads
from oracle.memory.store import ConversationStore
from oracle.memory.summarizer import (
    fold_into_profile,
    summarize_conversation,
    summarize_thread,
)


class ContextBuilder:
    """Builds the messages array: one system message (persona + memory +
    activity + summary), the history, then one user message (RAG + question).

    Order matters for latency: Ollama/llama.cpp reuse the KV cache for the
    longest byte-identical prefix of the previous prompt. Persona, memory
    and the summary are constant within a session and history only ever
    grows at the end, so everything up to the retrieved chunks is a cache
    hit on the next turn. The RAG block changes every turn and therefore
    goes *last*, right before the question. (Measured 2026-09-27: with the
    chunks inside the persona message the whole 2-3k-token prompt was
    re-prefilled every turn, ~5 s.)
    """

    def __init__(
        self,
        store: ConversationStore,
        session_id: str,
        user: str | None = None,
        embed_fn: threads.EmbedFn | None = None,
    ):
        self._store = store
        self._session_id = session_id
        self._user = user or settings.default_user
        # Survive a mid-session restart: reload whatever was persisted.
        self._summary: str | None = store.get_summary(session_id)
        self._long_term: str | None = self._load_long_term()
        self._bg_task: asyncio.Task | None = None
        journal.set_context(self._user, session_id)
        # Conversation threads: which topic each answered question belongs
        # to, across sessions. None when disabled or the db is unavailable.
        self.threads: threads.ThreadTracker | None = None
        if settings.threads_enabled:
            thread_store = threads.get_store()
            if thread_store is not None:
                self.threads = threads.ThreadTracker(thread_store, embed_fn or _resident_embed)
        self._resumed: str | None = None  # prompt block of a thread picked back up
        self._filing: set[asyncio.Task] = set()

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
        # Another person's thread is not theirs to continue.
        self._resumed = None
        if self.threads is not None:
            self.threads.leave()

    # --------------------------------------------------------------- threads

    def resume_thread(self, thread_id: int) -> tuple[threads.Thread, list[dict]] | None:
        """Pick a thread back up: its summary and last exchanges join the
        system prompt (one prefix-cache miss, then stable again) and the
        next question is filed into it."""
        if self.threads is None:
            return None
        thread = self.threads.store.get(thread_id)
        if thread is None or thread.user != self._user:
            return None
        turns = self.threads.store.turns(thread_id, limit=4)
        self._resumed = threads.prompt_block(thread, turns)
        self.threads.resume(thread_id)
        logger.info(f"Thread {thread_id} resumed: {thread.label!r}")
        return thread, turns

    def schedule_file_turn(self, question: str, answer: str, followup: bool = False) -> None:
        """File an answered question into its thread, off the turn's path
        (one embedding, ~64 ms on the Jetson, after the answer was spoken)."""
        if self.threads is None:
            return
        task = asyncio.create_task(
            asyncio.to_thread(
                self.threads.file_turn, self._user, self._session_id, question, answer, followup
            )
        )
        self._filing.add(task)
        task.add_done_callback(self._filing.discard)

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
            f"What you remember doing with {self._user.title()} (from the log, reliable; "
            f"background only — bring it up when asked or when it is what the question "
            f"is about):\n{text}"
            if text
            else None
        )

    def _load_long_term(self) -> str | None:
        """Compose the cross-session memory block injected into every turn."""
        parts: list[str] = []
        try:
            profile = self._store.get_profile(self._user)
            if profile:
                parts.append(
                    f"What you remember about {self._user.title()} (background; use only when "
                    f"relevant to what they ask):\n{profile}"
                )
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
            rag_context: Formatted RAG retrieval context (goes into the
                final user message, ahead of the question)
            user_text: The current question. Callers store it before
                building, so it is dropped from the history tail and
                appended once, after the RAG block. None = legacy callers
                that append the user message themselves.

        Returns:
            List of message dicts ready for Ollama
        """
        # One system message. Qwen3.5's chat template (and most others)
        # accepts a system message only at index 0, so persona, memory,
        # the activity block and the session summary are joined into it —
        # same text, same order, so the prefix cache still sees one stable
        # prefix.
        blocks = [system_prompt]

        # Cross-session memory (profile + last conversation)
        if self._long_term:
            blocks.append(self._long_term)

        # Recent books / music / questions — from the journal, not the LLM.
        recent_activity = self._recent_activity()
        if recent_activity:
            blocks.append(recent_activity)

        # A thread the user asked to pick back up. Set once per resume, so
        # the prefix is stable again from the next turn on.
        if self._resumed:
            blocks.append(self._resumed)

        # Session summary (if we've summarized older turns)
        if self._summary:
            blocks.append(f"Previous conversation summary: {self._summary}")

        messages: list[dict[str, str]] = [{"role": "system", "content": "\n\n".join(blocks)}]

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

        # The RAG block rides inside the final user message, ahead of the
        # question: it changes every turn, and llama-server takes a context
        # checkpoint at the start of the last user message, which is what
        # lets a hybrid/recurrent model (Qwen3.5) reuse everything before it.
        if user_text is not None:
            content = f"{rag_context}\n\nQuestion: {user_text}" if rag_context else user_text
            messages.append({"role": "user", "content": content})
        elif rag_context:
            # Legacy callers append the question themselves.
            messages.append({"role": "user", "content": rag_context})

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
        if self._filing:
            await asyncio.gather(*list(self._filing), return_exceptions=True)
        try:
            await finalize_session(self._store, self._session_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Session finalize failed: {e}")
        await summarize_threads()


def _resident_embed(text: str):
    """Embed with the retriever's model *if it is already loaded*. Threads
    never load a model of their own (8 GB box); without one, turns are
    filed by follow-up continuity only."""
    try:
        from oracle.rag.backends.faiss_ivfpq import resident_embedder
    except Exception:  # noqa: BLE001 — faiss not installed on this host
        return None
    embedder = resident_embedder()
    return None if embedder is None else embedder.embed_single(text)


async def summarize_threads(limit: int = 3) -> int:
    """Give threads with new exchanges a title and a rolling summary.
    Returns how many were updated. Never raises."""
    thread_store = threads.get_store() if settings.threads_enabled else None
    if thread_store is None:
        return 0
    done = 0
    try:
        pending = thread_store.unsummarized(limit=limit, min_turns=settings.thread_recall_min_turns)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Thread summarize: could not list threads: {e}")
        return 0
    for thread in pending:
        try:
            # Only the exchanges the last summary has not seen (capped).
            fresh = max(1, min(8, thread.turns - thread.summarized_turns))
            turns = thread_store.turns(thread.id, limit=fresh)
            title, summary = await summarize_thread(thread.summary, turns)
            if summary is None:
                logger.warning(f"Thread {thread.id}: summarizer returned nothing usable")
                continue
            # A title, once given, stays: it is what the user says to get back.
            thread_store.set_label(
                thread.id, None if thread.title else title, summary, thread.turns
            )
            logger.info(f"Thread {thread.id} labelled {thread.title or title!r}")
            done += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Thread {thread.id} summarize failed: {e}")
    return done


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
    await summarize_threads()


def _humanize_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%B %d, %Y")
    except ValueError:
        return iso
