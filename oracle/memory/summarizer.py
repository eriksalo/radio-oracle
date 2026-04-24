"""Conversation summarization using the LLM."""

from __future__ import annotations

from loguru import logger

from oracle.llm import chat

SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely, capturing key topics discussed, "
    "decisions made, and any important facts mentioned. Keep it under 200 words."
)


async def summarize_conversation(messages: list[dict[str, str]]) -> str:
    """Use the LLM to summarize a list of conversation messages."""
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
    )

    summary_messages = [
        {"role": "system", "content": SUMMARIZE_PROMPT},
        {"role": "user", "content": conversation_text},
    ]

    summary = await chat(summary_messages)
    logger.debug(f"Generated summary: {summary[:100]}...")
    return summary
