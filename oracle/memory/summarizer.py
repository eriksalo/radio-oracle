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


PROFILE_PROMPT = (
    "You maintain a compact long-term memory profile of a voice assistant's "
    "owner. Merge the existing profile with the new conversation summary. "
    "Keep durable facts: name, interests, ongoing projects, preferences, "
    "recurring topics. Drop one-off details and anything superseded. "
    "Under 150 words. Output only the profile text, no preamble."
)


async def fold_into_profile(existing: str | None, new_summary: str) -> str:
    """Merge a session summary into the rolling long-term profile."""
    body = (
        f"Existing profile:\n{existing or '(none yet)'}\n\n"
        f"New conversation summary:\n{new_summary}"
    )
    messages = [
        {"role": "system", "content": PROFILE_PROMPT},
        {"role": "user", "content": body},
    ]
    profile = await chat(messages)
    logger.debug(f"Updated profile: {profile[:100]}...")
    return profile.strip()
