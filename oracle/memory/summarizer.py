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
    "You maintain a compact long-term memory profile of one person who talks to a "
    "voice assistant (a radio that plays music, reads books aloud and answers "
    "questions). Merge the existing profile with the new session summary. Keep it "
    "under 150 words, as short labelled lines:\n"
    "Name: (only if they said it; never invent one)\n"
    "Music: artists/genres they ask for, like, or skip\n"
    "Books: what they are reading or have finished, and reactions\n"
    "Interests & recurring topics: what they keep asking about\n"
    "Projects & people: ongoing things and names they mention\n"
    "Preferences: how they like answers, pet peeves\n"
    "Keep durable facts, drop one-off details and anything superseded. Never pad with "
    "guesses. Output only the profile text, no preamble."
)


async def fold_into_profile(existing: str | None, new_summary: str) -> str:
    """Merge a session summary into the rolling long-term profile."""
    body = (
        f"Existing profile:\n{existing or '(none yet)'}\n\nNew conversation summary:\n{new_summary}"
    )
    messages = [
        {"role": "system", "content": PROFILE_PROMPT},
        {"role": "user", "content": body},
    ]
    profile = await chat(messages)
    logger.debug(f"Updated profile: {profile[:100]}...")
    return profile.strip()
