"""Conversation summarization using the LLM."""

from __future__ import annotations

import re

from loguru import logger

from oracle.llm import chat

SUMMARIZE_PROMPT = (
    "Summarize the following conversation concisely, capturing key topics discussed, "
    "decisions made, and any important facts mentioned. Keep it under 200 words."
)


async def summarize_conversation(messages: list[dict[str, str]], activity: str = "") -> str:
    """Use the LLM to summarize a list of conversation messages.

    ``activity`` is the session's journal (books started, music asked
    for, questions) rendered by ``journal.session_activity_text``; it goes
    in alongside the transcript so the profile learns what the user did,
    not only what they said.
    """
    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
    )
    if activity:
        conversation_text = (
            f"What happened this session:\n{activity}\n\nConversation:\n{conversation_text}"
        ).strip()

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


THREAD_PROMPT = (
    "You label one ongoing topic of conversation between a person and a voice "
    "assistant. Reply with exactly two lines:\n"
    "Title: a short name for the topic, 2 to 5 words, no quotes, the way you would "
    "say it aloud after 'we were talking about' (for example: the well pump)\n"
    "Summary: under 60 words. What was asked, what was established, and what was "
    "still open. Use only what is in the text; never invent."
)

_TITLE_RE = re.compile(r"^\s*title\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\s*summary\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE | re.DOTALL)


def parse_thread_label(raw: str) -> tuple[str | None, str | None]:
    """(title, summary) from the model's two lines; None for a part that
    is missing or unusable (a title has to be short enough to say)."""
    title = None
    m = _TITLE_RE.search(raw)
    if m:
        cand = m.group(1).strip().strip("\"'.").strip()
        if 0 < len(cand) <= 60 and len(cand.split()) <= 8:
            # Spoken mid-sentence: "we were talking about the well pump".
            # Headline capitals ("The Well Pump") are a model habit, not a
            # name; the title is only ever spoken.
            if cand.istitle() and len(cand.split()) > 1:
                title = cand.lower()
            else:
                title = "the " + cand[4:] if cand.startswith("The ") else cand
    m = _SUMMARY_RE.search(raw)
    summary = " ".join(m.group(1).split()) if m else None
    return title, (summary or None)


async def summarize_thread(
    prior_summary: str | None, turns: list[dict[str, str]]
) -> tuple[str | None, str | None]:
    """Title and rolling summary for one conversation thread. *turns* are
    ``{"question", "answer"}`` dicts, oldest first."""
    body = "\n".join(f"Q: {t['question']}\nA: {t['answer'][:500]}" for t in turns)
    if prior_summary:
        body = f"Summary so far: {prior_summary}\n\nNewer exchanges:\n{body}"
    raw = await chat(
        [
            {"role": "system", "content": THREAD_PROMPT},
            {"role": "user", "content": body},
        ]
    )
    return parse_thread_label(raw)
