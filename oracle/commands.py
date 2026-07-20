"""Wake-word command dispatcher — one surface for both channels.

Pipeline: record → STT → keyword match → question heuristic →
LLM-JSON fallback → action.

The same utterance vocabulary works whether music or a book is playing
(``context``): "pause"/"next" act on the current channel, "play music" /
"read my book" switch channels, "what music/books do you have" explores
the archives, and anything interrogative is a question for the oracle —
answered in place with a follow-up window, after which the channel
resumes. Common ops match keywords for sub-second latency; freeform
requests fall through to a one-shot LLM intent extractor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loguru import logger

from config.settings import settings
from oracle.audio import play_audio, record_until_silence
from oracle.llm import chat

if TYPE_CHECKING:
    from oracle.core import VoiceContext
    from oracle.hardware.leds import StatusLEDs
    from oracle.music.catalog import Catalog
    from oracle.music.player import Player

NextMode = Literal["radio", "librarian", "reader"]
Channel = Literal["music", "book"]
AbortCheck = Callable[[], bool] | None


@dataclass(frozen=True)
class DispatchResult:
    """What the dispatcher decided.

    ``next_mode`` is the channel to be on afterwards ("radio" = music,
    "reader" = book). ``resume_channel`` says whether that channel should
    keep playing — False when the user asked for silence.

    ``reader_query`` carries a requested book title/author into the book
    channel ("read me Moby Dick"); ``play_query`` carries a music request
    into the music channel ("play Pink Floyd" said mid-book).
    """

    next_mode: NextMode
    resume_channel: bool = True
    reader_query: str | None = None
    play_query: str | None = None


_LLM_SYSTEM_PROMPT = """You are a strict voice-command parser for a radio. \
Output ONE JSON object on a single line, no prose, no code fences.

Schema: {"action": <str>, "query": <str|null>}
action must be one of:
  "play"        — start music matching query (artist/album/genre/title)
  "music_on"    — switch to / resume the music (no specific request)
  "next"        — skip forward (track, or chapter when reading)
  "next_album"  — skip to a new album
  "next_chapter"— skip to the next chapter of the book
  "pause"       — pause playback
  "resume"      — resume playback
  "stop"        — stop playback / silence
  "read_book"   — read a book aloud; query is the book title or author
  "list_music"  — what music is available; query narrows it (artist/genre)
  "list_books"  — what books are available; query narrows it (author/title)
  "question"    — an information question or request for knowledge
  "none"        — not a command and not a question (fragments, noise)

Examples:
"play some jazz"         -> {"action":"play","query":"jazz"}
"put on Pink Floyd"      -> {"action":"play","query":"Pink Floyd"}
"put the music back on"  -> {"action":"music_on","query":null}
"skip this"              -> {"action":"next","query":null}
"another album"          -> {"action":"next_album","query":null}
"hush"                   -> {"action":"pause","query":null}
"read me Moby Dick"      -> {"action":"read_book","query":"Moby Dick"}
"read Sherlock Holmes to me" -> {"action":"read_book","query":"Sherlock Holmes"}
"what music do we have"  -> {"action":"list_music","query":null}
"any albums by the Beatles" -> {"action":"list_music","query":"Beatles"}
"what books are there by Mark Twain" -> {"action":"list_books","query":"Mark Twain"}
"why is the sky blue"    -> {"action":"question","query":null}
"how do I splint a broken arm" -> {"action":"question","query":null}
"umm never mind"         -> {"action":"none","query":null}
"""


@dataclass(frozen=True)
class _KeywordRule:
    pattern: re.Pattern[str]
    action: str  # action name; see _do_action


def _build_keyword_table() -> list[_KeywordRule]:
    # Order matters — first match wins. Boundaries (\b) keep "skip" from
    # firing on "skipper", etc.
    raw: list[tuple[str, str]] = [
        # Channel switches / conversation first so they don't get eaten
        # by the generic transport words below.
        (r"\bi\s+have\s+a\s+question\b", "mode_librarian"),
        (r"\bi'?d\s+like\s+to\s+read\s+a\s+book\b", "mode_reader"),
        (r"\b(?:read|listen\s+to)\s+(?:a|my|the)\s+book\b", "mode_reader"),
        (r"\b(?:continue|resume)\s+(?:my|the)\s+book\b", "mode_reader"),
        # "play music by X" is a search, not a resume — route it to play
        # with the qualifier as the query (resolved in dispatch).
        (r"\b(?:play|put\s+on)\s+(?:the\s+|some\s+)?music\s+(?:by|from|like)\b", "play_qualified"),
        # Bare "play music" (utterance ends there) = resume/switch channel.
        (r"\b(?:play|back\s+to|put\s+on)\s+(?:the\s+|some\s+)?music[.!]?\s*$", "music_on"),
        # Exploration.
        (r"\bwhat\s+music\b|\bwhat\s+(?:songs|albums|artists)\s+(?:do|are)\b", "list_music"),
        (r"\bwhat\s+books?\b|\bwhich\s+books?\b", "list_books"),
        # Chapter / track / album ops.
        (r"\bnext\s+chapter\b", "next_chapter"),
        (r"\bnext\s+(?:song|track)\b", "next"),
        (r"\bskip(?:\s+(?:this|song|track|chapter))?\b", "next"),
        (r"\b(?:next|new|change|another)\s+album\b", "next_album"),
        # Transport — with or without the noun, channel decides meaning.
        (r"\b(?:pause|stop)\s+(?:the\s+)?(?:music|reading|book)\b", "pause"),
        (r"\bresume\s+(?:the\s+)?(?:music|reading|book)\b", "resume"),
        (r"^\s*(?:pause|stop|quiet|silence|hush)[.!]?\s*$", "pause"),
        (r"^\s*(?:resume|continue)[.!]?\s*$", "resume"),
    ]
    return [_KeywordRule(re.compile(p, re.IGNORECASE), a) for p, a in raw]


_KEYWORD_RULES = _build_keyword_table()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _keyword_match(text: str) -> str | None:
    norm = _normalise(text)
    for rule in _KEYWORD_RULES:
        if rule.pattern.search(norm):
            return rule.action
    return None


# Cheap question detector — skips the LLM-intent round trip (~2-3s) for the
# common "wake word + ask something" flow. Anything interrogative that
# didn't match a music/book keyword is a question for the oracle.
_QUESTION_RE = re.compile(
    r"^(?:who|what|why|when|where|which|how|is|are|was|were|did|does|do|can|"
    r"could|should|would|will|tell me|explain)\b",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith("?") or bool(_QUESTION_RE.match(stripped))


async def _llm_intent(text: str) -> tuple[str, str | None]:
    """Ask the LLM to classify the command as JSON. Returns (action, query)."""
    messages = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        raw = await chat(messages)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM intent extraction failed: {e}")
        return ("none", None)

    # Strip code fences if the model wrapped its output despite the prompt.
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    # Take only the first {...} block in case the model added extra text.
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        logger.warning(f"LLM intent: no JSON in response: {raw!r}")
        return ("none", None)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        logger.warning(f"LLM intent: bad JSON: {m.group(0)!r}")
        return ("none", None)

    action = str(obj.get("action") or "none").lower()
    query = obj.get("query")
    if isinstance(query, str):
        query = query.strip() or None
    else:
        query = None
    return (action, query)


def _speak(vc: VoiceContext, text: str, should_abort: AbortCheck = None) -> None:
    audio = vc.tts.synthesize(text)
    play_audio(audio, vc.tts.sample_rate, should_abort=should_abort)


# Pre-synthesized "thinking" acknowledgments. A question turn takes ~6-10s
# to first spoken audio (retrieval + prompt processing + generation); an
# instant canned ack over that window converts dead air into feedback.
_ACK_PHRASES = ("Checking the archives.", "Consulting the archives.", "One moment.")
_ack_cache: list = []


def _play_thinking_ack(vc: VoiceContext, should_abort: AbortCheck = None) -> None:
    """Play a canned ack (synthesized once, then cached). Blocking ~1.5s —
    run via a thread alongside the turn, not in front of it."""
    import random

    try:
        if not _ack_cache:
            for phrase in _ACK_PHRASES:
                _ack_cache.append(vc.tts.synthesize(phrase))
        audio = random.choice(_ack_cache)
        play_audio(audio, vc.tts.sample_rate, should_abort=should_abort)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Thinking ack failed: {e}")


async def _question_turns(
    vc: VoiceContext,
    text: str,
    leds: StatusLEDs | None,
    should_abort: AbortCheck,
    window: float | None = None,
) -> None:
    """Answer a question, then hold the mic open for follow-ups.

    Each answer overlaps a canned 'checking the archives' ack with the
    retrieval/generation dead air. After answering, the mic stays open
    for *window* seconds (default settings.followup_window_s) — a
    follow-up needs no wake word; silence (or a button press) lets the
    interrupted channel resume.
    """
    import asyncio

    from oracle.core import voice_turn

    window = settings.followup_window_s if window is None else window

    def aborted() -> bool:
        return bool(should_abort and should_abort())

    while True:
        ack = asyncio.create_task(asyncio.to_thread(_play_thinking_ack, vc, should_abort))
        try:
            await voice_turn(vc, leds=leds, should_abort=should_abort, pre_text=text)
        finally:
            await ack

        if window <= 0 or aborted():
            return
        if leds is not None:
            leds.set_mode("librarian")  # solid blue: still listening
        try:
            audio = await asyncio.to_thread(
                record_until_silence,
                silence_duration=settings.vad_silence_duration_radio,
                onset_timeout=window,
                should_abort=should_abort,
            )
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable for follow-up: {e}")
            return
        if aborted() or len(audio) == 0:
            return  # no follow-up — the channel resumes
        vc.stt_fast.load()
        text = await asyncio.to_thread(vc.stt_fast.transcribe, audio)
        if not text.strip():
            return
        logger.info(f"Follow-up: {text!r}")


def _play_query(player: Player, catalog: Catalog, query: str) -> str | None:
    """Search and start playback. Returns a short human label on success."""
    import random

    hits = catalog.search(query)
    if not hits:
        return None
    # Random hit, not hits[0]: "play Pink Floyd" should feel like tuning
    # into that artist, not always the alphabetically first song.
    track = random.choice(hits)
    player.stop()
    player.play(track=track)
    label = track.artist or track.album or track.title
    return label


async def dispatch_radio_command(
    player: Player | None,
    catalog: Catalog | None,
    vc: VoiceContext,
    leds: StatusLEDs | None = None,
    should_abort: AbortCheck = None,
    context: Channel = "music",
    reader=None,
) -> DispatchResult:
    """One wake-word voice turn — the single dispatcher for both channels.

    ``context`` says which channel was playing ("music" or "book"): the
    same words act on the current channel ("pause", "next"), and the
    result's ``next_mode`` tells the caller which channel to be on after.

    Steps:
      1. record + STT (LED blue → blink)
      2. keyword match; question heuristic; else LLM JSON intent
      3. perform the action (questions hold a follow-up window)
      4. return the channel intent
    """

    def aborted() -> bool:
        return bool(should_abort and should_abort())

    here = "radio" if context == "music" else "reader"

    # 1. Capture user utterance.
    if leds is not None:
        leds.set_mode("librarian")  # solid blue while listening
    try:
        audio = record_until_silence(
            silence_duration=settings.vad_silence_duration_radio,
            should_abort=should_abort,
        )
    except (ValueError, OSError) as e:
        logger.warning(f"Mic unavailable: {e}")
        return DispatchResult(here)
    if aborted() or len(audio) == 0:
        return DispatchResult(here)

    # 2. STT — blink blue while we think. ``stt_fast`` is kept resident
    # across calls (with parakeet it's the same object as ``stt``) and
    # only unloaded around LLM-intent calls on the whisper backends.
    if leds is not None:
        leds.set_mode("thinking")
    vc.stt_fast.load()
    text = vc.stt_fast.transcribe(audio)
    if aborted() or not text.strip():
        return DispatchResult(here)
    logger.info(f"Voice command ({context}): {text!r}")

    # 3. Classify.
    action = _keyword_match(text)
    query: str | None = None
    if action is None and _looks_like_question(text):
        # Interrogative and not a music/book keyword → straight to the
        # oracle, no LLM-intent round trip.
        action = "question"
        logger.info("Question detected (regex)")
    elif action is None:
        # Falling through to the LLM — free STT RAM first.
        vc.stt_fast.unload()
        action, query = await _llm_intent(text)
        logger.info(f"LLM intent: action={action} query={query!r}")
        # Reload eagerly so the *next* command (almost always keyword-
        # matched) doesn't pay the reload itself.
        vc.stt_fast.load()
    else:
        if action == "play_qualified":
            action, query = "play", _extract_qualifier(text)
        logger.info(f"Keyword intent: action={action} query={query!r}")

    # 4. Act.
    if action == "question":
        # Oracle turn(s): answer with full RAG + memory + persona, hold
        # the mic open for wake-word-free follow-ups, then let the
        # channel resume.
        await _question_turns(vc, text, leds, should_abort)
        return DispatchResult(here)

    if action == "mode_librarian":
        # "I have a question" — same overlay, just an invitation first
        # and a longer follow-up window for open-ended conversation.
        _speak(vc, "What would you like to know?", should_abort)
        opening = await _listen_once(vc, onset_timeout=max(settings.followup_window_s * 2, 8.0))
        if opening:
            await _question_turns(
                vc,
                opening,
                leds,
                should_abort,
                window=max(settings.followup_window_s * 2, 8.0),
            )
        return DispatchResult(here)

    if leds is not None:
        leds.set_mode("speaking")
    return _do_action(
        action,
        query,
        player,
        catalog,
        vc,
        should_abort,
        context=context,
        reader=reader,
        raw_text=text,
    )


async def _listen_once(vc: VoiceContext, onset_timeout: float) -> str | None:
    """Record one utterance and transcribe it; None on silence."""
    import asyncio

    try:
        audio = await asyncio.to_thread(
            record_until_silence,
            silence_duration=settings.vad_silence_duration_radio,
            onset_timeout=onset_timeout,
        )
    except (ValueError, OSError) as e:
        logger.warning(f"Mic unavailable: {e}")
        return None
    if len(audio) == 0:
        return None
    vc.stt_fast.load()
    text = await asyncio.to_thread(vc.stt_fast.transcribe, audio)
    return text.strip() or None


# Pulls "…by Mark Twain" / "…about bees" out of keyword-matched
# exploration phrases (the keyword table doesn't capture queries).
_QUALIFIER_RE = re.compile(r"\b(?:by|from|about|like|of)\s+(.+?)[.?!]?\s*$", re.IGNORECASE)


def _extract_qualifier(text: str) -> str | None:
    m = _QUALIFIER_RE.search(text)
    return m.group(1).strip() if m else None


def _describe_music(catalog: Catalog | None, query: str | None) -> str:
    if catalog is None:
        return "The music archive isn't available."
    if query:
        hits = catalog.search(query)
        if not hits:
            return f"Nothing in the music archive matches {query}."
        artists = sorted({t.artist for t in hits if t.artist})[:4]
        who = ", ".join(artists) if artists else hits[0].title
        return f"{len(hits)} tracks match {query} — {who}. Say play and a name."
    s = catalog.stats()
    sample = ", ".join(catalog.sample_artists(6))
    return (
        f"The archive holds {s['tracks']} tracks — about {s['hours']:.0f} hours "
        f"from {s['artists']} artists. A few of them: {sample}. "
        "Ask again for other names, or say play and an artist."
    )


def _describe_books(query: str | None) -> str:
    try:
        from oracle.books.library import Library

        lib = Library()
        try:
            if query:
                hits = lib.search(query)[:4]
                if not hits:
                    return f"No books match {query}. Try an author or a title."
                titles = "; ".join(
                    f"{b.title} by {b.author}" if b.author else b.title for b in hits
                )
                return f"I have {titles}. Say read me, and a title."
            n = lib.count_books()
            sample = ", ".join(lib.sample_authors(5))
            return (
                f"The library holds {n} books. Authors include {sample}, "
                "and about sixty thousand more. Ask by author, title, or "
                "say what books, by someone."
            )
        finally:
            lib.close()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Book listing failed: {e}")
        return "The book archive isn't available."


def _do_action(
    action: str,
    query: str | None,
    player: Player | None,
    catalog: Catalog | None,
    vc: VoiceContext,
    should_abort: AbortCheck,
    context: Channel = "music",
    reader=None,
    raw_text: str = "",
) -> DispatchResult:
    here = "radio" if context == "music" else "reader"

    # ---- channel switches -------------------------------------------------
    if action in ("mode_reader",):
        if context == "book":
            return DispatchResult("reader")  # already here — just resume
        return DispatchResult("reader", resume_channel=False)
    if action == "read_book":
        # Reader announces the title itself; no generic ack needed.
        return DispatchResult("reader", resume_channel=False, reader_query=query)
    if action == "music_on":
        if context == "book":
            _speak(vc, "Back to the music.", should_abort)
            return DispatchResult("radio", resume_channel=False)
        if player is not None:
            player.resume()
        return DispatchResult("radio")
    if action == "play" and context == "book":
        # A specific music request mid-book: bookmark and switch.
        _speak(vc, "Switching to music.", should_abort)
        return DispatchResult("radio", resume_channel=False, play_query=query)

    # ---- exploration ------------------------------------------------------
    if action == "list_music":
        _speak(vc, _describe_music(catalog, query or _extract_qualifier(raw_text)), should_abort)
        return DispatchResult(here)
    if action == "list_books":
        _speak(vc, _describe_books(query or _extract_qualifier(raw_text)), should_abort)
        return DispatchResult(here)

    # ---- book channel transport --------------------------------------------
    if context == "book":
        if action in ("next", "next_chapter", "next_album"):
            if reader is not None and not reader.next_chapter():
                _speak(vc, "That's the last chapter.", should_abort)
            return DispatchResult("reader")
        if action in ("pause", "stop"):
            return DispatchResult("reader", resume_channel=False)
        if action == "resume":
            return DispatchResult("reader")
        logger.debug(f"No-op action {action!r} in book context")
        return DispatchResult("reader")

    # ---- music channel transport -------------------------------------------
    if action == "next_chapter":
        # Chapter words while music plays → treat as resuming the book.
        return DispatchResult("reader", resume_channel=False)
    if player is None:
        _speak(vc, "Music player isn't available.", should_abort)
        return DispatchResult("radio")

    if action == "next":
        player.next()
    elif action == "next_album":
        player.next_album()
    elif action == "pause":
        # Wake handler already paused music for STT; leave it paused
        # rather than letting the handler SIGCONT it on the way out.
        return DispatchResult("radio", resume_channel=False)
    elif action == "resume":
        player.resume()
    elif action == "stop":
        player.stop()
        return DispatchResult("radio", resume_channel=False)
    elif action == "play":
        if not query or catalog is None:
            _speak(vc, "What would you like to hear?", should_abort)
            return DispatchResult("radio")
        label = _play_query(player, catalog, query)
        if label is None:
            _speak(vc, f"I couldn't find anything for {query}.", should_abort)
        else:
            _speak(vc, f"Playing {label}.", should_abort)
    else:
        # "none" or unknown — quietly drop back to the channel.
        logger.debug(f"No-op action {action!r}")
    return DispatchResult("radio")
