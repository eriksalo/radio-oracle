"""Radio-mode wake-word command dispatcher.

Pipeline: record → STT → keyword match → LLM-JSON fallback → action.

The wake word in radio mode is the user's main control surface for the
music player and for entering the voice-conversation modes (librarian,
reader). Common ops ('next song', 'next album', 'pause music') match
keywords for sub-second latency; freeform requests like "play some jazz"
or "put on Pink Floyd" fall through to a one-shot LLM intent extractor
that emits JSON we route to ``Catalog.search()`` + ``Player.play()``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

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
AbortCheck = Callable[[], bool] | None


@dataclass(frozen=True)
class DispatchResult:
    """What the radio-mode dispatcher decided.

    ``resume_music`` is the hint the wake handler reads to decide whether
    to SIGCONT the previously-paused music. False when the user asked
    for silence ('pause'/'stop' commands) or for a mode change.

    ``reader_query`` carries a requested book title/author into reader
    mode ("read me Moby Dick"); None means the reader picks (resume the
    current book, or ask).
    """
    next_mode: NextMode
    resume_music: bool = True
    reader_query: str | None = None


_LLM_SYSTEM_PROMPT = """You are a strict voice-command parser for a radio. \
Output ONE JSON object on a single line, no prose, no code fences.

Schema: {"action": <str>, "query": <str|null>}
action must be one of:
  "play"        — start music matching query (artist/album/genre/title)
  "next"        — skip to next track
  "next_album"  — skip to a new album
  "pause"       — pause music
  "resume"      — resume music
  "stop"        — stop music
  "read_book"   — read a book aloud; query is the book title or author
  "none"        — request not a music or book command

Examples:
"play some jazz"         -> {"action":"play","query":"jazz"}
"put on Pink Floyd"      -> {"action":"play","query":"Pink Floyd"}
"play the white album"   -> {"action":"play","query":"white album"}
"skip this"              -> {"action":"next","query":null}
"another album"          -> {"action":"next_album","query":null}
"hush"                   -> {"action":"pause","query":null}
"read me Moby Dick"      -> {"action":"read_book","query":"Moby Dick"}
"read Sherlock Holmes to me" -> {"action":"read_book","query":"Sherlock Holmes"}
"what time is it"        -> {"action":"none","query":null}
"""


@dataclass(frozen=True)
class _KeywordRule:
    pattern: re.Pattern[str]
    action: str  # action name; see _do_action


def _build_keyword_table() -> list[_KeywordRule]:
    # Order matters — first match wins. Boundaries (\b) keep "skip" from
    # firing on "skipper", etc.
    raw: list[tuple[str, str]] = [
        # Mode transitions first so they don't get eaten by "stop".
        (r"\bi\s+have\s+a\s+question\b",        "mode_librarian"),
        (r"\bi'?d\s+like\s+to\s+read\s+a\s+book\b", "mode_reader"),
        (r"\b(?:read|listen\s+to)\s+a\s+book\b", "mode_reader"),
        # Track / album ops.
        (r"\bnext\s+(?:song|track)\b",           "next"),
        (r"\bskip(?:\s+(?:this|song|track))?\b", "next"),
        (r"\b(?:next|new|change|another)\s+album\b", "next_album"),
        # Transport.
        (r"\b(?:pause|stop)\s+(?:the\s+)?music\b", "pause"),
        (r"\bresume\s+(?:the\s+)?music\b",       "resume"),
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


def _speak(vc: "VoiceContext", text: str, should_abort: AbortCheck = None) -> None:
    audio = vc.tts.synthesize(text)
    play_audio(audio, vc.tts.sample_rate, should_abort=should_abort)


def _play_query(player: "Player", catalog: "Catalog", query: str) -> "str | None":
    """Search and start playback. Returns a short human label on success."""
    hits = catalog.search(query)
    if not hits:
        return None
    track = hits[0]
    player.stop()
    player.play(track=track)
    label = track.artist or track.album or track.title
    return label


async def dispatch_radio_command(
    player: "Player | None",
    catalog: "Catalog | None",
    vc: "VoiceContext",
    leds: "StatusLEDs | None" = None,
    should_abort: AbortCheck = None,
) -> DispatchResult:
    """One radio-mode voice turn. Returns the next state intent.

    Steps:
      1. record + STT (LED blue → blink)
      2. keyword match; else LLM JSON intent
      3. perform the action and optionally TTS-ack
      4. return next_mode + whether the wake handler should resume music
    """
    def aborted() -> bool:
        return bool(should_abort and should_abort())

    # 1. Capture user utterance. Use a short trailing-silence window —
    # radio commands ("next song", "pause music") have no internal
    # pauses, so 0.6 s is plenty and shaves ~1 s off the perceived turn.
    if leds is not None:
        leds.set_mode("librarian")  # solid blue while listening
    try:
        audio = record_until_silence(
            silence_duration=settings.vad_silence_duration_radio,
            should_abort=should_abort,
        )
    except (ValueError, OSError) as e:
        logger.warning(f"Mic unavailable: {e}")
        return DispatchResult("radio", resume_music=True)
    if aborted() or len(audio) == 0:
        return DispatchResult("radio", resume_music=True)

    # 2. STT — blink blue while we think. ``stt_fast`` is preloaded in
    # voice_init() with tiny.en; we keep it resident across calls so the
    # second-and-subsequent commands don't pay a model reload, and only
    # unload when falling through to the LLM (the 8 GB unified-memory
    # rule requires STT and LLM to stay sequential — see CLAUDE.md).
    if leds is not None:
        leds.set_mode("thinking")
    vc.stt_fast.load()
    text = vc.stt_fast.transcribe(audio)
    if aborted() or not text.strip():
        return DispatchResult("radio", resume_music=True)
    logger.info(f"Radio command: {text!r}")

    # 3. Classify.
    action = _keyword_match(text)
    query: str | None = None
    if action is None:
        # Falling through to the LLM — free STT RAM first.
        vc.stt_fast.unload()
        action, query = await _llm_intent(text)
        logger.info(f"LLM intent: action={action} query={query!r}")
        # Reload eagerly so the *next* command (almost always keyword-
        # matched) doesn't pay the reload itself.
        vc.stt_fast.load()
    else:
        logger.info(f"Keyword intent: action={action}")

    # 4. Act + ack.
    if leds is not None:
        leds.set_mode("speaking")
    return _do_action(action, query, player, catalog, vc, should_abort)


def _do_action(
    action: str,
    query: str | None,
    player: "Player | None",
    catalog: "Catalog | None",
    vc: "VoiceContext",
    should_abort: AbortCheck,
) -> DispatchResult:
    if action == "mode_librarian":
        _speak(vc, "Yes? What's your question?", should_abort)
        return DispatchResult("librarian", resume_music=False)
    if action == "mode_reader":
        _speak(vc, "Book reader mode.", should_abort)
        return DispatchResult("reader", resume_music=False)
    if action == "read_book":
        # Reader announces the title itself; no generic ack needed.
        return DispatchResult("reader", resume_music=False, reader_query=query)
    if player is None:
        _speak(vc, "Music player isn't available.", should_abort)
        return DispatchResult("radio", resume_music=True)

    if action == "next":
        player.next()
    elif action == "next_album":
        player.next_album()
    elif action == "pause":
        # Wake handler already paused music for STT; leave it paused
        # rather than letting the handler SIGCONT it on the way out.
        return DispatchResult("radio", resume_music=False)
    elif action == "resume":
        player.resume()
    elif action == "stop":
        player.stop()
        return DispatchResult("radio", resume_music=False)
    elif action == "play":
        if not query or catalog is None:
            _speak(vc, "What would you like to hear?", should_abort)
            return DispatchResult("radio", resume_music=True)
        label = _play_query(player, catalog, query)
        if label is None:
            _speak(vc, f"I couldn't find anything for {query}.", should_abort)
        else:
            _speak(vc, f"Playing {label}.", should_abort)
    else:
        # "none" or unknown — quietly drop back to music.
        logger.debug(f"No-op action {action!r}")
    return DispatchResult("radio", resume_music=True)
