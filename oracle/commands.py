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
from oracle import timing
from oracle.audio import play_audio
from oracle.llm import chat
from oracle.stt import listen

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
    # A chapter reference to apply once the book is open ("go to chapter
    # three" said while music plays → resume the current book there).
    reader_chapter: str | None = None
    # The user explicitly asked for music (play / music on / next…): the
    # only thing that lifts the power-on "wait quietly" hold.
    starts_music: bool = False


_LLM_SYSTEM_PROMPT = """You are a strict voice-command parser for a radio. \
Output ONE JSON object on a single line, no prose, no code fences.

Schema: {"action": <str>, "query": <str|null>}
action must be one of:
  "play"        — start music matching query (artist/album/genre/title)
  "music_on"    — switch to / resume the music (no specific request)
  "next"        — skip forward (track, or chapter when reading)
  "next_album"  — skip to a new album
  "next_chapter"— skip to the next chapter of the book
  "prev_chapter"— go back to the previous chapter
  "goto_chapter"— jump to a chapter; query is the reference: "three", "XII", "the last",
                  "the preface", or a chapter title
  "restart_book"— start the current book over from its first chapter
  "book_status" — what book / chapter is being read right now
  "pause"       — pause playback
  "resume"      — resume playback
  "stop"        — stop playback / silence
  "read_book"   — read a book aloud; query is the book title or author
  "list_music"  — what music is available; query narrows it (artist/genre)
  "list_books"  — what books are available; query narrows it (author/title)
  "question"    — an information question or request for knowledge
  "about_device"— what this radio is: who built it, what it holds
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
"start with chapter one"  -> {"action":"goto_chapter","query":"one"}
"go to the chapter called Loomings" -> {"action":"goto_chapter","query":"Loomings"}
"go back a chapter"      -> {"action":"prev_chapter","query":null}
"start the book over"    -> {"action":"restart_book","query":null}
"what am I reading"      -> {"action":"book_status","query":null}
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
        (
            r"\bi\s+have\s+a\s+question\b|\b(?:ask|have)\s+(?:you\s+)?(?:a\s+|some\s+)?questions?\b|\bquestion\s+mode\b",
            "mode_librarian",
        ),
        (
            r"\b(?:about|tell\s+me\s+about)\s+(?:this|the|your)\s+(?:device|radio|machine|box|hardware)\b|\bwho\s+(?:made|built|created)\s+you\b|\bwhat\s+are\s+you\b",
            "about_device",
        ),
        (r"\bi'?d\s+like\s+to\s+read\s+a\s+book\b", "mode_reader"),
        (r"\b(?:read|listen\s+to)\s+(?:a|my|the)\s+book\b", "mode_reader"),
        (r"\b(?:continue|resume)\s+(?:my|the)\s+book\b", "mode_reader"),
        # "play music by X" is a search, not a resume — route it to play
        # with the qualifier as the query (resolved in dispatch).
        (r"\b(?:play|put\s+on)\s+(?:the\s+|some\s+)?music\s+(?:by|from|like)\b", "play_qualified"),
        # Bare "play music" (utterance ends there) = resume/switch channel.
        (r"\b(?:play|back\s+to|put\s+on)\s+(?:the\s+|some\s+)?music[.!]?\s*$", "music_on"),
        (r"\bturn\s+(?:the\s+)?(?:radio|music)\s+(?:back\s+)?on\b", "music_on"),
        # "Can you play X" is a request, not a question for the oracle.
        (r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:play|put\s+on)\b", "play_request"),
        (r"\bi\s+don'?t\s+like\s+this\b|\bnot\s+this\s+(?:one|song)\b", "next"),
        # Exploration.
        (
            r"\bwhat\s+music\b|\bwhat\s+(?:songs|albums|artists)\s+(?:do|are)\b|\bexplore\s+(?:the\s+)?music\b",
            "list_music",
        ),
        (r"\bwhat\s+books?\b|\bwhich\s+books?\b|\bexplore\s+(?:the\s+)?books?\b", "list_books"),
        # Chapter / track / album ops.
        (r"\bnext\s+chapter\b", "next_chapter"),
        # "skip to the last chapter" is a jump; "the last chapter" alone is
        # the one before this ("read the last chapter again").
        (r"\b(?:skip|go|jump)\s+to\s+the\s+(?:last|final)\s+chapter\b", "goto_chapter"),
        (
            r"\b(?:previous|last|prior)\s+chapter\b|\b(?:go\s+)?back\s+(?:a|one)\s+chapter\b",
            "prev_chapter",
        ),
        (
            r"\b(?:start|read)\s+(?:it\s+|the\s+book\s+)?(?:over|again)\b|\bfrom\s+the\s+(?:beginning|top|start)\b|\brestart\s+(?:the\s+)?book\b",
            "restart_book",
        ),
        (
            r"\b(?:go|jump|skip)\s+to\s+(?:the\s+)?chapter\b|\bstart\s+(?:with|at|from)\s+(?:the\s+)?chapter\b|\bread\s+(?:me\s+)?(?:the\s+)?chapter\b|^\s*chapter\s+\w+",
            "goto_chapter",
        ),
        (
            r"\b(?:read|go\s+to|start\s+with)\s+(?:me\s+)?(?:the\s+)?(?:front\s+matter|preface|preamble|introduction)\b",
            "goto_chapter",
        ),
        (
            r"\bwhat\s+(?:am\s+i|are\s+we)\s+reading\b|\bwhere\s+(?:was|am|were)\s+(?:i|we)\b|\b(?:what|which)\s+chapter\b|\bwhat\s+book\s+(?:is\s+this|was\s+that|am\s+i)\b",
            "book_status",
        ),
        # "next slide" is what Parakeet hears for "next song" (2026-09-28).
        (r"\bnext\s+(?:song|track|slide|tune)\b", "next"),
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


# Questions *about the catalog* ("Do you have any Springsteen?", "Any
# albums by the Beatles?", "Is there any jazz?", "What kind of music is
# there?") start with an interrogative and used to be swept into the
# oracle as Wikipedia questions (quality eval 2026-09-30). They are
# exploration commands; the channel is read from the nouns, and a bare
# "Do you have any Springsteen?" goes to the LLM intent step instead of
# the oracle.
_CATALOG_Q_RE = re.compile(
    r"^\s*(?:"
    r"(?:do|did|have)\s+(?:you|we)\s+(?:have|got|own|carry|keep)\b|"
    r"(?:is|are)\s+there\s+(?:any|some|a|an)\b|"
    r"(?:any|what|which|how\s+many)\s+(?:songs?|albums?|artists?|music|tracks?|tunes?|bands?|"
    r"books?|novels?|stories|story|authors?|poems?|poetry|titles?)\b|"
    r"what\s+kind\s+of\s+(?:music|songs?|books?|stories)\b|"
    r"what\s+(?:music|books?)\s+(?:is|are)\s+(?:there|available)\b"
    r")",
    re.IGNORECASE,
)
_MUSIC_NOUN_RE = re.compile(
    r"\b(?:songs?|albums?|artists?|music|tracks?|tunes?|bands?|singers?|records?|jazz|blues|rock|"
    r"folk|country|classical|pop)\b",
    re.IGNORECASE,
)
_BOOK_NOUN_RE = re.compile(
    r"\b(?:books?|novels?|stories|story|authors?|poems?|poetry|titles?|read|reading|writers?)\b",
    re.IGNORECASE,
)
_OBJECT_STRIP_RE = re.compile(
    r"^\s*(?:"
    r"(?:do|did|have)\s+(?:you|we)\s+(?:have|got|own|carry|keep)|"
    r"(?:is|are)\s+there|what\s+kind\s+of|how\s+many|what|which|any"
    r")?\s*(?:any|some|a|an|the|more)?\s*"
    r"(?:songs?|albums?|artists?|music|tracks?|tunes?|bands?|books?|novels?|stories|story|"
    r"authors?|poems?|poetry|titles?)?\s*(?:by|from|of|about|like)?\s*",
    re.IGNORECASE,
)
_OBJECT_TAIL_RE = re.compile(
    r"\s*(?:(?:do|did|does)\s+(?:you|we|i)\s+(?:have|got|own|keep|carry)|(?:is|are)\s+there|"
    r"have\s+you\s+got|in\s+(?:the\s+)?(?:library|archive|collection)|available|around|"
    r"on\s+(?:this|the)\s+(?:radio|device|thing))?\s*[.?!]*\s*$",
    re.IGNORECASE,
)


def _catalog_question(text: str) -> tuple[str, str | None] | None:
    """(action, query) for a question about what the radio holds, or None.

    Returns ``("list_music", q)`` / ``("list_books", q)`` when the nouns
    say which shelf; ``("ask_llm", None)`` when they don't ("Do you have
    any Springsteen?") so the caller skips the oracle and lets the LLM
    intent step decide.
    """
    if not _CATALOG_Q_RE.match(text):
        return None
    music = bool(_MUSIC_NOUN_RE.search(text))
    book = bool(_BOOK_NOUN_RE.search(text))
    if re.search(r"\b(?:by|from|about)\b", text, re.IGNORECASE):
        query = _extract_qualifier(text)
    else:
        obj = _OBJECT_TAIL_RE.sub("", _OBJECT_STRIP_RE.sub("", text, count=1)).strip(" ,")
        query = obj or None
    if music and not book:
        return ("list_music", query)
    if book and not music:
        return ("list_books", query)
    return ("ask_llm", None)


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


def _chapter_announcement(title: str | None) -> str:
    if not title:
        return "That's as far as the book goes."
    t = title.strip().rstrip(".")
    return t if t.lower().startswith(("chapter", "part", "book", "act")) else f"Chapter: {t}"


def _speak(vc: VoiceContext, text: str, should_abort: AbortCheck = None) -> None:
    from oracle.activity import emit
    from oracle.tts import say

    emit("spoke", text=text)
    say(vc.tts, text, should_abort=should_abort)


# Pre-synthesized "thinking" acknowledgments. A question turn takes several
# seconds to first spoken audio (retrieval + prompt processing +
# generation); an instant canned ack over that window converts dead air
# into feedback. Only worth it when the archives are actually consulted.
_ACK_PHRASES = ("Checking the archives.", "Consulting the archives.", "One moment.")
_ack_cache: list = []


def warm_thinking_acks(vc: VoiceContext) -> None:
    """Synthesize the ack clips once, at boot, so the first question
    doesn't pay ~1.5 s of Kokoro before the ack can even start."""
    try:
        if not _ack_cache:
            for phrase in _ACK_PHRASES:
                _ack_cache.append(vc.tts.synthesize(phrase))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Thinking ack warm-up failed: {e}")


def _play_thinking_ack(vc: VoiceContext, should_abort: AbortCheck = None) -> None:
    """Play a canned ack (synthesized once, then cached). Blocking ~1.5s —
    run via a thread alongside the turn, not in front of it. Playback is
    serialized with the answer by oracle.audio's speaker lock."""
    import random

    try:
        warm_thinking_acks(vc)
        audio = random.choice(_ack_cache)
        play_audio(audio, vc.tts.sample_rate, should_abort=should_abort)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Thinking ack failed: {e}")


def _archives_available() -> bool:
    """Whether a question turn will actually run retrieval."""
    from oracle.core import _get_retriever

    try:
        return bool(_get_retriever())
    except Exception:  # noqa: BLE001
        return False


async def _question_turns(
    vc: VoiceContext,
    text: str,
    leds: StatusLEDs | None,
    should_abort: AbortCheck,
    window: float | None = None,
    player: Player | None = None,
    catalog: Catalog | None = None,
    context: Channel = "music",
    reader=None,
) -> DispatchResult | None:
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

    use_ack = _archives_available()
    while True:
        ack = (
            asyncio.create_task(asyncio.to_thread(_play_thinking_ack, vc, should_abort))
            if use_ack
            else None
        )
        try:
            await voice_turn(vc, leds=leds, should_abort=should_abort, pre_text=text)
        finally:
            if ack is not None:
                await ack

        if window <= 0 or aborted():
            return None
        if leds is not None:
            leds.set_mode("q_listen")  # blue slow blink: still listening
        # A follow-up is a turn of its own for timing purposes: the
        # previous timer was finished by voice_turn.
        timing.start("followup")
        try:
            _audio, text = await asyncio.to_thread(
                listen,
                vc.stt_fast,
                silence_duration=settings.vad_silence_duration_radio,
                onset_timeout=window,
                should_abort=should_abort,
            )
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable for follow-up: {e}")
            timing.clear()
            return
        if aborted() or not text.strip():
            timing.clear()
            return None  # no follow-up — the channel resumes
        logger.info(f"Follow-up: {text!r}")
        # A follow-up may be a command ("read me Moby Dick", "play some
        # jazz", "next song"): act on it instead of chatting about it.
        action, query = await classify(vc, text)
        if action not in ("question", "none"):
            from oracle.activity import emit

            emit("decided", action=action, query=query)
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


def _play_query(player: Player, catalog: Catalog, query: str, raw_text: str = "") -> str | None:
    """Search and start playback. Returns a short human label on success."""
    import random

    from oracle.activity import emit

    hint = _play_hint(raw_text) or _play_hint(query)
    # "the album Harvest" → the words "the album" are the hint, not the name.
    cleaned = re.sub(r"^\s*(?:the\s+)?(?:album|record|song|track|tune)\s+", "", query, flags=re.I)
    hits, tier = catalog.search_ranked(cleaned or query, hint=hint)
    emit("music_request", query=query, hits=len(hits), tier=tier)
    if not hits:
        return None
    # Random hit within the best tier, not hits[0]: "play Pink Floyd"
    # should feel like tuning into that artist, not always the
    # alphabetically first song.
    track = random.choice(hits)
    player.stop()
    player.play(track=track)
    if tier in ("artist", "genre", "substring"):
        return track.artist or track.album or track.title
    if tier.startswith("album"):
        return f"{track.album}, {track.artist}" if track.artist else track.album
    return f"{track.title}, {track.artist}" if track.artist else track.title


async def classify(vc: VoiceContext, text: str) -> tuple[str, str | None]:
    """Keyword table → question heuristic → LLM intent. Used for wake-word
    commands and for follow-ups in the question window alike, so "read me
    Moby Dick" said as a follow-up opens the book instead of being chatted
    about (2026-09-28: the LLM replied that it can't read books aloud)."""
    action = _keyword_match(text)
    query: str | None = None
    catalog_q = _catalog_question(text) if action is None else None
    if catalog_q is not None and catalog_q[0] != "ask_llm":
        action, query = catalog_q
        logger.info(f"Catalog question: action={action} query={query!r}")
    elif action is None and catalog_q is None and _looks_like_question(text):
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
        elif action == "play_request":
            action, query = "play", _extract_play_object(text)
        elif action in ("list_music", "list_books"):
            query = _extract_qualifier(text)
        logger.info(f"Keyword intent: action={action} query={query!r}")
    return action, query


async def dispatch_radio_command(
    player: Player | None,
    catalog: Catalog | None,
    vc: VoiceContext,
    leds: StatusLEDs | None = None,
    should_abort: AbortCheck = None,
    context: Channel = "music",
    reader=None,
    pre_text: str | None = None,
    pre_audio=None,
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
    timer = timing.start("command")

    # 1+2. Capture the utterance and transcribe it (streaming backends
    # decode during capture). ``stt_fast`` is kept resident across calls
    # (with parakeet/nemotron it's the same object as ``stt``) and only
    # unloaded around LLM-intent calls on the whisper backends.
    # LED colour follows the channel the turn started from: green while
    # music is up, purple in a book, blue when the radio is idle (a wake
    # from the quiet state is almost always a question).
    if context == "book":
        base = "book"
    elif player is not None and getattr(player, "is_playing", False):
        base = "music"
    else:
        base = "q"
    if pre_text is not None:
        # Already recorded and transcribed (the power-on welcome).
        audio_in, text = pre_audio, pre_text
        timer.speech_ended()
    else:
        if leds is not None:
            leds.set_mode(f"{base}_listen")
        try:
            audio_in, text = listen(
                vc.stt_fast,
                silence_duration=settings.vad_silence_duration_radio,
                should_abort=should_abort,
            )
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable: {e}")
            return DispatchResult(here)
    if leds is not None:
        leds.set_mode(f"{base}_think")
    if aborted() or not text.strip():
        return DispatchResult(here)
    logger.info(f"Voice command ({context}): {text!r}")

    # Who is this? Silent (~100 ms); an unknown voice gets asked *after*
    # the command completes (see the end of this function), never mid-turn.
    from oracle.speaker import maybe_ask, observe

    if audio_in is not None:
        await observe(vc, audio_in, reader=reader)

    # 3. Classify.
    action, query = await classify(vc, text)

    from oracle.activity import emit

    emit("decided", action=action, query=query)
    timer.mark("decide")
    timer.note(action=action)

    # 4. Act.
    if action == "question":
        # Oracle turn(s): answer with full RAG + memory + persona, hold
        # the mic open for wake-word-free follow-ups, then let the
        # channel resume. voice_turn inherits this timer and finishes it.
        timer.label = "question"
        switched = await _question_turns(
            vc,
            text,
            leds,
            should_abort,
            player=player,
            catalog=catalog,
            context=context,
            reader=reader,
        )
        if switched is not None:
            return switched
        if context == "music" and not aborted():
            await maybe_ask(vc)
        return DispatchResult(here)

    if action == "mode_librarian":
        # "I have a question" — same overlay, just an invitation first
        # and a longer follow-up window for open-ended conversation.
        _speak(vc, "What would you like to know?", should_abort)
        opening = await _listen_once(vc, onset_timeout=max(settings.followup_window_s * 2, 8.0))
        if not opening and not (should_abort and should_abort()):
            # Heard nothing usable: say so and give one more chance, rather
            # than dropping back to the music in silence (2026-09-30).
            _speak(vc, "Sorry, I didn't catch that. Go ahead.", should_abort)
            opening = await _listen_once(vc, onset_timeout=max(settings.followup_window_s * 2, 8.0))
        if opening:
            switched = await _question_turns(
                vc,
                opening,
                leds,
                should_abort,
                window=max(settings.followup_window_s * 2, 8.0),
                player=player,
                catalog=catalog,
                context=context,
                reader=reader,
            )
            if switched is not None:
                return switched
        return DispatchResult(here)

    if leds is not None:
        leds.set_mode(f"{base}_speak")
    try:
        result = _do_action(
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
    finally:
        timer.mark("act")
        timer.finish()
        timing.clear()
    # A book resumes right after its command; ask on the music side only.
    if context == "music" and result.next_mode == "radio" and action != "none" and not aborted():
        await maybe_ask(vc)
    return result


async def _listen_once(vc: VoiceContext, onset_timeout: float) -> str | None:
    """Record one utterance and transcribe it; None on silence."""
    import asyncio

    try:
        _audio, text = await asyncio.to_thread(
            listen,
            vc.stt_fast,
            silence_duration=settings.vad_silence_duration_radio,
            onset_timeout=onset_timeout,
        )
    except (ValueError, OSError) as e:
        logger.warning(f"Mic unavailable: {e}")
        return None
    return text.strip() or None


# Pulls "…by Mark Twain" / "…about bees" out of keyword-matched
# exploration phrases (the keyword table doesn't capture queries).
_QUALIFIER_RE = re.compile(
    r"\b(?:by|from|about|like|of)\s+(.+?)"
    r"(?:\s+(?:do|did|does)\s+(?:you|we|i)\s+(?:have|got|own|keep)|\s+(?:are|is)\s+there|"
    r"\s+have\s+you\s+got)?[.?!]?\s*$",
    re.IGNORECASE,
)
_PLAY_OBJECT_RE = re.compile(
    r"\b(?:play|put\s+on)\s+(?:me\s+|us\s+)?(?:some\s+|a\s+little\s+|a\s+bit\s+of\s+)?(.+?)"
    r"(?:\s+(?:for\s+me|please))?[.?!]?\s*$",
    re.IGNORECASE,
)


def _extract_play_object(text: str) -> str | None:
    """ "Can you play Mark Knopfler?" → "Mark Knopfler"."""
    m = _PLAY_OBJECT_RE.search(text)
    return m.group(1).strip() if m else None


_HINT_RE = re.compile(r"\b(album|record|song|track|tune)\b", re.IGNORECASE)


def _play_hint(raw_text: str) -> str | None:
    m = _HINT_RE.search(raw_text)
    if not m:
        return None
    w = m.group(1).lower()
    return "album" if w in ("album", "record") else "song"


_CHAPTER_SPEC_RE = re.compile(
    r"\bchapter\s+(?:called\s+|titled\s+|named\s+)?([a-z0-9 ]+?)[.?!]?\s*$", re.IGNORECASE
)
_FRONT_MATTER_RE = re.compile(r"\b(front\s+matter|preface|preamble|introduction)\b", re.IGNORECASE)


_ORDINAL_CHAPTER_RE = re.compile(r"\b(last|final|first|next|previous)\s+chapter\b", re.IGNORECASE)


def _extract_chapter_spec(text: str) -> str | None:
    """ "go to chapter twenty one" → "twenty one"; "read the preface" →
    "preface"; "skip to the last chapter" → "last"."""
    m = _CHAPTER_SPEC_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ORDINAL_CHAPTER_RE.search(text)
    if m:
        word = m.group(1).lower()
        return "last" if word == "final" else word
    m = _FRONT_MATTER_RE.search(text)
    return m.group(1) if m else None


def _book_in_progress() -> bool:
    """A bookmark exists — "next chapter" while music plays means resume it."""
    try:
        from oracle.books.bookmarks import BookmarkStore

        store = BookmarkStore()
        try:
            return bool(store.list_in_progress())
        finally:
            store.close()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"bookmark check failed: {e}")
        return False


def _current_book_status() -> str | None:
    """Status of the most recent book without opening the reader."""
    try:
        from oracle.books.session import ReaderSession

        session = ReaderSession()
        try:
            book = session.current_book()
            if book is None:
                return None
            bm = session._bookmarks.get(book.id)
            titles = [
                (i, sub or t) for i, t, sub in session._library.list_chapter_headings(book.id)
            ]
            first = session._library.first_content_chapter(book.id)
            ch = bm.chapter_idx if bm else first
            title = next((t for i, t in titles if i == ch), "")
            where = (
                f"chapter {ch - first + 1} of {max(len(titles) - first, 1)}"
                if ch >= first
                else "the front matter"
            )
            tail = f": {title.strip()}" if title and ch >= first else ""
            return f"{book.title}, {where}{tail}."
        finally:
            session.close()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"book status failed: {e}")
        return None


def _extract_qualifier(text: str) -> str | None:
    m = _QUALIFIER_RE.search(text)
    return m.group(1).strip() if m else None


def _describe_music(catalog: Catalog | None, query: str | None) -> str:
    if catalog is None:
        return "The music archive isn't available."
    if query:
        hits, tier = catalog.search_ranked(query)
        if not hits:
            return f"Nothing in the music archive matches {query}."
        artists = sorted({t.artist for t in hits if t.artist})
        albums = sorted({t.album for t in hits if t.album})
        n = len(hits)
        if tier == "artist" and len(artists) <= 2:
            who = " and ".join(artists)
            shelf = f", {len(albums)} album{'s' if len(albums) != 1 else ''}" if albums else ""
            plural = "s" if n != 1 else ""
            return f"Yes. {n} track{plural} by {who}{shelf}. Say play {artists[0]}."
        who = ", ".join(artists[:4]) if artists else hits[0].title
        return f"{n} tracks match {query} — {who}. Say play and a name."
    s = catalog.stats()
    sample = ", ".join(catalog.sample_artists(6))
    return (
        f"The archive holds {s['tracks']} tracks — about {s['hours']:.0f} hours "
        f"from {s['artists']} artists. A few of them: {sample}. "
        "Ask again for other names, or say play and an artist."
    )


DEVICE_DESCRIPTION = (
    "Hello. I'm glad you found me. I am the Librarian. I was built by Erik Salo, "
    "in Boulder, Colorado, in 2026. I don't know how long ago that was for you. "
    "I hold the knowledge of the old world: more than eleven million passages of "
    "encyclopedia, a medical reference, repair guides for the machines you'll find "
    "lying around, and lessons on almost any subject you could want to learn. "
    "If you're hurt, I can help you understand it. If something is broken, I can "
    "help you fix it. If you're simply lost, I can help you think. "
    "I also keep sixty thousand books, and four thousand songs from nearly "
    "eighteen hundred artists. "
    'Ask me anything. Or say "read a book," or "play some music." '
    "I'm listening."
)


def describe_device(catalog: Catalog | None = None) -> str:
    """ "Tell me about this device" — Erik's fixed passage (2026-09-30).

    The counts are written out in words on purpose: they are spoken, and the
    passage is addressed to whoever finds the radio, not to a spreadsheet.
    *catalog* is accepted for the callers that still pass it.
    """
    return DEVICE_DESCRIPTION


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
            return DispatchResult("radio", resume_channel=False, starts_music=True)
        if player is not None:
            player.resume()
        return DispatchResult("radio", starts_music=True)
    if action == "play" and context == "book":
        # A specific music request mid-book: bookmark and switch.
        _speak(vc, "Switching to music.", should_abort)
        return DispatchResult("radio", resume_channel=False, play_query=query, starts_music=True)

    # ---- exploration ------------------------------------------------------
    if action == "list_music":
        _speak(vc, _describe_music(catalog, query or _extract_qualifier(raw_text)), should_abort)
        return DispatchResult(here)
    if action == "list_books":
        _speak(vc, _describe_books(query or _extract_qualifier(raw_text)), should_abort)
        return DispatchResult(here)
    if action == "about_device":
        _speak(vc, describe_device(catalog), should_abort)
        return DispatchResult(here)

    # ---- book channel transport --------------------------------------------
    if context == "book":
        if action in ("next", "next_chapter", "next_album"):
            if reader is not None and not reader.next_chapter():
                _speak(vc, "That's the last chapter.", should_abort)
            return DispatchResult("reader")
        if action == "prev_chapter":
            title = reader.prev_chapter() if reader is not None else None
            _speak(vc, _chapter_announcement(title), should_abort)
            return DispatchResult("reader")
        if action == "goto_chapter":
            spec = query or _extract_chapter_spec(raw_text)
            title = reader.goto_chapter(spec) if (reader is not None and spec) else None
            if title is None:
                _speak(
                    vc,
                    f"I couldn't find chapter {spec}." if spec else "Which chapter?",
                    should_abort,
                )
            else:
                _speak(vc, _chapter_announcement(title), should_abort)
            return DispatchResult("reader")
        if action == "restart_book":
            title = reader.restart() if reader is not None else None
            _speak(vc, "From the beginning. " + _chapter_announcement(title), should_abort)
            return DispatchResult("reader")
        if action == "book_status":
            status = reader.status_text() if reader is not None else None
            _speak(vc, status or "Nothing is open right now.", should_abort)
            return DispatchResult("reader")
        if action in ("pause", "stop"):
            return DispatchResult("reader", resume_channel=False)
        if action == "resume":
            return DispatchResult("reader")
        logger.debug(f"No-op action {action!r} in book context")
        return DispatchResult("reader")

    # ---- music channel transport -------------------------------------------
    if action in ("next_chapter", "prev_chapter", "goto_chapter", "restart_book"):
        # Chapter words while music plays: resume the book *if there is
        # one* (and apply the jump once it's open). Without a bookmark,
        # "next chapter" is almost certainly a misheard "next song".
        if _book_in_progress():
            spec = {
                "next_chapter": "next",
                "prev_chapter": "previous",
                "restart_book": "beginning",
            }.get(action, query or _extract_chapter_spec(raw_text))
            return DispatchResult("reader", resume_channel=False, reader_chapter=spec)
        if action == "next_chapter":
            action = "next"
        else:
            _speak(
                vc,
                "You're not reading anything right now. Say 'read a book' to start one.",
                should_abort,
            )
            return DispatchResult("radio")
    if action == "book_status":
        _speak(vc, _current_book_status() or "You're not reading anything right now.", should_abort)
        return DispatchResult("radio")
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
        label = _play_query(player, catalog, query, raw_text)
        if label is None:
            _speak(vc, f"I couldn't find anything for {query}.", should_abort)
        else:
            _speak(vc, f"Playing {label}.", should_abort)
    else:
        # "none" or unknown — quietly drop back to the channel.
        logger.debug(f"No-op action {action!r}")
    return DispatchResult("radio", starts_music=action in ("next", "next_album", "resume", "play"))
