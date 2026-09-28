"""Core event loop — text REPL, voice mode, and hardware-driven mode."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from config.settings import settings
from oracle import timing
from oracle.llm import chat, check_ollama, stream_chat
from oracle.memory.context import ContextBuilder, catch_up_summaries
from oracle.memory.store import ConversationStore
from oracle.persona import build_system_prompt, get_greeting

if TYPE_CHECKING:
    from oracle.hardware.leds import StatusLEDs
    from oracle.music.player import Player
    from oracle.stt import WhisperSTT
    from oracle.tts import KokoroTTS

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# Clause boundary for the first TTS unit only: punctuation followed by
# whitespace, so "1,000" and "e.g." don't split.
_CLAUSE_END_RE = re.compile(r"(?<=[,;:—])\s+")


# Words before which a unit may be cut mid-sentence when no punctuation
# has shown up: conjunctions, relative pronouns and prepositions start a
# new prosodic phrase, so the join is least audible there.
_SOFT_CUT_WORDS = frozenset(
    {
        "and", "but", "or", "which", "that", "because", "so", "while", "although",
        "though", "where", "when", "as", "to", "into", "from", "with", "by", "for",
        "through", "than", "after", "before", "until", "if",
    }
)  # fmt: skip


class SpeechSplitter:
    """Cuts a token stream into units for TTS as soon as they are speakable.

    Kokoro on this CPU synthesizes at ~0.8× real time, so a 20-word
    sentence is ~6 s of dead air before it can start playing, and a long
    unit after a short one leaves a gap while it renders. Units are
    therefore kept short throughout: a sentence end (. ! ?) always closes
    a unit; a clause boundary (, ; : —) closes one once it holds
    ``clause_min_words``; without punctuation a unit is cut before a
    conjunction/preposition after ``soft_cut_words`` words, or at any word
    boundary after ``hard_cut_words``. Kokoro renders comma-terminated
    fragments naturally; the mid-phrase cuts are the price of a fast
    first word. ``clause_min_words=0`` disables the clause/soft/hard cuts
    (sentences only).
    """

    def __init__(
        self,
        clause_min_words: int | None = None,
        soft_cut_words: int | None = None,
        hard_cut_words: int | None = None,
    ) -> None:
        self._buf = ""
        self._min_words = (
            settings.tts_clause_min_words if clause_min_words is None else clause_min_words
        )
        self._soft = settings.tts_soft_cut_words if soft_cut_words is None else soft_cut_words
        self._hard = settings.tts_hard_cut_words if hard_cut_words is None else hard_cut_words

    def feed(self, token: str) -> list[str]:
        self._buf += token
        out: list[str] = []
        parts = _SENTENCE_END_RE.split(self._buf)
        if len(parts) > 1:
            out.extend(p.strip() for p in parts[:-1] if p.strip())
            self._buf = parts[-1]
        if self._min_words > 0:
            # The remainder may itself already be long enough to cut.
            while True:
                clause = _CLAUSE_END_RE.split(self._buf, maxsplit=1)
                if len(clause) > 1 and len(clause[0].split()) >= self._min_words:
                    out.append(clause[0].strip())
                    self._buf = clause[1]
                    continue
                cut = self._unit_cut()
                if cut is None:
                    break
                out.append(self._buf[:cut].strip())
                self._buf = self._buf[cut:]
        return out

    def _unit_cut(self) -> int | None:
        """Character offset to cut an over-long unit at, or None."""
        # Only complete words count: the last token may be mid-word.
        words = self._buf.split(" ")
        complete = words[:-1]
        if len(complete) < self._soft:
            return None
        for i in range(self._soft, len(complete)):
            if complete[i].lower().strip(",.;:") in _SOFT_CUT_WORDS:
                return len(" ".join(complete[:i])) + 1
        if len(complete) >= self._hard:
            return len(" ".join(complete[: self._hard])) + 1
        return None

    def flush(self) -> str:
        tail, self._buf = self._buf.strip(), ""
        return tail


async def _init_common() -> tuple[str, ConversationStore, str]:
    """Shared init: check Ollama, load persona, create session."""
    available = await check_ollama()
    if not available:
        logger.error("Ollama not available. Start Ollama and pull the model first.")
        sys.exit(1)

    system_prompt = build_system_prompt()
    store = ConversationStore()
    session_id = store.new_session()
    return system_prompt, store, session_id


# Retriever is expensive to construct (embedder + FAISS index loads from
# disk) — build once, reuse every turn. None = not yet tried; False = tried
# and unavailable (don't retry every turn).
_retriever: object | None = None


def _get_retriever():
    global _retriever
    if _retriever is False:
        return None
    if _retriever is None:
        try:
            from oracle.rag.retriever import Retriever

            r = Retriever()
            if not r.list_collections():
                logger.info("RAG: no collections found — retrieval disabled")
                _retriever = False
                return None
            # Warm the backends now: FAISS read_index + embedder load are
            # lazy, so without this the first real question after boot
            # pays ~30s of disk I/O (measured on the Jetson).
            r.query("warmup", top_k=1)
            _retriever = r
        except Exception as e:  # noqa: BLE001
            logger.debug(f"RAG unavailable: {e}")
            _retriever = False
            return None
    return _retriever


def _try_rag_query(user_input: str) -> str:
    """Attempt RAG retrieval. Returns empty string if RAG unavailable."""
    retriever = _get_retriever()
    if retriever is None:
        return ""
    try:
        from oracle.rag.modes import detect_mode

        # "tell me more" / "go deeper" style wording upgrades to deep mode:
        # wider candidate pool + cross-encoder rerank.
        results = retriever.query(user_input, mode=detect_mode(user_input))
        if results:
            from oracle.activity import emit

            titles = [
                (r.get("metadata") or {}).get("title") or r.get("source", "?") for r in results[:3]
            ]
            emit("consulted", sources=titles, hits=len(results))
        return retriever.format_context(results)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"RAG query failed: {e}")
        return ""


# Follow-ups like "where did he die?" embed uselessly on their own — the
# pronoun refers to the previous turn. Detect them and rewrite into a
# self-contained query with a quick LLM call before retrieval.
#
# The rewrite costs ~2 s on the Jetson *and* evicts the main prompt's
# prefix cache (single Ollama slot), so it must fire only on real
# follow-ups. The old rule (any of it/that/this/there/one, or ≤5 words)
# fired on "Who wrote Pride and Prejudice?". Now: a third-person pronoun,
# a deictic phrase, a follow-up opener, or a bare short wh-question —
# and only when there is a previous answer to refer back to.
_PRONOUN_RE = re.compile(
    r"\b(he|she|they|him|her|them|his|hers|their|theirs|it|its)\b", re.IGNORECASE
)
_DEICTIC_RE = re.compile(r"\b(that one|the same|the other|those|these)\b", re.IGNORECASE)
_CUE_RE = re.compile(
    r"^\W*(and|what about|how about|tell me more|more about|anything else|how come)\b",
    re.IGNORECASE,
)
_WH_RE = re.compile(r"^\W*(why|where|when|how|who|which|what)\b", re.IGNORECASE)

_REWRITE_PROMPT = (
    "Rewrite the user's latest message as one short, self-contained search "
    "query, resolving pronouns and references using the conversation. "
    "Output only the query, nothing else."
)


def _needs_rewrite(text: str, has_prior_answer: bool = True) -> bool:
    if not has_prior_answer:
        return False
    if _PRONOUN_RE.search(text) or _DEICTIC_RE.search(text) or _CUE_RE.search(text):
        return True
    # "Why is that?", "Where exactly?" — short wh-questions lean on context.
    return bool(_WH_RE.search(text)) and len(text.split()) <= 4


async def _rewrite_query(history: list[dict[str, str]], text: str) -> str:
    """Ask the LLM for a self-contained query; falls back to *text*."""
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    try:
        out = await chat(
            [
                {"role": "system", "content": _REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{hist}\n\nLatest message: {text}"},
            ],
            model=settings.ollama_rewrite_model,
            num_predict=settings.ollama_rewrite_num_predict,
        )
        out = out.strip().strip('"')
        if 0 < len(out) <= 200 and "\n" not in out:
            logger.debug(f"Retrieval query rewritten: {text!r} -> {out!r}")
            return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Query rewrite failed, using raw text: {e}")
    return text


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of lowercase word sets — 1.0 means the rewrite
    added nothing worth a second retrieval."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


async def _retrieve_for_turn(store: ConversationStore, session_id: str, text: str) -> str:
    """RAG context for *text*, rewriting follow-ups off the critical path.

    When a rewrite is needed, retrieval on the raw text runs concurrently
    with the LLM rewrite; the rewritten query only triggers a second
    retrieval if it actually differs (a pronoun resolved to a name).
    Returns the formatted context ("" when nothing relevant).
    """
    if not settings.rag_query_rewrite:
        return await asyncio.to_thread(_try_rag_query, text)
    # The current user message was already stored; history is everything before.
    recent = store.get_messages(session_id, limit=5)
    prior = recent[:-1] if recent else []
    has_prior_answer = any(m["role"] == "assistant" for m in prior)
    if not _needs_rewrite(text, has_prior_answer):
        return await asyncio.to_thread(_try_rag_query, text)

    rewritten, raw_context = await asyncio.gather(
        _rewrite_query(prior, text),
        asyncio.to_thread(_try_rag_query, text),
    )
    t = timing.current()
    if t is not None:
        t.note(rewrite=1)
    if rewritten == text or _token_overlap(rewritten, text) >= 0.8:
        return raw_context
    logger.debug("Rewrite changed the query — retrieving again")
    return await asyncio.to_thread(_try_rag_query, rewritten)


# ---------------------------------------------------------------------------
# Text REPL
# ---------------------------------------------------------------------------


async def text_repl() -> None:
    """Interactive text REPL — type queries, get streamed responses."""
    system_prompt, store, session_id = await _init_common()
    ctx = ContextBuilder(store, session_id)
    catch_up = asyncio.create_task(catch_up_summaries(store, session_id))

    greeting = get_greeting()
    print(f"\n=== The Oracle === (type 'quit' to exit)\n\nOracle: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOracle signing off.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Oracle signing off.")
            break

        store.add_message(session_id, "user", user_input)
        rag_context = await _retrieve_for_turn(store, session_id, user_input)
        messages = await ctx.build(system_prompt, rag_context, user_text=user_input)

        print("Oracle: ", end="", flush=True)
        full_response: list[str] = []
        async for token in stream_chat(messages):
            print(token, end="", flush=True)
            full_response.append(token)
        print()

        response_text = "".join(full_response)
        store.add_message(session_id, "assistant", response_text)
        ctx.schedule_summarize()

    if not catch_up.done():
        catch_up.cancel()
    await ctx.close()
    store.close()


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------


@dataclass
class VoiceContext:
    """Bundle of long-lived voice-mode resources.

    Two STT models, by design:
      - ``stt`` runs the larger model (small.en) for the librarian turn,
        where transcript quality feeds the LLM.
      - ``stt_fast`` runs a tiny model (tiny.en) for the radio dispatcher,
        which only keyword-matches the result. Kept loaded across calls so
        ``librarian, next song`` doesn't pay a per-command model reload.
    """

    stt: WhisperSTT
    stt_fast: WhisperSTT
    tts: KokoroTTS
    store: ConversationStore
    ctx_builder: ContextBuilder
    system_prompt: str
    session_id: str
    catch_up: asyncio.Task | None = None
    # Speaker identification (oracle/speaker.py); None when disabled/unavailable.
    speaker_id: object | None = None
    speaker: object | None = None

    @property
    def user(self) -> str:
        return self.speaker.user if self.speaker is not None else settings.default_user


async def voice_init() -> VoiceContext:
    """Initialize STT, TTS, conversation store, and persona."""
    from oracle.stt import create_stt
    from oracle.tts import KokoroTTS

    system_prompt, store, session_id = await _init_common()
    ctx_builder = ContextBuilder(store, session_id)
    stt = create_stt()
    if settings.stt_backend == "parakeet":
        # One Parakeet model serves both roles — share the instance so it
        # loads (and stays resident) exactly once.
        stt_fast = stt
    else:
        stt_fast = create_stt(model_name=settings.faster_whisper_radio_model)
    tts = KokoroTTS()

    # Warm the LLM into VRAM now (keep_alive=-1 then pins it): otherwise
    # the first question of the session pays the model load.
    async def _warm_llm() -> None:
        try:
            await chat([{"role": "user", "content": "hi"}])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LLM warmup failed: {e}")

    asyncio.create_task(_warm_llm())

    # Preload everything the first interaction needs, off the event loop:
    # the radio STT model (radio is the mode the user lands in), Kokoro
    # (first spoken reply otherwise pays a cold model load), and the RAG
    # retriever (embedder + FAISS indices — seconds of disk I/O).
    from oracle.endpoint import warm as warm_endpoint
    from oracle.speaker import SpeakerId, SpeakerSession

    speaker_id: SpeakerId | None = SpeakerId() if settings.speaker_id_enabled else None

    def _load_speaker() -> None:
        nonlocal speaker_id
        if speaker_id is None:
            return
        try:
            speaker_id.load()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Speaker ID disabled: {e}")
            speaker_id = None

    await asyncio.gather(
        asyncio.to_thread(stt_fast.load),
        asyncio.to_thread(tts.load),
        asyncio.to_thread(_get_retriever),
        asyncio.to_thread(warm_endpoint),
        asyncio.to_thread(_load_speaker),
    )

    vc_stub = VoiceContext(
        stt=stt,
        stt_fast=stt_fast,
        tts=tts,
        store=store,
        ctx_builder=ctx_builder,
        system_prompt=system_prompt,
        session_id=session_id,
    )
    # The "checking the archives" clips: synthesize now (~4 s of Kokoro on
    # this CPU) rather than in front of the first question.
    from oracle.commands import warm_thinking_acks

    await asyncio.to_thread(warm_thinking_acks, vc_stub)

    return VoiceContext(
        stt=stt,
        stt_fast=stt_fast,
        tts=tts,
        store=store,
        ctx_builder=ctx_builder,
        system_prompt=system_prompt,
        session_id=session_id,
        # Summarize sessions that ended without one (power-off usually
        # beats the in-session threshold) — background, off the boot path.
        catch_up=asyncio.create_task(catch_up_summaries(store, session_id)),
        speaker_id=speaker_id,
        speaker=SpeakerSession() if speaker_id is not None else None,
    )


async def voice_close(vc: VoiceContext) -> None:
    from oracle.llm import close_client

    if vc.catch_up is not None and not vc.catch_up.done():
        vc.catch_up.cancel()  # re-attempted at next boot
    await vc.ctx_builder.close()
    await close_client()
    vc.store.close()


async def speak_text(vc: VoiceContext, text: str) -> None:
    """Speak an announcement through the shared TTS (chunked + pipelined,
    off the event loop)."""
    from oracle.activity import emit
    from oracle.tts import say

    emit("spoke", text=text)
    await asyncio.to_thread(say, vc.tts, text)


async def wake_word_listen(
    vc: VoiceContext,
    leds: StatusLEDs | None = None,
    should_abort: Callable[[], bool] | None = None,
    player: Player | None = None,
) -> str | None:
    """Listen for the wake word. Returns text after the wake word, or None.

    Blocks until speech is detected, transcribes it, then checks for the
    wake word. If found, returns the remainder (possibly empty if the user
    only said the wake word). Returns None if the wake word wasn't spoken.
    """
    from oracle.audio import record_until_silence

    def aborted() -> bool:
        return should_abort() if should_abort is not None else False

    try:
        audio = record_until_silence(should_abort=should_abort)
    except (ValueError, OSError) as e:
        logger.warning(f"Mic unavailable for wake word: {e}")
        await asyncio.sleep(5)  # back off before retrying
        return None
    if aborted() or len(audio) == 0:
        return None

    if leds is not None:
        leds.set_mode("thinking")

    if aborted():
        return None

    vc.stt.load()
    # Pause music while STT runs so playback doesn't stutter under
    # GPU/CPU contention. The mic capture is already done; transcription
    # only needs `audio` in memory.
    was_playing = bool(player and player.is_playing and not player.is_paused)
    if was_playing:
        player.pause()
    try:
        text = vc.stt.transcribe(audio)
    finally:
        if was_playing:
            player.resume()
    vc.stt.unload()

    if aborted():
        return None

    if not text.strip():
        return None

    wake_word = settings.wake_word.lower()
    lower = text.lower()
    if wake_word not in lower:
        logger.debug(f"No wake word in: {text!r}")
        return None

    idx = lower.index(wake_word) + len(wake_word)
    remainder = text[idx:].strip().lstrip(",.!? ")
    logger.info(f"Wake word detected! Remainder: {remainder!r}")
    return remainder


async def voice_turn(
    vc: VoiceContext,
    leds: StatusLEDs | None = None,
    should_abort: Callable[[], bool] | None = None,
    pre_text: str | None = None,
) -> bool:
    """Run one voice conversation turn (record → transcribe → LLM → TTS).

    If *pre_text* is provided, skip recording/transcription and use it directly.
    Returns True if a turn completed, False if aborted or skipped (silence).
    """
    # The dispatcher may already own a timer (it did the record + STT);
    # otherwise this turn is the whole story. Always cleared on exit so
    # an aborted turn's timer never bleeds into the next one.
    timer = timing.get_or_start("librarian")
    try:
        return await _voice_turn(vc, leds, should_abort, pre_text, timer)
    finally:
        timing.clear()


async def _voice_turn(
    vc: VoiceContext,
    leds: StatusLEDs | None,
    should_abort: Callable[[], bool] | None,
    pre_text: str | None,
    timer: timing.TurnTimer,
) -> bool:
    from oracle.audio import play_audio
    from oracle.stt import listen

    def aborted() -> bool:
        return should_abort() if should_abort is not None else False

    if pre_text is not None:
        text = pre_text
        timer.speech_ended()
        if leds is not None:
            leds.set_mode("thinking")
    else:
        # Listening (+ transcribing as we go with a streaming backend)
        if leds is not None:
            leds.set_mode("librarian")
        logger.info("Listening...")
        try:
            audio_in, text = await asyncio.to_thread(listen, vc.stt, should_abort=should_abort)
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable for voice turn: {e}")
            return False
        vc.stt.unload()
        if leds is not None:
            leds.set_mode("thinking")
        if aborted():
            return False
        if text.strip():
            from oracle.speaker import check_in

            await check_in(vc, audio_in)

    if not text.strip():
        logger.debug("Empty transcription, skipping")
        return False

    logger.info(f"You: {text}")
    from oracle.activity import emit

    emit("asked", text=text)
    vc.store.add_message(vc.session_id, "user", text)

    rag_context = await _retrieve_for_turn(vc.store, vc.session_id, text)
    timer.mark("retrieve")
    messages = await vc.ctx_builder.build(vc.system_prompt, rag_context, user_text=text)
    timer.mark("build")
    timer.note(rag_chars=len(rag_context))

    response_parts: list[str] = []

    if leds is not None:
        leds.set_mode("speaking")

    # Three-stage pipeline: token stream → text units → synthesis → playback.
    # Synthesis and playback are separate workers so sentence N+1 is being
    # synthesized while N plays (Kokoro on this CPU is RTF ~0.8 — a single
    # synth-then-play worker left a synthesis-length gap between every
    # sentence). The first unit may be a clause so the first audio doesn't
    # wait for a whole sentence.
    text_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
    audio_q: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=3)

    async def _synth_worker() -> None:
        while True:
            unit = await text_q.get()
            if unit is None:
                await audio_q.put(None)
                return
            if aborted():
                continue  # keep draining so the producer never blocks
            await audio_q.put(await asyncio.to_thread(vc.tts.synthesize, unit))

    async def _play_worker() -> None:
        while True:
            audio_out = await audio_q.get()
            if audio_out is None:
                return
            if aborted():
                continue
            timer.mark_once("first_audio")
            await asyncio.to_thread(play_audio, audio_out, vc.tts.sample_rate, should_abort)

    synth = asyncio.create_task(_synth_worker())
    play = asyncio.create_task(_play_worker())
    splitter = SpeechSplitter()
    stats: dict = {}
    try:
        async for token in stream_chat(messages, stats=stats):
            if aborted():
                break
            response_parts.append(token)
            for unit in splitter.feed(token):
                await text_q.put(unit)
        if not aborted():
            tail = splitter.flush()
            if tail and stats.get("done_reason") == "length" and _SENTENCE_END_RE.split(tail):
                # Hit the token cap mid-sentence: don't speak a fragment
                # that trails off; the stored reply keeps only what was said.
                logger.info(f"Reply hit num_predict; dropping unfinished tail {tail[:40]!r}…")
                spoken = "".join(response_parts)
                response_parts[:] = [spoken[: len(spoken) - len(tail)].rstrip()]
            elif tail:
                await text_q.put(tail)
    finally:
        await text_q.put(None)
        await synth
        await play

    response_text = "".join(response_parts)
    logger.info(f"Oracle: {response_text}")
    emit("answered", text=response_text)
    vc.store.add_message(vc.session_id, "assistant", response_text)
    vc.ctx_builder.schedule_summarize()
    timer.finish()
    return True


async def voice_loop() -> None:
    """Voice mode (no hardware): wait for wake word, then converse."""
    vc = await voice_init()
    logger.info(f"Voice mode active — say '{settings.wake_word}' to begin")
    try:
        while True:
            remainder = await wake_word_listen(vc)
            if remainder is None:
                continue
            # Wake word detected — do one turn
            if remainder:
                await voice_turn(vc, pre_text=remainder)
            else:
                await voice_turn(vc)
    except KeyboardInterrupt:
        logger.info("Oracle signing off.")
    finally:
        await voice_close(vc)


# ---------------------------------------------------------------------------
# Mode dispatcher
# ---------------------------------------------------------------------------


async def run(mode: str = "text") -> None:
    """Main entry point for the Oracle."""
    if mode == "text":
        await text_repl()
    elif mode == "voice":
        await voice_loop()
    elif mode == "hardware":
        from oracle.app import OracleApp

        await OracleApp().run()
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)
