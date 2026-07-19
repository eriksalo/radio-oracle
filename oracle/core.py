"""Core event loop — text REPL, voice mode, and hardware-driven mode."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import settings
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
        return retriever.format_context(results)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"RAG query failed: {e}")
        return ""


# Follow-ups like "where did he die?" embed uselessly on their own — the
# pronoun refers to the previous turn. Detect them and rewrite into a
# self-contained query with a quick LLM call before retrieval.
_FOLLOWUP_RE = re.compile(
    r"\b(he|she|it|they|him|her|them|his|hers|its|their|theirs|"
    r"that|this|those|these|there|one)\b",
    re.IGNORECASE,
)

_REWRITE_PROMPT = (
    "Rewrite the user's latest message as one short, self-contained search "
    "query, resolving pronouns and references using the conversation. "
    "Output only the query, nothing else."
)


def _needs_rewrite(text: str) -> bool:
    return bool(_FOLLOWUP_RE.search(text)) or len(text.split()) <= 5


async def _retrieval_query(store: ConversationStore, session_id: str, text: str) -> str:
    """The text to embed for retrieval — rewritten if it's a follow-up."""
    if not settings.rag_query_rewrite or not _needs_rewrite(text):
        return text
    # The current user message was already stored; history is everything before.
    recent = store.get_messages(session_id, limit=5)
    prior = recent[:-1] if recent else []
    if not prior:
        return text
    history = "\n".join(f"{m['role']}: {m['content']}" for m in prior)
    try:
        out = await chat(
            [
                {"role": "system", "content": _REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation:\n{history}\n\nLatest message: {text}"},
            ]
        )
        out = out.strip().strip('"')
        if 0 < len(out) <= 200 and "\n" not in out:
            logger.debug(f"Retrieval query rewritten: {text!r} -> {out!r}")
            return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Query rewrite failed, using raw text: {e}")
    return text


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
        retrieval_text = await _retrieval_query(store, session_id, user_input)
        rag_context = _try_rag_query(retrieval_text)
        messages = await ctx.build(system_prompt, rag_context)
        messages.append({"role": "user", "content": user_input})

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


async def voice_init() -> VoiceContext:
    """Initialize STT, TTS, conversation store, and persona."""
    from oracle.stt import WhisperSTT
    from oracle.tts import KokoroTTS

    system_prompt, store, session_id = await _init_common()
    ctx_builder = ContextBuilder(store, session_id)
    stt = WhisperSTT()
    stt_fast = WhisperSTT(model_name=settings.faster_whisper_radio_model)
    tts = KokoroTTS()
    # Preload everything the first interaction needs, off the event loop:
    # the radio STT model (radio is the mode the user lands in), Kokoro
    # (first spoken reply otherwise pays a cold model load), and the RAG
    # retriever (embedder + FAISS indices — seconds of disk I/O).
    await asyncio.gather(
        asyncio.to_thread(stt_fast.load),
        asyncio.to_thread(tts.load),
        asyncio.to_thread(_get_retriever),
    )

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
    )


async def voice_close(vc: VoiceContext) -> None:
    from oracle.llm import close_client

    if vc.catch_up is not None and not vc.catch_up.done():
        vc.catch_up.cancel()  # re-attempted at next boot
    await vc.ctx_builder.close()
    await close_client()
    vc.store.close()


async def speak_text(vc: VoiceContext, text: str) -> None:
    """Synthesize and play a short announcement through the shared TTS."""
    from oracle.audio import play_audio

    audio = vc.tts.synthesize(text)
    play_audio(audio, vc.tts.sample_rate)


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
    from oracle.audio import play_audio, record_until_silence

    def aborted() -> bool:
        return should_abort() if should_abort is not None else False

    if pre_text is not None:
        text = pre_text
        if leds is not None:
            leds.set_mode("thinking")
    else:
        # Listening
        if leds is not None:
            leds.set_mode("librarian")
        logger.info("Listening...")
        try:
            audio = await asyncio.to_thread(record_until_silence, should_abort=should_abort)
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable for voice turn: {e}")
            return False
        if aborted() or len(audio) == 0:
            return False

        # Thinking (transcribe + LLM)
        if leds is not None:
            leds.set_mode("thinking")
        if aborted():
            return False
        vc.stt.load()
        text = await asyncio.to_thread(vc.stt.transcribe, audio)
        vc.stt.unload()
        if aborted():
            return False

    if not text.strip():
        logger.debug("Empty transcription, skipping")
        return False

    logger.info(f"You: {text}")
    vc.store.add_message(vc.session_id, "user", text)

    retrieval_text = await _retrieval_query(vc.store, vc.session_id, text)
    rag_context = await asyncio.to_thread(_try_rag_query, retrieval_text)
    messages = await vc.ctx_builder.build(vc.system_prompt, rag_context)
    messages.append({"role": "user", "content": text})

    response_parts: list[str] = []
    sentence_buffer = ""

    if leds is not None:
        leds.set_mode("speaking")

    # Pipeline TTS with generation: completed sentences go onto a queue and
    # a worker synthesizes/plays them in a thread, so the token stream keeps
    # flowing while earlier sentences are being spoken. Previously synthesis
    # + playback blocked the event loop and stalled the stream per sentence.
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)

    async def _tts_worker() -> None:
        while True:
            sentence = await tts_queue.get()
            if sentence is None:
                return
            if aborted():
                continue  # keep draining so the producer never blocks
            audio_out = await asyncio.to_thread(vc.tts.synthesize, sentence)
            if aborted():
                continue
            await asyncio.to_thread(play_audio, audio_out, vc.tts.sample_rate, should_abort)

    worker = asyncio.create_task(_tts_worker())
    try:
        async for token in stream_chat(messages):
            if aborted():
                break
            response_parts.append(token)
            sentence_buffer += token
            sentences = _SENTENCE_END_RE.split(sentence_buffer)
            if len(sentences) > 1:
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        await tts_queue.put(sentence)
                sentence_buffer = sentences[-1]

        if sentence_buffer.strip() and not aborted():
            await tts_queue.put(sentence_buffer.strip())
    finally:
        await tts_queue.put(None)
        await worker

    response_text = "".join(response_parts)
    logger.info(f"Oracle: {response_text}")
    vc.store.add_message(vc.session_id, "assistant", response_text)
    vc.ctx_builder.schedule_summarize()
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
