"""Core event loop — text REPL, voice mode, and hardware-driven mode."""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from loguru import logger

from config.settings import settings
from oracle.llm import check_ollama, stream_chat
from oracle.memory.context import ContextBuilder
from oracle.memory.store import ConversationStore
from oracle.persona import build_system_prompt, get_greeting

if TYPE_CHECKING:
    from oracle.hardware.leds import StatusLEDs
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


def _try_rag_query(user_input: str) -> str:
    """Attempt RAG retrieval. Returns empty string if RAG unavailable."""
    try:
        from oracle.rag.retriever import Retriever

        retriever = Retriever()
        collections = retriever.list_collections()
        if not collections:
            return ""
        results = retriever.query(user_input)
        return retriever.format_context(results)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"RAG unavailable: {e}")
        return ""


# ---------------------------------------------------------------------------
# Text REPL
# ---------------------------------------------------------------------------

async def text_repl() -> None:
    """Interactive text REPL — type queries, get streamed responses."""
    system_prompt, store, session_id = await _init_common()
    ctx = ContextBuilder(store, session_id)

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
        rag_context = _try_rag_query(user_input)
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
        await ctx.maybe_summarize()

    store.close()


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

@dataclass
class VoiceContext:
    """Bundle of long-lived voice-mode resources."""
    stt: "WhisperSTT"
    tts: "KokoroTTS"
    store: ConversationStore
    ctx_builder: ContextBuilder
    system_prompt: str
    session_id: str


async def voice_init() -> VoiceContext:
    """Initialize STT, TTS, conversation store, and persona."""
    from oracle.stt import WhisperSTT
    from oracle.tts import KokoroTTS

    system_prompt, store, session_id = await _init_common()
    ctx_builder = ContextBuilder(store, session_id)
    stt = WhisperSTT()
    tts = KokoroTTS()

    return VoiceContext(
        stt=stt,
        tts=tts,
        store=store,
        ctx_builder=ctx_builder,
        system_prompt=system_prompt,
        session_id=session_id,
    )


async def voice_close(vc: VoiceContext) -> None:
    vc.store.close()


async def wake_word_listen(
    vc: VoiceContext,
    leds: "StatusLEDs | None" = None,
    should_abort: Callable[[], bool] | None = None,
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

    vc.stt.load()
    text = vc.stt.transcribe(audio)
    vc.stt.unload()

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
    leds: "StatusLEDs | None" = None,
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
            audio = record_until_silence(should_abort=should_abort)
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable for voice turn: {e}")
            return False
        if aborted() or len(audio) == 0:
            return False

        # Thinking (transcribe + LLM)
        if leds is not None:
            leds.set_mode("thinking")
        vc.stt.load()
        text = vc.stt.transcribe(audio)
        vc.stt.unload()

    if not text.strip():
        logger.debug("Empty transcription, skipping")
        return False

    logger.info(f"You: {text}")
    vc.store.add_message(vc.session_id, "user", text)

    rag_context = _try_rag_query(text)
    messages = await vc.ctx_builder.build(vc.system_prompt, rag_context)
    messages.append({"role": "user", "content": text})

    response_parts: list[str] = []
    sentence_buffer = ""

    if leds is not None:
        leds.set_mode("speaking")

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
                    audio_out = vc.tts.synthesize(sentence)
                    play_audio(audio_out, vc.tts.sample_rate, should_abort=should_abort)
                    if aborted():
                        break
            sentence_buffer = sentences[-1]

    if sentence_buffer.strip() and not aborted():
        audio_out = vc.tts.synthesize(sentence_buffer.strip())
        play_audio(audio_out, vc.tts.sample_rate, should_abort=should_abort)

    response_text = "".join(response_parts)
    logger.info(f"Oracle: {response_text}")
    vc.store.add_message(vc.session_id, "assistant", response_text)
    await vc.ctx_builder.maybe_summarize()
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
