"""Core event loop — text REPL, voice mode, and hardware-driven mode."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from loguru import logger

from oracle.llm import check_ollama, stream_chat
from oracle.memory.context import ContextBuilder
from oracle.memory.store import ConversationStore
from oracle.persona import build_system_prompt, get_greeting

if TYPE_CHECKING:
    from oracle.hardware.leds import StatusLEDs
    from oracle.stt import WhisperSTT
    from oracle.tts import PiperTTS

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
    tts: "PiperTTS"
    store: ConversationStore
    ctx_builder: ContextBuilder
    system_prompt: str
    session_id: str


async def voice_init() -> VoiceContext:
    """Initialize STT, TTS, conversation store, persona, and play greeting."""
    from oracle.audio import apply_radio_filter, play_audio
    from oracle.stt import WhisperSTT
    from oracle.tts import PiperTTS

    system_prompt, store, session_id = await _init_common()
    ctx_builder = ContextBuilder(store, session_id)
    stt = WhisperSTT()
    tts = PiperTTS()

    greeting = get_greeting()
    logger.info(f"Oracle: {greeting}")
    greeting_audio = tts.synthesize(greeting)
    greeting_audio = apply_radio_filter(greeting_audio, tts.sample_rate)
    play_audio(greeting_audio, tts.sample_rate)

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


async def voice_turn(
    vc: VoiceContext,
    leds: "StatusLEDs | None" = None,
    should_abort: Callable[[], bool] | None = None,
) -> bool:
    """Run one voice conversation turn (record → transcribe → LLM → TTS).

    Returns True if a turn completed, False if aborted or skipped (silence).
    """
    from oracle.audio import apply_radio_filter, play_audio, record_until_silence

    def aborted() -> bool:
        return should_abort() if should_abort is not None else False

    # Listening
    if leds is not None:
        leds.set_mode("librarian")
    logger.info("Listening...")
    audio = record_until_silence()
    if aborted():
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
                    audio_out = apply_radio_filter(audio_out, vc.tts.sample_rate)
                    play_audio(audio_out, vc.tts.sample_rate)
                    if aborted():
                        break
            sentence_buffer = sentences[-1]

    if sentence_buffer.strip() and not aborted():
        audio_out = vc.tts.synthesize(sentence_buffer.strip())
        audio_out = apply_radio_filter(audio_out, vc.tts.sample_rate)
        play_audio(audio_out, vc.tts.sample_rate)

    response_text = "".join(response_parts)
    logger.info(f"Oracle: {response_text}")
    vc.store.add_message(vc.session_id, "assistant", response_text)
    await vc.ctx_builder.maybe_summarize()
    return True


async def voice_loop() -> None:
    """Voice mode (no hardware): record → transcribe → LLM → TTS in a loop."""
    vc = await voice_init()
    logger.info("Voice mode active — speak when ready")
    try:
        while True:
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
