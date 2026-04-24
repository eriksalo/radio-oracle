"""Core event loop — text REPL and voice mode."""

import re
import sys

from loguru import logger

from oracle.llm import check_ollama, stream_chat
from oracle.memory.context import ContextBuilder
from oracle.memory.store import ConversationStore
from oracle.persona import build_system_prompt, get_greeting

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
    except Exception as e:
        logger.debug(f"RAG unavailable: {e}")
        return ""


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

        # RAG retrieval
        rag_context = _try_rag_query(user_input)

        # Build context with memory
        messages = await ctx.build(system_prompt, rag_context)
        messages.append({"role": "user", "content": user_input})

        # Stream response
        print("Oracle: ", end="", flush=True)
        full_response: list[str] = []
        async for token in stream_chat(messages):
            print(token, end="", flush=True)
            full_response.append(token)
        print()

        response_text = "".join(full_response)
        store.add_message(session_id, "assistant", response_text)

        # Summarize if needed
        await ctx.maybe_summarize()

    store.close()


async def voice_loop() -> None:
    """Voice mode: record → transcribe → LLM → synthesize → play."""
    from oracle.audio import apply_radio_filter, play_audio, record_until_silence
    from oracle.stt import WhisperSTT
    from oracle.tts import PiperTTS

    system_prompt, store, session_id = await _init_common()
    ctx = ContextBuilder(store, session_id)

    stt = WhisperSTT()
    tts = PiperTTS()

    # Play greeting
    greeting = get_greeting()
    logger.info(f"Oracle: {greeting}")
    greeting_audio = tts.synthesize(greeting)
    greeting_audio = apply_radio_filter(greeting_audio, tts.sample_rate)
    play_audio(greeting_audio, tts.sample_rate)

    logger.info("Voice mode active — speak when ready")

    try:
        while True:
            # Record
            logger.info("Listening...")
            audio = record_until_silence()

            # Transcribe
            stt.load()
            text = stt.transcribe(audio)
            stt.unload()

            if not text.strip():
                logger.debug("Empty transcription, skipping")
                continue

            logger.info(f"You: {text}")
            store.add_message(session_id, "user", text)

            # RAG
            rag_context = _try_rag_query(text)

            # Build context
            messages = await ctx.build(system_prompt, rag_context)
            messages.append({"role": "user", "content": text})

            # Stream LLM, buffer by sentence, synthesize and play
            response_parts: list[str] = []
            sentence_buffer = ""

            async for token in stream_chat(messages):
                response_parts.append(token)
                sentence_buffer += token

                # Check for sentence boundary
                sentences = _SENTENCE_END_RE.split(sentence_buffer)
                if len(sentences) > 1:
                    # Synthesize and play completed sentences
                    for sentence in sentences[:-1]:
                        sentence = sentence.strip()
                        if sentence:
                            audio_out = tts.synthesize(sentence)
                            audio_out = apply_radio_filter(audio_out, tts.sample_rate)
                            play_audio(audio_out, tts.sample_rate)
                    sentence_buffer = sentences[-1]

            # Play any remaining text
            if sentence_buffer.strip():
                audio_out = tts.synthesize(sentence_buffer.strip())
                audio_out = apply_radio_filter(audio_out, tts.sample_rate)
                play_audio(audio_out, tts.sample_rate)

            response_text = "".join(response_parts)
            logger.info(f"Oracle: {response_text}")
            store.add_message(session_id, "assistant", response_text)

            await ctx.maybe_summarize()

    except KeyboardInterrupt:
        logger.info("Oracle signing off.")
    finally:
        store.close()


async def run(mode: str = "text") -> None:
    """Main entry point for the Oracle."""
    if mode == "text":
        await text_repl()
    elif mode == "voice":
        await voice_loop()
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)
