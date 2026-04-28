# Workstream 3: TTS & Conversation

The core voice interaction pipeline. Currently working end-to-end on Jetson.

## Scope

- Whisper STT (load/unload per utterance for memory)
- Piper TTS (CPU, sentence-level streaming)
- LLM streaming via Ollama
- Conversation memory (SQLite + summarization)
- Context window management
- Persona / system prompt
- AM radio audio filter
- VAD (voice activity detection)

## Key Files

```
oracle/
  core.py                  # Main event loop (text REPL + voice)
  llm.py                   # Async Ollama streaming client
  stt.py                   # Whisper STT wrapper
  stt_worker.py            # STT in subprocess (memory isolation)
  tts.py                   # Piper TTS wrapper
  audio.py                 # Mic capture, playback, VAD, radio filter
  persona.py               # System prompt builder
  memory/
    store.py               # SQLite conversation persistence
    context.py             # LLM context window builder
    summarizer.py          # Conversation summarization via LLM
config/
  settings.py              # All ORACLE_* settings
  persona.toml             # Personality config
```

## Dependencies

```
pip install -e ".[voice,tts,stt]"
```

## Interface Contract

**Consumes**:
- `oracle/rag/retriever.py` — RAG context (lazy import, graceful fallback if unavailable)
- `oracle/hardware/` — PTT button events, LED state (lazy import, falls back to keyboard)

**Provides**:
- `run(mode)` entry point called from `__main__.py`
- Text REPL mode works on any machine (no hardware deps)
- Voice mode requires audio hardware + models

## Testing

```bash
pytest tests/test_llm.py tests/test_persona.py tests/test_memory.py

# Text REPL (needs Ollama running)
make run

# Voice mode (needs Ollama + audio hardware + models)
make run-voice
```

## TODO

- [ ] Interrupt handling (stop TTS when PTT pressed mid-response)
- [ ] Better sentence boundary detection (abbreviations, numbers)
- [ ] Warm-up: pre-load TTS on startup for faster first response
- [ ] Conversation export (dump session to text file)
- [ ] Multi-turn RAG (use conversation context to refine queries)
