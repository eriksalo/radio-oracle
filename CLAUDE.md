# Radio Oracle

Offline voice assistant running on Jetson Orin Nano Super 8GB inside a vintage radio enclosure.

## Quick Start

```bash
make install    # create venv, install deps
make run        # python -m oracle
make lint       # ruff check + format
make test       # pytest
```

## Architecture

- `oracle/core.py` — main event loop (text REPL or voice mode)
- `oracle/llm.py` — async Ollama streaming client
- `oracle/stt.py` — Whisper STT (whisper.cpp, GPU)
- `oracle/tts.py` — Piper TTS (CPU, ONNX)
- `oracle/audio.py` — mic capture, speaker playback, VAD
- `oracle/rag/` — ChromaDB retrieval, embeddings, chunking
- `oracle/memory/` — conversation persistence (SQLite + ChromaDB)
- `oracle/persona.py` — system prompt builder from persona config
- `oracle/hardware/` — GPIO button, LEDs, audio routing
- `config/settings.py` — Pydantic BaseSettings, all `ORACLE_` prefixed env vars

## Key Design Decisions

- LLM: Ollama + Llama 3.2 3B Q4_K_M (~2.5GB VRAM)
- STT and LLM are sequential (never concurrent) to fit in 8GB unified memory
- TTS runs on CPU to avoid GPU contention
- ChromaDB embeddings (all-MiniLM-L6-v2) run on CPU
- Knowledge ingestion runs on workstation, rsync ChromaDB to Jetson
- Config via env vars with `ORACLE_` prefix (direnv-compatible)

## Conventions

- All Python: snake_case, type hints required
- Logging via loguru (never print())
- Error handling: explicit, never silent
- Config: Pydantic BaseSettings, env-driven
