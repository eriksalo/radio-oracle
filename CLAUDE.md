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

- `oracle/__main__.py` — CLI entry point, mode dispatch
- `oracle/app.py` — hardware-driven state machine (Standby/Radio/Librarian)
- `oracle/core.py` — text REPL + per-turn voice helper (`voice_init`/`voice_turn`/`voice_close`)
- `oracle/llm.py` — async Ollama streaming client
- `oracle/stt.py` — Whisper STT (whisper.cpp, GPU)
- `oracle/tts.py` — Piper TTS (CPU, ONNX)
- `oracle/audio.py` — mic capture, speaker playback, VAD, AM-radio filter
- `oracle/rag/` — ChromaDB retrieval, embeddings, chunking
- `oracle/memory/` — conversation persistence (SQLite + summarization)
- `oracle/persona.py` — system prompt builder from persona config
- `oracle/hardware/` — GPIO button, RGB LED, power switch, audio routing
- `oracle/music/` — music library + player (stub)
- `oracle/books/` — book reader (stub)
- `oracle/diagnostics/` — local web dashboard (stub)
- `config/settings.py` — Pydantic BaseSettings, all `ORACLE_` prefixed env vars

## Key Design Decisions

- LLM: Ollama + Llama 3.2 3B Q4_K_M (~2.5GB VRAM)
- STT and LLM are sequential (never concurrent) to fit in 8GB unified memory
- TTS runs on CPU to avoid GPU contention
- ChromaDB embeddings (all-MiniLM-L6-v2) auto-detect CUDA on workstation, CPU on Jetson
- Knowledge ingestion runs on workstation, rsync ChromaDB to Jetson
- Config via env vars with `ORACLE_` prefix (direnv-compatible)

## Workstreams

Project is split into **8 independent workstreams**. Each can be worked on in
isolation — see `docs/workstreams/README.md` for the index, dependency graph,
and per-workstream "standalone exercise" steps.

1. **Electronics & Wiring** — `oracle/hardware/`, `docs/wiring/`
2. **Large-data ingest / RAG** — `oracle/rag/`, `scripts/ingest_*.py`
3. **Music player** — `oracle/music/`
4. **Books & book reader** — `oracle/books/`
5. **Text-to-voice (TTS + audio I/O)** — `oracle/tts.py`, `oracle/audio.py`
6. **LLM behavior (chat, persona, memory)** — `oracle/llm.py`, `oracle/persona.py`, `oracle/memory/`
7. **Intro & working-flow (state machine, STT, deploy)** — `oracle/app.py`, `oracle/core.py`, `oracle/stt.py`, `systemd/`
8. **Diagnostic web page** — `oracle/diagnostics/`

When changing code, prefer to stay inside the workstream that owns the file.
Cross-workstream calls go through the *Interface contract* documented in each
workstream's doc, and use lazy imports so missing deps degrade gracefully.

## Conventions

- All Python: snake_case, type hints required
- Logging via loguru (never print())
- Error handling: explicit, never silent
- Config: Pydantic BaseSettings, env-driven
