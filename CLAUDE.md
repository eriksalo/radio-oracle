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
- `oracle/rag/` — FAISS IVF-PQ retrieval (nomic-v1.5), pluggable backends, tiered modes, cross-encoder rerank, query router
- `oracle/memory/` — conversation persistence (SQLite + summarization)
- `oracle/persona.py` — system prompt builder from persona config
- `oracle/hardware/` — GPIO button, RGB LED, power switch, audio routing
- `oracle/music/` — music library + player (stub)
- `oracle/books/` — book reader (stub)
- `oracle/web/` — Pip-Boy styled diagnostic web GUI (FastAPI, port 8080)
- `config/settings.py` — Pydantic BaseSettings, all `ORACLE_` prefixed env vars

## Key Design Decisions

- LLM: Ollama + Llama 3.2 3B Q4_K_M (~2.5GB VRAM)
- STT and LLM are sequential (never concurrent) to fit in 8GB unified memory
- TTS runs on CPU to avoid GPU contention
- RAG: FAISS IVF-PQ (PQ-64, METRIC_INNER_PRODUCT, score_scale=20.0) per collection, queried with `nomic-embed-text-v1.5` (768-d). Backend is pluggable per collection via `collection_backends` so old ChromaDB collections still work if needed.
- Tiered retrieval: snappy first-pass (`tier1_top_k`) returns immediately; deep mode adds a cross-encoder rerank on a larger candidate pool (workstation/CPU). See `oracle/rag/modes.py`.
- Workstation builds FAISS indices from ChromaDB-staged chunks; only `data/faiss/` rsyncs to the Jetson. ChromaDB is workstation-only after the FAISS cutover (2026-05-19).
- Embedder runs on CPU on the Jetson today (~1.2 s warm). cp311 CUDA torch wheels for JetPack 6.2 don't exist; see `docs/rag-migration-runbook.md` §"Known follow-up" for the three fix paths.
- Mic (ReSpeaker Lite, XMOS XU316, USB firmware v2.0.7) does AEC/IC/NS/AGC/VNR on-chip before audio reaches the Jetson. Do NOT add software AEC or noise suppression in `oracle/audio.py` — it would fight the on-board DSP. The host pipeline gets a single processed mono channel. Firmware bin + DFU procedure in `firmware/`.
- Config via env vars with `ORACLE_` prefix (direnv-compatible). The Jetson's `/opt/radio-oracle/.env` sets `ORACLE_COLLECTION_BACKENDS` to route every collection to FAISS.

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
8. **Diagnostic web page** — `oracle/web/`

When changing code, prefer to stay inside the workstream that owns the file.
Cross-workstream calls go through the *Interface contract* documented in each
workstream's doc, and use lazy imports so missing deps degrade gracefully.

## Deploying to the Jetson

The Jetson is at `erik@10.0.0.186`, project installed at `/opt/radio-oracle` (owned by `oracle` user).

```bash
# Push code changes (files owned by oracle, need sudo on remote)
rsync -avz -e ssh --rsync-path="sudo rsync" <files> erik@10.0.0.186:/opt/radio-oracle/...

# Restart after deploy
ssh erik@10.0.0.186 "sudo systemctl restart radio-oracle"

# Check logs
ssh erik@10.0.0.186 "sudo journalctl -u radio-oracle -f"

# Push FAISS indices
rsync -av data/faiss/ erik@10.0.0.186:/opt/radio-oracle/data/faiss/
```

Always commit and push after code changes. Always deploy and restart the service to verify on hardware.

## Conventions

- All Python: snake_case, type hints required
- Logging via loguru (never print())
- Error handling: explicit, never silent
- Config: Pydantic BaseSettings, env-driven
