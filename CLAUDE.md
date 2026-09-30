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
- `oracle/stt.py` — STT factory + `listen()` (the one record→transcribe path); backends: `stt_parakeet.py` (default on the Jetson), `stt_streaming.py` (Nemotron, opt-in), Whisper
- `oracle/tts.py` — Kokoro TTS client (in-process CPU, or the GPU sidecar `oracle/tts_server.py` when `ORACLE_TTS_BACKEND=server`)
- `oracle/audio.py` — mic capture, speaker playback (serialized), AM-radio filter
- `oracle/endpoint.py` — end-of-utterance: energy (default) / Silero VAD / Silero + Smart Turn v3 (`ORACLE_VAD_BACKEND`)
- `oracle/timing.py` — per-turn stage timer (`TURN …` log line + `timing` activity event; `ttfa` = end of speech → first audio)
- `oracle/rag/` — FAISS IVF-PQ retrieval (nomic-v1.5), pluggable backends, tiered modes, cross-encoder rerank, query router
- `oracle/memory/` — conversation persistence (SQLite + summarization), `journal.py` (durable activity events → the "What you remember doing" prompt block), `users.py` (users + voiceprints); profiles, sessions, events and bookmarks are all per user
- `oracle/welcome.py` — power-on routine: chime + listen (7 s) → "This is the Librarian…" (5 s) → the four options (5 s) → music; anything heard goes through the dispatcher (`about_device` = who built it + live counts)
- `oracle/speaker.py` — speaker identification (TitaNet via sherpa-onnx); asks "Is this Erik?" once per session when unsure and enrols the answer
- `oracle/persona.py` — system prompt builder from persona config
- `oracle/hardware/` — ADS1115 button / power switch / pot, RGB status LED (`leds.py`; scheme in `docs/led-behaviour.md`), audio routing
- `oracle/music/` — music library + player (`mpg123` subprocess → PulseAudio speaker sink)
- `oracle/books/` — book library + reader (FTS5 search, per-user bookmarks, chapter navigation by voice; fresh books start past the Gutenberg preamble; paragraphs spoken as ≤30-word pipelined units)
- `oracle/diag/` — phosphor-CRT styled diagnostic web GUI (FastAPI, port 8000; page + favicon + fonts in `oracle/diag/static/`)
- `config/settings.py` — Pydantic BaseSettings, all `ORACLE_` prefixed env vars

## Key Design Decisions

- LLM: Qwen3-4B-Instruct-2507 Q4_K_M via Ollama (default) or llama.cpp's `llama-server` in NVIDIA's JP6 container (`ORACLE_LLM_BACKEND=llama-server`, `systemd/llama-server.service`, same GGUF blob); llama3.2:3b is the rollback model
- STT and LLM are sequential (never concurrent) to fit in 8GB unified memory
- LLM calls always set num_ctx (8192) — Ollama's 2048 default silently truncates
- Memory: sessions are summarized at close (or caught up at next boot) and folded
  into a rolling profile row; both are injected into every turn's context
- TTS: Kokoro on the Orin CPU is RTF ~0.8 (measured 2026-09-27) — too slow for a snappy first word — so on the Jetson it runs on the GPU in a cp310 sidecar venv (`.venv-tts`, onnxruntime-gpu from pypi.jetson-ai-lab.io; the app venv is cp311 and has no CUDA onnxruntime). Replies are cut into short units (`oracle.core.SpeechSplitter`) and synthesized/played in a 3-stage pipeline.
- Speech playback (`oracle/audio.py::_stream_play`) uses PortAudio **blocking writes** into a 250 ms buffer, never a Python callback: the USB DAC's own buffer is 32 ms and any interpreter stall (wake-word inference, a fork, a page fault) underran a callback stream (1,018 underflows in one 9 s clip, 2026-09-30). The volume bridge smooths the pot and only calls `pactl` on ≥2 % moves. To test speech end to end without a person: `POST /api/speak` "Librarian." then the question through the diag server; the radio hears its own speaker.
- Prompt layout is prefix-cache friendly: persona, memory, summary, history, *then* the RAG block, then the question once (`oracle/memory/context.py`). Keep anything that changes per turn at the end.
- Latency regression harness: `sudo /opt/radio-oracle/scripts/sim_turn.sh` on the Jetson (stops/starts the service; `docs/golden_questions.txt`); results in `docs/deploy-2026-09-latency.md`.
- RAG: FAISS IVF-PQ (PQ-64, METRIC_INNER_PRODUCT, score_scale=20.0) per collection, queried with `nomic-embed-text-v1.5` (768-d). Backend is pluggable per collection via `collection_backends` so old ChromaDB collections still work if needed.
- Tiered retrieval: snappy first-pass (`tier1_top_k`) returns immediately; deep mode adds a cross-encoder rerank on a larger candidate pool (workstation/CPU). See `oracle/rag/modes.py`.
- Workstation builds FAISS indices from ChromaDB-staged chunks; only `data/faiss/` rsyncs to the Jetson. ChromaDB is workstation-only after the FAISS cutover (2026-05-19).
- Query embedder on the Jetson is nomic-v1.5 fp32 ONNX via onnxruntime (`ORACLE_EMBEDDING_RUNTIME=onnx`, ~64 ms/query, no torch in the process); sentence-transformers stays the workstation/ingest path. The vectors are **un-normalized** mean-pooled outputs — the FAISS `score_scale`/distance gate are calibrated on that; never L2-normalize query vectors.
- Audio architecture (see `docs/SETUP.md` §1.6): **asymmetric routing.** Mic capture goes through PulseAudio's `module-echo-cancel` (`aec_source`) for NS/AGC; music + TTS go *direct* to the real USB speaker sink at 48 kHz, bypassing AEC. Music is decoded by an `mpg123` subprocess at ~1 % CPU (the prior in-process miniaudio+scipy+sounddevice pipeline pegged 100 %+ and underran constantly). Trade-off: wake-word reliability degrades during music since AEC has no music reference; the action button is the reliable wake during playback. On-chip AEC on the XU316 doesn't apply either — separate USB devices, no shared reference. IC/NS/AGC/VNR on the XU316 still help (mic-input-only DSP). Pulse config tracked at `systemd/pulse-default.pa`; firmware bin + DFU procedure in `firmware/`.
- The GPU TTS sidecar's CUDA arena must shrink after every run (`ORACLE_TTS_ARENA_SHRINK=1`, kSameAsRequested) and stays capped at 512 MB: variable-length units fragment the BFC arena and, without shrinkage, after a few long units every request fails and the radio goes mute until a restart. Units are sized by *spoken* words (`oracle.tts.spoken_words`: digits count) — a number-heavy 24-word sentence is 13 s of speech and needs more arena than a paragraph. `/health` shows `fails=`/`rebuilds=`; `scripts/probe_say.py` is the regression check.
- Speech playback (`oracle/audio.py::_stream_play`) writes a buffer's worth of silence behind every clip: `stream.stop()` on the pulse path drops the ring buffer instead of draining it, which clipped the last ~270 ms of every unit. Check with `scripts/probe_tail.py` (records the sink monitor; stop the radio first).
- Memory budget on the Jetson is the hard constraint (llama-server ~3.2 GB + TTS sidecar ~1 GB pinned; app ~2 GB). zram swap (`nvzramconfig`) is **disabled** — it held swapped pages compressed in RAM and starved the box; the 8 GB NVMe swapfile takes the overflow. Never add a resident model without measuring `free -m`.
- The ADS1115 (pot, button, power switch) has ONE reader: the app's SharedAdcPoller. A second reader interleaves on the mux and yields another channel's voltage (phantom button presses that cut speech off, pot jumps). The dashboard never touches the chip unless its "DIRECT ADS1115 READS" box is ticked (off after every restart, and only honoured while no `oracle --mode hardware` process exists); its readers are one-shot `DigitalSwitch` objects, never the `make_*_switch()` factories, which auto-start a poller thread. ADC probes: stop `radio-oracle-diag` too.
- Status LED (`oracle/hardware/leds.py`, full table in `docs/led-behaviour.md`): colour = context, pattern = activity.

  | | listen (mic open) | think | speak / output | idle |
  |---|---|---|---|---|
  | questions — **blue** | 1 s blink | 0.3 s blink | solid | — |
  | music — **green** | 1 s blink | 0.3 s blink | solid (playing or announcing) | — |
  | books — **purple** | 1 s blink | 0.3 s blink | solid (reading or announcing) | paused: 2 s blink |
  | standby **off** · boot **amber** 2 s · waiting **white** 2 s · error **red** 0.5 s | | | | |

  Modes are `q_/music_/book_` + `listen/think/speak`, plus `music_play`, `book_read`, `book_paused`, `boot`, `waiting`, `error`, `off`. The base colour is picked from the command's context (`commands.py`: reader → book, music playing → music, else q). Keep `Color` a dataclass — the table is built from it at import.
- On-device probes must not leave state behind: use a scratch `ORACLE_DB_PATH`, and restore bookmarks (see `scripts/probe_reader.py`). A stray bookmark once made the radio resume a book nobody asked for.
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
8. **Diagnostic web page** — `oracle/diag/`

When changing code, prefer to stay inside the workstream that owns the file.
Cross-workstream calls go through the *Interface contract* documented in each
workstream's doc, and use lazy imports so missing deps degrade gracefully.

## Deploying to the Jetson

The Jetson is at `erik@radio-oracle.local`, project installed at `/opt/radio-oracle` (owned by `oracle` user).

```bash
# Push code changes (files owned by oracle, need sudo on remote)
rsync -avz -e ssh --rsync-path="sudo rsync" <files> erik@radio-oracle.local:/opt/radio-oracle/...

# Restart after deploy
ssh erik@radio-oracle.local "sudo systemctl restart radio-oracle"

# Check logs
ssh erik@radio-oracle.local "sudo journalctl -u radio-oracle -f"

# Push FAISS indices
rsync -av data/faiss/ erik@radio-oracle.local:/opt/radio-oracle/data/faiss/
```

Always commit and push after code changes. Always deploy and restart the service to verify on hardware.

## Conventions

- All Python: snake_case, type hints required
- Logging via loguru (never print())
- Error handling: explicit, never silent
- Config: Pydantic BaseSettings, env-driven
