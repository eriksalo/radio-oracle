# Radio Oracle

A fully offline voice assistant housed in a vintage radio enclosure. Ask it anything — if the world ends, this box can teach you how to survive.

Powered by a Jetson Orin Nano Super with a 1TB knowledge store spanning Wikipedia, iFixit, Project Gutenberg, survival manuals, and more. The Oracle speaks with a retro-futuristic Fallout-style personality, treating all knowledge as declassified archive entries.

> *"Good day, citizen. The Oracle is online and operational. State your inquiry, and I shall consult the archives."*

## Hardware

| Component | Role |
|-----------|------|
| Jetson Orin Nano Super 8GB | Brain (40 TOPS, CUDA 12.2) |
| 1TB NVMe SSD | Knowledge storage |
| Seeed reSpeaker Lite (XMOS XU316) | Voice input — see [`firmware/`](firmware/) for on-chip DSP notes |
| USB DAC + speaker (UACDemoV1.0) | Voice output through vintage radio enclosure. Echo cancelled in software via PulseAudio (`module-echo-cancel`); see [`docs/SETUP.md`](docs/SETUP.md#16-echo-cancellation-pulseaudio-aec-stack). |
| PTT button (GPIO) | Push-to-talk |
| Status LEDs (GPIO) | Idle / listening / thinking / speaking |
| Enclosure | Vintage radio ([zionbrock.com/radio](https://zionbrock.com/radio)) |

## Software Stack

| Layer | Choice |
|-------|--------|
| LLM | Ollama + Qwen3-4B-Instruct-2507 (Q4_K_M) |
| STT | whisper.cpp (small.en, GPU) |
| TTS | Kokoro (am_michael, CPU, ONNX) |
| RAG | FAISS IVF-PQ + nomic-embed-text-v1.5 (768-d) |
| Memory | SQLite + conversation summaries |
| Config | Pydantic BaseSettings (`ORACLE_` env prefix) |

Everything runs locally. No network required after setup.

## Quick Start

### Development (any machine)

```bash
git clone https://github.com/eriksalo/radio-oracle.git
cd radio-oracle

make install        # create venv, install core + dev deps
make test           # run tests
make run            # text REPL mode (requires Ollama running)
```

### Jetson Deployment

```bash
# One-time setup (run as root)
sudo ./scripts/setup_jetson.sh --dry-run   # preview
sudo ./scripts/setup_jetson.sh             # install everything

# Download models
./scripts/download_models.sh
ollama pull qwen3:4b-instruct-2507-q4_K_M

# Install all runtime deps
.venv/bin/pip install -e ".[all]"

# Run
python -m oracle --mode voice
```

### Knowledge Base Setup

Three-stage pipeline on a workstation with an NVIDIA GPU, then rsync the
FAISS artifacts to the Jetson. See [`docs/rag-migration-runbook.md`](docs/rag-migration-runbook.md)
for full command-by-command detail.

```bash
# 1. Download knowledge sources (~60 GB of ZIMs)
./scripts/download_knowledge.sh --dry-run   # preview
./scripts/download_knowledge.sh

# 2. Stage chunk text in ChromaDB (source of truth for chunk content)
pip install -e ".[ingest]"
python scripts/ingest_zim.py data/knowledge/wikipedia_en_all_nopic_latest.zim
python scripts/ingest_zim.py data/knowledge/ifixit_en_all_latest.zim --collection ifixit
# ...one ingest_zim.py call per ZIM source

# 3. Re-embed each collection with nomic-v1.5 (768-d) to flat .f32 + .sqlite
python scripts/reembed_collection.py \
    --source wikipedia --target wikipedia \
    --model nomic-ai/nomic-embed-text-v1.5 --dim 768 \
    --db-path data/chroma --out-dir data/embeddings \
    --workers 12 --batch-size 2000 --encode-batch-size 256

# 4. Build the per-collection FAISS IVF-PQ index
python scripts/build_faiss_ivfpq.py \
    --name wikipedia --in-dir data/embeddings --out-dir data/faiss --dim 768

# 5. Copy FAISS artifacts to Jetson (chunk text is embedded in the FAISS sqlite)
rsync -av data/faiss/ jetson:/opt/radio-oracle/data/faiss/
```

ChromaDB stays on the workstation as the read-only source of truth for
chunk text + the legacy MiniLM-L6 embeddings; only the FAISS layer ships
to the Jetson. Music has its own pipeline (no ZIM; tags from `mutagen`
into `data/music.db`, then the same flat-file → FAISS build).

## Project Structure

```
oracle/
  __main__.py          # CLI entry point
  core.py              # Main loop (text REPL + voice mode)
  llm.py               # Async Ollama streaming client
  stt.py               # Whisper speech-to-text
  tts.py               # Piper text-to-speech
  audio.py             # Mic capture, playback, VAD, AM radio filter
  persona.py           # System prompt builder
  health.py            # Subsystem health checks
  rag/
    retriever.py       # Pluggable backend dispatch + tiered modes
    backends/
      chroma.py        # ChromaDB backend (legacy collections)
      faiss_ivfpq.py   # FAISS IVF-PQ backend (production)
    embedder.py        # Sentence-transformer / nomic embeddings
    chunker.py         # Text chunking for ingestion
    modes.py           # Snappy vs. deep retrieval params
    reranker.py        # Cross-encoder rerank for deep mode
    router.py          # Per-query collection routing
  memory/
    store.py           # SQLite conversation persistence
    context.py         # LLM context window builder
    summarizer.py      # Conversation summarization
  hardware/
    button.py          # GPIO push-to-talk
    leds.py            # Status LED control
    audio_routing.py   # USB audio auto-detection
config/
  settings.py          # Pydantic BaseSettings
  persona.toml         # Oracle personality (editable)
scripts/               # Download, ingest, and setup scripts
systemd/               # Service file for boot-on-startup
```

## Configuration

All settings use the `ORACLE_` env prefix and can be set via `.env` / `direnv`:

```bash
ORACLE_OLLAMA_HOST=http://localhost:11434
ORACLE_OLLAMA_MODEL=qwen3:4b-instruct-2507-q4_K_M
ORACLE_MODE=voice              # or "text"
ORACLE_LOG_LEVEL=INFO
ORACLE_RAG_TOP_K=5
ORACLE_VAD_SILENCE_DURATION=1.5
ORACLE_PTT_GPIO_PIN=18
```

See [`config/settings.py`](config/settings.py) for all options.

## Memory Budget (8GB unified)

| Component | Memory |
|-----------|--------|
| Ollama LLM | ~3 GB |
| Whisper STT | ~1 GB (load/unload per utterance) |
| Kokoro TTS | ~0.3 GB (CPU only) |
| FAISS indices + nomic embedder | ~1.5 GB (CPU only; ~1.65 GB of indices mmap'd lazily) |
| Python + OS | ~1.5 GB |
| **Total** | **~7.3 GB** |

STT and LLM run sequentially — never concurrent — so peak GPU usage stays around 3GB.

## Knowledge Sources

| Source | Size | Description |
|--------|------|-------------|
| Wikipedia EN | 22 GB | Full text, no images (ZIM) |
| iFixit | 2.5 GB | Repair guides (ZIM) |
| Project Gutenberg | 20 GB | Public domain books |
| Wikibooks | 2 GB | Textbooks and manuals (ZIM) |
| WikiMed | 1 GB | Medical encyclopedia (ZIM) |
| Stack Exchange | 10 GB | Q&A subset (XML) |
| Army field manuals | 2 GB | Survival and technical manuals |

## License

MIT
