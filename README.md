# Radio Oracle

A fully offline voice assistant housed in a vintage radio enclosure. Ask it anything — if the world ends, this box can teach you how to survive.

Powered by a Jetson Orin Nano Super with a 1TB knowledge store spanning Wikipedia, iFixit, Project Gutenberg, survival manuals, and more. The Oracle speaks with a retro-futuristic Fallout-style personality, treating all knowledge as declassified archive entries.

> *"Good day, citizen. The Oracle is online and operational. State your inquiry, and I shall consult the archives."*

## Hardware

| Component | Role |
|-----------|------|
| Jetson Orin Nano Super 8GB | Brain (40 TOPS, CUDA 12.2) |
| 1TB NVMe SSD | Knowledge storage |
| USB microphone array | Voice input |
| USB DAC + speaker | Voice output through vintage radio enclosure |
| PTT button (GPIO) | Push-to-talk |
| Status LEDs (GPIO) | Idle / listening / thinking / speaking |
| Enclosure | Vintage radio ([zionbrock.com/radio](https://zionbrock.com/radio)) |

## Software Stack

| Layer | Choice |
|-------|--------|
| LLM | Ollama + Llama 3.2 3B (Q4_K_M) |
| STT | whisper.cpp (small.en, GPU) |
| TTS | Piper (lessac-medium, CPU) |
| RAG | ChromaDB + all-MiniLM-L6-v2 |
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
ollama pull llama3.2:3b

# Install all runtime deps
.venv/bin/pip install -e ".[all]"

# Run
python -m oracle --mode voice
```

### Knowledge Base Setup

Run on a workstation with a GPU for faster embedding, then rsync to the Jetson:

```bash
# Download knowledge sources (~60GB)
./scripts/download_knowledge.sh --dry-run   # preview
./scripts/download_knowledge.sh

# Ingest into ChromaDB
pip install -e ".[ingest]"
python scripts/ingest_wikipedia.py data/knowledge/wikipedia_en_all_nopic_latest.zim
python scripts/ingest_generic_zim.py data/knowledge/ifixit_en_all_latest.zim --collection ifixit
python scripts/ingest_gutenberg.py data/knowledge/gutenberg/

# Copy to Jetson
rsync -av data/chroma/ jetson:/opt/radio-oracle/data/chroma/
```

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
    retriever.py       # ChromaDB semantic search
    embedder.py        # Sentence-transformer embeddings
    chunker.py         # Text chunking for ingestion
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
ORACLE_OLLAMA_MODEL=llama3.2:3b
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
| Piper TTS | ~0.3 GB (CPU only) |
| ChromaDB + embeddings | ~0.5 GB (CPU only) |
| Python + OS | ~1.5 GB |
| **Total** | **~6.3 GB** |

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
