# Workstream 7: Introduction & working flow

The integration layer. Owns the top-level state machine, the boot/shutdown
lifecycle, the STT input pipeline, wake word detection, the entry-point CLI,
and the deployment/systemd glue.

## Status

Working end-to-end. Three modes: `text`, `voice`, and `hardware`. The
hardware state machine (Standby/Radio/Librarian) is in `oracle/app.py`
and ties together every other workstream. Wake word ("librarian") detection
active in radio mode. Power switch immediately halts all I/O.

## Scope

- Entry point and mode dispatcher (`__main__.py`)
- Top-level hardware state machine (`OracleApp`: Standby/Radio/Librarian)
- Wake word detection (record → Whisper STT → check for keyword)
- Per-turn voice loop (record → transcribe → LLM → speak), refactored so
  the app can drive one turn at a time
- Whisper STT (load/unload per utterance for VRAM hygiene)
- Mode-aware LED transitions (calls Workstream 1 to set colors)
- Abort propagation: power switch → recording/playback/LLM immediate halt
- Logging configuration (loguru)
- systemd service + deployment scripts (Jetson setup)
- Health-check primitives consumed by Workstream 8

## File ownership

```
oracle/
  __main__.py              # CLI, --mode dispatch
  __init__.py
  core.py                  # voice_init/voice_turn/wake_word_listen/voice_close + text_repl
  app.py                   # OracleApp — Standby/Radio/Librarian state machine
  stt.py                   # WhisperSTT (load/unload per utterance)
  stt_worker.py            # STT in subprocess (memory isolation)
  log.py                   # loguru setup
systemd/
  radio-oracle.service     # main service unit
scripts/
  setup_jetson.sh          # one-time Jetson provisioning
  download_models.sh       # fetch Kokoro + Whisper models
docs/
  SETUP.md
```

## Settings

```bash
ORACLE_MODE=hardware              # text | voice | hardware
ORACLE_LOG_LEVEL=INFO
ORACLE_WHISPER_MODEL_PATH=models/whisper-base.en.bin
ORACLE_WHISPER_FORCE_CPU=true
ORACLE_WHISPER_LANGUAGE=en
ORACLE_WAKE_WORD=librarian
```

## Dependencies

```bash
pip install -e ".[stt,voice,rag,tts]"
./scripts/download_models.sh
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

For the production unit on Jetson: `[all]` plus the steps in
`scripts/setup_jetson.sh`.

## Interface contract

**Provides** (consumed by the user / the systemd unit):
- `python -m oracle --mode {text|voice|hardware}` is the entry point.
  `text` and `voice` work on any laptop; `hardware` requires the Jetson.

**Consumes** (everything else):
- WS 1: `ActionButton`, `PowerSwitch`, `StatusLEDs`, `VolumeControl`
- WS 2: `Retriever` (lazy)
- WS 3: `Player` (lazy, once it exists)
- WS 4: `Reader` (lazy, once it exists)
- WS 5: `KokoroTTS`, `play_audio`, `record_until_silence`, `apply_radio_filter`
- WS 6: `stream_chat`, `ConversationStore`, `ContextBuilder`,
  `build_system_prompt`, `get_greeting`

**State machine** (in `oracle/app.py::OracleApp`):
- power switch open      → STANDBY (LED off, all I/O halted immediately)
- power switch closed    → RADIO (default; LED green, wake word listening)
- wake word detected     → one voice turn, then back to RADIO
- long-press button      → RADIO ↔ LIBRARIAN (continuous voice conversation)
- short-press in RADIO   → next track / next book paragraph (when those exist)

## Standalone exercise

```bash
# Pure-orchestration smoke test (REPL — no audio, no hardware)
python -m oracle --mode text

# Voice loop without hardware (laptop with mic + speakers)
python -m oracle --mode voice

# Full hardware loop (Jetson)
python -m oracle --mode hardware

# Install + enable as a systemd service
sudo cp systemd/radio-oracle.service /etc/systemd/system/
sudo systemctl enable --now radio-oracle
journalctl -u radio-oracle -f
```

## TODO

- [ ] systemd watchdog integration (sd_notify) so a hung process restarts
- [ ] Auto-restart backoff (avoid crash loops)
- [ ] OTA update script: git pull → pip install → systemctl restart
- [ ] Disk-space monitoring (ChromaDB + music + books on 1 TB NVMe)
- [ ] Nightly SQLite vacuum / WAL checkpoint
- [ ] Short-press in Librarian = interrupt current TTS playback (currently no-op)
- [ ] Boot greeting customization per power-on time of day
- [x] Wake-word listener so the toggle-+-button ritual is optional
