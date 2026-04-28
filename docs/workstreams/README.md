# Workstreams

Radio Oracle is split into 5 independent workstreams. Each can be developed and tested in isolation.

| # | Workstream | Status | Key Paths |
|---|-----------|--------|-----------|
| 1 | [RAG Ingest](1-rag-ingest.md) | In progress | `scripts/ingest_*.py`, `oracle/rag/` |
| 2 | [Music Player](2-music-player.md) | Not started | `oracle/music/`, `scripts/ingest_music.py` |
| 3 | [TTS & Conversation](3-tts-conversation.md) | Working E2E | `oracle/core.py`, `oracle/tts.py`, `oracle/stt.py`, `oracle/llm.py` |
| 4 | [Electronics & Wiring](4-electronics.md) | Basic GPIO done | `oracle/hardware/` |
| 5 | [Updates & Reliability](5-reliability.md) | Not started | `systemd/`, `oracle/health.py`, `scripts/setup_jetson.sh` |

## Interface Contracts

Workstreams communicate through well-defined boundaries:

- **RAG -> Conversation**: `Retriever.query(text) -> list[dict]` returns ranked chunks. Conversation code calls this; RAG ingest never touches conversation code.
- **Music -> Conversation**: `core.py` detects music intent via LLM and delegates to `oracle/music/player.py`. Music player owns audio output while playing; conversation pauses.
- **Hardware -> Conversation**: `core.py` reads PTT events from `hardware/button.py` and sets LED state via `hardware/leds.py`. Hardware modules expose simple async interfaces.
- **Reliability -> All**: `health.py` checks subsystem status. systemd manages process lifecycle. No code changes needed in other workstreams.
