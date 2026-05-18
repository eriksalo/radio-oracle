# Workstreams

Radio Oracle is split into **8 independent workstreams**. Each has its own
folder of code, its own tests, its own settings prefix, and a documented way
to develop and exercise it in isolation. Pick one, ignore the others.

| #   | Workstream                                | Status         | Code lives in                              |
|-----|-------------------------------------------|----------------|--------------------------------------------|
| [1](1-electronics.md)  | Electronics & Wiring                | Done (volume + power switch) | `oracle/hardware/`, `docs/wiring/` |
| [2](2-rag.md)          | Large-data ingest / RAG             | Working E2E (workstation) | `oracle/rag/`, `scripts/ingest_*.py`   |
| [3](3-music.md)        | Music player                        | Stub            | `oracle/music/`                          |
| [4](4-books.md)        | Books & e-reader                    | Working E2E (60k books) | `oracle/books/`, `scripts/index_books.py` |
| [5](5-tts.md)          | Text-to-voice (TTS + audio I/O)     | Working E2E (Kokoro) | `oracle/tts.py`, `oracle/audio.py`  |
| [6](6-llm.md)          | LLM behavior (chat, persona, memory)| Working E2E     | `oracle/llm.py`, `oracle/persona.py`, `oracle/memory/` |
| [7](7-orchestration.md)| Intro & working-flow (state machine, STT, deploy) | Working | `oracle/app.py`, `oracle/core.py`, `oracle/stt.py`, `systemd/` |
| [8](8-diagnostics.md)  | Diagnostic web page                 | Working | `oracle/web/`, `systemd/radio-oracle-web.service` |

## How to pick a workstream and start

1. Open the workstream's doc.
2. Skim **Scope** and **File ownership** — those tell you exactly what's
   yours to change.
3. Run the **Standalone exercise** section to confirm the workstream works
   end-to-end on your machine before touching anything.
4. Make changes only inside that workstream's owned files. If you find
   yourself editing a file owned by another workstream, the change probably
   belongs in *its* doc as a TODO instead.

## Cross-workstream rules

- **Settings**: each workstream owns an env-var prefix under `ORACLE_*`
  (e.g., `ORACLE_LED_*`, `ORACLE_MUSIC_*`). All defaults live in
  `config/settings.py`.
- **Lazy imports**: where one workstream consumes another, the import is
  done inside the function that needs it and wrapped in a try/except, so
  a missing dep degrades gracefully instead of crashing startup. See
  `oracle/core.py::_try_rag_query` for the pattern.
- **Interfaces are documented in each workstream's doc** under
  *Interface contract*. Don't add a new cross-workstream call without
  updating both docs.
- **Tests** go in `tests/test_<workstream-thing>.py`. Each workstream's doc
  lists its own tests under *Standalone exercise*.

## Dependency graph

```
                  ┌──────────────────────┐
                  │ 7. Orchestration     │
                  │  (app, core, STT)    │
                  └──┬─────┬─────┬──────┘
              uses  │     │     │  uses
                    ▼     ▼     ▼
       ┌─────────┐ ┌────┐ ┌────────┐ ┌───────┐
       │ 5. TTS  │ │ 6. │ │ 1. HW  │ │ 3.    │
       │ + audio │ │LLM │ │+wiring │ │Music  │
       └─────────┘ └─┬──┘ └────────┘ └───┬───┘
                    │ uses              │ uses TTS
                    ▼                    ▼
                 ┌────────┐          ┌─────────┐
                 │ 2. RAG │          │ 4. Books│
                 └────────┘          └─────────┘

8. Diagnostics — reads metrics from all of the above; depended on by none.
```

Workstreams above the line consume those below. Nothing flows the other way,
so you can stub or break a downstream workstream without affecting upstream
ones (the lazy-import pattern handles missing deps).
