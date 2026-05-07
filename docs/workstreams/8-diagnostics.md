# Workstream 8: Diagnostic web page

A FastAPI-based diagnostic web UI showing live system + app health,
designed for tuning, debugging, and the "is it actually doing anything?"
check. Bound by default to `0.0.0.0:8000` (LAN-accessible) and runnable
either standalone or under its own systemd unit.

## Status

Working dashboard with audio I/O test cards, streaming LLM ask, system
health checks, live state from a running `radio-oracle` service, GPU
metrics from `tegrastats`, and a log tail (own process + sibling
service via `journalctl`). All wired into the Pip-Boy themed UI.

## Scope

- Local HTTP server (FastAPI/uvicorn) with a single-page dashboard
- Subsystem health checks: Ollama, ChromaDB collections, audio device
  enumeration, GPIO availability
- Live state: current mode, last button event, last LLM latency, last
  transcription, queue depths
- System metrics: CPU temp, GPU temp, GPU memory, disk free, uptime
- "Talk to me" debug panel — type a message, see the full pipeline response
- Recent log tail (loguru sink → ring buffer → endpoint)
- TTS smoke test that runs Piper in a per-call subprocess so RSS is
  released between invocations (`oracle/diag/tts_worker.py`)
- Coordination with the main `radio-oracle.service`: warns if it's active
  (mic/speaker would be taken)

## File ownership

```
oracle/diag/
  __init__.py
  __main__.py              # `python -m oracle.diag` — starts uvicorn
  server.py                # FastAPI app, routes, single-page dashboard
  tts_worker.py            # subprocess Piper invocation for RSS hygiene
  tegrastats.py            # background tegrastats poller + parser
oracle/
  log.py                   # ring-buffer sink for /api/logs (in-process)
  state.py                 # cross-process state file (running app → diag)
systemd/
  radio-oracle-diag.service
```

## Settings

```bash
# Set on the systemd unit or via env
ORACLE_DIAG_HOST=0.0.0.0     # default; bind interface
ORACLE_DIAG_PORT=8000        # default
```

## Dependencies

```toml
diagnostics = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
]
```

`pip install -e ".[diagnostics]"`. Already declared as a `[diagnostics]`
extra in `pyproject.toml`.

## Interface contract

**Provides** (HTTP, browser- or curl-consumable):
- `GET /`                     → dashboard (single-page HTML)
- `POST /api/record`          → mic capture → WAV
- `POST /api/speak`           → Piper synth (subprocess) + Jetson playback
- `GET /api/speak.wav`        → synth only, return WAV (no playback)
- `POST /api/ask`             → blocking LLM (+ optional RAG) — returns answer
- `POST /api/ask/stream`      → streaming SSE: `meta` event then `token` events then `done`
- `GET /api/health`           → `{ollama, chroma, audio, gpio}` with up/down + detail
- `GET /api/state`            → snapshot of the running `radio-oracle.service`
                                 (mode, power, last button, last transcription, pid liveness)
- `GET /api/logs?tail=N`      → in-process loguru ring buffer (own logs)
- `GET /api/journal?unit=…&tail=N` → systemd journal tail for `radio-oracle` or `radio-oracle-diag`
- `GET /api/gpu`              → tegrastats snapshot (gpu%, freq, temps, RAM)
- `GET /api/stats`            → CPU/mem/swap/load avg/temps via psutil
- `GET /api/persona`          → user_name, assistant_name
- `POST /api/persona`         → set user_name (persists to persona.toml)

**Consumes** (read-only):
- WS 2: `Retriever.list_collections()` + counts (health + ask)
- WS 5: `oracle.audio.record_until_silence`, `play_wav_bytes`,
         `sounddevice.query_devices` (audio health)
- WS 6: `check_ollama`, `stream_chat`, `chat`, `build_system_prompt`,
         persona getters/setters
- WS 7: `oracle.state.read_state()` — non-shared, file-backed

**Cross-process state**: the running app (`oracle/app.py::OracleApp`)
publishes a snapshot to `$XDG_RUNTIME_DIR/radio-oracle-state.json` (or
`/tmp/radio-oracle-state.json`) on every transition. `/api/state`
reads + checks `pid_exists` so a stale file is reported as "not running".

**Coordination with the main app**: the diagnostics service detects
`radio-oracle.service` running and warns the operator that the mic and
speaker are held — debug TTS calls go through the subprocess worker so
they don't fight Piper for the audio device.

## Standalone exercise

```bash
# Stop the main app first if it's running, then:
python -m oracle.diag --host 0.0.0.0 --port 8000

# Or via systemd:
sudo systemctl start radio-oracle-diag

# From any LAN device:
curl http://<jetson>:8000/api/health | jq
curl http://<jetson>:8000/api/metrics | jq
# Or just open http://<jetson>:8000 in a browser
```

## TODO

- [ ] mDNS / Bonjour so the page is discoverable as `oracle.local:8000`
- [ ] Per-collection HNSW memory footprint chart
- [ ] Tegrastats: per-rail GPU power (currently only GPU%, freq, temps)
- [ ] Auth / token on `POST /api/ask*` (LAN-trust assumption today)
- [ ] Persist last-N pipeline traces for postmortem
- [ ] Split inline HTML/JS/CSS into `oracle/diag/static/` with FastAPI
      `StaticFiles` so iteration on the dashboard isn't a Python edit
