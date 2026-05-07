# Workstream 8: Diagnostic web page

A FastAPI-based diagnostic web UI showing live system + app health,
designed for tuning, debugging, and the "is it actually doing anything?"
check. Bound by default to `0.0.0.0:8000` (LAN-accessible) and runnable
either standalone or under its own systemd unit.

## Status

First cut landed (`oracle/diag/`): FastAPI server, page templates,
TTS-worker subprocess for free-RSS testing, and `radio-oracle-diag.service`
systemd unit. Iterating on the dashboard panels, metrics, and a
"talk to me" debug path.

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
  server.py                # FastAPI app, routes, dashboard
  tts_worker.py            # subprocess Piper invocation for RSS hygiene
systemd/
  radio-oracle-diag.service
oracle/
  health.py                # subsystem check primitives (consumed by diag)
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
- `GET /` → dashboard (single-page HTML)
- `GET /api/health` → subsystem checks
- `GET /api/state` → current mode + last events
- `GET /api/metrics` → CPU/GPU/disk/memory snapshot
- `GET /api/logs?tail=N` → recent log lines
- `POST /api/debug/chat` → streamed pipeline response

**Consumes** (read-only):
- WS 1: `StatusLEDs.mode`, `PowerSwitch.is_on`
- WS 2: `Retriever.list_collections()` + per-collection counts
- WS 5: device enumeration (`sounddevice.query_devices`)
- WS 6: `check_ollama()`, `oracle.persona.build_system_prompt`
- WS 7: log ring buffer; current `OracleApp` state when running

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
- [ ] Tegrastats parsing for per-rail GPU power
- [ ] Auth on the `POST /api/debug/chat` endpoint (LAN-trust assumption today)
- [ ] Persist last-N pipeline traces for postmortem
- [ ] Lightweight JS, no build step (single hand-written `app.js`)
