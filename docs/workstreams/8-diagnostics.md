# Workstream 8: Diagnostic web page

A FastAPI-based diagnostic web UI showing live system + app health,
designed for tuning, debugging, and the "is it actually doing anything?"
check. Bound by default to `0.0.0.0:8000` (LAN-accessible) and runnable
either standalone or under its own systemd unit
(`radio-oracle-diag.service`, which Conflicts= the main app unit).

> History: an earlier, separate dashboard lived in `oracle/web/` on port
> 8080 with its own `oracle-web.service`. It was removed 2026-07 — the
> `oracle/diag` app on port 8000 is the only diagnostics stack; its one
> unique endpoint (`/api/conversations`) was ported over.

## Status

Working dashboard with audio I/O test cards, streaming LLM ask, system
health checks, live state from a running `radio-oracle` service, GPU
metrics from `tegrastats`, hardware controls (LED, pot, switches), a
per-service memory budget, the last turn's stage timing (ttfa), and a
log tail. Phosphor-CRT themed; the page is `oracle/diag/static/index.html`
served by FastAPI with self-hosted fonts and a favicon (`favicon.ico`
16/32/48 + SVG + apple-touch-icon; the mascot's head).

Refreshed 2026-09-28. Measured cost on the Jetson before the refresh:
one open browser tab produced ~6.7 requests/s (the hardware card polled
at 250 ms), the diag process sat at ~3.7 % of a core, uvicorn's access log
wrote ~62k journal lines/day, and a backgrounded tab kept polling forever.
Now: polling stops while the tab is hidden, the hardware card polls at
1 s, stats/GPU at 3 s, activity/state at 2 s, memory at 5 s, logs at 5 s,
health at 30 s (~2.6 requests/s visible, 0 hidden). Measured after: one
tab at the new cadence adds ~1.2 % of a core (was ~3.7 % + 0.4 % journald
for the access log). The access log is off;
the Retriever is built once per process instead of per health check.
Per-request CPU on the Orin is 2–7 ms for the read-only endpoints and
~60 ms for `/api/health`. The process is ~50 MB resident; a stale process
started before the ONNX-embedder fix had libtorch mapped (~150 MB of it
swapped out), so restart the unit after deploying embedder changes.

## Scope

- Local HTTP server (FastAPI/uvicorn) with a single-page dashboard
- Subsystem health checks: LLM (llama-server or Ollama), archives (FAISS
  collections), voice (TTS sidecar `/health` or local model files), audio
  device enumeration, GPIO availability
- Live state: current mode, power, last button event, last LLM latency,
  last transcription, queue depths
- System metrics: CPU temp, GPU temp, GPU memory, disk free, uptime
- Memory budget: per-service cgroup memory (resident + swap) for
  llama-server / radio-oracle / TTS sidecar / diag — the box's binding
  constraint, on one stacked bar
- Last turn: ttfa and per-stage seconds from the `timing` activity event,
  with a small ttfa history
- Hardware controls: LED color picker, pot/switch readings
- "Talk to me" debug panel — type a message, see the full pipeline response
- Recent log tail (loguru sink → ring buffer → endpoint)
- TTS smoke test that runs Kokoro in a per-call subprocess so RSS is
  released between invocations (`oracle/diag/tts_worker.py`)
- Coordination with the main `radio-oracle.service`: warns if it's active
  (mic/speaker would be taken)

## File ownership

```
oracle/diag/
  __init__.py
  __main__.py              # `python -m oracle.diag` — starts uvicorn
  server.py                # FastAPI app + routes
  static/index.html        # the single-page UI (served at /)
  static/favicon.*         # favicon.ico (16/32/48), favicon.svg, apple-touch-icon.png, icon-192.png
  static/*.woff2           # VT323 + Share Tech Mono, latin subsets (OFL), self-hosted
  tts_worker.py            # persistent-subprocess TTS worker
  tegrastats.py            # background tegrastats poller + parser
oracle/
  log.py                   # ring-buffer sink for /api/logs (in-process)
  state.py                 # cross-process state file (running app → diag)
  health.py                # health check primitives
systemd/
  radio-oracle-diag.service
```

## Settings

```bash
# CLI flags on the unit: python -m oracle.diag --host 0.0.0.0 --port 8000
```

## Dependencies

```toml
diag = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "psutil>=5.9",
    ...
]
```

`pip install -e ".[diag]"` (see `pyproject.toml` for the full list).

## Interface contract

**Provides** (HTTP, browser- or curl-consumable):
- `GET /`                     → dashboard (`static/index.html`, no templating)
- `GET /favicon.ico`, `GET /static/{name}` → icon + fonts, 1-day cache
- `POST /api/record`          → mic capture → WAV
- `POST /api/speak`           → Kokoro synth (subprocess) + Jetson playback
- `GET /api/speak.wav`        → synth only, return WAV (no playback)
- `POST /api/ask`             → blocking LLM (+ optional RAG) — returns answer
- `POST /api/ask/stream`      → streaming SSE: `meta` event then `token` events then `done`
- `GET /api/health`           → `{ok, llm, rag, tts, audio, gpio}` each with `ok` + `detail` (+ `latency_ms`)
- `GET /api/state`            → snapshot of the running `radio-oracle.service`
                                 (mode, power, last button, last transcription, pid liveness)
- `GET /api/logs?tail=N`      → in-process loguru ring buffer (own logs)
- `GET /api/journal?unit=…&tail=N` → systemd journal tail
- `GET /api/gpu`              → tegrastats snapshot (gpu%, freq, temps, RAM)
- `GET /api/stats`            → CPU/mem/swap/load avg/temps/uptime/hostname via psutil
- `GET /api/procs`            → per-service cgroup memory (`memory.current`, `memory.swap.current`, anon/file)
- `GET /api/persona`          → user_name, assistant_name
- `POST /api/persona`         → set user_name (persists to persona.toml)
- `GET /api/conversations`    → recent sessions + summaries
- `GET /api/hardware/inputs` / `POST /api/hardware/led` → hardware tab

**Consumes** (read-only):
- WS 1: `Potentiometer`, `DigitalSwitch`, `StatusLEDs` (hardware tab)
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

**Coordination with the main app**: the two units run side by side. The
page shows a banner either way (running: live panels are fed by the
radio, test cards will get device-busy; stopped: test cards own the
hardware, live panels are stale). Debug TTS calls go through the CPU
subprocess worker so they never touch the radio's GPU sidecar.

**Journal access**: the unit adds `systemd-journal` to
`SupplementaryGroups`; without it `journalctl -u radio-oracle` returns
"No journal files were opened" and the RADIO-ORACLE log tab is empty.

## Standalone exercise

```bash
# Stop the main app first if it's running, then:
python -m oracle.diag --host 0.0.0.0 --port 8000

# Or via systemd (Conflicts= stops the main app):
sudo systemctl start radio-oracle-diag

# From any LAN device:
curl http://<jetson>:8000/api/health | jq
# Or just open http://<jetson>:8000 in a browser
```

## TODO

- [x] Custom favicon (2026-09-28)
- [ ] mDNS / Bonjour so the page is discoverable as `oracle.local:8000`
- [ ] Per-collection HNSW memory footprint chart
- [ ] Tegrastats: per-rail GPU power (currently only GPU%, freq, temps)
- [ ] Auth / token on `POST /api/ask*` (LAN-trust assumption today)
- [ ] Persist last-N pipeline traces for postmortem
