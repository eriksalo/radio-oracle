"""FastAPI diagnostic server: mic, speaker, LLM, system stats, health, logs."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import subprocess
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from config.settings import settings
from oracle.diag import tegrastats
from oracle.log import attach_ring_buffer, get_recent_logs
from oracle.state import read_state

# Lazy hardware singletons used only by the diag I/O card. Imported here so
# the ``HardwareInputs`` / LED state lives for the lifetime of the process.
_hw_inputs: _HardwareInputs | None = None
_hw_leds = None  # type: ignore[var-annotated]
_hw_pot = None  # type: ignore[var-annotated]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Don't call setup_logging() here — that adds a file sink at
    # data/oracle.log which can fail under systemd if the data dir
    # isn't writable, and would kill startup before uvicorn binds.
    # Just attach the ring buffer to whatever sinks loguru already has.
    try:
        attach_ring_buffer()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"diag: ring buffer attach failed: {e}")
    try:
        tegrastats.start()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"diag: tegrastats start failed: {e}")
    logger.info("diag server up")
    try:
        yield
    finally:
        await tegrastats.stop()
        await _tts_worker.aclose()


app = FastAPI(title="Radio Oracle Diagnostics", lifespan=_lifespan)

_THERMAL_ROOT = Path("/sys/devices/virtual/thermal")


class SpeakRequest(BaseModel):
    text: str
    radio_filter: bool = False


class AskRequest(BaseModel):
    text: str
    use_rag: bool = True


class RecordRequest(BaseModel):
    silence_duration: float | None = None


class PersonaUpdate(BaseModel):
    user_name: str


# ---------------------------------------------------------------------------
# /api/record — capture mic on Jetson, return WAV
# ---------------------------------------------------------------------------


@app.post("/api/record")
async def record(req: RecordRequest) -> Response:
    from oracle.audio import audio_to_wav_bytes, record_until_silence

    loop = asyncio.get_running_loop()
    audio = await loop.run_in_executor(
        None,
        lambda: record_until_silence(silence_duration=req.silence_duration),
    )
    wav = audio_to_wav_bytes(audio)
    duration = len(audio) / settings.audio_sample_rate
    logger.info(f"diag: recorded {duration:.2f}s, {len(wav)} bytes")
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"X-Duration-Sec": f"{duration:.2f}"},
    )


# ---------------------------------------------------------------------------
# /api/speak — synthesize via persistent Kokoro worker, play on Jetson speaker
# ---------------------------------------------------------------------------

# Serializes synthesis + playback. Kokoro's ONNX session lives in a long-lived
# worker subprocess so we pay the ~2-4 s model load only once, not per request.
# Playback is exclusive anyway (single speaker), so we use one shared lock for
# both the worker request/response framing and the audio output.
_speak_lock = asyncio.Lock()


class _PersistentTTSWorker:
    """Long-lived ``oracle.diag.tts_worker --persistent`` subprocess.

    Lazily started on first call and restarted on crash. Callers must hold
    ``_speak_lock`` (the protocol is not safe under concurrent requests).
    """

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None

    async def synth(self, text: str, radio_filter: bool) -> bytes:
        # One retry: if the worker died between calls, restart and try again.
        for attempt in (0, 1):
            try:
                await self._ensure_started()
                return await self._call(text, radio_filter)
            except (
                BrokenPipeError,
                ConnectionResetError,
                asyncio.IncompleteReadError,
                RuntimeError,
            ) as e:
                self._reset()
                if attempt == 1:
                    raise RuntimeError(f"tts worker failed: {e}") from e
                logger.warning(f"diag: tts worker unhealthy ({e}); restarting")
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (TimeoutError, ProcessLookupError):
            proc.kill()
            await proc.wait()

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "oracle.diag.tts_worker",
            "--persistent",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # stderr inherits from parent → goes to journalctl
        )
        ready = await proc.stdout.readline()
        if ready.strip() != b"READY":
            await proc.wait()
            raise RuntimeError(f"tts worker did not signal READY (got {ready!r})")
        self._proc = proc
        logger.info(f"diag: persistent tts worker started (pid={proc.pid})")

    async def _call(self, text: str, radio_filter: bool) -> bytes:
        assert (
            self._proc is not None
            and self._proc.stdin is not None
            and self._proc.stdout is not None
        )
        text_bytes = text.encode("utf-8")
        flag = "1" if radio_filter else "0"
        header = f"{flag} {len(text_bytes)}\n".encode("ascii")
        self._proc.stdin.write(header)
        self._proc.stdin.write(text_bytes)
        await self._proc.stdin.drain()

        resp_header = await self._proc.stdout.readline()
        if not resp_header:
            raise RuntimeError("tts worker exited unexpectedly")
        try:
            status, len_str = resp_header.decode("ascii").strip().split()
            length = int(len_str)
        except ValueError as e:
            raise RuntimeError(f"bad worker response header: {resp_header!r}") from e
        body = await self._proc.stdout.readexactly(length)
        if status == "OK":
            return body
        if status == "ERR":
            raise RuntimeError(body.decode("utf-8", errors="replace"))
        raise RuntimeError(f"unknown worker status: {status}")

    def _reset(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


_tts_worker = _PersistentTTSWorker()


async def _synth_via_worker(text: str, radio_filter: bool) -> bytes:
    """Synthesize WAV bytes using the persistent Kokoro worker."""
    return await _tts_worker.synth(text, radio_filter)


def _wav_duration_sec(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


@app.post("/api/speak")
async def speak(req: SpeakRequest) -> dict:
    from oracle.audio import play_wav_bytes

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    async with _speak_lock:
        wav = await _synth_via_worker(text, req.radio_filter)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, play_wav_bytes, wav)

    duration = _wav_duration_sec(wav)
    logger.info(f"diag: spoke {len(text)} chars, {duration:.2f}s audio")
    return {"ok": True, "duration_sec": duration, "chars": len(text)}


@app.get("/api/speak.wav")
async def speak_wav(text: str, radio_filter: bool = False) -> Response:
    """Return synthesized audio as WAV without playing on the Jetson."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty text")

    async with _speak_lock:
        wav = await _synth_via_worker(text, radio_filter)
    return Response(content=wav, media_type="audio/wav")


# ---------------------------------------------------------------------------
# /api/ask — LLM (+ optional RAG)
# ---------------------------------------------------------------------------


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    from oracle.llm import chat
    from oracle.persona import build_system_prompt

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    system_prompt = build_system_prompt()
    rag_context = ""
    rag_sources: list[str] = []
    if req.use_rag:
        try:
            from oracle.rag.retriever import Retriever

            retriever = Retriever()
            collections = retriever.list_collections()
            if collections:
                results = retriever.query(text)
                rag_context = retriever.format_context(results)
                rag_sources = sorted({r.get("source", "?") for r in results})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"diag: RAG unavailable: {e}")

    full_system = system_prompt
    if rag_context:
        full_system = f"{system_prompt}\n\n{rag_context}"

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": text},
    ]
    response = await chat(messages)
    return {
        "answer": response,
        "rag_used": bool(rag_context),
        "rag_sources": rag_sources,
    }


# ---------------------------------------------------------------------------
# /api/persona — get/set the name the assistant addresses the user by
# ---------------------------------------------------------------------------


@app.get("/api/persona")
def get_persona() -> dict:
    from oracle.persona import get_user_name, load_persona

    persona = load_persona()
    return {
        "user_name": get_user_name(persona),
        "assistant_name": persona["oracle"]["name"],
    }


@app.post("/api/persona")
def update_persona(req: PersonaUpdate) -> dict:
    from oracle.persona import set_user_name

    try:
        saved = set_user_name(req.user_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"user_name": saved}


# ---------------------------------------------------------------------------
# /api/ask/stream — same as /api/ask but streams tokens via Server-Sent Events
# ---------------------------------------------------------------------------


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    from oracle.llm import stream_chat
    from oracle.persona import build_system_prompt

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    system_prompt = build_system_prompt()
    rag_context = ""
    rag_sources: list[str] = []
    if req.use_rag:
        try:
            from oracle.rag.retriever import Retriever

            retriever = Retriever()
            collections = retriever.list_collections()
            if collections:
                results = retriever.query(text)
                rag_context = retriever.format_context(results)
                rag_sources = sorted({r.get("source", "?") for r in results})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"diag: RAG unavailable for stream: {e}")

    full_system = system_prompt + (f"\n\n{rag_context}" if rag_context else "")
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": text},
    ]

    async def gen():
        # Emit a meta event up-front so the UI can label sources before tokens arrive
        meta = {"type": "meta", "rag_used": bool(rag_context), "rag_sources": rag_sources}
        yield f"data: {json.dumps(meta)}\n\n"
        try:
            async for token in stream_chat(messages):
                yield f"data: {json.dumps({'type': 'token', 'value': token})}\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as e:  # noqa: BLE001
            logger.warning(f"diag: ask stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# /api/health — subsystem reachability checks
# ---------------------------------------------------------------------------


async def _check_ollama() -> dict:
    from oracle.llm import check_ollama

    t0 = time.time()
    try:
        ok = await check_ollama()
        return {
            "ok": bool(ok),
            "detail": settings.ollama_model if ok else "unreachable",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": repr(e), "latency_ms": None}


def _check_chroma() -> dict:
    try:
        from oracle.rag.retriever import Retriever

        r = Retriever()
        cols = r.list_collections()
        return {
            "ok": True,
            "detail": f"{len(cols)} collection(s)",
            "collections": cols,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": repr(e), "collections": []}


def _check_audio() -> dict:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        names = [d["name"] for d in devices]
        in_dev = settings.audio_input_device
        out_dev = settings.audio_output_device
        in_ok = any(in_dev in n for n in names)
        out_ok = any(out_dev in n for n in names)
        ok = in_ok and out_ok
        missing = []
        if not in_ok:
            missing.append(f"input '{in_dev}'")
        if not out_ok:
            missing.append(f"output '{out_dev}'")
        detail = "input + output present" if ok else f"missing: {', '.join(missing)}"
        return {
            "ok": ok,
            "detail": detail,
            "input_device": in_dev,
            "output_device": out_dev,
            "devices": names,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": repr(e), "devices": []}


def _check_gpio() -> dict:
    try:
        import Jetson.GPIO as GPIO  # noqa: F401  # type: ignore[import-not-found]

        return {"ok": True, "detail": "Jetson.GPIO importable"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"unavailable: {e!r}"}


@app.get("/api/health")
async def health() -> dict:
    """Snapshot of subsystem reachability — green/red lights for the UI."""
    ollama_task = asyncio.create_task(_check_ollama())
    loop = asyncio.get_running_loop()
    chroma = await loop.run_in_executor(None, _check_chroma)
    audio = await loop.run_in_executor(None, _check_audio)
    gpio = await loop.run_in_executor(None, _check_gpio)
    ollama = await ollama_task
    overall = all((ollama["ok"], chroma["ok"], audio["ok"]))  # GPIO optional
    return {
        "ok": overall,
        "ollama": ollama,
        "chroma": chroma,
        "audio": audio,
        "gpio": gpio,
    }


# ---------------------------------------------------------------------------
# /api/hardware — live raw GPIO inputs + pot reading + direct LED control
# ---------------------------------------------------------------------------


class _HardwareInputs:
    """Read the action button and power switch via the ADS1115.

    The switches are wired to ADC inputs (10 kΩ pull-up to 3V3, switch shorts
    to GND) rather than GPIO because the Tegra234 GPIO INPUT register exhibits
    a loopback bug on JP 6.2.x for these pads. The ADC reading is thresholded
    to a boolean.
    """

    def __init__(self) -> None:
        from oracle.hardware.switch_adc import (
            make_action_button_switch,
            make_power_switch_switch,
        )

        self._button = make_action_button_switch()
        self._power = make_power_switch_switch()

    def read(self) -> dict:
        if not self._button.available:
            return {
                "available": False,
                "detail": self._button.error or "ADS1115 unavailable",
                "button": None,
                "switch": None,
            }
        btn = self._button.read()
        sw = self._power.read()
        out: dict = {"available": True}
        if btn is None:
            out["button"] = None
        else:
            out["button"] = {
                "channel": btn.channel,
                "voltage": btn.voltage,
                "level": "LOW" if btn.closed else "HIGH",
                "pressed": btn.closed,
            }
        if sw is None:
            out["switch"] = None
        else:
            out["switch"] = {
                "channel": sw.channel,
                "voltage": sw.voltage,
                "level": "LOW" if sw.closed else "HIGH",
                "on": sw.closed,
            }
        return out


def _get_inputs() -> _HardwareInputs:
    global _hw_inputs
    if _hw_inputs is None:
        _hw_inputs = _HardwareInputs()
    return _hw_inputs


def _get_pot():
    global _hw_pot
    if _hw_pot is None:
        from oracle.hardware.pot import Potentiometer

        _hw_pot = Potentiometer()
    return _hw_pot


def _get_leds():
    global _hw_leds
    if _hw_leds is None:
        from oracle.hardware.leds import StatusLEDs

        _hw_leds = StatusLEDs()
    return _hw_leds


class LEDRequest(BaseModel):
    r: bool = False
    g: bool = False
    b: bool = False


@app.get("/api/hardware/inputs")
def hw_inputs() -> dict:
    # While radio-oracle runs, it owns the ADS1115 — reading the chip from
    # a second process interleaves on its single mux register and corrupts
    # both readers (the dashboard pot jumped; the radio's button/switch
    # reads could glitch too). Use the app's published telemetry instead.
    snap = read_state()
    if snap and time.time() - snap.get("updated_at", 0) < 3 and snap.get("hw"):
        hw = snap["hw"]
        out: dict = {"available": True, "via_app": True}
        pot = hw.get("pot")
        if pot:
            out["pot"] = {"available": True, **pot}
        else:
            out["pot"] = {"available": False, "detail": "no pot telemetry"}
        out["switch"] = {"channel": "-", "on": bool(hw.get("power_on"))}
        lb = snap.get("last_button") or {}
        out["button"] = {"channel": "-", "pressed": False, "last": lb.get("kind")}
        return out

    inputs = _get_inputs().read()
    pot = _get_pot()
    if not pot.available:
        inputs["pot"] = {"available": False, "detail": pot.error or "unavailable"}
    else:
        reading = pot.read()
        if reading is None:
            inputs["pot"] = {"available": False, "detail": pot.error or "read failed"}
        else:
            inputs["pot"] = {
                "available": True,
                "raw": reading.raw,
                "voltage": reading.voltage,
                "pct": reading.pct,
            }
    return inputs


@app.post("/api/hardware/led")
def hw_led(req: LEDRequest) -> dict:
    leds = _get_leds()
    color = leds.set_rgb(req.r, req.g, req.b)
    logger.info(f"diag: LED set R={color.r} G={color.g} B={color.b}")
    return {"ok": True, "r": color.r, "g": color.g, "b": color.b}


# ---------------------------------------------------------------------------
# /api/state — read shared state file written by the running radio-oracle
# ---------------------------------------------------------------------------


@app.get("/api/state")
def app_state() -> dict:
    snap = read_state()
    if snap is None:
        return {"ok": False, "running": False, "detail": "no state file"}
    pid = snap.get("pid")
    running = False
    if pid:
        try:
            running = psutil.pid_exists(int(pid))
        except (TypeError, ValueError):
            running = False
    return {"ok": True, "running": running, **snap}


# ---------------------------------------------------------------------------
# /api/activity — live event feed from the running app (heard / decided /
# spoke / answered / playing / reading / phase)
# ---------------------------------------------------------------------------


@app.get("/api/activity")
def activity(after: int = 0, limit: int = 100) -> dict:
    from oracle.activity import read_events

    events = read_events(after=after, limit=min(limit, 300))
    return {"events": events, "last_id": events[-1]["id"] if events else after}


# ---------------------------------------------------------------------------
# /api/conversations — recent sessions with summaries (ported from the
# retired oracle/web app)
# ---------------------------------------------------------------------------


@app.get("/api/conversations")
def recent_conversations() -> dict:
    try:
        from oracle.memory.store import ConversationStore

        store = ConversationStore()
        sessions = store.get_recent_sessions(limit=10)
        result = []
        for s in sessions:
            msgs = store.get_messages(s["session_id"], limit=4)
            result.append(
                {
                    "session_id": s["session_id"][:8],
                    "started": s["started_at"],
                    "summary": s.get("summary", ""),
                    "message_count": store.count_messages(s["session_id"]),
                    "preview": msgs[:2] if msgs else [],
                }
            )
        store.close()
        return {"sessions": result}
    except Exception as e:  # noqa: BLE001
        return {"sessions": [], "error": str(e)}


# ---------------------------------------------------------------------------
# /api/logs — tail of in-process loguru ring buffer
# ---------------------------------------------------------------------------


@app.get("/api/logs")
def logs(tail: int = 200, level: str | None = None) -> dict:
    return {"entries": get_recent_logs(tail=tail, level=level)}


# ---------------------------------------------------------------------------
# /api/journal — tail of systemd journal for a sibling unit
# ---------------------------------------------------------------------------

_ALLOWED_UNITS = {"radio-oracle", "radio-oracle-diag"}


@app.get("/api/journal")
def journal(unit: str = "radio-oracle", tail: int = 200) -> dict:
    if unit not in _ALLOWED_UNITS:
        raise HTTPException(status_code=400, detail=f"unit must be one of {sorted(_ALLOWED_UNITS)}")
    if shutil.which("journalctl") is None:
        return {"available": False, "entries": [], "detail": "journalctl not on PATH"}
    try:
        out = subprocess.check_output(
            [
                "journalctl",
                "-u",
                unit,
                "-n",
                str(int(tail)),
                "--no-pager",
                "--output=short-iso",
            ],
            stderr=subprocess.STDOUT,
            timeout=5,
            text=True,
        )
        lines = out.splitlines()
        return {"available": True, "unit": unit, "entries": lines}
    except subprocess.CalledProcessError as e:
        return {"available": True, "unit": unit, "entries": [], "detail": e.output.strip()[-400:]}
    except subprocess.TimeoutExpired:
        return {"available": True, "unit": unit, "entries": [], "detail": "journalctl timed out"}


# ---------------------------------------------------------------------------
# /api/gpu — Jetson GPU + RAM stats from background tegrastats poller
# ---------------------------------------------------------------------------


@app.get("/api/gpu")
def gpu() -> dict:
    return tegrastats.snapshot()


# ---------------------------------------------------------------------------
# /api/stats — CPU, memory, swap, load avg, temperatures
# ---------------------------------------------------------------------------


def _read_temps() -> dict[str, float]:
    out: dict[str, float] = {}
    if not _THERMAL_ROOT.exists():
        return out
    for zone in sorted(_THERMAL_ROOT.glob("thermal_zone*")):
        try:
            zone_type = (zone / "type").read_text().strip()
            raw = (zone / "temp").read_text().strip()
            if not raw:
                continue
            out[zone_type] = round(int(raw) / 1000.0, 1)
        except Exception:  # noqa: BLE001 - some Jetson zones return None / EINVAL
            continue
    return out


@app.get("/api/stats")
def stats() -> dict:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    overall = sum(per_cpu) / len(per_cpu) if per_cpu else 0.0
    try:
        load1, load5, load15 = psutil.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = load15 = 0.0
    return {
        "cpu": {
            "overall_pct": round(overall, 1),
            "per_cpu_pct": [round(c, 1) for c in per_cpu],
            "count": psutil.cpu_count(logical=True),
            "load_avg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        },
        "memory": {
            "total_mb": round(vm.total / 1024 / 1024, 0),
            "used_mb": round(vm.used / 1024 / 1024, 0),
            "available_mb": round(vm.available / 1024 / 1024, 0),
            "pct": vm.percent,
        },
        "swap": {
            "total_mb": round(sm.total / 1024 / 1024, 0),
            "used_mb": round(sm.used / 1024 / 1024, 0),
            "pct": sm.percent,
        },
        "temps_c": _read_temps(),
        "uptime_sec": int(psutil.boot_time()),
    }


# ---------------------------------------------------------------------------
# /  — single-page UI
# ---------------------------------------------------------------------------

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VAULT-TEC PIP-BOY 3000 :: ORACLE DIAGNOSTICS</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --grn: #3dff7a;        /* primary phosphor */
      --grn-dim: #1f8a3f;    /* dim phosphor */
      --grn-deep: #0a2010;   /* darker fill */
      --grn-glow: rgba(61, 255, 122, 0.55);
      --bg: #02100a;
      --bg-2: #051b0f;
    }
    * { box-sizing: border-box; }
    html, body { background: var(--bg); }
    body {
      font-family: 'VT323', 'Share Tech Mono', 'Courier New', monospace;
      color: var(--grn);
      margin: 0;
      padding: 18px;
      font-size: 18px;
      line-height: 1.35;
      text-shadow: 0 0 4px var(--grn-glow);
      min-height: 100vh;
      background:
        radial-gradient(ellipse at center, rgba(61,255,122,0.06) 0%, transparent 70%),
        var(--bg);
      position: relative;
      overflow-x: hidden;
    }
    /* CRT scan lines */
    body::before {
      content: '';
      position: fixed; inset: 0;
      background: repeating-linear-gradient(
        to bottom,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,0.18) 3px,
        rgba(0,0,0,0) 4px
      );
      pointer-events: none;
      z-index: 100;
      mix-blend-mode: multiply;
    }
    /* CRT vignette + flicker */
    body::after {
      content: '';
      position: fixed; inset: 0;
      box-shadow: inset 0 0 180px rgba(0,0,0,0.85);
      pointer-events: none;
      z-index: 99;
      animation: flicker 6s infinite;
    }
    @keyframes flicker {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.97; }
      52% { opacity: 0.9; }
      53% { opacity: 1; }
    }

    .header {
      display: flex;
      align-items: center;
      gap: 16px;
      border: 2px solid var(--grn);
      padding: 10px 14px;
      box-shadow: 0 0 12px var(--grn-glow), inset 0 0 12px rgba(61,255,122,0.08);
      margin-bottom: 14px;
    }
    .header svg { flex: 0 0 auto; filter: drop-shadow(0 0 5px var(--grn-glow)); }
    .header .title { flex: 1; }
    .header h1 {
      margin: 0; font-size: 26px; letter-spacing: 0.08em;
    }
    .header .sub {
      font-size: 15px; color: var(--grn-dim); letter-spacing: 0.1em;
    }
    .header .clock {
      text-align: right; font-size: 14px; color: var(--grn-dim);
      font-variant-numeric: tabular-nums;
    }

    .banner {
      border: 1px dashed var(--grn-dim);
      padding: 6px 10px;
      margin-bottom: 14px;
      font-size: 15px;
      color: var(--grn-dim);
    }
    .banner::before { content: '! '; color: var(--grn); }

    .addressbar {
      display: flex; align-items: center; gap: 10px;
      padding: 6px 10px; margin-bottom: 14px;
      border: 1px solid var(--grn-dim);
      background: rgba(0,0,0,0.25);
    }
    .addressbar input { flex: 0 1 220px; padding: 4px 8px; font-size: 16px; }
    .addressbar label { margin: 0; }

    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
    .card {
      background: linear-gradient(180deg, var(--bg-2), var(--bg));
      border: 2px solid var(--grn);
      padding: 12px 14px 14px;
      box-shadow: 0 0 8px rgba(61,255,122,0.18), inset 0 0 8px rgba(61,255,122,0.04);
      position: relative;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0.1em;
      border-bottom: 1px solid var(--grn-dim);
      padding-bottom: 4px;
    }
    h2::before { content: '> '; color: var(--grn); }

    button {
      font-family: inherit;
      font-size: 17px;
      background: transparent;
      color: var(--grn);
      border: 2px solid var(--grn);
      padding: 4px 14px;
      cursor: pointer;
      letter-spacing: 0.08em;
      text-shadow: 0 0 4px var(--grn-glow);
      transition: background 0.1s, transform 0.05s;
    }
    button:hover:not(:disabled) {
      background: var(--grn);
      color: var(--bg);
      text-shadow: none;
      box-shadow: 0 0 12px var(--grn-glow);
    }
    button:active:not(:disabled) { transform: translateY(1px); }
    button:disabled { color: var(--grn-dim); border-color: var(--grn-dim); cursor: wait; text-shadow: none; }
    button.secondary { border-style: dashed; }

    textarea, input[type=text] {
      width: 100%;
      background: rgba(0,0,0,0.4);
      color: var(--grn);
      border: 1px solid var(--grn-dim);
      padding: 8px 10px;
      font: inherit;
      text-shadow: 0 0 3px var(--grn-glow);
      caret-color: var(--grn);
    }
    textarea { min-height: 64px; resize: vertical; }
    textarea:focus, input:focus { outline: 0; border-color: var(--grn); box-shadow: 0 0 8px var(--grn-glow); }
    textarea::placeholder, input::placeholder { color: var(--grn-dim); }

    label { font-size: 15px; color: var(--grn-dim); display: block; margin: 10px 0 4px; letter-spacing: 0.05em; }
    label.inline {
      display: inline-flex; align-items: center; gap: 6px;
      margin: 0; cursor: pointer;
    }
    /* terminal-style checkbox */
    input[type=checkbox] {
      appearance: none; -webkit-appearance: none;
      width: 18px; height: 18px;
      border: 1px solid var(--grn-dim);
      background: transparent;
      position: relative; cursor: pointer;
      vertical-align: middle;
    }
    input[type=checkbox]:checked { border-color: var(--grn); }
    input[type=checkbox]:checked::after {
      content: 'X';
      position: absolute; inset: 0;
      display: flex; align-items: center; justify-content: center;
      color: var(--grn); font-size: 16px; line-height: 1;
      text-shadow: 0 0 4px var(--grn-glow);
    }

    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
    .hint { color: var(--grn-dim); font-size: 14px; margin-top: 6px; }
    .answer {
      background: rgba(0,0,0,0.4);
      border: 1px solid var(--grn-dim);
      padding: 10px 12px;
      white-space: pre-wrap;
      max-height: 320px;
      overflow: auto;
      min-height: 50px;
      font-size: 17px;
    }
    .answer:empty::before { content: '[ awaiting query ]'; color: var(--grn-dim); }

    .stat-row {
      display: flex; justify-content: space-between;
      padding: 2px 0; font-size: 16px;
      font-variant-numeric: tabular-nums;
    }
    .stat-row span:first-child { color: var(--grn-dim); }
    .stat-row span:first-child::before { content: '· '; }

    .bar {
      background: rgba(0,0,0,0.5);
      border: 1px solid var(--grn-dim);
      height: 16px;
      overflow: hidden;
      margin: 3px 0;
      position: relative;
    }
    .bar > div {
      height: 100%;
      background:
        repeating-linear-gradient(
          90deg,
          var(--grn) 0px,
          var(--grn) 8px,
          rgba(0,0,0,0.3) 8px,
          rgba(0,0,0,0.3) 10px
        );
      box-shadow: 0 0 8px var(--grn-glow);
      transition: width 0.3s;
    }
    .meter-label {
      display: flex; justify-content: space-between;
      font-size: 14px; color: var(--grn-dim);
      margin-bottom: 2px; letter-spacing: 0.05em;
    }

    audio {
      width: 100%; margin-top: 8px;
      filter: hue-rotate(85deg) saturate(2) brightness(1.1) sepia(0.4);
    }

    .status { font-size: 14px; color: var(--grn-dim); }
    .status.ok { color: var(--grn); }
    .status.err { color: #ff5e5e; text-shadow: 0 0 4px rgba(255,80,80,0.6); }
    .status::before { content: '> '; }

    .cpu-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 4px; margin-top: 6px; }
    .cpu-cell {
      background: rgba(0,0,0,0.4);
      border: 1px solid var(--grn-dim);
      padding: 3px 2px;
      font-size: 13px;
      text-align: center;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0.04em;
    }
    .cpu-cell .v { font-size: 16px; color: var(--grn); }

    /* Health dots */
    .dot {
      display: inline-block;
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--grn-dim);
      box-shadow: 0 0 6px transparent;
      margin-right: 8px; vertical-align: middle;
    }
    .dot.ok  { background: var(--grn);   box-shadow: 0 0 8px var(--grn-glow); }
    .dot.err { background: #ff5e5e;      box-shadow: 0 0 8px rgba(255,80,80,0.6); }
    .dot.warn{ background: #f5c518;      box-shadow: 0 0 8px rgba(245,197,24,0.55); }
    .health-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 4px 0; border-bottom: 1px dashed rgba(31,138,63,0.25);
      font-size: 16px;
    }
    .health-row:last-child { border-bottom: 0; }
    .health-row .name { color: var(--grn-dim); letter-spacing: 0.06em; }
    .health-row .detail { color: var(--grn); font-size: 14px; max-width: 60%;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    /* Log tail */
    .log-tabs { display: flex; gap: 8px; margin-bottom: 6px; }
    .log-tabs button { padding: 2px 10px; font-size: 14px; }
    .log-tabs button.active {
      background: var(--grn); color: var(--bg);
      text-shadow: none; box-shadow: 0 0 12px var(--grn-glow);
    }
    .log {
      background: rgba(0,0,0,0.55);
      border: 1px solid var(--grn-dim);
      padding: 6px 8px;
      max-height: 280px;
      overflow: auto;
      font-size: 13px;
      line-height: 1.25;
      font-family: 'Share Tech Mono', 'Courier New', monospace;
      white-space: pre;
    }
    .log .lvl-WARNING { color: #f5c518; }
    .log .lvl-ERROR, .log .lvl-CRITICAL { color: #ff5e5e; }
    .log .lvl-DEBUG { color: var(--grn-dim); }

    .footer {
      margin-top: 18px;
      text-align: center;
      font-size: 14px;
      color: var(--grn-dim);
      letter-spacing: 0.15em;
    }
    .footer .blink { animation: blink 1.1s steps(2, jump-none) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
  </style>
</head>
<body>

  <div class="header">
    <!-- Original retro tin-robot mascot — own design, no franchise IP.
         Mid-century atomic-age styling, monochrome phosphor green w/ yellow accents. -->
    <svg width="84" height="108" viewBox="0 0 100 130" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="eye">
          <stop offset="0%" stop-color="#bfffd4"/>
          <stop offset="60%" stop-color="#3dff7a"/>
          <stop offset="100%" stop-color="#1f8a3f"/>
        </radialGradient>
      </defs>
      <g stroke="#3dff7a" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" fill="#0a1f12">
        <!-- antennae with bulb tips -->
        <line x1="34" y1="16" x2="28" y2="4"/>
        <circle cx="28" cy="4" r="3" fill="#f5c518" stroke-width="1.5"/>
        <line x1="66" y1="16" x2="72" y2="4"/>
        <circle cx="72" cy="4" r="3" fill="#f5c518" stroke-width="1.5"/>
        <!-- head -->
        <rect x="26" y="16" width="48" height="40" rx="8" stroke-width="2.5"/>
        <!-- forehead lamp -->
        <circle cx="50" cy="22" r="2" fill="#f5c518" stroke-width="1"/>
        <!-- eyes -->
        <circle cx="40" cy="34" r="6"/>
        <circle cx="40" cy="34" r="3.5" fill="url(#eye)" stroke="none"/>
        <circle cx="60" cy="34" r="6"/>
        <circle cx="60" cy="34" r="3.5" fill="url(#eye)" stroke="none"/>
        <!-- speaker grille mouth -->
        <rect x="36" y="46" width="28" height="7" rx="1" stroke-width="1.5"/>
        <line x1="42" y1="46" x2="42" y2="53" stroke-width="1"/>
        <line x1="48" y1="46" x2="48" y2="53" stroke-width="1"/>
        <line x1="54" y1="46" x2="54" y2="53" stroke-width="1"/>
        <line x1="60" y1="46" x2="60" y2="53" stroke-width="1"/>
        <!-- neck -->
        <rect x="42" y="56" width="16" height="6" stroke-width="2"/>
        <!-- torso -->
        <path d="M20 64 L 80 64 L 84 120 L 16 120 Z" stroke-width="2.5"/>
        <!-- chest gauge -->
        <circle cx="50" cy="88" r="12" stroke-width="2"/>
        <circle cx="50" cy="88" r="9" fill="none" stroke="#1f8a3f" stroke-width="1"/>
        <line x1="50" y1="78" x2="50" y2="80" stroke-width="1"/>
        <line x1="60" y1="88" x2="58" y2="88" stroke-width="1"/>
        <line x1="50" y1="98" x2="50" y2="96" stroke-width="1"/>
        <line x1="40" y1="88" x2="42" y2="88" stroke-width="1"/>
        <line x1="50" y1="88" x2="57" y2="82" stroke="#f5c518" stroke-width="1.8"/>
        <circle cx="50" cy="88" r="1.5" fill="#f5c518" stroke="none"/>
        <!-- rivets -->
        <circle cx="24" cy="68" r="1.2" fill="#3dff7a" stroke="none"/>
        <circle cx="76" cy="68" r="1.2" fill="#3dff7a" stroke="none"/>
        <circle cx="20" cy="116" r="1.2" fill="#3dff7a" stroke="none"/>
        <circle cx="80" cy="116" r="1.2" fill="#3dff7a" stroke="none"/>
        <!-- left arm at side (viewer's right) -->
        <circle cx="80" cy="70" r="4" stroke-width="2"/>
        <rect x="76" y="73" width="8" height="18" rx="2" stroke-width="2"/>
        <circle cx="80" cy="93" r="3" stroke-width="2"/>
        <rect x="76" y="94" width="8" height="20" rx="2" stroke-width="2"/>
        <!-- pincer at end -->
        <path d="M76 113 L 72 122 L 78 122 L 80 116 L 82 116 L 84 122 L 90 122 L 86 113 Z" stroke-width="2"/>
        <!-- right arm raised (viewer's left) — thumbs-up claw -->
        <circle cx="20" cy="70" r="4" stroke-width="2"/>
        <path d="M16 71 L 12 50 L 22 47 L 26 68 Z" stroke-width="2"/>
        <circle cx="17" cy="50" r="3" stroke-width="2"/>
        <rect x="13" y="26" width="8" height="22" rx="2" stroke-width="2"/>
        <!-- thumb sticking up -->
        <path d="M13 26 Q 9 14, 13 10 Q 18 10, 18 16 L 18 24 Z" stroke-width="2"/>
        <!-- folded fingers -->
        <path d="M21 26 Q 24 22, 22 18 L 19 22 Z" stroke-width="2"/>
      </g>
    </svg>

    <div class="title">
      <h1>VAULT-TEC PIP-BOY 3000</h1>
      <div class="sub">ORACLE DIAGNOSTIC SUITE :: VAULT 124 :: AUTH OK</div>
    </div>
    <div class="clock" id="clock">00:00:00</div>
  </div>

  <div class="banner" id="banner">
    Audio devices are exclusive — halt the radio-oracle service before running mic/speaker tests.
  </div>

  <div class="addressbar">
    <label class="inline" for="userName">ADDRESS ME AS</label>
    <input id="userName" type="text" maxlength="40" placeholder="Erik" />
    <button id="saveUserName" class="secondary">[ SAVE ]</button>
    <span class="status" id="userNameStatus"></span>
  </div>

  <div class="grid">

    <!-- Health -->
    <div class="card">
      <h2>[ 0 ] SYSTEM HEALTH</h2>
      <div id="healthList">
        <div class="health-row"><span class="name"><span class="dot"></span>OLLAMA</span><span class="detail">…</span></div>
        <div class="health-row"><span class="name"><span class="dot"></span>CHROMA</span><span class="detail">…</span></div>
        <div class="health-row"><span class="name"><span class="dot"></span>AUDIO</span><span class="detail">…</span></div>
        <div class="health-row"><span class="name"><span class="dot"></span>GPIO</span><span class="detail">…</span></div>
      </div>
      <div class="row"><button id="refreshHealthBtn" class="secondary">[ REFRESH ]</button>
        <span class="status" id="healthStatus"></span></div>
    </div>

    <!-- Live state -->
    <div class="card">
      <h2>[ 0 ] LIVE STATE</h2>
      <div id="liveState">
        <div class="stat-row"><span>service</span><span id="lsRunning">—</span></div>
        <div class="stat-row"><span>mode</span><span id="lsMode">—</span></div>
        <div class="stat-row"><span>power</span><span id="lsPower">—</span></div>
        <div class="stat-row"><span>last button</span><span id="lsButton">—</span></div>
        <div class="stat-row"><span>last transcription</span><span id="lsLastTx">—</span></div>
        <div class="stat-row"><span>updated</span><span id="lsUpdated">—</span></div>
      </div>
      <div class="hint">Populated when <code>radio-oracle.service</code> is running.
        Stop it to free the mic/speaker for the cards below.</div>
    </div>

    <!-- Live activity feed -->
    <div class="card" style="grid-column: 1 / -1">
      <h2>[ * ] LIVE ACTIVITY</h2>
      <div class="stat-row"><span>doing</span><span id="actPhase">—</span></div>
      <div class="stat-row"><span>now playing</span><span id="actMedia">—</span></div>
      <div id="actFeed" style="margin-top:8px;max-height:340px;overflow-y:auto;
           font-size:12px;line-height:1.55"></div>
      <div class="hint">Everything the oracle hears, decides, and says — live
        from <code>radio-oracle.service</code>.</div>
    </div>

    <!-- Hardware I/O -->
    <div class="card" style="grid-column: 1 / -1">
      <h2>[ ! ] HARDWARE I/O DIAGNOSTIC</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px">
        <div>
          <div class="meter-label"><span>MOMENTARY (AIN<span id="hwBtnPin">—</span>)</span><span id="hwBtnLevel">—</span></div>
          <div class="stat-row"><span>state</span><span id="hwBtnState">—</span></div>
          <div class="stat-row"><span>voltage</span><span id="hwBtnVolt">—</span></div>
        </div>
        <div>
          <div class="meter-label"><span>POWER SWITCH (AIN<span id="hwSwPin">—</span>)</span><span id="hwSwLevel">—</span></div>
          <div class="stat-row"><span>state</span><span id="hwSwState">—</span></div>
          <div class="stat-row"><span>voltage</span><span id="hwSwVolt">—</span></div>
        </div>
        <div>
          <div class="meter-label"><span>POTENTIOMETER</span><span id="hwPotPct">—</span></div>
          <div class="bar"><div id="hwPotBar" style="width:0%"></div></div>
          <div class="stat-row"><span>raw</span><span id="hwPotRaw">—</span></div>
          <div class="stat-row"><span>voltage</span><span id="hwPotV">—</span></div>
        </div>
        <div>
          <div class="meter-label"><span>RGB LED</span><span id="hwLedSwatch" style="display:inline-block;width:18px;height:14px;border:1px solid var(--grn-dim);background:#000;vertical-align:middle"></span></div>
          <div class="row" style="margin-top:4px">
            <label class="inline"><input type="checkbox" id="hwLedR" /> R</label>
            <label class="inline"><input type="checkbox" id="hwLedG" /> G</label>
            <label class="inline"><input type="checkbox" id="hwLedB" /> B</label>
          </div>
          <div class="row" style="margin-top:6px">
            <button id="hwLedApply">[ APPLY ]</button>
            <button id="hwLedOff" class="secondary">[ OFF ]</button>
          </div>
          <span class="status" id="hwLedStatus"></span>
        </div>
      </div>
      <div class="hint">Live raw GPIO + ADS1115 reads. Halt <code>radio-oracle.service</code> if its
        own LED writes are stomping yours.</div>
      <div class="status" id="hwInputsStatus"></div>
    </div>

    <!-- Mic -->
    <div class="card">
      <h2>[ 1 ] AUDIO INPUT TEST</h2>
      <div class="row">
        <button id="recBtn">[ RECORD ]</button>
        <span class="status" id="recStatus">Captures until silence detected (VAD).</span>
      </div>
      <audio id="recPlayer" controls></audio>
    </div>

    <!-- TTS -->
    <div class="card">
      <h2>[ 2 ] VOX SYNTHESIS</h2>
      <textarea id="ttsText" placeholder="Type transmission for the Oracle to broadcast...">Testing one two three. The Oracle is operational.</textarea>
      <div class="row">
        <button id="speakBtn">[ TRANSMIT ]</button>
        <button id="speakBrowserBtn" class="secondary">[ PREVIEW ]</button>
        <label class="inline"><input type="checkbox" id="filterChk" /> RADIO FILTER</label>
        <span class="status" id="ttsStatus"></span>
      </div>
      <audio id="ttsPlayer" controls></audio>
    </div>

    <!-- Ask -->
    <div class="card" style="grid-column: 1 / -1">
      <h2>[ 3 ] ORACLE QUERY (LLM + RAG)</h2>
      <textarea id="askText" placeholder="State your inquiry, citizen..."></textarea>
      <div class="row">
        <button id="askBtn">[ INTERROGATE ]</button>
        <label class="inline"><input type="checkbox" id="ragChk" checked /> ARCHIVES (RAG)</label>
        <label class="inline"><input type="checkbox" id="speakAnswerChk" /> VOICE RESPONSE</label>
        <span class="status" id="askStatus"></span>
      </div>
      <label>RESPONSE</label>
      <div class="answer" id="answer"></div>
      <div class="hint" id="ragSources"></div>
    </div>

    <!-- Stats -->
    <div class="card" style="grid-column: 1 / -1">
      <h2>[ 4 ] VITALS</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px">
        <div>
          <div class="meter-label"><span>CPU LOAD</span><span id="cpuPct">—</span></div>
          <div class="bar"><div id="cpuBar" style="width:0%"></div></div>
          <div class="cpu-grid" id="cpuGrid"></div>
          <div class="stat-row" style="margin-top:6px"><span>load avg</span><span id="loadAvg">—</span></div>
        </div>
        <div>
          <div class="meter-label"><span>MEMORY</span><span id="memPct">—</span></div>
          <div class="bar"><div id="memBar" style="width:0%"></div></div>
          <div class="stat-row"><span>used</span><span id="memUsed">—</span></div>
          <div class="stat-row"><span>available</span><span id="memAvail">—</span></div>
          <div class="meter-label" style="margin-top:8px"><span>SWAP</span><span id="swapPct">—</span></div>
          <div class="bar"><div id="swapBar" style="width:0%"></div></div>
        </div>
        <div>
          <div class="meter-label"><span>CORE TEMP (°C)</span><span></span></div>
          <div id="temps"></div>
        </div>
        <div>
          <div class="meter-label"><span>GPU</span><span id="gpuPct">—</span></div>
          <div class="bar"><div id="gpuBar" style="width:0%"></div></div>
          <div class="stat-row"><span>freq</span><span id="gpuFreq">—</span></div>
          <div class="stat-row"><span>tegrastats</span><span id="gpuAvail">—</span></div>
        </div>
      </div>
    </div>

    <!-- Logs -->
    <div class="card" style="grid-column: 1 / -1">
      <h2>[ 5 ] TELEMETRY LOG</h2>
      <div class="log-tabs">
        <button id="tabDiag" class="active">[ DIAG ]</button>
        <button id="tabRadio" class="secondary">[ RADIO-ORACLE ]</button>
        <button id="tabRefresh" class="secondary" style="margin-left:auto">[ REFRESH ]</button>
        <label class="inline" style="margin-left:8px"><input type="checkbox" id="logAuto" checked /> AUTO</label>
      </div>
      <div class="log" id="logBox">[ awaiting telemetry ]</div>
    </div>

  </div>

  <div class="footer">
    PROPERTY OF VAULT-TEC INDUSTRIES &nbsp;·&nbsp; ROBCO INDUSTRIES UNIFIED OS<span class="blink">_</span>
  </div>

<script>
const $ = (id) => document.getElementById(id);

async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r;
}

function setStatus(el, msg, cls) {
  el.textContent = msg;
  el.className = 'status' + (cls ? ' ' + cls : '');
}

// --- mic ---
$('recBtn').onclick = async () => {
  const btn = $('recBtn'), st = $('recStatus');
  btn.disabled = true;
  setStatus(st, 'Listening on Jetson mic (speak now, then go quiet)…');
  try {
    const r = await postJSON('/api/record', {});
    const blob = await r.blob();
    const dur = r.headers.get('X-Duration-Sec') || '?';
    $('recPlayer').src = URL.createObjectURL(blob);
    setStatus(st, `Captured ${dur}s. Press play.`, 'ok');
  } catch (e) {
    setStatus(st, 'Error: ' + e.message, 'err');
  } finally { btn.disabled = false; }
};

// --- tts ---
$('speakBtn').onclick = async () => {
  const btn = $('speakBtn'), st = $('ttsStatus');
  const text = $('ttsText').value.trim();
  if (!text) return;
  btn.disabled = true; setStatus(st, 'Synthesizing & playing on Jetson…');
  try {
    const r = await postJSON('/api/speak', { text, radio_filter: $('filterChk').checked });
    const j = await r.json();
    setStatus(st, `Played ${j.duration_sec.toFixed(2)}s.`, 'ok');
  } catch (e) {
    setStatus(st, 'Error: ' + e.message, 'err');
  } finally { btn.disabled = false; }
};

$('speakBrowserBtn').onclick = async () => {
  const btn = $('speakBrowserBtn'), st = $('ttsStatus');
  const text = $('ttsText').value.trim();
  if (!text) return;
  btn.disabled = true; setStatus(st, 'Synthesizing for browser preview…');
  try {
    const url = `/api/speak.wav?text=${encodeURIComponent(text)}&radio_filter=${$('filterChk').checked}`;
    $('ttsPlayer').src = url;
    $('ttsPlayer').play().catch(()=>{});
    setStatus(st, 'Loaded into player.', 'ok');
  } catch (e) {
    setStatus(st, 'Error: ' + e.message, 'err');
  } finally { btn.disabled = false; }
};

// --- ask (streaming via SSE / fetch ReadableStream) ---
let _askAbort = null;
$('askBtn').onclick = async () => {
  const btn = $('askBtn'), st = $('askStatus');
  const text = $('askText').value.trim();
  if (!text) return;
  btn.disabled = true; setStatus(st, 'Streaming…');
  $('answer').textContent = ''; $('ragSources').textContent = '';

  // Use fetch + ReadableStream (POST is required; EventSource only does GET).
  const ctrl = new AbortController();
  _askAbort = ctrl;
  let answer = '';
  try {
    const resp = await fetch('/api/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, use_rag: $('ragChk').checked }),
      signal: ctrl.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(await resp.text() || resp.statusText);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      // SSE frames split by blank line.
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const dataLines = frame.split('\\n').filter(l => l.startsWith('data: ')).map(l => l.slice(6));
        if (!dataLines.length) continue;
        let evt;
        try { evt = JSON.parse(dataLines.join('\\n')); } catch { continue; }
        if (evt.type === 'token') {
          answer += evt.value;
          $('answer').textContent = answer;
        } else if (evt.type === 'meta') {
          if (evt.rag_used) {
            $('ragSources').textContent = 'RAG sources: ' + (evt.rag_sources.join(', ') || '(none)');
          } else if ($('ragChk').checked) {
            $('ragSources').textContent = 'RAG requested but no collections matched.';
          }
        } else if (evt.type === 'done') {
          setStatus(st, 'Done.', 'ok');
        } else if (evt.type === 'error') {
          setStatus(st, 'Error: ' + evt.message, 'err');
        }
      }
    }
    if ($('speakAnswerChk').checked && answer) {
      setStatus(st, 'Speaking…', 'ok');
      await postJSON('/api/speak', { text: answer, radio_filter: $('filterChk').checked });
      setStatus(st, 'Done.', 'ok');
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      setStatus(st, 'Cancelled.', 'ok');
    } else {
      setStatus(st, 'Error: ' + e.message, 'err');
    }
  } finally { btn.disabled = false; _askAbort = null; }
};

// --- stats poll ---
function fmtMB(mb) { return mb >= 1024 ? (mb/1024).toFixed(2) + ' GB' : mb + ' MB'; }
async function pollStats() {
  try {
    const r = await fetch('/api/stats'); const j = await r.json();
    $('cpuPct').textContent = j.cpu.overall_pct.toFixed(1) + '%';
    $('cpuBar').style.width = j.cpu.overall_pct + '%';
    $('loadAvg').textContent = j.cpu.load_avg.join(' / ');
    $('cpuGrid').innerHTML = j.cpu.per_cpu_pct.map((v,i) =>
      `<div class="cpu-cell">cpu${i}<div class="v">${v.toFixed(0)}%</div></div>`).join('');
    $('memPct').textContent = j.memory.pct + '%';
    $('memBar').style.width = j.memory.pct + '%';
    $('memUsed').textContent = fmtMB(j.memory.used_mb);
    $('memAvail').textContent = fmtMB(j.memory.available_mb);
    $('swapPct').textContent = j.swap.pct + '%';
    $('swapBar').style.width = j.swap.pct + '%';
    $('temps').innerHTML = Object.entries(j.temps_c).map(([k,v]) =>
      `<div class="stat-row"><span>${k}</span><span>${v.toFixed(1)}</span></div>`).join('');
  } catch (e) { /* ignore transient */ }
}
pollStats();
setInterval(pollStats, 2000);

// --- gpu poll (tegrastats) ---
async function pollGpu() {
  try {
    const r = await fetch('/api/gpu'); const j = await r.json();
    $('gpuAvail').textContent = j.available ? (j.stale_sec != null ? `${j.stale_sec.toFixed(1)}s ago` : 'live') : 'unavailable';
    if (j.gpu_pct != null) {
      $('gpuPct').textContent = j.gpu_pct + '%';
      $('gpuBar').style.width = j.gpu_pct + '%';
    } else {
      $('gpuPct').textContent = '—';
      $('gpuBar').style.width = '0%';
    }
    $('gpuFreq').textContent = j.gpu_freq_mhz != null ? j.gpu_freq_mhz + ' MHz' : '—';
  } catch (e) { /* ignore */ }
}
pollGpu();
setInterval(pollGpu, 2000);

// --- health poll ---
function setHealthRow(el, name, info) {
  const dotCls = info?.ok ? 'ok' : 'err';
  const detail = info?.detail || '';
  const lat = info?.latency_ms != null ? ` (${info.latency_ms} ms)` : '';
  el.innerHTML = `<span class="name"><span class="dot ${dotCls}"></span>${name}</span><span class="detail" title="${detail}">${detail}${lat}</span>`;
}
async function pollHealth() {
  const st = $('healthStatus');
  setStatus(st, 'checking…');
  try {
    const r = await fetch('/api/health'); const j = await r.json();
    const rows = $('healthList').children;
    setHealthRow(rows[0], 'OLLAMA', j.ollama);
    setHealthRow(rows[1], 'CHROMA', j.chroma);
    setHealthRow(rows[2], 'AUDIO',  j.audio);
    setHealthRow(rows[3], 'GPIO',   j.gpio);
    setStatus(st, j.ok ? 'all systems nominal' : 'subsystem failures', j.ok ? 'ok' : 'err');
  } catch (e) {
    setStatus(st, 'error: ' + e.message, 'err');
  }
}
$('refreshHealthBtn').onclick = pollHealth;
pollHealth();
setInterval(pollHealth, 15000);

// --- live state poll ---
function fmtAgo(ts) {
  if (!ts) return '—';
  const d = Math.max(0, (Date.now() / 1000) - ts);
  if (d < 60) return d.toFixed(0) + 's ago';
  if (d < 3600) return (d / 60).toFixed(0) + 'm ago';
  return (d / 3600).toFixed(1) + 'h ago';
}
async function pollState() {
  try {
    const r = await fetch('/api/state'); const j = await r.json();
    if (!j.ok || !j.running) {
      $('lsRunning').textContent = j.running === false ? 'stale state file' : 'not running';
      $('lsMode').textContent = j.mode || '—';
      $('lsPower').textContent = j.power_on != null ? (j.power_on ? 'on' : 'off') : '—';
      $('lsButton').textContent = '—';
      $('lsLastTx').textContent = '—';
      $('lsUpdated').textContent = j.updated_at ? fmtAgo(j.updated_at) : '—';
      return;
    }
    $('lsRunning').textContent = `pid ${j.pid} ✓`;
    $('lsMode').textContent = (j.mode || '—').toUpperCase();
    $('lsPower').textContent = j.power_on ? 'ON' : 'OFF';
    $('lsButton').textContent = j.last_button
      ? `${j.last_button.kind} (${j.last_button.duration.toFixed(2)}s, ${fmtAgo(j.last_button.ts)})`
      : '—';
    $('lsLastTx').textContent = j.last_transcription
      ? `"${j.last_transcription.text}" — ${fmtAgo(j.last_transcription.ts)}`
      : '—';
    $('lsUpdated').textContent = fmtAgo(j.updated_at);
  } catch (e) { /* ignore */ }
}
pollState();
setInterval(pollState, 1500);

// ---- live activity feed ----
let actAfter = 0;
const ACT_LABELS = {
  phase: 'PHASE', wake: 'WAKE', heard: 'HEARD', decided: 'DECIDED',
  spoke: 'SPOKE', asked: 'ASKED', answered: 'ANSWERED',
  consulted: 'ARCHIVES', playing: 'PLAYING', reading: 'READING', error: 'ERROR',
};
const PHASE_TEXT = {
  radio: 'playing music', librarian: 'listening', thinking: 'thinking',
  speaking: 'speaking', reader: 'reading a book', off: 'standby',
};
function actLine(ev) {
  const t = new Date(ev.ts * 1000).toTimeString().slice(0, 8);
  let detail = '';
  if (ev.kind === 'phase') detail = PHASE_TEXT[ev.phase] || ev.phase;
  else if (ev.kind === 'heard' || ev.kind === 'spoke' || ev.kind === 'asked' || ev.kind === 'answered') detail = '\u201c' + (ev.text || '') + '\u201d';
  else if (ev.kind === 'decided') detail = ev.action + (ev.query ? ' \u2192 ' + ev.query : '');
  else if (ev.kind === 'consulted') detail = (ev.sources || []).join(' / ') + ' (' + ev.hits + ' hits)';
  else if (ev.kind === 'playing') detail = (ev.artist ? ev.artist + ' \u2014 ' : '') + ev.title;
  else if (ev.kind === 'reading') detail = (ev.book ? ev.book + ', ' : '') + 'chapter ' + ev.chapter + (ev.chapter_title ? ': ' + ev.chapter_title : '');
  else detail = JSON.stringify(ev);
  const label = ACT_LABELS[ev.kind] || ev.kind.toUpperCase();
  return '<div><span style="color:var(--grn-dim)">' + t + '</span> ' +
         '<span style="display:inline-block;min-width:76px;color:var(--amber, #ffb641)">' + label + '</span> ' +
         detail.replace(/</g, '&lt;') + '</div>';
}
async function pollActivity() {
  try {
    const r = await fetch('/api/activity?after=' + actAfter + '&limit=100');
    const j = await r.json();
    if (!j.events || !j.events.length) return;
    const feed = $('actFeed');
    const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
    for (const ev of j.events) {
      feed.insertAdjacentHTML('beforeend', actLine(ev));
      if (ev.kind === 'phase') $('actPhase').textContent = PHASE_TEXT[ev.phase] || ev.phase;
      if (ev.kind === 'playing') $('actMedia').textContent = (ev.artist ? ev.artist + ' \u2014 ' : '') + ev.title;
      if (ev.kind === 'reading' && ev.book) $('actMedia').textContent = 'reading: ' + ev.book + ' (ch ' + ev.chapter + ')';
    }
    while (feed.childElementCount > 300) feed.removeChild(feed.firstChild);
    if (atBottom) feed.scrollTop = feed.scrollHeight;
    actAfter = j.last_id;
  } catch (e) { /* server restarting */ }
}
pollActivity();
setInterval(pollActivity, 1500);

// --- hardware I/O poll + LED control ---
async function pollHwInputs() {
  const st = $('hwInputsStatus');
  try {
    const r = await fetch('/api/hardware/inputs'); const j = await r.json();
    const volt = v => (v == null ? '\u2014' : v.toFixed(3) + ' V');
    if (j.button) {
      $('hwBtnPin').textContent = j.button.channel;
      $('hwBtnLevel').textContent = j.button.level ?? '\u2014';
      $('hwBtnState').textContent = j.button.pressed ? 'PRESSED' : (j.button.last ? 'last: ' + j.button.last : 'released');
      $('hwBtnVolt').textContent = volt(j.button.voltage);
    }
    if (j.switch) {
      $('hwSwPin').textContent = j.switch.channel;
      $('hwSwLevel').textContent = j.switch.level ?? '\u2014';
      $('hwSwState').textContent = j.switch.on ? 'ON' : 'OFF';
      $('hwSwVolt').textContent = volt(j.switch.voltage);
    }
    if (j.pot && j.pot.available) {
      $('hwPotPct').textContent = j.pot.pct.toFixed(1) + '%';
      $('hwPotBar').style.width = j.pot.pct + '%';
      $('hwPotRaw').textContent = j.pot.raw;
      $('hwPotV').textContent = volt(j.pot.voltage);
    } else {
      $('hwPotPct').textContent = '—';
      $('hwPotBar').style.width = '0%';
      $('hwPotRaw').textContent = j.pot?.detail || 'unavailable';
      $('hwPotV').textContent = '—';
    }
    if (j.available === false) {
      setStatus(st, 'GPIO unavailable: ' + (j.detail || ''), 'err');
    } else if (j.via_app) {
      setStatus(st, 'live via radio-oracle telemetry (direct chip access paused)');
    } else {
      setStatus(st, '');
    }
  } catch (e) {
    setStatus(st, 'poll error: ' + e.message, 'err');
  }
}
function updateLedSwatch() {
  const r = $('hwLedR').checked, g = $('hwLedG').checked, b = $('hwLedB').checked;
  // Common-anode digital: each channel is on/off only (polarity handled server-side).
  const css = `rgb(${r?255:0}, ${g?255:0}, ${b?255:0})`;
  $('hwLedSwatch').style.background = css;
}
['hwLedR','hwLedG','hwLedB'].forEach(id => $(id).addEventListener('change', updateLedSwatch));
async function applyLed(r, g, b) {
  const st = $('hwLedStatus');
  setStatus(st, 'writing…');
  try {
    const resp = await postJSON('/api/hardware/led', { r, g, b });
    const j = await resp.json();
    setStatus(st, `R=${j.r?1:0} G=${j.g?1:0} B=${j.b?1:0}`, 'ok');
  } catch (e) {
    setStatus(st, 'error: ' + e.message, 'err');
  }
}
$('hwLedApply').onclick = () => applyLed($('hwLedR').checked, $('hwLedG').checked, $('hwLedB').checked);
$('hwLedOff').onclick = () => {
  $('hwLedR').checked = $('hwLedG').checked = $('hwLedB').checked = false;
  updateLedSwatch();
  applyLed(false, false, false);
};
updateLedSwatch();
pollHwInputs();
setInterval(pollHwInputs, 250);

// --- log tail ---
let _logTab = 'diag';
function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' })[c]); }
async function pollLogs() {
  const box = $('logBox');
  try {
    if (_logTab === 'diag') {
      const r = await fetch('/api/logs?tail=200'); const j = await r.json();
      box.innerHTML = (j.entries || []).map(e =>
        `<span class="lvl-${e.level}">[${e.ts}] ${e.level.padEnd(7)} ${e.name} :: ${escapeHtml(e.message)}</span>`
      ).join('\\n') || '[ empty ]';
    } else {
      const r = await fetch('/api/journal?unit=radio-oracle&tail=200'); const j = await r.json();
      if (!j.available) { box.textContent = j.detail || 'journalctl unavailable'; return; }
      box.innerHTML = (j.entries || []).map(escapeHtml).join('\\n') || '[ no entries — service inactive? ]';
    }
    box.scrollTop = box.scrollHeight;
  } catch (e) { box.textContent = 'log fetch error: ' + e.message; }
}
$('tabDiag').onclick = () => { _logTab = 'diag';
  $('tabDiag').classList.add('active'); $('tabRadio').classList.remove('active'); pollLogs(); };
$('tabRadio').onclick = () => { _logTab = 'radio';
  $('tabRadio').classList.add('active'); $('tabDiag').classList.remove('active'); pollLogs(); };
$('tabRefresh').onclick = pollLogs;
pollLogs();
setInterval(() => { if ($('logAuto').checked) pollLogs(); }, 3000);

// --- persona / user name ---
async function loadUserName() {
  try {
    const r = await fetch('/api/persona');
    const j = await r.json();
    $('userName').value = j.user_name || '';
  } catch (e) { /* ignore */ }
}
$('saveUserName').onclick = async () => {
  const btn = $('saveUserName'), st = $('userNameStatus');
  const name = $('userName').value.trim();
  if (!name) { setStatus(st, 'name required', 'err'); return; }
  btn.disabled = true; setStatus(st, 'saving…');
  try {
    const r = await postJSON('/api/persona', { user_name: name });
    const j = await r.json();
    setStatus(st, `saved: ${j.user_name}`, 'ok');
  } catch (e) {
    setStatus(st, 'error: ' + e.message, 'err');
  } finally { btn.disabled = false; }
};
$('userName').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('saveUserName').click();
});
loadUserName();

// --- clock ---
function tickClock() {
  const d = new Date();
  const z = (n) => String(n).padStart(2, '0');
  $('clock').textContent = `${z(d.getHours())}:${z(d.getMinutes())}:${z(d.getSeconds())}`;
}
tickClock();
setInterval(tickClock, 1000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE
