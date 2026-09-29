"""FastAPI diagnostic server: mic, speaker, LLM, system stats, health, logs."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import socket
import subprocess
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
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
            retriever = _get_retriever()
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
            retriever = _get_retriever()
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


async def _check_llm() -> dict:
    from oracle.llm import check_ollama

    if settings.llm_backend == "llama-server":
        where = f"llama-server · {settings.llama_server_model}"
    else:
        where = f"ollama · {settings.ollama_model}"
    t0 = time.time()
    try:
        ok = await check_ollama()
        return {
            "ok": bool(ok),
            "detail": where if ok else f"{where} unreachable",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": repr(e), "latency_ms": None}


async def _check_tts() -> dict:
    """The voice the *radio* uses: GPU sidecar when configured, else the local model files."""
    if settings.tts_backend == "server":
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.tts_server_url}/health")
            ok = resp.status_code == 200
            return {
                "ok": ok,
                "detail": f"sidecar · {resp.text.strip()[:40]}"
                if ok
                else f"sidecar HTTP {resp.status_code}",
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except httpx.HTTPError as e:
            return {
                "ok": False,
                "detail": f"sidecar unreachable: {e.__class__.__name__}",
                "latency_ms": None,
            }
    model_ok = settings.tts_model_path.is_file()
    voices_ok = settings.tts_voices_path.is_file()
    return {
        "ok": model_ok and voices_ok,
        "detail": f"in-process kokoro · {settings.tts_voice}"
        if model_ok and voices_ok
        else "model files missing",
    }


_retriever = None


def _get_retriever():
    """One Retriever for the process: building it per request re-read every FAISS index."""
    global _retriever
    if _retriever is None:
        from oracle.rag.retriever import Retriever

        _retriever = Retriever()
    return _retriever


def _check_rag() -> dict:
    try:
        cols = _get_retriever().list_collections()
        kinds = settings.collection_backends
        backend = "faiss" if cols and all(kinds.get(c) == "faiss" for c in cols) else "mixed"
        return {
            "ok": bool(cols),
            "detail": f"{backend} · {len(cols)} collection(s): {', '.join(cols)}"
            if cols
            else "no collections",
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
    llm_task = asyncio.create_task(_check_llm())
    tts_task = asyncio.create_task(_check_tts())
    loop = asyncio.get_running_loop()
    rag = await loop.run_in_executor(None, _check_rag)
    audio = await loop.run_in_executor(None, _check_audio)
    gpio = await loop.run_in_executor(None, _check_gpio)
    gpio["optional"] = True
    llm = await llm_task
    tts = await tts_task
    overall = all((llm["ok"], rag["ok"], tts["ok"], audio["ok"]))  # GPIO optional
    return {
        "ok": overall,
        "llm": llm,
        "rag": rag,
        "tts": tts,
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
        # Direct one-shot readers, NO poller: the app's factories
        # (make_action_button_switch / make_power_switch_switch) register a
        # SharedAdcPoller that auto-starts a background thread and polls the
        # chip for the life of the process. In the dashboard that thread
        # interleaved with the radio's own poller for hours — phantom button
        # presses cutting the radio off mid-sentence (2026-09-28).
        from config.settings import settings
        from oracle.hardware.switch_adc import DigitalSwitch, shared_adc

        self._button = DigitalSwitch(
            channel=settings.action_button_ads1115_channel, adc=shared_adc()
        )
        self._power = DigitalSwitch(channel=settings.power_switch_ads1115_channel, adc=shared_adc())

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


_proc_check: dict = {"ts": 0.0, "alive": False}


def _radio_process_running() -> bool:
    """Is the radio app itself running? Decided from the process table, not
    only the state file's pid: a direct ADC read from here while the app
    owns the ADS1115 interleaves on the chip's mux and produces phantom
    button presses / pot jumps in the radio (2026-09-28). Cached 2 s."""
    now = time.time()
    if now - _proc_check["ts"] < 2.0:
        return _proc_check["alive"]
    alive = False
    try:
        for proc in psutil.process_iter(["cmdline"]):
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "-m oracle" in cmd and "--mode hardware" in cmd:
                alive = True
                break
    except Exception:  # noqa: BLE001
        alive = True  # unknown → assume alive, hands off the chip
    _proc_check.update(ts=now, alive=alive)
    return alive


@app.get("/api/hardware/inputs")
def hw_inputs() -> dict:
    # While radio-oracle runs, it owns the ADS1115 — reading the chip from
    # a second process interleaves on its single mux register and corrupts
    # both readers (the dashboard pot jumped; the radio's button/switch
    # reads could glitch too). Use the app's published telemetry instead.
    snap = read_state()
    app_alive = _radio_process_running()
    if snap and snap.get("pid"):
        try:
            app_alive = app_alive or psutil.pid_exists(int(snap["pid"]))
        except (TypeError, ValueError):
            pass
    if app_alive and not snap:
        # Alive but no readable snapshot (e.g. just starting): never touch the chip.
        return {
            "available": True,
            "via_app": True,
            "pot": {"available": False, "detail": "waiting for app telemetry"},
            "switch": {"channel": "-", "on": None},
            "button": {"channel": "-", "pressed": False},
        }
    if snap and app_alive and snap.get("hw"):
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
        age = time.time() - snap.get("updated_at", 0)
        if age > 3:
            out["stale_s"] = round(age, 1)
        return out
    if snap and app_alive:
        # App alive but no telemetry yet — still must not touch the chip.
        return {
            "available": True,
            "via_app": True,
            "pot": {"available": False, "detail": "waiting for app telemetry"},
            "switch": {"channel": "-", "on": bool(snap.get("power_on"))},
            "button": {"channel": "-", "pressed": False},
        }

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
        "uptime_sec": int(time.time() - psutil.boot_time()),
        "hostname": socket.gethostname(),
    }


# ---------------------------------------------------------------------------
# /api/procs — memory of each service cgroup: the Jetson's binding constraint
# ---------------------------------------------------------------------------

_CGROUP_ROOT = Path("/sys/fs/cgroup/system.slice")
_MEMORY_UNITS = ("llama-server", "ollama", "radio-oracle", "radio-oracle-tts", "radio-oracle-diag")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_memory(unit: str) -> dict:
    """Resident and swapped bytes of one systemd service, from cgroup v2 files (no /proc walk)."""
    cg = _CGROUP_ROOT / f"{unit}.service"
    current = _read_int(cg / "memory.current")
    if current is None:
        return {"unit": unit, "present": False}
    swap = _read_int(cg / "memory.swap.current") or 0
    anon = file = 0
    try:
        for line in (cg / "memory.stat").read_text().splitlines():
            key, _, val = line.partition(" ")
            if key == "anon":
                anon = int(val)
            elif key == "file":
                file = int(val)
    except (OSError, ValueError):
        pass
    return {
        "unit": unit,
        "present": True,
        "resident_mb": round(current / 1024 / 1024, 1),
        "anon_mb": round(anon / 1024 / 1024, 1),
        "file_mb": round(file / 1024 / 1024, 1),
        "swap_mb": round(swap / 1024 / 1024, 1),
    }


@app.get("/api/procs")
def procs() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / 1024 / 1024, 0),
        "used_mb": round(vm.used / 1024 / 1024, 0),
        "available_mb": round(vm.available / 1024 / 1024, 0),
        "services": [_cgroup_memory(u) for u in _MEMORY_UNITS],
    }


# ---------------------------------------------------------------------------
# /  — single-page UI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# /  — single-page UI (oracle/diag/static/index.html) + its assets
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_CACHE = "public, max-age=86400"
_STATIC_TYPES = {
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".css": "text/css",
    ".js": "text/javascript",
}
_page_cache: tuple[float, str] | None = None


def _load_page() -> str:
    """The dashboard HTML, re-read only when the file changes (dev convenience)."""
    global _page_cache
    path = _STATIC_DIR / "index.html"
    mtime = path.stat().st_mtime
    if _page_cache is None or _page_cache[0] != mtime:
        _page_cache = (mtime, path.read_text(encoding="utf-8"))
    return _page_cache[1]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _load_page()


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(
        _STATIC_DIR / "favicon.ico",
        media_type="image/x-icon",
        headers={"Cache-Control": _STATIC_CACHE},
    )


@app.get("/static/{name}", include_in_schema=False)
def static_file(name: str) -> FileResponse:
    # Flat directory, explicit allow-list of types: no traversal, no surprises.
    path = _STATIC_DIR / name
    if Path(name).name != name or path.suffix not in _STATIC_TYPES or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        path,
        media_type=_STATIC_TYPES[path.suffix],
        headers={"Cache-Control": _STATIC_CACHE},
    )
