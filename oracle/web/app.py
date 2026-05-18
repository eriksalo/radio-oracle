"""Diagnostic web GUI — Pip-Boy terminal style."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from config.settings import settings

app = FastAPI(title="Oracle Radio Diagnostics", docs_url=None, redoc_url=None)

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_TEMPLATES_DIR / "index.html").read_text()


@app.get("/api/health")
async def health_checks():
    from oracle.health import run_health_checks

    results = await run_health_checks()
    return results


@app.get("/api/system")
async def system_info():
    info: dict = {}

    # CPU
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            info["load_avg"] = {"1m": parts[0], "5m": parts[1], "15m": parts[2]}
    except OSError:
        info["load_avg"] = None

    # Memory
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0])
            total_mb = meminfo["MemTotal"] / 1024
            avail_mb = meminfo["MemAvailable"] / 1024
            info["memory"] = {
                "total_mb": round(total_mb),
                "available_mb": round(avail_mb),
                "used_mb": round(total_mb - avail_mb),
                "percent": round((1 - avail_mb / total_mb) * 100, 1),
            }
    except OSError:
        info["memory"] = None

    # Swap
    try:
        with open("/proc/swaps") as f:
            lines = f.readlines()[1:]  # skip header
            total_kb = sum(int(l.split()[2]) for l in lines if l.strip())
            used_kb = sum(int(l.split()[3]) for l in lines if l.strip())
            info["swap"] = {
                "total_mb": round(total_kb / 1024),
                "used_mb": round(used_kb / 1024),
            }
    except OSError:
        info["swap"] = None

    # Disk
    usage = shutil.disk_usage("/")
    info["disk"] = {
        "total_gb": round(usage.total / (1024**3), 1),
        "used_gb": round(usage.used / (1024**3), 1),
        "free_gb": round(usage.free / (1024**3), 1),
        "percent": round(usage.used / usage.total * 100, 1),
    }

    # Uptime
    try:
        with open("/proc/uptime") as f:
            uptime_secs = float(f.read().split()[0])
            hours = int(uptime_secs // 3600)
            minutes = int((uptime_secs % 3600) // 60)
            info["uptime"] = f"{hours}h {minutes}m"
    except OSError:
        info["uptime"] = None

    # CPU temperature (Jetson)
    try:
        temp_paths = list(Path("/sys/devices/virtual/thermal/thermal_zone0").glob("temp"))
        if temp_paths:
            temp_mc = int(temp_paths[0].read_text().strip())
            info["cpu_temp_c"] = round(temp_mc / 1000, 1)
        else:
            info["cpu_temp_c"] = None
    except (OSError, ValueError):
        info["cpu_temp_c"] = None

    # GPU info (Jetson unified memory)
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100", "--stop", "1"],
            capture_output=True, text=True, timeout=3,
        )
        info["tegrastats"] = result.stdout.strip() if result.stdout else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["tegrastats"] = None

    # Hostname
    try:
        info["hostname"] = Path("/etc/hostname").read_text().strip()
    except OSError:
        info["hostname"] = "unknown"

    info["timestamp"] = datetime.now(UTC).isoformat()
    return info


@app.get("/api/ollama")
async def ollama_status():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / (1024**3), 2),
                    "modified": m.get("modified_at", ""),
                })

            # Check if model is loaded (running)
            ps_resp = await client.get(f"{settings.ollama_host}/api/ps")
            running = []
            if ps_resp.status_code == 200:
                ps_data = ps_resp.json()
                running = [m["name"] for m in ps_data.get("models", [])]

            return {
                "status": "online",
                "host": settings.ollama_host,
                "models": models,
                "running": running,
                "configured_model": settings.ollama_model,
            }
    except Exception as e:
        return {"status": "offline", "error": str(e)}


@app.get("/api/chroma")
async def chroma_status():
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        collections = []
        for coll in client.list_collections():
            name = coll if isinstance(coll, str) else coll.name
            try:
                c = client.get_collection(name)
                count = c.count()
                collections.append({"name": name, "chunks": count, "status": "ok"})
            except Exception as e:
                collections.append({"name": name, "chunks": 0, "status": f"error: {e}"})

        total = sum(c["chunks"] for c in collections)
        return {
            "path": str(settings.chroma_path),
            "collections": collections,
            "total_chunks": total,
        }
    except Exception as e:
        return {"path": str(settings.chroma_path), "error": str(e), "collections": []}


@app.get("/api/audio")
async def audio_devices():
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 or d["max_output_channels"] > 0:
                result.append({
                    "index": i,
                    "name": d["name"],
                    "inputs": d["max_input_channels"],
                    "outputs": d["max_output_channels"],
                    "sample_rate": d["default_samplerate"],
                })
        return {"devices": result}
    except Exception as e:
        return {"devices": [], "error": str(e)}


@app.get("/api/service")
async def service_status():
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "radio-oracle"],
            capture_output=True, text=True, timeout=5,
        )
        active = result.stdout.strip()

        result2 = subprocess.run(
            ["systemctl", "show", "radio-oracle",
             "--property=ActiveEnterTimestamp,MainPID,MemoryCurrent"],
            capture_output=True, text=True, timeout=5,
        )
        props = {}
        for line in result2.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                props[k] = v

        return {
            "status": active,
            "pid": props.get("MainPID", ""),
            "memory_bytes": props.get("MemoryCurrent", ""),
            "started": props.get("ActiveEnterTimestamp", ""),
        }
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


@app.get("/api/journal")
async def journal_lines(n: int = 50):
    try:
        result = subprocess.run(
            ["journalctl", "-u", "radio-oracle", "-n", str(min(n, 200)),
             "--no-pager", "--output=short-iso"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().split("\n") if result.stdout else []
        return {"lines": lines}
    except Exception as e:
        return {"lines": [], "error": str(e)}


@app.get("/api/conversations")
async def recent_conversations():
    try:
        from oracle.memory.store import ConversationStore

        store = ConversationStore()
        sessions = store.get_recent_sessions(limit=10)
        result = []
        for s in sessions:
            msgs = store.get_messages(s["session_id"], limit=4)
            result.append({
                "session_id": s["session_id"][:8],
                "started": s["started_at"],
                "summary": s.get("summary", ""),
                "message_count": len(store.get_messages(s["session_id"])),
                "preview": msgs[:2] if msgs else [],
            })
        store.close()
        return {"sessions": result}
    except Exception as e:
        return {"sessions": [], "error": str(e)}


@app.get("/api/models")
async def model_files():
    models_dir = Path("models")
    files = []
    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024**2), 1),
                })
    return {"files": files}


# --- Hardware singletons ---

_pot: object | None = None
_action_btn: object | None = None
_power_sw: object | None = None
_leds: object | None = None


def _get_pot():
    global _pot
    if _pot is None:
        from oracle.hardware.pot import Potentiometer
        _pot = Potentiometer()
    return _pot


def _get_switches():
    global _action_btn, _power_sw
    if _action_btn is None:
        from oracle.hardware.switch_adc import make_action_button_switch, make_power_switch_switch
        _action_btn = make_action_button_switch()
        _power_sw = make_power_switch_switch()
    return _action_btn, _power_sw


def _get_leds():
    global _leds
    if _leds is None:
        from oracle.hardware.leds import StatusLEDs
        _leds = StatusLEDs()
    return _leds


@app.get("/api/hardware")
async def hardware_status():
    """Read all hardware inputs: pot, button, switch, LED mode."""
    result: dict = {}

    # Potentiometer via ADS1115
    pot = _get_pot()
    if pot.available:
        reading = pot.read()
        if reading:
            result["potentiometer"] = {
                "raw": reading.raw, "voltage": reading.voltage, "percent": reading.pct,
            }
        else:
            result["potentiometer"] = {"error": "read failed"}
    else:
        result["potentiometer"] = {"error": pot.error or "unavailable"}

    # Switches via ADS1115
    action_btn, power_sw = _get_switches()

    btn_reading = action_btn.read()
    if btn_reading:
        result["button"] = {
            "channel": btn_reading.channel, "pressed": btn_reading.closed,
            "voltage": btn_reading.voltage,
        }
    else:
        result["button"] = {"error": action_btn.error or "unavailable"}

    sw_reading = power_sw.read()
    if sw_reading:
        result["switch"] = {
            "channel": sw_reading.channel, "on": sw_reading.closed,
            "voltage": sw_reading.voltage,
        }
    else:
        result["switch"] = {"error": power_sw.error or "unavailable"}

    # LED current mode
    leds = _get_leds()
    result["led"] = {"mode": leds.mode}

    return result


@app.post("/api/hardware/led")
async def set_led(request: Request):
    """Set LED mode or direct RGB. Expects {"mode": "off|radio|..."} or {"r": bool, "g": bool, "b": bool}."""
    body = await request.json()
    leds = _get_leds()

    if "r" in body or "g" in body or "b" in body:
        color = leds.set_rgb(body.get("r", False), body.get("g", False), body.get("b", False))
        return {"ok": True, "r": color.r, "g": color.g, "b": color.b}

    mode = body.get("mode", "off")
    from oracle.hardware.leds import MODE_COLORS
    if mode not in MODE_COLORS:
        return JSONResponse({"error": f"unknown mode: {mode}", "valid": list(MODE_COLORS)}, status_code=400)
    leds.set_mode(mode)
    return {"ok": True, "mode": mode}


# --- Test tools ---

@app.post("/api/test/rag")
async def test_rag(request: Request):
    body = await request.json()
    query = body.get("query", "")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    try:
        from oracle.rag.retriever import Retriever

        retriever = Retriever()
        results = retriever.query(query, top_k=5)
        return {
            "query": query,
            "results": [
                {
                    "source": r["source"],
                    "distance": round(r["distance"], 4),
                    "text": r["text"][:300],
                    "title": r.get("metadata", {}).get("title", ""),
                }
                for r in results
            ],
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/test/llm")
async def test_llm(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)
    try:
        from oracle.llm import chat

        messages = [
            {"role": "system", "content": "Answer concisely in 1-2 sentences."},
            {"role": "user", "content": prompt},
        ]
        response = await chat(messages)
        return {"prompt": prompt, "response": response}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/test/tts")
async def test_tts(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    try:
        from oracle.tts import PiperTTS

        tts = PiperTTS()
        audio = tts.synthesize(text)
        # Return as WAV
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(tts.sample_rate)
            wf.writeframes((audio * 32767).astype("int16").tobytes())
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def main():
    import uvicorn

    uvicorn.run(
        "oracle.web.app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )


if __name__ == "__main__":
    main()
