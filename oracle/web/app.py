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


# --- Hardware ---

def _read_gpio(pin: int) -> bool | None:
    """Read a GPIO pin state. Returns True (HIGH), False (LOW), or None if unavailable."""
    try:
        import Jetson.GPIO as GPIO

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        val = GPIO.input(pin)
        return bool(val)
    except Exception:
        # Try sysfs fallback
        try:
            # Map board pin to sysfs GPIO number (Jetson Orin Nano)
            sysfs = Path(f"/sys/class/gpio/gpio{_board_to_sysfs(pin)}/value")
            if sysfs.exists():
                return sysfs.read_text().strip() == "1"
        except Exception:
            pass
    return None


def _board_to_sysfs(board_pin: int) -> int:
    """Map Jetson Orin Nano board pin to sysfs GPIO number (common mappings)."""
    # These are approximate — Jetson Orin Nano Super specific
    mapping = {
        31: 316,  # momentary button
        33: 317,  # on/off switch
    }
    return mapping.get(board_pin, board_pin)


def _read_ads1115() -> dict | None:
    """Read ADS1115 ADC channel 0 (potentiometer) via I2C."""
    try:
        import smbus2

        bus = smbus2.SMBus(1)
        addr = 0x48

        # Config: AINp=A0, AINn=GND, FSR=+/-4.096V, single-shot, 128SPS
        config = 0xC383
        bus.write_i2c_block_data(addr, 0x01, [(config >> 8) & 0xFF, config & 0xFF])

        import time
        time.sleep(0.01)

        # Read conversion
        data = bus.read_i2c_block_data(addr, 0x00, 2)
        raw = (data[0] << 8) | data[1]
        if raw > 32767:
            raw -= 65536

        # Convert to voltage (FSR = 4.096V, 16-bit signed)
        voltage = raw * 4.096 / 32767
        # Convert to 0-100% (3.3V pot rail)
        percent = max(0.0, min(100.0, (voltage / 3.3) * 100))

        bus.close()
        return {"raw": raw, "voltage": round(voltage, 3), "percent": round(percent, 1)}
    except Exception as e:
        return {"error": str(e)}


def _set_led_pwm(duty: int) -> bool:
    """Set LED brightness via PWM on pin 32 (PWM0). duty: 0-100."""
    try:
        import Jetson.GPIO as GPIO

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(32, GPIO.OUT)
        pwm = GPIO.PWM(32, 1000)  # 1kHz
        pwm.start(max(0, min(100, duty)))
        # Store reference so it doesn't get garbage collected
        _set_led_pwm._pwm = pwm
        return True
    except Exception:
        # sysfs fallback for basic on/off
        try:
            pin_path = Path(f"/sys/class/gpio/gpio{_board_to_sysfs(32)}")
            if not pin_path.exists():
                Path("/sys/class/gpio/export").write_text(str(_board_to_sysfs(32)))
            (pin_path / "direction").write_text("out")
            (pin_path / "value").write_text("1" if duty > 0 else "0")
            return True
        except Exception:
            return False


@app.get("/api/hardware")
async def hardware_status():
    """Read all hardware inputs: pot, button, switch."""
    result: dict = {}

    # Potentiometer via ADS1115
    result["potentiometer"] = _read_ads1115()

    # Momentary button (pin 31, active LOW with pull-up)
    btn_val = _read_gpio(31)
    if btn_val is not None:
        result["button"] = {"pin": 31, "pressed": not btn_val, "raw": btn_val}
    else:
        result["button"] = {"pin": 31, "error": "GPIO unavailable"}

    # On/off switch (pin 33, active LOW with pull-up)
    sw_val = _read_gpio(33)
    if sw_val is not None:
        result["switch"] = {"pin": 33, "on": not sw_val, "raw": sw_val}
    else:
        result["switch"] = {"pin": 33, "error": "GPIO unavailable"}

    return result


@app.post("/api/hardware/led")
async def set_led(request: Request):
    """Set LED brightness. Expects {"brightness": 0-100} or {"color": "off|dim|bright|pulse"}."""
    body = await request.json()

    if "brightness" in body:
        duty = int(body["brightness"])
        ok = _set_led_pwm(duty)
        return {"brightness": duty, "ok": ok}

    color = body.get("color", "off")
    presets = {"off": 0, "dim": 15, "medium": 50, "bright": 100}
    duty = presets.get(color, 0)
    ok = _set_led_pwm(duty)
    return {"color": color, "brightness": duty, "ok": ok}


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
