"""FastAPI diagnostic server: mic, speaker, LLM, system stats."""

from __future__ import annotations

import asyncio
import io
import sys
import wave
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from loguru import logger
from pydantic import BaseModel

from config.settings import settings

app = FastAPI(title="Radio Oracle Diagnostics")

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
# /api/speak — synthesize via Piper subprocess, play through Jetson speaker
# ---------------------------------------------------------------------------

# Serializes synthesis + playback. The synth subprocess loads ~300 MB of Piper
# weights; running two in parallel doubles peak RSS and risks Ollama OOM on the
# Jetson's unified memory pool. Playback is also exclusive (single speaker).
_speak_lock = asyncio.Lock()


async def _synth_via_worker(text: str, radio_filter: bool) -> bytes:
    """Run oracle.diag.tts_worker as a subprocess; return WAV bytes.

    Subprocess exit reclaims the Piper model RSS so the diag process stays
    small between calls.
    """
    args = [sys.executable, "-m", "oracle.diag.tts_worker"]
    if radio_filter:
        args.append("--radio-filter")
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(text.encode("utf-8"))
    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"tts worker failed: {msg}")
    return stdout


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
      </div>
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

// --- ask ---
$('askBtn').onclick = async () => {
  const btn = $('askBtn'), st = $('askStatus');
  const text = $('askText').value.trim();
  if (!text) return;
  btn.disabled = true; setStatus(st, 'Thinking…');
  $('answer').textContent = ''; $('ragSources').textContent = '';
  try {
    const r = await postJSON('/api/ask', { text, use_rag: $('ragChk').checked });
    const j = await r.json();
    $('answer').textContent = j.answer;
    if (j.rag_used) {
      $('ragSources').textContent = 'RAG sources: ' + (j.rag_sources.join(', ') || '(none)');
    } else if ($('ragChk').checked) {
      $('ragSources').textContent = 'RAG requested but no collections matched.';
    }
    setStatus(st, 'Done.', 'ok');
    if ($('speakAnswerChk').checked) {
      await postJSON('/api/speak', { text: j.answer, radio_filter: $('filterChk').checked });
    }
  } catch (e) {
    setStatus(st, 'Error: ' + e.message, 'err');
  } finally { btn.disabled = false; }
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
  } catch (e) {
    // ignore transient
  }
}
pollStats();
setInterval(pollStats, 2000);

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
