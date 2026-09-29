"""Kokoro TTS sidecar: runs the ONNX model on the GPU in its own interpreter.

Why a sidecar: the app venv is Python 3.11 and the only JetPack 6 CUDA
build of onnxruntime is a cp310 wheel (pypi.jetson-ai-lab.io/jp6/cu126).
Kokoro-82M on the Orin's CPU is ~0.8× real time — the single largest
piece of a turn's latency — and 0.13–0.2× on the GPU (measured
2026-09-27, scripts/probe_kokoro_gpu.py).

This module has *no* project dependencies (stdlib + numpy + kokoro-onnx)
so it can run from the sidecar venv:

    /opt/radio-oracle/.venv-tts/bin/python -m oracle.tts_server

Protocol (loopback HTTP, one request at a time):
    POST /synth   body: UTF-8 text; query: voice=am_michael&speed=1.0
                  → 200, body: little-endian float32 PCM mono at 24 kHz,
                    header X-Sample-Rate
    GET  /health  → 200 "ok <provider>"

Config via env: ORACLE_TTS_SERVER_PORT (8781), ORACLE_TTS_MODEL_PATH,
ORACLE_TTS_VOICES_PATH, ORACLE_TTS_PROVIDER (CUDAExecutionProvider),
ORACLE_TTS_GPU_MEM_MB (CUDA arena cap; 0 = unlimited).
"""

from __future__ import annotations

import gc
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

PORT = int(os.environ.get("ORACLE_TTS_SERVER_PORT", "8781"))
MODEL = os.environ.get("ORACLE_TTS_MODEL_PATH", "models/kokoro-v1.0.fp16.onnx")
VOICES = os.environ.get("ORACLE_TTS_VOICES_PATH", "models/voices-v1.0.bin")
PROVIDER = os.environ.get("ORACLE_TTS_PROVIDER", "CUDAExecutionProvider")
GPU_MEM_MB = int(os.environ.get("ORACLE_TTS_GPU_MEM_MB", "0"))
# The CUDA arena and variable-length inputs (2026-09-28): every unit has
# a different shape, so the BFC arena fragments and, whatever the extend
# strategy, after a few long units (the device description: numbers,
# 8 s of audio per unit) it reports "Available memory of 0" under the
# cap and *every* later request fails — the radio goes silent until the
# sidecar is restarted. Two defences:
#   1. shrink the arena after every run (ORT run option
#      memory.enable_memory_arena_shrinkage, which wants kSameAsRequested):
#      memory beyond the initial chunk goes back to the system per unit;
#   2. if a unit still fails, rebuild the session once and retry it —
#      the failure is then one delayed unit, never a mute radio.
ARENA_SHRINK = os.environ.get("ORACLE_TTS_ARENA_SHRINK", "1") == "1"
ARENA_STRATEGY = os.environ.get(
    "ORACLE_TTS_ARENA_STRATEGY", "kSameAsRequested" if ARENA_SHRINK else "kNextPowerOfTwo"
)

_lock = threading.Lock()
_kokoro = None
_provider_used = "?"
_stats = {"units": 0, "fails": 0, "rebuilds": 0}
# A rebuild only helps when the arena has gone bad *since* it last worked
# (fragmentation). A unit that fails right after a rebuild simply needs
# more than the cap: fail it fast instead of rebuilding for every unit.
_ok_since_rebuild = True


class _ShrinkingSession:
    """Stand-in for the InferenceSession that passes the arena-shrink run
    option on every ``run`` (kokoro-onnx calls ``sess.run(None, feed)``
    and offers no hook). A proxy rather than a patched bound method: the
    patch made a reference cycle that kept a dropped session — and its
    whole arena — alive across rebuilds (2.8 GB after six)."""

    def __init__(self, sess, run_options):
        self._sess = sess
        self._ro = run_options

    def run(self, output_names, input_feed, run_options=None):
        return self._sess.run(output_names, input_feed, run_options or self._ro)

    def __getattr__(self, name):
        return getattr(self._sess, name)


def _shrinking(sess):
    import onnxruntime as ort

    ro = ort.RunOptions()
    ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return _ShrinkingSession(sess, ro)


def _load() -> None:
    global _kokoro, _provider_used
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    t = time.monotonic()
    _kokoro = None  # drop the old session (and its arena) before building a new one
    gc.collect()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    # Build the session ourselves: the JP6 onnxruntime-gpu wheel also
    # exposes TensorrtExecutionProvider, which kokoro-onnx's own default
    # (all available providers) would pick first — minutes of engine
    # building and ~1 GB of RAM before it fails on unsupported ops.
    providers: list = ["CPUExecutionProvider"]
    if PROVIDER != "CPUExecutionProvider":
        opts: dict[str, object] = {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",
            "arena_extend_strategy": ARENA_STRATEGY,
        }
        if GPU_MEM_MB:
            opts["gpu_mem_limit"] = GPU_MEM_MB * 1024 * 1024
        providers = [(PROVIDER, opts), "CPUExecutionProvider"]
    sess = ort.InferenceSession(MODEL, so, providers=providers)
    _provider_used = sess.get_providers()[0]
    if ARENA_SHRINK and _provider_used == "CUDAExecutionProvider":
        sess = _shrinking(sess)
    k = Kokoro.from_session(sess, VOICES)
    # Warm the CUDA kernels / cuDNN algo search so the first turn doesn't pay it.
    k.create("The library is open.", voice=os.environ.get("ORACLE_TTS_VOICE", "am_michael"))
    _kokoro = k
    print(
        f"tts_server: {MODEL} on {_provider_used} ready in {time.monotonic() - t:.1f}s "
        f"(arena {ARENA_STRATEGY}, shrink={'on' if ARENA_SHRINK else 'off'}, "
        f"cap {GPU_MEM_MB or 'none'} MB)",
        file=sys.stderr,
        flush=True,
    )


def _synth(text: str, voice: str, speed: float):
    """One unit, serialized; on failure rebuild the session and retry once
    (only if something has synthesized since the last rebuild)."""
    global _ok_since_rebuild
    with _lock:
        _stats["units"] += 1
        try:
            out = _kokoro.create(text, voice=voice, speed=speed)
            _ok_since_rebuild = True
            return out
        except Exception as e:  # noqa: BLE001
            first = str(e).splitlines()[0][:200]
            if not _ok_since_rebuild:
                raise
            print(
                f"tts_server: synth failed for {text[:60]!r}: {first}; rebuilding the session",
                file=sys.stderr,
                flush=True,
            )
            _stats["rebuilds"] += 1
            _ok_since_rebuild = False
            _load()
            out = _kokoro.create(text, voice=voice, speed=speed)
            _ok_since_rebuild = True
            return out


_LETTER_RE = re.compile(r"[A-Za-z0-9]")


def _speakable(text: str) -> str:
    """Strip leading punctuation/whitespace; "" when there is nothing to
    voice (". " or "—" units made the CUDA graph fail on empty input)."""
    t = text.strip()
    t = re.sub(r"^[^A-Za-z0-9\(\[\"'$]+", "", t).strip()
    return t if _LETTER_RE.search(t) else ""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def do_GET(self):  # noqa: N802
        if urlparse(self.path).path == "/health":
            body = (
                f"ok {_provider_used} units={_stats['units']} "
                f"fails={_stats['fails']} rebuilds={_stats['rebuilds']}"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/synth":
            self.send_error(404)
            return
        q = parse_qs(url.query)
        voice = q.get("voice", ["am_michael"])[0]
        speed = float(q.get("speed", ["1.0"])[0])
        n = int(self.headers.get("Content-Length", "0"))
        text = _speakable(self.rfile.read(n).decode("utf-8"))
        sr = 24000
        err = ""
        if not text:
            samples = np.zeros(int(0.15 * sr), dtype=np.float32)  # nothing to say
        else:
            try:
                samples, sr = _synth(text, voice, speed)
            except Exception as e:  # noqa: BLE001
                # A failed unit is a short silence, not a 500: the client
                # must never abandon the GPU for the CPU over one bad input.
                _stats["fails"] += 1
                err = str(e).splitlines()[0][:200]
                print(
                    f"tts_server: synth failed for {text[:60]!r}: {err}",
                    file=sys.stderr,
                    flush=True,
                )
                samples = np.zeros(int(0.3 * sr), dtype=np.float32)
        body = np.asarray(samples, dtype="<f4").tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", str(sr))
        if err:
            self.send_header("X-Error", err)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    _load()
    srv = HTTPServer(("127.0.0.1", PORT), _Handler)
    print(f"tts_server: listening on 127.0.0.1:{PORT}", file=sys.stderr, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
