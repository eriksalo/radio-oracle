"""KokoroTTS ↔ GPU sidecar client: health check, remote synthesis, fallback."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from config.settings import settings
from oracle.tts import KokoroTTS


class _Resp:
    def __init__(self, content=b"", text="ok CUDAExecutionProvider", status=200):
        self.content = content
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def _fake_httpx(monkeypatch, *, health_ok=True, synth=None):
    calls: list[tuple] = []
    fake = types.ModuleType("httpx")

    def get(url, timeout=None):
        calls.append(("get", url))
        if not health_ok:
            raise ConnectionError("refused")
        return _Resp()

    def post(url, params=None, content=None, timeout=None):
        calls.append(("post", url, params, content))
        if synth is None:
            raise ConnectionError("refused")
        return _Resp(content=np.asarray(synth, dtype="<f4").tobytes())

    fake.get = get
    fake.post = post
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return calls


@pytest.fixture
def server_mode(monkeypatch):
    monkeypatch.setattr(settings, "tts_backend", "server")
    monkeypatch.setattr(settings, "tts_server_url", "http://127.0.0.1:8781")
    monkeypatch.setattr(settings, "tts_peak", 0.0)  # no normalisation in these tests


def test_remote_synthesis_decodes_float32(server_mode, monkeypatch):
    calls = _fake_httpx(monkeypatch, synth=[0.1, -0.2, 0.3])
    tts = KokoroTTS()
    tts.load()
    assert tts._server == "http://127.0.0.1:8781"
    out = tts.synthesize("Hello there.")
    np.testing.assert_allclose(out, np.array([0.1, -0.2, 0.3], dtype=np.float32), rtol=1e-6)
    post = [c for c in calls if c[0] == "post"][0]
    assert post[1].endswith("/synth")
    assert post[2]["voice"] == settings.tts_voice
    assert post[3] == b"Hello there."


def test_unreachable_sidecar_falls_back_to_local(server_mode, monkeypatch):
    _fake_httpx(monkeypatch, health_ok=False)
    tts = KokoroTTS()
    assert tts._try_server() is False
    assert tts._server is None


def test_failed_remote_call_drops_to_local(server_mode, monkeypatch):
    _fake_httpx(monkeypatch, synth=None)  # health ok, synth refused
    tts = KokoroTTS()
    tts.load()
    assert tts._server is not None

    # The fallback must load the local model itself (a stub here), not
    # just re-run load(), which would re-attach to the still-healthy
    # sidecar and leave _kokoro None (the 2026-09-27 harness crash).
    def fake_local():
        tts._kokoro = types.SimpleNamespace(
            create=lambda text, voice, speed: (np.array([0.5], dtype=np.float32), 24000)
        )

    monkeypatch.setattr(tts, "_load_local", fake_local)
    out = tts.synthesize("x")
    assert tts._server is None
    np.testing.assert_allclose(out, np.array([0.5], dtype=np.float32))
    # Subsequent calls stay local without re-probing the sidecar.
    out2 = tts.synthesize("y")
    np.testing.assert_allclose(out2, np.array([0.5], dtype=np.float32))


def test_local_backend_never_touches_http(monkeypatch):
    monkeypatch.setattr(settings, "tts_backend", "local")
    calls = _fake_httpx(monkeypatch)
    tts = KokoroTTS()
    assert tts._try_server() is False
    assert calls == []
