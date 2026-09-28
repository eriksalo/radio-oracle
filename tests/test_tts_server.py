"""KokoroTTS ↔ GPU sidecar client: health check, remote synthesis, fallback."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from config.settings import settings
from oracle.tts import KokoroTTS


class _Resp:
    headers: dict = {}

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


def test_unit_failure_keeps_sidecar_and_skips(server_mode, monkeypatch):
    """A 5xx / X-Error for one unit yields a short silence; the sidecar stays."""
    calls = _fake_httpx(monkeypatch, synth=[0.1])

    class _Err(_Resp):
        headers = {"X-Error": "boom"}
        status_code = 500

    import httpx as fake

    fake.post = lambda url, params=None, content=None, timeout=None: _Err()
    tts = KokoroTTS()
    tts.load()
    out = tts.synthesize("x")
    assert tts._server is not None
    assert len(out) > 0 and not out.any()


def test_failed_remote_call_drops_to_local(server_mode, monkeypatch):
    _fake_httpx(monkeypatch, synth=None)  # health ok, connection refused on synth
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


# ------------------------------------------------------------ say() / units


def test_speech_units_chunks_and_sanitizes(monkeypatch):
    from oracle.tts import speech_units

    monkeypatch.setattr(settings, "reading_unit_max_words", 8)
    text = ". LOOMINGS. Call me Ishmael. Some years ago, never mind how long precisely, I sailed. — Yes!"
    units = speech_units(text)
    assert units[0] == "LOOMINGS. Call me Ishmael."
    assert all(len(u.split()) <= 8 for u in units)  # long sentences are cut at clauses
    assert units[-1] == "I sailed. Yes!"  # the 10-word sentence was cut at its clauses
    assert speech_units("... — .") == []


def test_say_pipelines_units_in_order(monkeypatch):
    from oracle import audio
    from oracle.tts import say

    monkeypatch.setattr(settings, "reading_unit_max_words", 6)
    played: list[int] = []
    monkeypatch.setattr(
        audio, "play_audio", lambda a, sr=None, should_abort=None: played.append(int(a[0]))
    )

    class T:
        sample_rate = 24000
        calls: list[str] = []

        def synthesize(self, text):
            T.calls.append(text)
            return np.array([len(T.calls)], dtype=np.float32)

    say(T(), "One two three four five six. Eight nine ten. Eleven twelve.")
    assert T.calls == ["One two three four five six.", "Eight nine ten. Eleven twelve."]
    assert played == [1, 2]


def test_say_aborts_between_units(monkeypatch):
    from oracle import audio
    from oracle.tts import say

    monkeypatch.setattr(settings, "reading_unit_max_words", 3)
    played: list[str] = []
    flags = iter([False, True, True, True, True])
    monkeypatch.setattr(
        audio, "play_audio", lambda a, sr=None, should_abort=None: played.append("x")
    )

    class T:
        sample_rate = 24000

        def synthesize(self, text):
            return np.zeros(4, dtype=np.float32)

    say(T(), "A b c. D e f. G h i.", should_abort=lambda: next(flags))
    assert len(played) <= 1


def test_speech_units_never_exceed_limit(monkeypatch):
    from oracle.tts import speech_units

    monkeypatch.setattr(settings, "reading_unit_max_words", 6)
    long = "The knowledge base has about eleven million passages from Wikipedia, about ten million from Gutenberg, plus WikiMed and iFixit repair guides for everyone."
    units = speech_units(long)
    assert all(len(u.split()) <= 6 for u in units)
    assert " ".join(units).split() == long.split()
    nopunct = " ".join(f"w{i}" for i in range(15))
    assert [len(u.split()) for u in speech_units(nopunct)] == [6, 6, 3]
