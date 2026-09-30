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
    _fake_httpx(monkeypatch, synth=[0.1])

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
    text = ". LOOMINGS. Call me Ishmael. Some years ago, never mind how long precisely, I sailed. — Yes!"  # noqa: E501
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
    long = "The knowledge base has about eleven million passages from Wikipedia, about ten million from Gutenberg, plus WikiMed and iFixit repair guides for everyone."  # noqa: E501
    units = speech_units(long)
    assert all(len(u.split()) <= 6 for u in units)
    assert " ".join(units).split() == long.split()
    nopunct = " ".join(chr(ord("a") + i) * 2 for i in range(15))  # letters only: digits weigh more
    assert [len(u.split()) for u in speech_units(nopunct)] == [6, 6, 3]


@pytest.mark.parametrize(
    "raw,spoken",
    [
        ("I'd suggest *Moby Dick* by Melville.", "I'd suggest Moby Dick by Melville."),
        ("**The Odyssey** and _The Iliad_", "The Odyssey and The Iliad"),
        ("- *Dracula*\n- *Frankenstein*", "Dracula\nFrankenstein"),
        ("## Books\nTry `Emma`.", "Books\nTry Emma."),
        ("See [Moby Dick](http://x) now", "See Moby Dick now"),
        ("2 * 3 = 6", "2 3 = 6"),
    ],
)
def test_clean_for_speech_strips_markdown(raw, spoken):
    from oracle.tts import clean_for_speech

    assert clean_for_speech(raw) == spoken


def test_speech_units_use_clean_text(monkeypatch):
    from oracle.tts import speech_units

    assert speech_units("Read *Moby Dick*. It's **great**.") == ["Read Moby Dick. It's great."]


# --- the sidecar process itself (oracle/tts_server.py, no project deps) ---


def _fresh_server(monkeypatch):
    import importlib

    import oracle.tts_server as srv

    importlib.reload(srv)
    return srv


def test_sidecar_shrinks_arena_on_every_run(monkeypatch):
    """kokoro-onnx calls sess.run(None, feed): the wrapper must inject the
    arena-shrink run option so memory goes back after each unit."""
    srv = _fresh_server(monkeypatch)
    seen = {}

    class _RO:
        def __init__(self):
            self.entries = {}

        def add_run_config_entry(self, k, v):
            self.entries[k] = v

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(RunOptions=_RO))

    class _Sess:
        def run(self, names, feed, run_options=None):
            seen["ro"] = run_options
            return [np.zeros(3)]

        def get_inputs(self):
            return ["tokens"]

    sess = srv._shrinking(_Sess())
    sess.run(None, {"x": 1})
    assert seen["ro"].entries == {"memory.enable_memory_arena_shrinkage": "gpu:0"}
    assert sess.get_inputs() == ["tokens"]  # everything else passes through
    assert srv.ARENA_STRATEGY == "kSameAsRequested"  # what shrinkage wants


def test_sidecar_rebuilds_session_and_retries_once(monkeypatch, capsys):
    """An arena failure must cost one rebuild, not a mute radio: the unit
    is retried on a fresh session and reported as synthesized."""
    srv = _fresh_server(monkeypatch)
    builds = []

    class _Kokoro:
        def __init__(self, fail_first):
            self.fail_first = fail_first

        def create(self, text, voice="am_michael", speed=1.0):
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("bfc_arena.cc: Available memory of 0 is smaller than requested")
            return np.ones(24, dtype=np.float32), 24000

    def fake_load():
        builds.append(1)
        srv._kokoro = _Kokoro(fail_first=False)

    monkeypatch.setattr(srv, "_load", fake_load)
    srv._kokoro = _Kokoro(fail_first=True)
    samples, sr = srv._synth("The book collection has 60,030 books.", "am_michael", 1.0)
    assert sr == 24000 and len(samples) == 24 and builds == [1]
    assert srv._stats["rebuilds"] == 1 and srv._stats["fails"] == 0
    assert "rebuilding the session" in capsys.readouterr().err

    # A unit that fails even on a fresh session propagates (the handler
    # answers with silence + X-Error and counts the failure) — and the
    # next failure does NOT rebuild again until something has succeeded:
    # a unit that needs more than the cap must not cost a rebuild each.
    class _Broken:
        def create(self, *a, **k):
            raise RuntimeError("still broken")

    srv._kokoro = _Broken()
    monkeypatch.setattr(srv, "_load", lambda: None)
    with pytest.raises(RuntimeError):
        srv._synth("x", "am_michael", 1.0)
    assert srv._stats["rebuilds"] == 2
    with pytest.raises(RuntimeError):
        srv._synth("y", "am_michael", 1.0)
    assert srv._stats["rebuilds"] == 2  # no third rebuild


def test_speech_units_weigh_numbers_as_spoken(monkeypatch):
    from oracle.tts import speech_units, spoken_words

    assert spoken_words("The book collection has 60,030 books.") == 5 + 6
    assert spoken_words("about 11.5 million passages") == 1 + 4 + 2
    monkeypatch.setattr(settings, "reading_unit_max_words", 12)
    text = (
        "The knowledge base has about 11.5 million passages from Wikipedia, "
        "plus about 10.3 million passages from the Project Gutenberg books, "
        "plus iFixit repair guides."
    )
    units = speech_units(text)
    assert len(units) >= 3 and all(spoken_words(u) <= 12 for u in units)
    # prose is unchanged: 12 plain words stay one unit
    assert speech_units("one two three four five six seven eight nine ten eleven twelve") == [
        "one two three four five six seven eight nine ten eleven twelve"
    ]
