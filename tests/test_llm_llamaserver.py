"""llama-server backend of oracle.llm: SSE streaming, timings → turn timer,
finish reason → stats, non-streaming chat, health."""

from __future__ import annotations

import json

import pytest

from config.settings import settings
from oracle import llm, timing
from oracle.llm import chat, check_ollama, stream_chat


class _StreamResp:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class _JsonResp:
    def __init__(self, data, status=200, text=""):
        self._data = data
        self.status_code = status
        self.text = text or json.dumps(data)

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Client:
    def __init__(self, lines=None, post_json=None, get_json=None):
        self._lines = lines or []
        self._post_json = post_json
        self._get_json = get_json
        self.requests: list[tuple] = []
        self.is_closed = False

    def stream(self, method, url, json=None):
        self.requests.append((method, url, json))
        return _StreamResp(self._lines)

    async def post(self, url, json=None):
        self.requests.append(("POST", url, json))
        return _JsonResp(self._post_json)

    async def get(self, url, timeout=None):
        self.requests.append(("GET", url, None))
        return _JsonResp(self._get_json)


@pytest.fixture
def llama(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "llama-server")
    timing.clear()
    yield
    timing.clear()


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


@pytest.mark.asyncio
async def test_stream_parses_sse_and_records_timings(llama, monkeypatch):
    lines = [
        _sse({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
        "",
        _sse({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
        _sse(
            {
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "timings": {
                    "prompt_n": 300,
                    "cache_n": 900,
                    "prompt_ms": 500.0,
                    "predicted_n": 140,
                    "predicted_ms": 7000.0,
                },
            }
        ),
        "data: [DONE]",
    ]
    client = _Client(lines=lines)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    t = timing.start("t")
    stats: dict = {}
    tokens = [tok async for tok in stream_chat([{"role": "user", "content": "hi"}], stats=stats)]
    assert tokens == ["Hel", "lo"]
    assert stats["done_reason"] == "length" and stats["eval_count"] == 140
    s = t.summary()
    assert s["prefill"] == pytest.approx(0.5)
    assert s["prompt_tokens"] == 1200 and s["cached_tokens"] == 900
    assert s["decode_tps"] == pytest.approx(20.0)
    assert "first_token" in s
    method, url, payload = client.requests[0]
    assert url.endswith("/v1/chat/completions")
    assert payload["cache_prompt"] is True
    assert payload["max_tokens"] == settings.ollama_num_predict
    assert payload["stream"] is True


@pytest.mark.asyncio
async def test_chat_non_streaming(llama, monkeypatch):
    client = _Client(
        post_json={
            "choices": [{"message": {"content": "Where did Nikola Tesla die?"}}],
            "timings": {
                "prompt_ms": 120.0,
                "prompt_n": 50,
                "predicted_n": 8,
                "predicted_ms": 400.0,
            },
        }
    )
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    out = await chat([{"role": "user", "content": "rewrite"}], num_predict=32)
    assert out == "Where did Nikola Tesla die?"
    assert client.requests[0][2]["max_tokens"] == 32


@pytest.mark.asyncio
async def test_health(llama, monkeypatch):
    client = _Client(get_json={"status": "ok"})
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    assert await check_ollama() is True
    assert client.requests[0][1].endswith("/health")


@pytest.mark.asyncio
async def test_ollama_backend_untouched(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    lines = [json.dumps({"message": {"content": "x"}, "done": False}), json.dumps({"done": True})]
    client = _Client(lines=lines)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    assert [t async for t in stream_chat([{"role": "user", "content": "hi"}])] == ["x"]
    assert client.requests[0][1].endswith("/api/chat")
