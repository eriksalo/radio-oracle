import json

import pytest

from config.settings import settings
from oracle import llm
from oracle.llm import stream_chat


class MockStreamResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockClient:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.requests: list[dict] = []
        self.is_closed = False

    def stream(self, method, url, json=None):
        self.requests.append(json)
        return MockStreamResponse(self._lines)


@pytest.mark.asyncio
async def test_stream_chat_tokens_and_payload(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " world"}, "done": False}),
        json.dumps({"done": True}),
    ]
    client = MockClient(lines)
    monkeypatch.setattr(llm, "_get_client", lambda: client)

    messages = [{"role": "user", "content": "hi"}]
    tokens = [token async for token in stream_chat(messages)]
    assert tokens == ["Hello", " world"]

    payload = client.requests[0]
    assert payload["keep_alive"] == -1
    # Ollama defaults num_ctx to 2048 and silently truncates — the payload
    # must always pin the window explicitly.
    assert payload["options"]["num_ctx"] == settings.ollama_num_ctx
    assert payload["options"]["temperature"] == settings.ollama_temperature
