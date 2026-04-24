import json

import httpx
import pytest

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


class MockStreamClient:
    def __init__(self, lines: list[str]):
        self._lines = lines

    def stream(self, method, url, json=None):
        return MockStreamResponse(self._lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_stream_chat(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " world"}, "done": False}),
        json.dumps({"done": True}),
    ]

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: MockStreamClient(lines),
    )

    messages = [{"role": "user", "content": "hi"}]
    tokens = []
    async for token in stream_chat(messages):
        tokens.append(token)

    assert tokens == ["Hello", " world"]
