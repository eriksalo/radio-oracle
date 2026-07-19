import json
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from config.settings import settings

_CHAT_URL = f"{settings.ollama_host}/api/chat"

# One shared client: connection reuse saves a TCP+HTTP handshake per call
# (every turn makes at least one, often two LLM calls).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=settings.ollama_timeout)
    return _client


def _build_payload(
    messages: list[dict[str, str]], model: str | None, stream: bool
) -> dict:
    # keep_alive=-1 pins the model in VRAM. On 8GB unified memory, allowing
    # Ollama's default 5-min unload causes cudaMalloc OOM on reload because
    # the 588MB compute buffer needs a contiguous block that fragments after
    # other allocators (STT, etc.) churn memory.
    #
    # num_ctx must be set explicitly: Ollama's default (2048 for most models)
    # silently truncates the prompt — persona + RAG chunks + history easily
    # exceed it, and the model loses whichever end Ollama drops.
    return {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": stream,
        "keep_alive": -1,
        "options": {
            "num_ctx": settings.ollama_num_ctx,
            "temperature": settings.ollama_temperature,
            "top_p": settings.ollama_top_p,
        },
    }


async def stream_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream chat completions from Ollama, yielding token strings."""
    payload = _build_payload(messages, model, stream=True)
    client = _get_client()
    async with client.stream("POST", _CHAT_URL, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                break
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> str:
    """Non-streaming chat — collects full response."""
    payload = _build_payload(messages, model, stream=False)
    client = _get_client()
    response = await client.post(_CHAT_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


async def check_ollama() -> bool:
    """Check if Ollama is reachable and the model is available."""
    try:
        client = _get_client()
        resp = await client.get(f"{settings.ollama_host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        tags = resp.json()
        models = [m["name"] for m in tags.get("models", [])]
        if settings.ollama_model in models:
            logger.info(f"Ollama ready, model '{settings.ollama_model}' loaded")
            return True
        logger.warning(
            f"Ollama reachable but model '{settings.ollama_model}' not found. "
            f"Available: {models}"
        )
        return False
    except httpx.HTTPError as e:
        logger.error(f"Ollama unreachable: {e}")
        return False


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
