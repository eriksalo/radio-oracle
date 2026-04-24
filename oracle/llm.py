from collections.abc import AsyncIterator

import httpx
from loguru import logger

from config.settings import settings

_CHAT_URL = f"{settings.ollama_host}/api/chat"


async def stream_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream chat completions from Ollama, yielding token strings."""
    model = model or settings.ollama_model
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        async with client.stream("POST", _CHAT_URL, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json

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
    model = model or settings.ollama_model
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        response = await client.post(_CHAT_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


async def check_ollama() -> bool:
    """Check if Ollama is reachable and the model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
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
