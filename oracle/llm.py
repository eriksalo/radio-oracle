import json
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from config.settings import settings
from oracle import timing

_CHAT_URL = f"{settings.ollama_host}/api/chat"
_LLAMA_CHAT_URL = f"{settings.llama_server_url}/v1/chat/completions"


def _use_llama_server() -> bool:
    return settings.llm_backend == "llama-server"


# One shared client: connection reuse saves a TCP+HTTP handshake per call
# (every turn makes at least one, often two LLM calls).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=settings.ollama_timeout)
    return _client


def _build_payload(
    messages: list[dict[str, str]],
    model: str | None,
    stream: bool,
    num_predict: int | None = None,
) -> dict:
    # keep_alive=-1 pins the model in VRAM. On 8GB unified memory, allowing
    # Ollama's default 5-min unload causes cudaMalloc OOM on reload because
    # the 588MB compute buffer needs a contiguous block that fragments after
    # other allocators (STT, etc.) churn memory.
    #
    # num_ctx must be set explicitly: Ollama's default (2048 for most models)
    # silently truncates the prompt — persona + RAG chunks + history easily
    # exceed it, and the model loses whichever end Ollama drops.
    options = {
        "num_ctx": settings.ollama_num_ctx,
        "temperature": settings.ollama_temperature,
        "top_p": settings.ollama_top_p,
    }
    if num_predict:
        options["num_predict"] = num_predict
    return {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": stream,
        "keep_alive": -1,
        "options": options,
    }


async def stream_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    stats: dict | None = None,
) -> AsyncIterator[str]:
    """Stream chat completions from Ollama, yielding token strings.

    If *stats* is given, the final chunk's ``done_reason`` ("stop" or
    "length") and token counts are copied into it for the caller.
    """
    # Streamed replies are spoken aloud — cap length so the radio doesn't
    # monologue (the persona asks for brevity; this enforces it).
    if _use_llama_server():
        async for token in _stream_llama_server(messages, model, stats):
            yield token
        return
    payload = _build_payload(messages, model, stream=True, num_predict=settings.ollama_num_predict)
    client = _get_client()
    timer = timing.current()
    first = True
    async with client.stream("POST", _CHAT_URL, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                _record_done_stats(timer, chunk)
                if stats is not None:
                    stats["done_reason"] = chunk.get("done_reason", "stop")
                    stats["eval_count"] = chunk.get("eval_count", 0)
                break
            token = chunk.get("message", {}).get("content", "")
            if token:
                if first and timer is not None:
                    timer.mark("first_token")
                    first = False
                yield token


def _record_done_stats(timer: timing.TurnTimer | None, chunk: dict) -> None:
    """Copy Ollama's own prefill/decode accounting onto the turn timer.

    ``prompt_eval_count`` is the *whole* prompt even on a prefix-cache hit
    (measured 2026-09-27: identical prompt → same count, 0.08s); only
    ``prompt_eval_duration`` reveals what was actually recomputed.
    """
    if timer is None:
        return
    pd = chunk.get("prompt_eval_duration")
    if pd:
        timer.set("prefill", pd / 1e9)
    ed = chunk.get("eval_duration")
    ec = chunk.get("eval_count")
    if ed and ec:
        timer.note(decode_tps=round(ec / (ed / 1e9), 1))
    timer.note(
        prompt_tokens=chunk.get("prompt_eval_count", 0),
        reply_tokens=ec or 0,
    )


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    num_predict: int | None = None,
) -> str:
    """Non-streaming chat — collects full response."""
    if _use_llama_server():
        payload = _build_llama_payload(messages, model, stream=False, num_predict=num_predict)
        response = await _get_client().post(_LLAMA_CHAT_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        _record_llama_timings(timing.current(), data.get("timings"))
        return data["choices"][0]["message"]["content"]
    payload = _build_payload(messages, model, stream=False, num_predict=num_predict)
    client = _get_client()
    response = await client.post(_CHAT_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


async def check_ollama() -> bool:
    """Check if the LLM server (Ollama or llama-server) is reachable and has the model."""
    if _use_llama_server():
        try:
            resp = await _get_client().get(f"{settings.llama_server_url}/health", timeout=5.0)
            ok = resp.status_code == 200 and resp.json().get("status") == "ok"
            (logger.info if ok else logger.warning)(
                f"llama-server at {settings.llama_server_url}: {resp.text.strip()}"
            )
            return ok
        except httpx.HTTPError as e:
            logger.error(f"llama-server unreachable: {e}")
            return False
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
            f"Ollama reachable but model '{settings.ollama_model}' not found. Available: {models}"
        )
        return False
    except httpx.HTTPError as e:
        logger.error(f"Ollama unreachable: {e}")
        return False


# ------------------------------------------------------------ llama-server
#
# OpenAI-compatible endpoint of llama.cpp's server (Phase 4: replaces
# Ollama on the Jetson — same GGUF, +20% decode, and an explicit prompt
# cache with a RAM cap). cache_prompt keeps the KV prefix across turns;
# timings_per_token makes the final chunk carry prompt_ms / cache_n so the
# turn timer sees real prefill cost and cache hits.


def _build_llama_payload(
    messages: list[dict[str, str]],
    model: str | None,
    stream: bool,
    num_predict: int | None = None,
) -> dict:
    payload = {
        "model": model or settings.llama_server_model,
        "messages": messages,
        "stream": stream,
        "temperature": settings.ollama_temperature,
        "top_p": settings.ollama_top_p,
        "cache_prompt": True,
        "timings_per_token": True,
    }
    if num_predict:
        payload["max_tokens"] = num_predict
    return payload


def _record_llama_timings(timer: timing.TurnTimer | None, t: dict | None) -> None:
    if timer is None or not t:
        return
    if t.get("prompt_ms") is not None:
        timer.set("prefill", t["prompt_ms"] / 1000.0)
    if t.get("predicted_ms") and t.get("predicted_n"):
        timer.note(decode_tps=round(t["predicted_n"] / (t["predicted_ms"] / 1000.0), 1))
    timer.note(
        prompt_tokens=int(t.get("prompt_n", 0)) + int(t.get("cache_n", 0) or 0),
        cached_tokens=int(t.get("cache_n", 0) or 0),
        reply_tokens=int(t.get("predicted_n", 0) or 0),
    )


async def _stream_llama_server(
    messages: list[dict[str, str]],
    model: str | None,
    stats: dict | None,
) -> AsyncIterator[str]:
    payload = _build_llama_payload(
        messages, model, stream=True, num_predict=settings.ollama_num_predict
    )
    client = _get_client()
    timer = timing.current()
    first = True
    async with client.stream("POST", _LLAMA_CHAT_URL, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "timings" in chunk:
                _record_llama_timings(timer, chunk["timings"])
                if stats is not None:
                    stats["eval_count"] = chunk["timings"].get("predicted_n", 0)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            token = (choices[0].get("delta") or {}).get("content") or ""
            reason = choices[0].get("finish_reason")
            if reason and stats is not None:
                stats["done_reason"] = "length" if reason == "length" else "stop"
            if token:
                if first and timer is not None:
                    timer.mark("first_token")
                    first = False
                yield token


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
