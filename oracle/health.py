"""Health checks for Oracle subsystems."""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from config.settings import settings
from oracle.llm import check_ollama


async def run_health_checks() -> dict[str, bool]:
    """Run all health checks, return results dict."""
    results: dict[str, bool] = {}

    # Ollama
    results["ollama"] = await check_ollama()

    # Whisper model
    results["whisper_model"] = settings.whisper_model_path.exists()
    if not results["whisper_model"]:
        logger.warning(f"Whisper model not found: {settings.whisper_model_path}")

    # Piper model
    results["piper_model"] = settings.piper_model_path.exists()
    if not results["piper_model"]:
        logger.warning(f"Piper model not found: {settings.piper_model_path}")

    # ChromaDB directory
    results["chroma_db"] = settings.chroma_path.exists()
    if not results["chroma_db"]:
        logger.warning(f"ChromaDB not found: {settings.chroma_path}")

    # Audio devices
    results["audio"] = _check_audio()

    # Disk space
    results["disk_space"] = _check_disk_space()

    # Summary
    all_ok = all(results.values())
    if all_ok:
        logger.info("All health checks passed")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.warning(f"Health checks failed: {', '.join(failed)}")

    return results


def _check_audio() -> bool:
    """Check if audio devices are available."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        has_input = any(d["max_input_channels"] > 0 for d in devices)
        has_output = any(d["max_output_channels"] > 0 for d in devices)
        if not has_input:
            logger.warning("No audio input device found")
        if not has_output:
            logger.warning("No audio output device found")
        return has_input and has_output
    except Exception as e:
        logger.warning(f"Audio check failed: {e}")
        return False


def _check_disk_space(min_gb: float = 5.0) -> bool:
    """Check if there's enough free disk space."""
    usage = shutil.disk_usage(Path("/"))
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        logger.warning(f"Low disk space: {free_gb:.1f}GB free (minimum {min_gb}GB)")
        return False
    return True
