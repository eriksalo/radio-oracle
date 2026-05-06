"""Run: python -m oracle.diag [--host 0.0.0.0] [--port 8000]"""

from __future__ import annotations

import argparse
import subprocess

import uvicorn
from loguru import logger


def _service_active(name: str) -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", name], capture_output=True, text=True, timeout=2
        )
        return r.stdout.strip() == "active"
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    if _service_active("radio-oracle"):
        logger.warning(
            "radio-oracle.service is active — it holds the mic/speaker. "
            "Stop it first: sudo systemctl stop radio-oracle"
        )

    logger.info(f"Diagnostics server starting on http://{args.host}:{args.port}")
    uvicorn.run(
        "oracle.diag.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
