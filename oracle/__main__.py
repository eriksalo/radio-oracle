import argparse
import asyncio

from oracle.log import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="The Oracle - Offline Voice Assistant")
    parser.add_argument(
        "--mode",
        choices=["text", "voice"],
        default=None,
        help="Run mode (default: from ORACLE_MODE env or 'text')",
    )
    args = parser.parse_args()

    setup_logging()

    from config.settings import settings
    from oracle.core import run

    mode = args.mode or settings.mode
    asyncio.run(run(mode))


if __name__ == "__main__":
    main()
