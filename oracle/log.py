import sys

from loguru import logger

from config.settings import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
            "<cyan>{name}</cyan> - {message}"
        ),
    )
    logger.add(
        "data/oracle.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )
