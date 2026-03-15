from __future__ import annotations

import logging
import sys

from loguru import logger


def setup_logging(level: str) -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    for noisy_logger_name in ("asyncio", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
