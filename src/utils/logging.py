"""Logging configuration for pipeline scripts."""

import logging
from typing import Final

DEFAULT_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")
    logging.basicConfig(level=numeric_level, format=DEFAULT_FORMAT, force=True)
