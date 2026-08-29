"""Logging foundation, configured from the loaded LoggingConfig."""

from __future__ import annotations

import logging

from anvesh.config import LoggingConfig


def configure_logging(logging_config: LoggingConfig) -> None:
    """Configure the root logger per the given LoggingConfig."""
    logging.basicConfig(
        level=logging_config.level,
        format=logging_config.format,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
