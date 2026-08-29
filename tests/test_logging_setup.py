import logging

from anvesh.config import LoggingConfig
from anvesh.logging_setup import configure_logging, get_logger


def test_configure_logging_sets_level():
    configure_logging(LoggingConfig(level="DEBUG", format="%(message)s"))
    assert logging.getLogger().level == logging.DEBUG

    configure_logging(LoggingConfig(level="WARNING", format="%(message)s"))
    assert logging.getLogger().level == logging.WARNING


def test_get_logger_returns_named_logger():
    logger = get_logger("anvesh.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "anvesh.test"
