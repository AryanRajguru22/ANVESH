import logging

from anvesh.main import main


def test_main_returns_zero():
    assert main() == 0


def test_main_configures_root_logger():
    main()
    assert logging.getLogger().level == logging.INFO
    assert len(logging.getLogger().handlers) > 0
