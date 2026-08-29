from pathlib import Path

import pytest

from anvesh.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads():
    config = load_config()
    assert config.app.name == "anvesh"
    assert config.app.environment == "development"
    assert config.logging.level == "INFO"


def test_default_config_path_exists():
    assert DEFAULT_CONFIG_PATH.is_file()


def test_missing_config_file_raises(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.toml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_path)


def test_config_missing_section_raises(tmp_path: Path):
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text('[app]\nname = "x"\nversion = "0.1.0"\nenvironment = "dev"\n')
    with pytest.raises(ValueError):
        load_config(bad_config)


def test_config_missing_key_raises(tmp_path: Path):
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text(
        '[app]\nname = "x"\nversion = "0.1.0"\nenvironment = "dev"\n'
        '[logging]\nlevel = "INFO"\n'
    )
    with pytest.raises(ValueError):
        load_config(bad_config)
