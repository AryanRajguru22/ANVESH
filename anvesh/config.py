"""Configuration loading foundation.

Reads the TOML config (default: configs/default.toml) using the stdlib
tomllib parser -- no extra dependency required.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class AppConfig:
    name: str
    version: str
    environment: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str


@dataclass(frozen=True)
class Config:
    app: AppConfig
    logging: LoggingConfig


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate the application config from a TOML file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    try:
        app_data = data["app"]
        logging_data = data["logging"]
    except KeyError as exc:
        raise ValueError(f"Config file missing required section: {exc}") from exc

    try:
        app = AppConfig(
            name=app_data["name"],
            version=app_data["version"],
            environment=app_data["environment"],
        )
        logging_cfg = LoggingConfig(
            level=logging_data["level"],
            format=logging_data["format"],
        )
    except KeyError as exc:
        raise ValueError(f"Config file missing required key: {exc}") from exc

    return Config(app=app, logging=logging_cfg)
