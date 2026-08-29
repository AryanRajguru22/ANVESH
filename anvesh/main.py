"""Minimal application entry point for the M0 foundation.

This does not run any perception, fusion, or reasoning pipeline --
those are introduced in later milestones (see
ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md, Part 14).
"""

from __future__ import annotations

from anvesh.config import load_config
from anvesh.logging_setup import configure_logging, get_logger


def main() -> int:
    config = load_config()
    configure_logging(config.logging)
    logger = get_logger(__name__)
    logger.info(
        "%s v%s starting (environment=%s)",
        config.app.name,
        config.app.version,
        config.app.environment,
    )
    logger.info("M0 foundation is up. No pipeline modules are implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
