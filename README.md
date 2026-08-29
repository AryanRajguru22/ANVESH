# ANVESH V1

Corridor-level, multi-camera traffic congestion-cause reasoning system.
Implementation authority: [`docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md`](docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md).

**Current status: M0.5 -- repository foundation + frozen data contracts.**
No perception, fusion, hypothesis, propagation, feedback, or dashboard logic
exists yet. The `anvesh/` subpackages (`perception/`, `world/`, `evidence/`,
`fusion/`, `hypotheses/`, `propagation/`, `feedback/`, `api/`, `evaluation/`)
are empty boundaries reserved for later milestones. The Part 5 data
contracts (schema only, no persistence) are frozen in
[`anvesh/storage/schemas.py`](anvesh/storage/schemas.py).

## Requirements

- Python 3.11+

## Install

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

## Run

```bash
python -m anvesh.main
# or, after install:
anvesh
```

## Test

```bash
pytest
```

## Layout

```
anvesh/            application package (config, logging, entry point, module boundaries)
configs/           TOML configuration (default.toml)
tests/             pytest suite
pyproject.toml     project + dependency metadata
```
