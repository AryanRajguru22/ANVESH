# ANVESH V1

Corridor-level, multi-camera traffic congestion-cause reasoning system.
Implementation authority: [`docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md`](docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md).

**Current status: M0 -> M6.5 complete.** Single- and two-camera perception,
world alignment, corridor-state assembly, cross-camera evidence fusion,
propagation + feedback, a controlled synthetic evaluation harness, and a
real-world validation/hardening checkpoint (see
[`docs/M6_5_REAL_WORLD_VALIDATION.md`](docs/M6_5_REAL_WORLD_VALIDATION.md))
are implemented. This is a research prototype: candidate-cause ranking
only, never causal inference, with no validated real-world accuracy claim.
The Part 5 data contracts (schema only) are frozen in
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
