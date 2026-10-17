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

**Milestone M-A (streaming single-camera world model):** see
[`docs/M-A_FOUNDATION_REPORT.md`](docs/M-A_FOUNDATION_REPORT.md) for what is
implemented versus what is actually validated, the speed-safety invariant, the
identity design, and the CityFlow / latency / soak results.

## Requirements

- Python 3.11+ (the evaluated Windows / CUDA 13.0 environment is recorded in
  `requirements-lock.txt`; the CUDA torch wheels must be installed separately
  first, see the header of that file. The lock is not claimed to be portable
  to other OSes, Python versions, or CPU-only machines)

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

## M-A evaluation scripts (need the CityFlow dataset at `E:\traffic dataset`, `models/yolov8n.pt`, a GPU)

```bash
python scripts/cityflow_cache_perception.py --camera c006   # cache real detector+tracker output once
python scripts/eval_cityflow.py --camera c006               # detector / tracker / TrackManager identity vs ground truth
python scripts/stream_parity.py                             # batch/stream parity on real data
python scripts/stream_bench.py --streams c006,c009 --tag dual_unpaced   # latency + throughput (1 or 2 streams)
python scripts/stream_soak.py                               # 34-minute demo.mp4 memory soak
python scripts/identity_before_after.py --video demo.mp4 --seconds 240  # duplicate-identity before/after
python scripts/run_s02_c006_showcase.py --sample-seconds 30 --output outputs/m_a/c006.json && python scripts/c006_safety_report.py
```

## Layout

```
anvesh/            application package (config, logging, entry point, module boundaries)
configs/           TOML configuration (default.toml)
tests/             pytest suite
pyproject.toml     project + dependency metadata
```
