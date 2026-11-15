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

**Operational MVP (traffic-police application layer):** `anvesh/app/` adds a dashboard + HTTP API on top of the
perception core -- camera/session management, incident detection with preserved evidence, plate entry/OCR boundary,
registry and e-challan adapters, an officer-reviewed case state machine, alert routing and a hash-chained audit log.
Quick start (needs `models/yolov8n.pt`; **no extra dependencies**, standard library only):

```bash
.venv\Scripts\python.exe scripts\run_app.py --demo-users      # then open http://127.0.0.1:8750/
# credentials are generated randomly into outputs\app\initial_credentials.txt (gitignored)
.venv\Scripts\python.exe scripts\demo_e2e.py                  # reproducible end-to-end demo through the HTTP API
```

What is real, simulated and not connected is documented in [`docs/ANVESH_ACCEPTANCE_REPORT.md`](docs/ANVESH_ACCEPTANCE_REPORT.md)
and [`docs/ANVESH_APP_GUIDE.md`](docs/ANVESH_APP_GUIDE.md). In short: detection/tracking/stop/congestion events run on real
footage; OCR, the vehicle registry and the official e-challan system are **not connected** (sandbox adapters are labelled
test data); the officer remains the final authority and nothing issues a challan.

**Officer screen (video-first live view):** one click starts a real processing session and shows the annotated video as it is
processed (detector -> tracker -> lanes / speed / wrong-way / accident / jam-cause candidate / plate reader), with the live
state, session id and source kind above the picture and the alerts beside it. The full dashboard stays at `/advanced`.

```bash
"Start ANVESH Demo.bat"                                                    # Windows: starts the demo and opens the browser
.venv\Scripts\python.exe scripts\run_demo.py --reset --one-click --open    # same thing from a terminal (disposable data in demo\data)
python demo\tools\ui_viewports.py --base http://127.0.0.1:8760            # responsive check at 320-1440 px (needs a running demo)
```

The processed picture is the live MJPEG stream of the session (`/api/v1/sessions/{id}/stream.mjpg`): each frame is the one the
detector/tracker just handled, the stream ends when the session stops and a closed browser tab frees its server thread. Scripted
test scenes are labelled **SIMULATED**; plates are read only when a vehicle is wide enough in the picture and are candidates for an
officer to confirm. This is a research/demonstration build, not an enforcement system. Session history and the measured results:
[`docs/ANVESH_SESSION8_REPORT.md`](docs/ANVESH_SESSION8_REPORT.md).

## Requirements

- Python 3.11+ (the evaluated Windows / CUDA 13.0 environment is recorded in
  `requirements-lock.txt`; the CUDA torch wheels must be installed separately
  first, see the header of that file. The lock is not claimed to be portable
  to other OSes, Python versions, or CPU-only machines)

## Install

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate      # Windows
# 1. CUDA torch first, from the PyTorch index (a plain `pip install -e .` would pull a different torch via ultralytics)
pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
# 2. the rest of the evaluated environment
pip install -r requirements-lock.txt
# 3. the project itself (--no-deps so nothing re-resolves torch)
pip install --no-deps -e ".[dev]"
```

This sequence was run once from a clean archive of the committed tree on the author's Windows machine (CPython 3.11.9,
NVIDIA GPU): `pip check` was clean, `pip freeze` matched the lock, and the suite gave 768 passed / 4 skipped (the 4 need
`models/yolov8n.pt`), also with the GPU hidden. It has not been run on any other OS, Python version or machine.

## Run

```bash
python -m anvesh.main
# or, after install:
anvesh
```

## Test

```bash
pytest                                                  # core tests (one process over the whole suite can be killed by the memory guard)
bash demo/tools/run_test_groups.sh demo/runs/tests.log  # the whole suite as 9 sequential processes (1,403 passed at the Session 8 final run: 1,381 in the original groups + 22 in the officer-layer and stream test files, now part of group app-4)
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

## Safety behaviours (M-A hardening)

These are defaults or opt-ins a user of the streaming API needs to know about. None of them is evidence of real-camera, RTSP or metric-speed validation.

- **Metric speed needs evidence.** `fit_planar_homography` returns an UNVALIDATED profile (image-space output, no m/s or km/h) unless a `MetricValidationRecord` is supplied. Test and demo code opts in through `anvesh.world.synthetic_fixture.fit_synthetic_validated_homography`; such profiles report `is_synthetic_validated`, never `is_real_camera_validated`. No code here produces real survey evidence yet.
- **`TrafficState` declares its geometry.** New states must set `geometry_validity`. Stored states from before that field load as unverified: never metric, and their speed is withheld (`measured_mean_speed()` returns `None`).
- **Data loss lowers trust.** Each window reports `frame_completeness`; the reasoner does not advance congestion hysteresis on a window below `min_update_coverage` (0.9), and reliability is multiplied by the window's observation completeness (neutral at 1.0).
- **Silence is detected automatically for sources with `stream_time()`** (the RTSP source). `idle_poll_s=None` opts out; requesting a poll on a source without `stream_time()` raises. The live frame policy is unchanged (`BLOCK` by default).
- **Stop episodes: the displacement criterion is opt-in** (`StreamingPerceptionEngine(stop_displacement=DisplacementCriterion())`). The default per-step speed rule is unchanged and is known to miss jittery parked vehicles.

## Layout

```
anvesh/            application package (config, logging, entry point, module boundaries)
configs/           TOML configuration (default.toml)
tests/             pytest suite
pyproject.toml     project + dependency metadata
```
