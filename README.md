# ANVESH V1

Corridor-level, multi-camera traffic congestion-cause reasoning system.
Implementation authority: [`docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md`](docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md).

**Status (Session 9, 15 November 2026): a finished research / demonstration system, not a production or enforcement system.**
It combines a corridor-level congestion-cause reasoning core (single- and two-camera perception, world alignment, evidence fusion,
candidate-cause ranking that is allowed to abstain, propagation, feedback, evaluation harness) with a streaming single-camera
world model and an operational MVP (HTTP API, officer screen, full dashboard, evidence, cases, audit). It is **implemented and
tested**; it is **not validated against ground truth** for speed, congestion, stop incidents, wrong-way, collision or cause
ranking, and no accuracy figure is claimed for them. Detector and tracker were scored on CityFlow / MIO-TCD (see the reports).
Candidate-cause ranking only, never causal inference. The Part 5 data contracts (schema only) are frozen in
[`anvesh/storage/schemas.py`](anvesh/storage/schemas.py). Final status, benchmark and limitations:
[`docs/ANVESH_FINALIZATION_REPORT.md`](docs/ANVESH_FINALIZATION_REPORT.md).

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

## Configuration (environment variables; nothing secret has a default)

| Variable | Meaning (default) |
|---|---|
| `ANVESH_DATA_DIR` | application data (SQLite, evidence, audit) (`outputs\app`) |
| `ANVESH_HOST` / `ANVESH_PORT` | listen address (`127.0.0.1` / `8750`; the demo uses 8760) |
| `ANVESH_MODEL` / `ANVESH_FINETUNED_MODEL` | detector weights (`models\yolov8n.pt`); the fine-tuned model is opt-in and **not promoted** |
| `ANVESH_SECURE_COOKIES` | `1` behind HTTPS |
| `ANVESH_SESSION_IDLE_S`, `ANVESH_DISTINCT_APPROVER` | sign-in idle timeout; four-eyes approval |
| `ANVESH_ALERT_WEBHOOK_URL`, `ANVESH_SMTP_*`, `ANVESH_REGISTRY_*`, `ANVESH_CHALLAN_*` | optional integrations; unset = "not connected" (none ships with an endpoint) |

Model weights (`models\yolov8n.pt`, `models\plates\...`) and videos (`demo.mp4`) are not in the repository. `demo.mp4`'s origin and licence are unverified
(see `demo/SOURCE.md`): it is never committed. Browser checks in `demo/tools` and `docs/viva_assets/capture_screens.py` need
`pip install playwright` and Chrome (system Python); they are tools, not application dependencies.

## Documentation

- `docs/ANVESH_FINALIZATION_REPORT.md` - final state, tests, benchmark, limitations (start here).
- `docs/ANVESH_APP_GUIDE.md`, `docs/ANVESH_RELEASE_READINESS.md`, `docs/ANVESH_SESSION6_REPORT.md` ... `SESSION9`, `demo/README.md` (demo runbook).
- The technical viva handbook (`ANVESH_COMPLETE_TECHNICAL_VIVA_HANDBOOK.pdf/.docx`, 22 chapters) and the panel guide
  (`ANVESH_PANEL_PRESENTATION_GUIDE.pdf/.docx`) are built from `docs/viva_assets/` and are **not committed** because they embed frames of a video
  whose licence is unverified. Rebuild them locally: `python docs\viva_assets\capture_screens.py --base http://127.0.0.1:8760` (with the demo
  running, `--one-click`), then `python docs\viva_assets\build_handbook.py` and `python docs\viva_assets\build_guide.py` (system Python with
  reportlab, python-docx and matplotlib).
- Benchmarks: `docs/benchmarks/` (raw 10 s samples and summaries). Measure again with `demo/tools/monitor_run.py`, `demo/tools/sample_cpu.py`,
  `demo/tools/summarize_realtime.py`.

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
bash demo/tools/run_test_groups.sh demo/runs/tests.log  # the whole suite as 9 sequential processes (1,403 passed at the Session 9 final run; one timing-bounded ingestion test is flaky when the machine is slow, see the finalization report)
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
anvesh/perception, world, safety   detection, tracking, traffic state, calibration, stop episodes
anvesh/evidence, fusion, hypotheses, propagation, feedback   evidence, Dempster-Shafer fusion, ranked candidate causes, shockwave, feedback
anvesh/streaming   streaming engine, bounded queues, windows, latency recorder, RTSP source
anvesh/app         HTTP API + dashboard + officer screen (static/), sessions, incidents, evidence, cases, audit, plates
anvesh/training    detector-evaluation and benchmark helpers (data and weights are not committed)
scripts/           run_demo.py, run_app.py, evaluation and diagnostic scripts
demo/              demo runbook, camera config, test/benchmark/browser tools (demo/tools)
docs/              reports, benchmarks/, viva_assets/ (handbook and guide build scripts)
tests/             pytest suite (run it in groups: demo/tools/run_test_groups.sh)
```
