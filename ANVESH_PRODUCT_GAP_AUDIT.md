# ANVESH — Product Gap Audit

**Audit date:** 2026-10-03
**Audited state:** `main` @ `042ea4b` **plus an uncommitted working-tree diff** (5 modified tracked files, ~25 untracked paths). All line numbers below are **working-tree** line numbers. This matters: the speed `OUTLIER` plausibility gate (`anvesh/perception/speed_estimation.py:119,262-270`) exists **only in the uncommitted diff**; HEAD does not contain it.
**Mode:** read-only. No source file was modified, deleted, renamed or committed. The only artifact created is this report.
**Method:** the perception, world (calibration/lane), pipeline and candidate-vehicle code was read in full; evidence, fusion, ranking, safety, storage, propagation, scripts and configs were read in part; feedback, evaluation, corridor-state, rendering and the UI template were only outlined or keyword-searched (exact coverage in the Post-Audit Report at the end); the exported UI JSON was analysed programmatically; the test suite was run; a short detection+tracking benchmark was run from a scratch directory outside the repo; the exported JSON artifacts (`ui/data/*.json`) were analysed programmatically. Statements marked **[measured]** come from something I ran; **[read]** from code I read; **[not verified]** are things I could not establish from the repository.

Capability labels used throughout (exactly the seven requested):
`IMPLEMENTED + TESTED` · `IMPLEMENTED BUT INCOMPLETE` · `DEMO/SYNTHETIC ONLY` · `PARTIAL FOUNDATION` · `MISSING` · `BROKEN / NEEDS REPLACEMENT` · `NOT APPLICABLE YET`.
"Tested" below means *unit/integration tests exist and pass*. It never means *validated against real-world ground truth*. I report implemented / tested / validated / deployment-ready as four separate columns because the repository itself blurs them in places.

---

## 1. Executive Summary

**What ANVESH is today.** A well-disciplined, honestly-labelled **offline research prototype** for *corridor congestion-cause ranking* from **recorded video**, built around one question ("which of six congestion-cause hypotheses H1–H6 is best supported by the evidence, with Dempster-Shafer abstention"). The blueprint says so itself: live feeds, RBAC beyond authentication, 3+ cameras and a real road-graph are explicitly "LATER" (`docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md:52`). It is **not** a live road-intelligence, investigation or enforcement platform, and it is **not close** to one: of the 32 product capabilities in the vision, **0 are validated, 0 are deployment-ready, 15 are `MISSING` outright, and 4 more are not yet buildable** (matrix, §4).

**The ten findings that matter most**

1. **Nothing runs live; the architecture is file-batch by construction.** `VideoFileFrameSource` rejects anything that is not an existing file (`perception/ingestion.py:37-38`), timestamps are `index/fps` (`ingestion.py:68`), and `run_m1_pipeline` builds `VehicleTrack`s only *after* the whole file has been consumed and `track_manager.finalize()` has run (`perception/pipeline.py:76-117`). Time-to-first-event = length of the file. The "live dashboard" replays precomputed JSON (`"perception_mode": "precomputed_replay"`, `scripts/export_ui_data.py:783`). Concurrency primitives (threads/asyncio/queues/multiprocessing): **zero hits** in `anvesh/` and `scripts/`.

2. **Unvalidated speeds already reach the reasoning layer, bypassing the repo's own "safety" guarantee. [measured]** The repo ships a provenance-aware estimator (`VehicleSpeedEstimate`, with `APPROXIMATE`/`OUTLIER`/`UNAVAILABLE`) and a regression test (`tests/perception/test_speed_evidence_safety.py`) asserting the reasoning layer never imports it. But that test only checks *imports*. A **second, parallel speed path** feeds the reasoning layer directly:
   - `export_ui_data.py:748` sets `motion_space = WORLD` whenever `calibration_profile.is_calibrated` — it never consults `metric_validation`.
   - `schema_conversion.build_vehicle_track` (`schema_conversion.py:81-85`) then projects the **centroid** trajectory (not the ground point) through the CityFlow homography and labels the result **m/s**, with no plausibility gate.
   - `aggregate_traffic_state` (`traffic_state.py:251-257`) averages those, `CongestionThresholds` (`traffic_state.py:100-103`, `free=8 m/s`, `congested=3 m/s`) classify on them, `evidence_builder._stopped_tracks` (`evidence_builder.py:111-120`, `≤1.0 m/s`) uses them, `evidence/reliability.py:74-78` awards `calibration=1.0` because `motion_space==WORLD`, and H2/H3 become `FULL` completeness.
   - In the exported c006 artifact (`ui/data/anvesh_s02_c006_data.json`), per-window **median track "speed" is 381 / 348 / 260 "m/s"** and **83% / 75% / 71% of tracks exceed 55.6 m/s (200 km/h)**. Those windows still produced `H2: contradicts / full` and **`H1 supports, belief 0.85–0.86, confidence_tier "high"`** (windows 0 and 1).
   - The repo's own audit (`ANVESH_C006_SPEED_VALIDATION_AUDIT.md`) found the c006 homography's output is 2–28 "units" per image pixel — not metres. The OUTLIER gate fixed the *labels on boxes*; it did not touch the path that matters for cause reasoning.
   Classification: the WORLD `VehicleTrack.speed_estimate` path is **BROKEN / NEEDS REPLACEMENT**; the provenance-aware estimator is **PARTIAL FOUNDATION**. Each track currently carries **two mutually inconsistent speeds** (centroid-based in the track, ground-point-based in the estimator).

3. **In image-space (every non-calibrated camera, incl. the 34-minute demo) cause ranking is structurally degenerate. [measured]** `_stopped_tracks` returns `[]` for `IMAGE` (`evidence_builder.py:112-113`) and H2–H6 are all `NONE` completeness. In all 13 "ranked" demo windows, evidence for H2–H6 is `none` and H1 is `partial`; **13/13 ranked windows are "H1, confidence high"**. That is not an inference — H1 is the only hypothesis that *can* win without a calibrated camera. Meanwhile the *safety* module emits 23 `STOPPED` events from a 5 px/s image-space heuristic (`stopped_vehicle.py:81`) that the *evidence* module refuses to use. Two modules disagree about whether image-space stopped detection is admissible.

4. **Track identity is not reliable. [measured]** ByteTrack IDs are per-run small integers (`tracking.py:66`, `str(int(track_id))`). `TrackManager` finalizes a track after `max_missed_frames=5` *frames* (0.2 s at 25 fps) while ByteTrack's own buffer is longer and can reuse the ID; `_active.get()` then misses and a **new `TrackState` is created with the same ID** (`track_manager.py:77-88,100-101`). Result in the exported artifacts: **196 of 1,598 per-window track entries (12%) in the demo and 24 of 160 (15%) in c006 are duplicate IDs inside a single window.** In the demo, **49% of tracks last <2 s and 43% <1 s**; `total_tracks=1108` for 240 s of video. Any join keyed on `track_id` (candidate vehicle, lane, speed, plate) is unsafe until this is fixed. There is no appearance re-ID, no global/stable vehicle ID, no persistence across runs or cameras.

5. **"Involved vehicle" resolution exists only as "the stopped vehicle named by H2/H3 evidence".** `hypotheses/candidate_vehicle.py:99-148` is carefully designed (RESOLVED / AMBIGUOUS / NO_VEHICLE / NOT_RANKED; never picks "closest/fastest"), and the *cause-vs-association* discipline in design is good. But the only evidence that ever carries a `track_id` is H2/H3, which require `WORLD` speed (finding 2). On the real c006 run the status is `no_vehicle_for_hypothesis` / `not_ranked` in 3/3 windows. There is no event→vehicle resolution for any behaviour event, because there are no behaviour events.

6. **Behaviour intelligence is absent.** No direction-of-travel logic, no wrong-side, wrong-lane, overspeeding, rash driving, sudden-braking, lane-change or accident detection (keyword search over `anvesh/` + `scripts/`: wrong-way/overspeed/accident/lane-change code hits = **0**; only docstrings saying "this is not X"). `H3 collision_like_obstruction` is a *congestion-cause hypothesis* ("multiple stopped vehicles"), **not accident detection**. Lane polygons are *not* wrong-lane detection. The docstring on `perception/lane_assignment.py:26` and `speed_estimation.py:14` explicitly state no lane-change logic exists. (`evidence_builder.py:21-22` still says "V1 perception has no lane geometry at all" — stale since Commit 4.)

7. **The entire back half of the product has zero code.** ANPR, plate detection, OCR, registry lookup, owner data, case/violation records, alerts, SOS, challan, authentication, RBAC, encryption, privacy controls, audit log, Dockerfile/CI: **0 files** (`anpr`=0, `ocr`=0, `registry`=0, `owner`=0, `challan`=0, `alert`=0, `sos`=0, `rbac`=0, `password`=0, `encrypt`=0, `docker`=0). `anvesh/api/__init__.py` is a 3-line docstring stub; `anvesh/main.py` logs "No pipeline modules are implemented yet" and exits.

8. **Validation is synthetic or self-referential.** The evaluation harness's own label says its scenarios are "NOT real-world ground truth" (`evaluation/scenarios.py:18-27`). 606 tests pass in ~6 s **[measured]** because **no test runs a real YOLO model or touches real footage** (`ByteTrackYoloTracker` is monkeypatched to `object()`; `tests/test_export_ui_data.py:255,276,311,336`). There is **no accuracy measurement against any labelled data** for detection, tracking, lane, speed, stopped-vehicle, or ranking — even though the CityFlow dataset at `E:\traffic dataset` ships `gt/`, `mtsc/`, `det/`, and `roi.jpg` that the repo never uses.

9. **Throughput is not the problem; architecture is. [measured]** On the RTX 3050 6 GB laptop GPU, detection+tracking (YOLOv8n + ByteTrack, decode included) ran at **80 FPS on 640×360 (3.2× real time of the 25 fps source)** and **44.6 FPS on 1920×1080 (4.5× real time of the 10 fps source)**. Scope of that number: decode + detect + track + `TrackManager` only — *no* lane assignment, speed, ANPR, evidence, encoding or I/O, one stream, nano model, no device argument passed (`device=` has 0 hits; GPU selection is Ultralytics' default). It says a single stream is feasible; it says nothing about N streams or ANPR.

10. **Operational fragility.** Dependencies are unpinned (`ultralytics>=8.0` → installed 8.4.132, `opencv-python>=4.8` → 5.0.0.93, `numpy>=1.24` → 2.4.6; no lockfile). `ByteTrackYoloTracker.reset()` (`tracking.py:93-98`) **crashes the next `update()` with `TypeError: 'NoneType' object is not subscriptable` on the installed Ultralytics [measured]** — latent only because nothing calls `reset()`. `scripts/serve_presentation.py` serves the **entire repo root (including `.git`, configs, models)** over HTTP with no auth, though bound to `localhost` only (`serve_presentation.py:97`).

**What is genuinely valuable and should be kept.** The repo's *epistemic conventions* are unusually good and map directly onto the vision's safety principle: explicit `UNAVAILABLE` / `AMBIGUOUS` / `OUTLIER` / `NOT_RANKED` / `INSUFFICIENT_EVIDENCE` / `HIGH_CONFLICT` states instead of guessed values; `Measurement(value, error)`; `CalibrationProvenance` + `MetricValidationStatus`; frozen schemas with `schema_version`; deterministic IDs; "never a probability" Dempster-Shafer outputs; abstention as a first-class result; single-camera operation as a first-class path (`pipeline.py:160-162`, `183-190`). These are reusable. The *code on top of them* mostly is not product-grade, and the *claims in some places* (finding 2, `ui/anvesh_dashboard.template.html:1237` "Real-time evidential scoring", `H1 … high` on a 13/13 structural result) are stronger than the evidence.

**Recommended first milestone (detail in §23):** *Build a real-time-capable single-camera "world model" core with trustworthy identity and a validated metric-speed gate*, evaluated against CityFlow ground truth — **before** adding any new behaviour detector, ANPR, or enforcement code.

---

## 2. Current Architecture

### 2.1 Repository shape (working tree)

| Area | Size | Notes |
|---|---|---|
| `anvesh/` | 8,259 LOC, ~50 modules | the package |
| `tests/` | 9,048 LOC, 606 tests | all hermetic; fakes for model/tracker |
| `scripts/` | 3,661 LOC, 15 scripts | demos, exporters, runners, dashboard builders, static server |
| `configs/` | 8 TOML | default, 1 pipeline, 2 validation, 2 lane configs (c006, c009) |
| `docs/` | blueprint (499 lines), `LANE_SPEED_IMPLEMENTATION_NOTES.md` (1232), `M6_5_REAL_WORLD_VALIDATION.md` (152), screenshots, a PDF | |
| root `ANVESH_*.md` | 8 files (untracked except `ANVESH_C006_LANE_GEOMETRY_ITERATION.md`) | prior forensic audits |
| `ui/`, `stitch/` | **untracked** | static dashboard (130 MB embedded-video HTML), design mock-ups |
| `demo.mp4` (88 MB), `traffic.mp4` (32 MB) | **untracked** | 34-min demo video (640×360 @25 fps, 51,201 frames); source/licence not recorded **[not verified]** |

`git ls-files` = 145 tracked files. Git history = 21 commits, 2026-08-29 → 2026-09-14. The whole product-relevant UI, demo videos and data JSON are untracked and so unversioned.

### 2.2 Layered module map (as actually wired)

```
 recorded file ──► perception/ingestion.py (VideoFileFrameSource, cv2, is_file() only)
                      │ Frame(index, timestamp=index/fps)
                      ▼
   perception/tracking.py  ByteTrackYoloTracker  (Ultralytics YOLOv8n COCO + built-in ByteTrack; 5 vehicle classes)
                      │ TrackedDetection(track_id = str(int(bytetrack_id)))
                      ▼
   perception/track_manager.py  TrackManager (in-memory; centroid + ground-point trajectories; VISIBLE/PARTIAL/LOST)
                      │  ── runs for WHOLE FILE; finalize() at EOF ──
                      ▼
   perception/schema_conversion.py  build_vehicle_track → VehicleTrack (IMAGE px/s  or  WORLD "m/s" via homography on CENTROID)
                      │
        ┌─────────────┴─────────────────────────────────────────────────────────────┐
        ▼                                                                           ▼
 perception/windowing.py (fixed 10 s windows)                  perception/lane_assignment.py + world/lane_geometry.py
 perception/traffic_state.py (count, occupancy PROXY,          (point-in-polygon on ground point; per-observation)
   mean speed, hysteresis congestion FSM)                     perception/speed_estimation.py + motion.compute_robust_speed
        │                                                      (provenance-aware APPROXIMATE/OUTLIER/UNAVAILABLE; UI labels only)
        ▼
 anvesh/pipeline.py  run_corridor_window  (ONE window, 1 or 2 cameras)
   evidence/reliability.py → evidence/evidence_builder.py (H1–H6, hard-coded scores)
   → fusion/ds_fusion.py (Dempster-Shafer, ≤2 cameras) → hypotheses/ranking.py (belief/plausibility, abstain)
   → hypotheses/candidate_vehicle.py (top hypothesis → track_id, only if H2/H3 named one)
   → world/corridor_state.py → propagation/shockwave.py (Rankine-Hugoniot) → feedback/update_engine.py
   → safety/stopped_vehicle.py (STOPPED → SafetyEvent)
        ▼
 scripts/export_ui_data.py → ui/data/*.json → static HTML dashboard (ui/*.html, Tailwind, replay)
 storage/db.py (SQLite: 3 tables)   evaluation/ (synthetic scenario harness)
```

### 2.3 Entry points discovered

| Entry point | What it really does |
|---|---|
| `anvesh` console script → `anvesh/main.py:main` (`pyproject.toml:[project.scripts]`) | Loads TOML, configures logging, prints "No pipeline modules are implemented yet", returns 0. **A stub.** |
| `scripts/run_pipeline.py` | M1 demo: YOLO+ByteTrack on one recorded file, optional annotated video |
| `scripts/export_ui_data.py` (`main`) | The **only** full real-video chain; analyses first 240 s of `demo.mp4`, plus 3 hand-authored synthetic scenarios → `ui/data/anvesh_demo_data.json` (24.7 MB) |
| `scripts/run_s02_c006_showcase.py` | Same chain on 30 s of CityFlow S02/c006 with its homography + lane polygons → `ui/data/anvesh_s02_c006_data.json`. Hard-codes `E:\traffic dataset` via lane config `source_video` |
| `scripts/analyze_video.py` | "Judge-demo" CLI: arbitrary video → same chain → self-contained HTML dashboard (`ui/live/`) |
| `scripts/run_real_world_validation.py` | M6: 2 Intel sample videos, **not** a real corridor (config says so) |
| `scripts/run_evaluation.py` | Synthetic scenario comparison, Baselines A/B/C vs ANVESH |
| `scripts/run_*_demo.py` (5) | Synthetic demos of world-alignment, corridor state, fusion, feedback, stopped vehicle, end-to-end |
| `scripts/serve_presentation.py` | Range-capable static server over the whole repo root, `localhost:8000` |
| `scripts/build_ui_artifact.py`, `build_offline_assets.py`, `rebuild_*_dashboard*.py` | Template substitution / vendoring for the dashboard |

### 2.4 Data models, storage, config
- **Data models:** `storage/schemas.py` (433 lines) — frozen Part-5 dataclasses: `Camera`, `CameraObservation`, `VehicleTrack`, `TrafficState`, `Evidence`, `CauseHypothesis`, `CandidateCauseHypothesis`, `PropagationPrediction/Observation`, `HypothesisUpdate`, `CorridorState`, `EvaluationRecord`, `SafetyEvent`. No Vehicle, Event, Incident, Case, Plate, User, Alert, AuditEntry entity.
- **Storage:** SQLite, **three tables only** — `corridor_state_records` (`db.py:48`), `ranking_records` (`db.py:235`), `feedback_records` (`db.py:358`), one JSON payload per row. `VehicleTrack`, `CameraObservation`, `SafetyEvent`, `EvidenceItem` and lane/speed results are **not persisted anywhere**. `*.db`/`*.sqlite3` are gitignored; `outputs/corridor_state_demo.sqlite3` (24 KB) is a demo leftover.
- **Config:** TOML via `tomllib` (`config.py`, `perception/config.py`, `world/lane_config.py`). Thresholds are code defaults and per-experiment TOML. No env/secret handling; no runtime config reload.
- **ML models:** one: `models/yolov8n.pt` (COCO-pretrained, gitignored). `COCO_TO_VEHICLE_CLASS` = bicycle, car, motorcycle, bus, truck (`detection.py:19-25`). `VehicleClass.AUTO_RICKSHAW` exists in the schema but **can never be produced** by this detector. No fine-tuning, no ReID model, no plate model, no OCR model.
- **Datasets:** Intel IoT sample videos (CC BY 4.0, untracked mp4), CityFlowV2/AIC22 at `E:\traffic dataset` (external; has `gt/`, `mtsc/`, `det/`, `roi.jpg`, `cam_timestamp`; licence PDF present — **check before any commercial use [not verified]**), `demo.mp4`, `traffic.mp4`.
- **Frontend:** a single 116 KB HTML template (`ui/anvesh_dashboard.template.html`) with embedded precomputed JSON; modes "Real Video Replay", "V1 Dataset Showcase", "Synthetic Demonstration". The UI labels itself "Precomputed" and "Informational only — not dispatch or enforcement automation" (honest), except `:1237` "Real-time evidential scoring". `stitch/` holds six design mock-ups ("Forensic Precision Terminal" design system) — design only; the only product-feature strings in them are an "ALERTS" nav label and the same disclaimer.
- **Deployment:** none. No Dockerfile, compose, CI, `.github`, requirements/lock file, service definition, health endpoint, metrics.

---

## 3. Current Runtime / Data Flow

**Actual flow, one run of `export_ui_data.real_video_limited_chain` (`scripts/export_ui_data.py:547-787`):**

1. `cv2.VideoCapture` probe → fps/size/frame count (`:623-630`). Optionally trims the file to a temp copy (`_write_trimmed_copy`).
2. `ByteTrackYoloTracker(...)` constructed (`:657-662`). `run_m1_pipeline` iterates **every frame of the file synchronously on one thread**: decode → YOLO+ByteTrack (`model.track(..., persist=True)`) → `TrackManager.update` → `CameraObservation` per frame appended to an unbounded list (`perception/pipeline.py:76-93`).
3. At EOF: `track_manager.finalize()`; `VehicleTrack`s are built from all `TrackState`s (`pipeline.py:112-117`). **Nothing downstream of tracking can run until here.**
4. Per-track `estimate_vehicle_speed` is run once on finished trajectories (`:714`); lane assignment per captured box (`:716-730`).
5. Fixed 10 s windows (`generate_windows`); per window: `aggregate_traffic_state` → `run_corridor_window` (reliability → evidence → fusion → ranking → candidate vehicle → stopped-vehicle events) (`:753-770`). `CongestionStateTracker` is the only cross-window state.
6. Everything serialised to one JSON (24.7 MB for 240 s) with `"perception_mode": "precomputed_replay"`; the dashboard `fetch()`es it and syncs to a `<video>` element.

**Consequences (all verified in code):** event latency = file length; memory grows with video length (all frames' observations and trajectories retained); a crash loses everything; one stream per process; no back-pressure, no frame dropping, no reconnection, no clock sync (CityFlow's own README warns frames are skipped/stuck and timestamps are not aligned — `index/fps` ignores that, which biases any speed derived from it); the `corridor_topology`/`segment_length_m`/`propagation_observation` for the 2-camera path are always **caller-supplied or synthetic** (`pipeline.py:20-34`) — no real two-camera run exists in the repo.

**Persistence/failure handling:** `CorridorStateStore`/`RankingStore`/`FeedbackRecordStore` are called only from synthetic demos (`run_corridor_state_demo.py`, tests); the real-video path writes JSON files, not the DB. Failure handling is limited to fail-fast validation (`FileNotFoundError`, `RuntimeError("Failed to open video")`, `ValueError`s) — no retry, supervision, checkpointing, or degraded modes.

---

## 4. Capability Matrix

Columns: **I**mplemented / **T**ested (unit) / **V**alidated (against labelled real data) / **D**eployment-ready. `Y` / `part` / `N`.

| # | Vision capability | Classification | I / T / V / D | Evidence (working-tree) |
|---|---|---|---|---|
| 1 | Vehicle detection | **IMPLEMENTED BUT INCOMPLETE** | Y / part / N / N | `detection.py`, `tracking.py:71-108`; COCO YOLOv8n, 5 classes, no auto-rickshaw, no night/rain/plate-size data; tests use fakes (`tests/perception/test_detection.py`, `test_tracking.py`); no mAP/recall anywhere |
| 2 | Multi-object tracking | **IMPLEMENTED BUT INCOMPLETE** (identity defect) | Y / part / N / N | ByteTrack via Ultralytics; **12–15% duplicate IDs/window, 49% tracks <2 s [measured]**; `track_manager.py:77-101`; no re-ID |
| 3 | Persistent vehicle state | **PARTIAL FOUNDATION** | part / part / N / N | `TrackState` in RAM only; per-camera; no vehicle entity/table; lost on exit |
| 4a | Lane **assignment** (given polygons) | **IMPLEMENTED + TESTED** (primitive only) | Y / Y / N / N | `world/lane_geometry.py:116-168` deterministic point-in-polygon with AMBIGUOUS/UNKNOWN states; `tests/world/test_lane_geometry.py`, `tests/perception/test_lane_assignment.py` |
| 4b | Lane **detection** / robust lane geometry | **PARTIAL FOUNDATION** | N / N / N / N | Hand-authored image-space polygons for **2 cameras** (`configs/lanes/s02_c006.toml` L1/L2 only, L2 "PROVISIONAL"; `s02_c009.toml`). In the c006 run **3,126 of 3,904 boxes (80%) are `unknown_outside_regions`** [measured]; far-side carriageway not modelled; no automatic lane/road-structure learning; ground-point polygons break under occlusion/camera shake |
| 5 | Direction detection | **MISSING** | N / N / N / N | No heading/flow-direction code; lane regions carry no travel direction attribute (`LaneRegion` = `lane_id` + `vertices`) |
| 6a | Real **metric** speed — validated path | **MISSING** | N / N / N / N | `MANUAL_SURVEY` homography fit exists (`calibration.py:178-225`) and is auto-stamped `VALIDATED` "by construction" (`:224`) from *whatever points the caller supplied* — no hold-out check, no reprojection threshold, no ground-truth speed test, no real survey ever performed |
| 6b | Provenance-aware speed estimator | **PARTIAL FOUNDATION** | Y / Y / N / N | `speed_estimation.py` (uncommitted gate `:119,262-270`); right *idea* (UNAVAILABLE/APPROXIMATE/OUTLIER; never `VALIDATED` for CityFlow); c006: 66 outlier / 34 unavailable / 7 approximate of 107 tracks (**6.5% usable**) [measured]; used only for UI labels |
| 6c | WORLD `VehicleTrack.speed_estimate` feeding reasoning | **BROKEN / NEEDS REPLACEMENT** | Y / Y / N / N | See finding 2: centroid, unit-unvalidated, ungated, mislabelled m/s, consumed by congestion/H2/H3/reliability |
| 7 | Wrong-side driving | **MISSING** | N | no code |
| 8 | Wrong-lane driving | **MISSING** | N | polygons ≠ rule violation; no lane-permission model, no turn-pocket semantics |
| 9 | Overspeeding | **MISSING** | N | no limit model; no validated speed to compare against |
| 10 | Rash / dangerous driving | **MISSING** | N | no lane-change, headway, jerk, or weaving features |
| 11 | Sudden braking / abnormal manoeuvre | **MISSING** | N | no acceleration estimation |
| 12 | Stopped-vehicle detection | **IMPLEMENTED BUT INCOMPLETE** | Y / Y / N / N | `safety/stopped_vehicle.py:235`; dwell+coverage logic is thoughtful; thresholds "illustrative… not derived" (`:70`); image-space 5 px/s is not physical; no context (red light, queue, parking, lane/shoulder); 23 events on demo with no ground truth; second, *different* stopped definition in `evidence_builder.py:111-120` |
| 13 | Traffic-jam / congestion detection | **IMPLEMENTED BUT INCOMPLETE** | Y / Y / N / N | `traffic_state.py`: FSM with hysteresis is sound; but occupancy = `mean_concurrency / max_capacity_vehicles` with an arbitrary constant 20 (`:257`, `export_ui_data.py:503`); speed branch consumes unit-invalid WORLD speeds on c006; whole-track speed (not in-window speed) is attributed to a window (`:251`) |
| 14 | Accident detection | **MISSING** | N | `H3 collision_like_obstruction` is a congestion-cause label, not detection |
| 15 | Involved/contributing vehicle identification | **PARTIAL FOUNDATION** | part / Y / N / N | `candidate_vehicle.py:99-148` — stopped-vehicle-only, via H2/H3; resolved 0/3 windows on real c006; correct *status vocabulary* |
| 16 | Evidence-based investigation | **PARTIAL FOUNDATION** | part / Y / N / N | `EvidenceItem` + DS fusion rationale; **no frames/clips/crops/hash/retention**, no per-event timeline, evidence not persisted |
| 17 | Candidate-cause reasoning (H1–H6) | **DEMO/SYNTHETIC ONLY** | Y / Y / N / N | Works on hand-authored WORLD evidence; on real video degenerates to H1 (finding 3); score constants hard-coded (0.8 / 0.75 / 0.15 / 0.05 …, `evidence_builder.py:237-283`) and never calibrated; H4/H5/H6 always `NONE` on real video |
| 18 | Cross-camera reasoning | **DEMO/SYNTHETIC ONLY** | Y / Y / N / N | `fuse_two_cameras`, `world/corridor.py`, `fusion/alignment.py`: hard-capped at 1–2 cameras (`pipeline.py:160-162`); topology & clock offset caller-supplied; **no real multi-camera run**, no cross-camera vehicle re-ID (blueprint defers it, Part 4) |
| 19 | Congestion propagation / impact prediction | **DEMO/SYNTHETIC ONLY** | Y / Y / N / N | `propagation/shockwave.py` Rankine-Hugoniot between two TrafficStates; only ever fed synthetic WORLD data; feedback loop requires a caller-supplied `PropagationObservation` |
| 20 | License-plate detection | **MISSING** | N | 0 code |
| 21 | ANPR / OCR | **MISSING** | N | 0 code |
| 22 | Authorized registry lookup | **NOT APPLICABLE YET** | N | needs ANPR + legal authorization + an authorized integration contract first |
| 23 | Authorized owner/registration information | **NOT APPLICABLE YET** | N | as above + data-protection regime; no data model |
| 24 | Incident / violation case generation | **MISSING** | N | `SafetyEvent` is a 7-field record (`schemas.py:511-533`), `event_type="stopped_vehicle"` only, **not persisted**; no case lifecycle, status, assignee, evidence bundle |
| 25 | Live authority alerts | **MISSING** | N | 0 code; UI explicitly "informational only" |
| 26 | Accident / SOS escalation | **NOT APPLICABLE YET** | N | no accident detector, no integration |
| 27 | Challan / enforcement workflow | **NOT APPLICABLE YET** | N | no validated violation detectors, no plates, no legal basis wired |
| 28 | Continuous monitoring (while cases reviewed) | **MISSING** | N | batch-to-EOF architecture; no concurrent event handling |
| 29 | Authentication / RBAC | **MISSING** | N | blueprint defers; no user model; static server unauthenticated |
| 30 | Auditability / evidence provenance | **PARTIAL FOUNDATION** | part / part / N / N | Good: deterministic IDs (`pipeline.py:105-122`), `model_id/version` on observations, `CalibrationProvenance`, `schema_version`. Missing: append-only/tamper-evident log, user/action trail, artifact hashes, config+model version pinning per decision, retention |
| 31 | Privacy / security controls | **MISSING** | N | no redaction, no encryption, no access control, no retention policy, no PII classification; whole-repo static server |
| 32 | Production deployment | **MISSING** | N | no container/CI/lock/monitoring/health; unpinned deps; Windows+local-path assumptions (`E:\…`) |
| — | Live stream ingestion (RTSP/RTMP/webcam) | **MISSING** | N | `ingestion.py:37-38` `is_file()` only |
| — | State persistence for the real pipeline | **PARTIAL FOUNDATION** | part / Y / N / N | SQLite stores for 3 synthetic-fed entities only |
| — | Dashboard / operator UI | **DEMO/SYNTHETIC ONLY** | Y / N / N / N | static replay of precomputed JSON; no backend; no UI tests |

**Tally (32 numbered capabilities; rows 4 and 6 are counted once each by their *weakest* sub-row):** IMPLEMENTED + TESTED: 0 · IMPLEMENTED BUT INCOMPLETE: 4 (1, 2, 12, 13) · DEMO/SYNTHETIC ONLY: 3 (17, 18, 19) · PARTIAL FOUNDATION: 5 (3, 4, 15, 16, 30) · BROKEN / NEEDS REPLACEMENT: 1 (6) · MISSING: 15 (5, 7–11, 14, 20, 21, 24, 25, 28, 29, 31, 32) · NOT APPLICABLE YET: 4 (22, 23, 26, 27). Total 32. Row 4 splits into 4a (`IMPLEMENTED + TESTED`, primitive only) and 4b (partial); row 6 into 6a (missing), 6b (partial foundation), 6c (broken). The only `IMPLEMENTED + TESTED` item anywhere is the lane point-in-polygon primitive — and it is tested, not validated.

---

## 5. What Is Actually Production-Ready

**Nothing, as a product.** No component is validated against real labelled data, deployed, monitored, secured, or run live. What *is* reusable is a set of **tested building blocks and conventions**. I separate them by how much I would trust them going forward:

| Reuse as-is (sound, small, tested) | Reuse with rework | Do not carry forward as-is |
|---|---|---|
| `world/lane_geometry.assign_lane` — deterministic, order-independent, explicit AMBIGUOUS/UNKNOWN/INVALID states (`lane_geometry.py:116-168`) | `perception/track_manager.py` — lifecycle logic is fine; the *ID-reuse/finalize* semantics are wrong (finding 4) | `schema_conversion.build_vehicle_track` WORLD branch (centroid + unvalidated units, `:81-85`) |
| `perception/motion.compute_robust_speed` — time-weighted, explicit-status, gap-excluding (`motion.py:214-308`) | `perception/speed_estimation.py` — good provenance model; needs a validated-metric precondition, not a plausibility ceiling, as its gate | The `WORLD == is_calibrated` coupling (`export_ui_data.py:748`) |
| `world/calibration.fit_planar_homography` + `CalibrationProfile` invariants (`calibration.py`) | `world/calibration` — `VALIDATED` must be *earned* (hold-out reprojection, ground-truth distance check), not asserted at construction (`:224`) | Hard-coded evidence scores (`evidence_builder.py:237-283`) |
| `fusion/ds_fusion.py` — textbook Dempster-Shafer, discounting, conflict handling (38 tests) | `traffic_state.py` — FSM keep; replace occupancy proxy and whole-track speed attribution | The H1–H6 *closed* hypothesis set as the product's reasoning core (it is a congestion-cause research model) |
| `perception/traffic_state.CongestionStateTracker` hysteresis FSM | `safety/stopped_vehicle.py` — keep dwell/coverage logic; add context + unify with evidence's stopped test | `perception/pipeline.run_m1_pipeline` batch loop |
| `storage/schemas.py` validation style, enums, `schema_version` | `storage/db.py` — pattern fine for a prototype; wrong substrate for multi-writer live ops | `ui/` replay dashboard as an operator surface |
| Epistemic vocabulary: `UNAVAILABLE`, `AMBIGUOUS`, `OUTLIER`, `NOT_RANKED`, `INSUFFICIENT_EVIDENCE`, `HIGH_CONFLICT`, `Measurement(value,error)` | `hypotheses/candidate_vehicle.py` — right shape; needs the observed/involved/contributor/causal ladder and temporal/spatial tests | |

---

## 6. What Is Only Demo / Synthetic

- **All multi-hypothesis, multi-camera, propagation and feedback behaviour.** Exercised only on hand-authored WORLD-space tracks (`export_ui_data.py:268-300`, `scenario_*` at `:321-445`) and `world/synthetic_fixture.py`. The H5 capacity of 1.6 veh/s is *stipulated* to fit the scenario (`export_ui_data.py:83`).
- **The evaluation harness** (`anvesh/evaluation/`, `scripts/run_evaluation.py`): scenario "design labels" are authored by the same team that wrote the engine; the file says so (`scenarios.py:18-27`). It demonstrates mechanism, not accuracy.
- **The "real" runs.** `demo.mp4` (240 s of 2,048 s analysed), c006 (30 s of 211 s, 3 windows), two Intel clips explicitly *not* a corridor (`datasets/recorded/validation/SOURCE.md`, `configs/experiment/m6_validation.toml:1-8`). No two-camera *real* run exists; the c006+c009 pair exists only as analysis prose (`ANVESH_S02_CAMERA_PAIR_ANALYSIS.md`).
- **The dashboard.** Replays precomputed JSON against a `<video>`; there is no backend, no API, no live data path. It is honest about this in most places (`"Precomputed"`, `"Informational only"`), but `ui/anvesh_dashboard.template.html:1237` says "Real-time evidential scoring", and the cause-ranking panel presents `H1 · high` on structurally-forced results.
- **Reliability scoring** (`evidence/reliability.py`): four equally-weighted (0.25) heuristic factors, never calibrated against anything; docstring says so.

---

## 7. What Is Partially Implemented

| Item | What exists | What is missing to be usable |
|---|---|---|
| Lane geometry | `LaneRegion`, TOML loader, per-observation assignment, c006 L1/L2 (data-fitted from 30 s of tracks) | Cover the full carriageway (80% of c006 boxes outside any lane), far-side carriageway, per-lane *direction & permitted-movement* attributes, robustness to shake/occlusion, a way to author/validate polygons that is not "fit to the tracker's own output" (`s02_c006.toml:40-90` — polygons were derived from ByteTrack ground points, so lane assignment is partly circular as a validation) |
| Speed | Robust estimator + provenance enum + plausibility ceiling | A validated metric calibration (survey or known-length marking), per-class plausibility, error propagation from detector jitter, speed from the *same* ground-point trajectory everywhere, removal of the centroid path |
| Candidate vehicle | Status vocabulary, "never guess" rules, tests | Any behaviour event to resolve against; temporal precedence & spatial tests; multi-vehicle involvement; cross-camera association |
| Stopped vehicle | Dwell/coverage logic, schema event | Context (signals, queues, parking, shoulder), physical thresholds, ground truth, unified definition with evidence layer |
| Persistence | 3 SQLite stores + JSON round-trip tests | Tracks, observations, events, evidence artifacts, cases; multi-writer store; migrations |
| Provenance | `CalibrationProvenance`, `model_id/model_version`, deterministic IDs | Per-decision record of {model hash, config hash, calibration id, code version, input clip hash}; append-only log |
| Propagation + feedback | Rankine-Hugoniot between two states; 4-branch feedback | Real two-camera data; real observed propagation; not a prerequisite for single-camera product value |

---

## 8. Missing Product Capabilities (dependency-ordered)

**Foundation (blocks everything):** live frame ingestion (RTSP/RTMP/file-as-stream) · incremental (streaming) track/state engine · stable vehicle identity · validated per-camera calibration · persistent event/track/evidence store · service API · configuration & model registry.

**Perception/behaviour:** direction-of-travel · wrong-side · wrong-lane/turn-restriction · overspeed · rash driving (weaving, tailgating, jerk) · sudden braking · lane-change · accident detection · per-vehicle persistent state (speed/lane/heading history) · automatic anomaly signals.

**Product workflow:** involved/contributing-vehicle resolution for *each* event type · case/incident entity & lifecycle · evidence bundle capture · plate detection · ANPR/OCR with multi-frame voting · authorized registry integration · owner-data handling · alerting · SOS/escalation integration · challan/enforcement workflow · review queue/human-in-the-loop · continuous monitoring concurrent with open cases.

**Platform:** authentication · RBAC · audit log · privacy/redaction · encryption · retention · deployment (container, CI, config mgmt, secrets) · observability (metrics, health, alerting on the system itself) · load/soak testing · failure recovery.

---

## 9. Architecture Problems / Technical Debt

1. **Batch-to-EOF core** (`perception/pipeline.py:76-117`): tracks only become `VehicleTrack`s at end of file; windows, evidence and events all depend on that. The fix is architectural, not incremental.
2. **Two speed paths, two stopped-vehicle definitions, two notions of "calibrated".** (a) `VehicleTrack.speed_estimate` (centroid, mean of per-step speeds, `motion.py:131-148`) vs `VehicleSpeedEstimate` (ground point, robust, `speed_estimation.py`); (b) `evidence_builder._stopped_tracks` (lifetime-mean speed ≤ 1 m/s & age ≥ 3 s, WORLD only) vs `StoppedVehicleClassification` (per-step dwell, image or world); (c) `is_calibrated` (a matrix exists) vs `metric_validation` (the units are confirmed). Only (c) is modelled correctly, and then ignored where it matters.
3. **Lifetime-average speed attributed to every window the track overlaps** (`traffic_state.py:251`). A vehicle that stopped for 5 s of a 60 s track is "fast"; a window sees none of that. `stopped_vehicle.py`'s own docstring (`:18-29`) explains exactly why this is wrong — and the evidence layer still uses it.
4. **`MotionSpace.WORLD` is a type-level promise ("metres") granted on `is_calibrated`.** The schema comment says WORLD means metres (`schemas.py:28-35`). CityFlow profiles have `units="unknown"`. The type system cannot express "world frame, scale unvalidated".
5. **Identity/finalization mismatch.** `TrackManager` has its own occlusion clock (`max_missed_frames=5` frames) layered over ByteTrack's own buffer (Ultralytics' default `bytetrack.yaml`, not in the repo **[not verified]**). Frame-count-based clocks also change meaning with fps (0.2 s @25 fps, 0.5 s @10 fps).
6. **Fixed, non-overlapping 10 s windows** (`export_ui_data.py:72`, `windowing.py:41-61`): event timing resolution is 10 s; an event straddling a boundary is split.
7. **Hard-coded, un-calibrated constants masquerading as evidence:** score table (0.8/0.75/0.2/0.15/0.1/0.05) in `evidence_builder.py`, `max_capacity_vehicles=20` (`export_ui_data.py:503`), congestion speed thresholds, `MAX_PLAUSIBLE_SPEED_KMH=200`. The M6.5 note (`m6_real_camera_thresholds.toml`) is admirably honest about this; the *consequence* (confidence tiers still render "high") is not.
8. **Frozen-schema discipline forced "companion objects"** for lane, speed, candidate, classification (`lane_assignment.py:13-17`, `speed_estimation.py:62-66`). There is therefore **no single per-vehicle world-model object**; each result is joined by string `track_id` — which is not unique (finding 4).
9. **Application layer lives in a script.** `scripts/export_ui_data.py` (705 lines) is the de-facto orchestrator + serialiser + synthetic-scenario author; its tests monkeypatch the tracker. `anvesh/pipeline.py` is explicitly "one window at a time, not a streaming engine" (`pipeline.py:11-14`).
10. **Unbounded memory:** `result.camera_observations` and per-track trajectories grow with file length (`pipeline.py:82-93`); one full-chain run holds all frames' boxes (`export_ui_data.py:673-688`).
11. **Environment coupling:** absolute `E:\traffic dataset\…` path in `configs/lanes/s02_c006.toml` (`source_video`), consumed by `run_s02_c006_showcase._calibration_path_for`; gitignored model/data files mean a fresh clone cannot run anything beyond unit tests without manual downloads.
12. **Unversioned deliverables.** `ui/`, `stitch/`, the exported JSON, `demo.mp4`, `traffic.mp4`, 7 audit docs, 3 dashboard builders and `tests/perception/test_speed_evidence_safety.py` are untracked or uncommitted; the OUTLIER gate exists only in the working tree. `ui/anvesh_dashboard.html` is a 130 MB single file with the video base64-embedded.
13. **Documentation drift.** `README.md` says "M0 → M6.5 complete" and shows an M0 layout, while the repo is at M8 + Commit 9; `anvesh/__init__.py` docstring says "M0 foundation only"; `anvesh/main.py` says no pipeline exists; `evidence_builder.py:21-22` says no lane geometry exists. Duplicate builders `rebuild_dashboard_live.py` / `rebuild_live_dashboard.py`.
14. **Dependency hygiene:** unpinned, brand-new majors (cv2 5.0.0.93, numpy 2.4.6, torch 2.13, ultralytics 8.4.132); `ByteTrackYoloTracker.reset()` already broken against them (measured). `requires-python >=3.11`, installed 3.11.9.
15. **Safety guarantee tested at the wrong layer.** `test_speed_evidence_safety.py` AST-checks imports of five reasoning modules; it cannot see data flowing in via `VehicleTrack`. It gives false assurance (finding 2).

---

## 10. ML / Computer-Vision Gaps

- **Detector:** YOLOv8n COCO-pretrained, conf 0.25 (`m1_pipeline.toml`, `detection.py`). No domain fine-tuning, no mAP/recall/class-confusion measurement, no per-class thresholds, no night/rain/glare/dust data, nothing for auto-rickshaws (class exists but is unreachable — significant if the target roads are Indian, as terms like *challan* suggest **[inference, not from the repo]**). Small, distant vehicles at 640×360 (demo) are a known weak point of a nano model; not measured here.
- **Tracking:** stock ByteTrack, motion-only association, no appearance features. Identity defects measured (finding 4). No MOT metrics (IDF1/HOTA/ID-switches) despite CityFlow `mtsc/` & `gt/` being on disk.
- **Calibration/geometry:** manual planar homography only. No auto-calibration, no vanishing-point/lane-width/known-length priors, no distortion correction applied (`parse_cityflow_calibration_text` *reads* intrinsics/distortion when present but `from_cityflow_homography` ignores them; c006's file has neither). The c006 homography's own `Reprojection error: 10.25` is stated in **pixels** by the dataset README but stored in a field named `reprojection_error_m` (`calibration.py:413`) — a latent unit mislabel. Ground contact is the bbox bottom-centre (`types.py:104-105`), biased by occlusion and viewing angle; no per-class/per-pose correction, no 3D box.
- **Camera-motion/synchronisation:** no stabilisation, no detection of camera move/defocus/obstruction. `timestamp = index/fps` ignores dropped/stuck frames that the dataset README documents.
- **Lane/road understanding:** none learned. Static hand polygons; no drivable-area, no lane-direction, no stop-line, no signal-state, no road-user-specific rules.
- **Behaviour models:** none (direction, lane-change, braking, weaving, near-miss, collision). Nothing to build on except trajectories.
- **Re-ID / cross-camera association:** none (deliberately deferred by blueprint Part 4).
- **Plate detector / OCR:** none; and see §13 on resolution feasibility.
- **Performance engineering:** no ONNX/TensorRT/FP16, no batching across streams, no explicit device selection, no warm-up/steady-state benchmark in-repo. **[measured, scratch script]:** 12.1 ms/frame YOLO+ByteTrack at 640×360, 15.3 ms at 1920×1080 (plus 7.1 ms decode) — i.e. one nano-model stream uses a small fraction of the GPU; a larger detector + ANPR crops + multiple streams will not.
- **Ground-truth evaluation data in hand but unused:** CityFlow `gt/`, `mtsc/`, `det/`, `roi.jpg`, `cam_timestamp`. (A speed ground truth is *not* part of what the README describes; validating metric speed needs a dataset with measured speeds or an on-site survey **[not in repo]**.)

---

## 11. Real-Time / Streaming Gaps

| Requirement | Today | Gap |
|---|---|---|
| Live source | `VideoFileFrameSource` only; `is_file()` guard (`ingestion.py:37-38`) | RTSP/RTMP/USB/NVDEC ingest, reconnect, jitter buffer |
| Timebase | `index/fps` from t=0 | Wall-clock/PTS capture time, NTP/PTP sync, per-frame skew handling |
| Incremental state | tracks finalised at EOF (`pipeline.py:112`) | Streaming track store; incremental windows; sliding (overlapping) windows |
| Concurrency | single thread, single process | Decode ‖ inference ‖ analytics ‖ I/O pipeline; per-camera workers; event handlers independent of the perception loop |
| Back-pressure/overload | none | Frame dropping policy, queue bounds, load shedding with *logged* degradation |
| Memory | grows with duration | Ring buffers, bounded history, spill-to-store |
| Multi-stream | one tracker object per process | Batched multi-stream inference, GPU scheduling |
| Event latency | file length | Target: seconds from occurrence to event (see §19) |
| Concurrent events while monitoring | impossible (batch) | Event bus + case manager running beside the perception loop |
| Failure handling | fail-fast `ValueError`/`RuntimeError`; crash = lose run | Supervision, checkpointing, camera health states, degraded-mode flags carried into evidence reliability |
| GPU | implicit Ultralytics default; `cuda_available` reported in JSON | explicit device/precision config; telemetry |
| Measured single-stream throughput | **80 FPS @640×360 (3.2× RT@25), 44.6 FPS @1080p (4.5× RT@10)** *(decode+YOLOv8n+ByteTrack+TrackManager, RTX 3050 6 GB laptop; measured by me, nothing else in the loop)* | Not representative of the full product loop; re-measure per stage as ANPR, lane/speed, and storage are added |

---

## 12. Vehicle Identity & Event-Resolution Gaps

**Identity layers the product needs (and what exists):**

| Layer | Needed | Exists |
|---|---|---|
| Detection | per-frame box+class | yes |
| Tracklet | short continuous track | yes, but ID reuse produces duplicate keys (finding 4) |
| Vehicle | stitched tracklets across occlusion/gaps in one camera | **no** |
| Cross-camera vehicle | re-identified across cameras | **no** (deferred) |
| Plate identity | ANPR read with confidence + voting | **no** |
| Registered identity | registry record | **no** |

**Specific defects:** track IDs are `str(int(bytetrack_id))` (`tracking.py:66`), unique only within one run and one process; they are camera-prefixed at `VehicleTrack` build (`schema_conversion`: `f"{camera_id}:{id}"`) but **not run-prefixed**, so two runs collide. Duplicates within a window (measured). `TrackManager` PARTIAL/LOST states are derived from *tracker output absence*, not from true occlusion (`stopped_vehicle.py:30-38` documents that LOST often means "detector dipped").

**Event→vehicle resolution:** only `candidate_vehicle.resolve_candidate_vehicle` (`:99-148`), which returns the `track_id`s named by the top hypothesis' evidence — i.e. only H2/H3 "stopped" tracks. Good rules (never guess; abstain on ambiguity). Gaps: (a) it cannot resolve for any event other than "stopped"; (b) no temporal-precedence test (did the stoppage *precede* the queue?), no spatial test (is the vehicle at the queue head?) — the blueprint lists these as belief-raising evidence (`cause_model.py:83`, "confirmed at the queue head in a later window") but they are **not implemented**; (c) no notion of *multiple* involved/contributing vehicles per event; (d) status names (`RESOLVED`) are easy to misread downstream as "responsible".

**Observed / involved / contributor / causal.** The vision's four-rung ladder is *not* encoded anywhere. The nearest structure is `CandidateVehicleStatus` + DS `belief/plausibility`. The *design intent* (docstrings: "never confirmed responsible vehicle") is correct; the *data model* needs an explicit `AttributionLevel` enum and rules for what evidence may promote a vehicle from rung to rung — with `causally established` effectively unreachable from video alone.

**Congestion responsibility.** Cannot currently be established. H1/H5/H6 never name a vehicle; H2/H3 name a stopped vehicle by speed threshold only. There is no queue-head localisation, no onset timing, no counterfactual or flow-continuity analysis. This is correct *restraint*, but it also means the "contributing vehicle" capability (#15) is effectively absent for congestion.

---

## 13. ANPR / Registry / Owner-Data Gaps

- **Code:** none — no plate detector, OCR, voting, plate-quality score, plate store, registry client, owner model, or lookup audit. (`plate` grep hits are unrelated English words, e.g. "template".)
- **Input feasibility is unproven and likely poor for the current footage:** `demo.mp4` is **640×360** and its detected *vehicle* boxes have a **median width of 28 px (p10 16, p90 74; n=71,593)** [measured from the export]; a licence plate is a fraction of that, i.e. a handful of pixels. c006 is 1920×1080 but vehicle boxes are **median 73 px wide (p10 47, p90 186; n=3,904)** [measured], putting a plate at roughly 10–15 px — also below what OCR can read **[the plate-to-vehicle ratio is my estimate]**. Plate legibility depends on pixels-on-plate, angle, shutter, IR/night; ANPR normally needs dedicated geometry/optics. A plate-readability study on real crops must precede any ANPR commitment.
- **Registry/owner:** would require a lawful, authorised integration, purpose-limited access, encrypted storage, field-level RBAC, per-lookup audit, retention limits, and a data-protection assessment — none of which have any representation in the repo. Classification: `NOT APPLICABLE YET`.
- **Safety coupling:** plate reads are probabilistic; a misread plate mapped to a registry owner is an accusation against the wrong person. The pipeline must carry plate-read confidence and character-level uncertainty all the way to the enforcement gate, and must require independent corroboration before any automated action.

---

## 14. Investigation & Evidence Gaps

- **No evidence artifacts.** Evidence is a record of *numbers* (`EvidenceItem`: score, direction, `supporting_track_refs`), not of *media*: no stored frames, crops, clips, pre/post-roll, plate images, calibration snapshot, or model hash. `Evidence` references tracks that are not persisted.
- **No event/case timeline.** `SafetyEvent` has `window` (10 s, quantised), `track_id`, `confidence_tier` — no start/end/peak time, location, lane, speed, vehicle class, severity, status, reviewer, or related-evidence list (`schemas.py:511-533`).
- **Explainability is DS-structural only.** `supporting_evidence_refs`/`contradicting_evidence_refs` exist; but the *scores* behind them are hard-coded constants, so the explanation is faithful to a model that was never fit.
- **Confidence → action mapping missing.** `ConfidenceTier` {low,medium,high} derives from belief thresholds with no calibration against outcomes; there is no defined mapping from tier to {discard, review, alert, enforce}, and no measured false-positive/false-negative rates to set one.
- **Chain of custody / integrity:** no hashing/signing of clips or records, no immutable log, no clock-provenance.
- **Counter-evidence & abstention are first-class (a strength)** — `HIGH_CONFLICT`, `INSUFFICIENT_EVIDENCE`, `contradicting` — but are surfaced only as labels in a replay UI, with no reviewer workflow to act on them.

---

## 15. Alert / SOS Gaps

- **Code:** none. No alert entity, severity model, routing, deduplication/suppression, acknowledgement, escalation timers, delivery channel (SMS/push/webhook/control-room integration), or retry/idempotency. The dashboard states "Informational only — not dispatch or enforcement automation" (`ui/anvesh_dashboard.template.html:537,1450`), which is accurate.
- **SOS/accident escalation** has two missing prerequisites: (a) an **accident detector** (does not exist; `H3` is not one), and (b) an **authorised integration contract** with an emergency/control-room system. Until (a) has measured precision/recall on real accident footage, wiring (b) would convert false alarms into real-world dispatches.
- **Needed design properties** (for later): severity × certified-capability gating, per-camera suppression windows, human acknowledgement tracking, escalation on non-acknowledgement, "system degraded" alerts about ANVESH itself, and a kill-switch.

## 16. Enforcement / Challan Gaps

- **Code:** none (no challan record, no rule→offence mapping, no evidence-pack builder, no submission, no appeal/dispute state).
- **Prerequisites missing, in order:** certified metric speed (6a) → certified violation detector(s) (7–11) → vehicle identity continuity (2/3) → plate read (20/21) → authorised registry (22/23) → evidence bundle (§14) → case workflow (24) → legal authority/jurisdiction rules.
- **Proposed automation tiers** (policy, not code today): **T0** observe/log · **T1** hold for human review · **T2** alert authority · **T3** automated enforcement. The vision's rule — *high-confidence, independently validated cases may auto-proceed; uncertain → review; contradictory/insufficient → never enforce* — needs a machine-checkable gate: `capability_certified(camera, capability)` ∧ `confidence ≥ threshold` ∧ `independent_corroboration` ∧ `no_contradiction` ∧ `plate_confidence ≥ threshold`. **Nothing in the repo can currently satisfy even T1 for a violation.**

## 17. Security / Privacy / RBAC Gaps

| Area | State |
|---|---|
| Authentication / RBAC | None. Blueprint defers "RBAC beyond authentication" (`blueprint:52`); there is no authentication either |
| Audit log | None (no user/action trail; DB rows have `created_at` only) |
| Data at rest | SQLite files unencrypted; exported JSON/HTML/video unencrypted; no retention/deletion policy |
| PII | Nothing is classified or minimised. Videos contain vehicles, pedestrians and (potentially) faces/plates; the pipeline stores full-frame observations and ships whole source videos (including base64 inside a 130 MB HTML) |
| Secrets | `.env` is gitignored; there is no secrets management and no secrets in the repo that I saw |
| Network surface | `scripts/serve_presentation.py` serves the **whole repo root** (incl. `.git`, `models/`, `configs/`) without auth, bound to `localhost` (`:97`). Fine for a demo laptop; unsafe if ever rebound |
| Third-party loads | The "protected" demo dashboard loads Tailwind/Google Fonts from CDNs (the *live* variant vendors them; `build_offline_assets.py` docstring). A deployed operator UI must not |
| Supply chain | Unpinned deps, no lockfile, no SBOM/vulnerability scan; **Ultralytics is AGPL-3.0 unless separately licensed** — a deployment/commercial-licensing question **[external knowledge, verify]** |
| Dataset licences | Intel clips CC BY 4.0 (recorded); AIC2022 licence PDF present at `E:\traffic dataset\Dataset License AIC2022.pdf` — **terms not reviewed here**; `demo.mp4`/`traffic.mp4` provenance not recorded **[not verified]** |
| Registry/owner data | No design; see §13 |

## 18. Testing / Validation Gaps

**What exists [measured]:** `pytest` → **606 passed in 5.96 s**. 9,048 LOC of tests; per-area counts: perception 149, world 81, hypotheses 52, evaluation 51, fusion 38, evidence 32, feedback 21, propagation 19, safety 18, storage 16, top-level 129. No skips/xfails (`grep` = 0). No coverage tooling, no CI.

**What the number means:** the suite is hermetic and fast because **no test runs a real model or real footage** — `ByteTrackYoloTracker` is stubbed (`tests/test_export_ui_data.py:255,276,311,336`; `tests/test_real_world_validation_runner.py:9`), ingestion uses a 16×12-pixel synthetic clip (`tests/perception/conftest.py`), and the c006 showcase test uses `tmp_path` and asserts the calibration file does *not* exist (`tests/test_run_s02_c006_showcase.py:59`). It appears not to need the `E:\` dataset, GPU, or weights **[not verified on a clean machine]**. It proves *plumbing and invariants*, not *correctness on real traffic*.

**Gaps**
1. **Zero ground-truth validation** for detection, tracking, lane assignment, speed, stopped-vehicle, congestion level, or ranking. CityFlow `gt/`/`mtsc/` are on disk, unused.
2. **A safety test at the wrong layer** (`test_speed_evidence_safety.py`, imports only) — it passes while 71–83% of c006 tracks carry >200 km/h "m/s" speeds into evidence (finding 2). Needed: a *data-flow* test (build a CITYFLOW_SUPPLIED profile → run the chain → assert no m/s-thresholded decision reads it).
3. **No test for ID reuse / duplicate track IDs** (finding 4), for tracker/TrackManager buffer mismatch, or for fps-dependence of `max_missed_frames`.
4. **No real-data regression snapshots** (the exported JSONs are neither golden-tested nor reproducible without the external dataset).
5. **No end-to-end test of the real model path**; the `reset()` crash on installed Ultralytics went unnoticed because nothing exercises it.
6. **No performance, soak, memory, or failure-injection tests.**
7. **No calibration of the confidence output** (reliability diagram, abstention rate vs. error rate), though the code is designed around calibrated-looking tiers.
8. **No UI tests**; the dashboard is a 116 KB hand-written template.
9. **Validation circularity:** c006 lane polygons were fit to ByteTrack ground points from the same 30 s that the dashboard then evaluates (`s02_c006.toml` revision notes) — in-sample.

## 19. Recommended Target Architecture

**Design rules**
1. **Replay = live.** Recorded files enter through the same `FrameSource` streaming interface as RTSP, paced or accelerated. Every demo/test exercises the production path.
2. **One world model per camera, one identity per vehicle.** A single `VehicleState` (stable `vehicle_id`, class, lane, heading, speed ± error with *validation status*, quality flags, history) replaces today's string-joined companion objects.
3. **Capability certification gates automation.** A registry `(camera, capability) → {calibration status, validation metrics, error bounds, date, approver}` is consulted before any action beyond T0. This operationalises "capability-specific validation".
4. **Epistemic ladder is a type.** `AttributionLevel ∈ {observed, involved, evidence_supported_contributor, causally_established}` with explicit promotion rules; `causally_established` is unreachable from video alone by policy.
5. **Abstain by default; keep every existing explicit-status enum** and extend it.
6. **Single-camera complete; multi-camera additive** — cross-camera modules enrich corroboration/continuity but are never required.
7. **Modular monolith first.** Bounded in-process queues and one process per camera/GPU; introduce a broker only when a second node is needed (the blueprint's V1 "no message broker" stance should be revisited deliberately, not by accident).

```
RTSP / file / USB ─► Ingest worker (PTS clock, reconnect, jitter buffer, health)
                         │ frames (bounded queue, drop policy logged)
                         ▼
                 Perception worker  [GPU: detect + track, batched; ReID optional]
                         │ tracklets
                         ▼
        World-model service (per camera)
           calibration + validation status · lane/road model (direction, permissions, limits)
           VehicleState store (stable id, lane, heading, speed±err) · sliding windows
                         │
      ┌──────────────────┼───────────────────────────┐
      ▼                  ▼                           ▼
 Behaviour engines   Congestion/ranking         Evidence recorder
 (stopped, wrong-    (existing DS H1–H6,        ring buffer → clip + crops + calib + model hash
  side, overspeed,    kept as one reasoner)      content-hash → object store
  wrong-lane, brake…)        │                           │
      └──────────────┬───────┘                           │
                     ▼                                   ▼
        Event bus ─► Attribution (involved / contributor) ─► Case manager (state machine)
                                                              │
        ANPR service (gated, optional) ──► Policy engine ◄── capability-certification registry
        Registry adapter (authorised, isolated, audited)          │
                                                                  ▼
              Alert / SOS / enforcement gateway (idempotent, kill-switch)
 Cross-cutting: AuthN/Z (OIDC+RBAC) · append-only hash-chained audit · PII redaction/retention
                · metrics/health/tracing · config+model registry · CI replay regression
 Store: Postgres (metadata, cases, audit) + object store (media) ; SQLite remains for dev only
 UI: live operator console (SSE/WebSocket) + case review; replay is a mode, not a separate app
```

## 20. Recommended Implementation Phases

Sizes are relative (S/M/L), not dated; exit criteria are what must be *measured* before moving on.

| Phase | Scope | Exit criteria |
|---|---|---|
| **P0 Truth hygiene** (S) | Decouple WORLD from `is_calibrated`; remove centroid speed path; replace AST test with data-flow test; fix `reset()`; pin deps/lockfile; commit/organise untracked assets; sync README/docstrings; remove "real-time"/"high" claims that are structural | No m/s-labelled value reachable from unvalidated calibration (test); H1-only degeneracy surfaced as `degenerate_ranking` not `high` |
| **P1 Streaming single-camera world model** (L) — **first milestone, §23** | Stream sources, stable identity, streaming store, sliding windows, GT evaluation harness (IDF1, fragmentation) | duplicate-ID rate = 0; IDF1 reported vs. CityFlow GT; batch↔stream parity; bounded memory soak; latency report |
| **P2 Validated geometry & speed** (M–L) | Surveyed or measurement-based metric calibration with hold-out check; `VALIDATED` earned not asserted; lane model with direction/permissions; automatic lane-coverage QA; error-bound propagation | Per-camera measured speed error bound against reference speeds; capability registry populated for ≥1 camera |
| **P3 Behaviour detectors v1** (L) | Harden stopped-vehicle (context), then direction → wrong-side → overspeed (only where speed certified) → wrong-lane → sudden braking; each with its own labelled eval set and **operating point** | Per-detector precision/recall on real labelled clips; documented failure modes; abstention rate |
| **P4 Persistence, API, case manager, auth** (L) | Postgres + object store; event/case schema; evidence recorder; REST/stream API; OIDC + RBAC; audit log; review-only UI | Every event has a media bundle + audit trail; role-based access tested; T0/T1 only |
| **P5 Attribution & investigation** (M) | `AttributionLevel`; queue-head/onset/precedence tests; multi-vehicle involvement; reviewer explanations | Reviewer-agreement study on real incidents |
| **P6 ANPR pilot** (M, gated) | Readability feasibility on real crops/optics → detector+OCR+voting → confidence propagation; **no registry yet** | Plate read precision at stated confidence; feasibility verdict per camera |
| **P7 Registry/owner (authorised), alerts, SOS, enforcement** (L, gated) | Only after legal/authority sign-off; T2 before T3; human-in-loop first | Authority-signed policy; red-team of mis-ID scenarios; kill-switch drill |
| **P8 Multi-camera enhancement** (M) | Re-ID, corridor continuity, propagation on **real** pairs (c006+c009 is the obvious test bed) | Corroboration measurably improves precision without being required |
| **P9 Production hardening** (L) | Containers, GPU runtime, CI replay, load/soak, DR, security review, licensing (Ultralytics) | SLOs met in soak; pen-test/threat-model closed |

## 21. Dependencies and Risks

**Dependencies:** labelled real data per capability (violations, accidents, plates) — mostly not in hand; per-site calibration surveys and reference speeds; camera optics/placement suitable for ANPR; GPU capacity beyond a 6 GB laptop for multi-stream + ANPR; legal authority, registry access agreement and data-protection assessment; a decision on detector licensing (Ultralytics AGPL-3.0 vs enterprise licence vs an alternative detector); operations owner (on-call, retention, review staff).

**Risks**

| Risk | Why it matters | Mitigation |
|---|---|---|
| Unvalidated speed/lane feeds a decision | Already happening in cause reasoning (finding 2); would become false accusations | P0 gating; capability registry; data-flow tests |
| Identity errors (ID reuse/switches) | Wrong vehicle ↔ wrong plate ↔ wrong owner | P1 identity fix + IDF1 gate; plate/track voting |
| Plate misreads | Enforcement against the wrong person | Confidence propagation; independent corroboration; human review for T3 |
| Over-scope (32 capabilities) | Repo has 21 commits over ~16 days (2026-08-29 → 09-14) and a research-prototype culture | Strict phase gates; one certified capability at a time |
| Domain shift (night, rain, camera shake, Indian traffic mix) | COCO nano model, no fine-tune, no auto-rickshaw | Eval first (P1), then targeted fine-tune |
| Camera geometry drift | Homography/lane polygons silently go stale | Calibration health monitoring; re-validation triggers |
| Dependency churn | Brand-new majors, unpinned; `reset()` already broken | Lockfile + CI replay |
| Legal/licence | AGPL, dataset terms, registry law | Early legal review; keep ANPR/registry behind feature gates |
| Demo-driven claims | Dashboard copy ("Real-time", `H1 high`) outruns evidence | Claim lint + review before any external demo |

## 22. What We Should NOT Build Yet

- **ANPR/OCR, registry lookup, owner data, challan, SOS automation** — before plate readability is proven, legal authority exists, and violation detectors are certified.
- **Any automated enforcement path (T3)** of any kind.
- **Cross-camera Re-ID, GNN/learned propagation, copula fusion, auto-generated hypotheses** — blueprint-LATER and not needed for single-camera product value.
- **More congestion hypotheses or more H1–H6 tuning** — it is a research model; its inputs are not yet trustworthy.
- **More synthetic scenarios, baselines, or dashboard polish** — the evaluation need is *real ground truth*, not more authored scenarios.
- **Kubernetes/service mesh/broker clusters** — until one stream works end-to-end live.
- **Hand-drawn lane polygons for more cameras** — decide the lane-model approach (and QA) first; polygons fit to the tracker's own output are not independent.
- **Accident-detection ML** — before the event/evidence/case infrastructure and a labelled accident set exist.
- **A fine-tuned detector** — before a GT harness can tell whether it helped.

## 23. Exact First Engineering Milestone

**M-A — "Streaming single-camera world model with trustworthy identity and gated speed, scored against CityFlow ground truth."**

**Deliverables (all single-camera, no new behaviour detectors):**

1. **Truth fixes (P0 subset, each small, each with a test)**
   a. WORLD only when *scale is validated* (or introduce `WORLD_UNVALIDATED` that every m/s-thresholded consumer — `traffic_state`, `evidence_builder._stopped_tracks`, `reliability._calibration_factor`, `stopped_vehicle` — treats as IMAGE-equivalent). Delete the centroid→homography path (`schema_conversion.py:81-85`).
   b. Replace `tests/perception/test_speed_evidence_safety.py`'s import check with a data-flow test (CITYFLOW_SUPPLIED profile → full chain → no consumer reads an m/s value).
   c. Identity: make the `VehicleTrack` key unique per `(camera, session, tracklet serial)`; stop creating a second `TrackState` under a live key; align `TrackManager`'s occlusion clock to seconds; add a ByteTrack-ID-reuse regression test.
   d. Pin dependencies (lockfile); fix or remove `ByteTrackYoloTracker.reset()`.
2. **Streaming core**
   `FrameSource` implementations: file-as-stream (real-time-paced and accelerated) and RTSP; PTS/wall-clock timestamps; bounded queues with logged drop policy; a `StreamingTrackStore` emitting per-frame `TrackUpdate`s; **sliding** windows feeding the *existing* `aggregate_traffic_state` / `run_corridor_window`; bounded memory.
3. **Ground-truth evaluation harness** using CityFlow `gt/`, `mtsc/`, `det/`, `roi.jpg` for c006 (+ c009): detection precision/recall @IoU 0.5, MOT IDF1 / ID-switches / fragmentation, and the duplicate-ID rate — with before/after numbers for fix 1c. Report baselines first; **set thresholds from the first measurement, not in advance**.
4. **Latency & throughput report:** per-stage ms/frame (decode, detect+track, track store, windowing), end-to-end event latency p50/p95 at 1× replay, two concurrent streams, GPU utilisation.

**Acceptance criteria (measurable):** (i) duplicate `track_id` per session = 0 (test + on c006 and `demo.mp4`); (ii) no m/s-labelled quantity reachable from an unvalidated calibration (data-flow test green); (iii) streaming output ≡ batch output on the same file within a stated tolerance (parity test); (iv) 34-minute `demo.mp4` soak with flat memory; (v) IDF1 and detection P/R reported against GT for c006; (vi) latency report published with p50/p95; (vii) c006 export no longer displays `H1 high` driven by unit-invalid speeds.

**Explicitly out of scope:** new violation detectors, ANPR, registry, UI work, auth, multi-camera.

**Why this first:** every downstream capability (violations, involvement, plates, cases) joins on *vehicle identity*, depends on a *trustworthy speed/lane gate*, and needs a *streaming* substrate; and nothing can be honestly claimed until a ground-truth baseline exists. It also touches the smallest amount of existing code while removing the most dangerous overclaim.

---

## CURRENT STATE → TARGET STATE

| | **CURRENT** | **TARGET** |
|---|---|---|
| Nature | Offline research prototype: recorded-file, batch-to-EOF, congestion-cause ranking | Continuous live road-intelligence & investigation platform, enforcement-*support* with certified gates |
| Input | `is_file()` only; `index/fps` clock | RTSP/stream with PTS clock; replay = live |
| Perception | YOLOv8n + ByteTrack, unvalidated; ~12–15% duplicate IDs/window; 49% tracks <2 s | Evaluated detector/tracker (IDF1 vs GT), stable vehicle identity |
| Geometry/speed | Hand polygons for 2 cameras; speed = unit-invalid on CityFlow, 6.5% usable; WORLD = "is_calibrated" | Validated per-camera calibration, lane model with direction/permissions, speed ± error with earned `VALIDATED` |
| Behaviour | Stopped-vehicle heuristic + congestion FSM; 0 of 7 violation/behaviour detectors | Certified detectors (wrong-side, overspeed, wrong-lane, braking, accident) each with measured operating points |
| Reasoning | H1–H6 DS ranking; on real video degenerates to "H1, high"; candidate vehicle = stopped track | Attribution ladder (observed → involved → contributor; causal unreachable), queue-head/onset tests, DS retained as one reasoner |
| Evidence | Numbers only; nothing persisted | Hash-chained media bundles (clip, crops, calibration, model hash) + append-only audit |
| Product workflow | None: no ANPR, registry, cases, alerts, SOS, challan | Gated: ANPR → authorised registry → case → alert → T1/T2, T3 only with legal + certification |
| Platform | No auth/RBAC/privacy/deploy; stub API; replay UI | OIDC + RBAC, PII controls, containers/CI/observability, live operator console |
| Validation | 606 hermetic tests, synthetic labels, 0 real GT | Per-capability labelled evaluation, calibrated confidence, soak/failure tests |
| Honesty culture | Excellent *conventions* (UNAVAILABLE/OUTLIER/abstain), uneven *application* | Same conventions enforced by data-flow tests and a capability-certification gate |

---

# Post-Audit Report

## Files inspected

I did **not** read every file end-to-end; here is the true coverage.

- **Read in full:** `anvesh/{main,config,logging_setup,__init__,pipeline}.py`, `anvesh/api/__init__.py`; `anvesh/perception/{config,ingestion,detection,tracking,track_manager,types,pipeline,windowing,lane_assignment,speed_estimation,motion,schema_conversion,traffic_state}.py`; `anvesh/world/{calibration,lane_geometry,lane_config}.py`; `anvesh/hypotheses/{candidate_vehicle,cause_model}.py`; `scripts/{run_s02_c006_showcase,serve_presentation}.py`; `configs/{default.toml, experiment/*.toml}`; `pyproject.toml`, `README.md`, `.gitignore`; both `datasets/recorded/*/SOURCE.md`.
- **Read in part (headers, key functions, or targeted ranges):** `anvesh/evidence/{evidence_builder (≈1–290),reliability (1–100)}.py`, `anvesh/hypotheses/ranking.py (1–60)`, `anvesh/fusion/{ds_fusion (1–70),alignment (1–50)}.py`, `anvesh/propagation/shockwave.py (1–50)`, `anvesh/safety/stopped_vehicle.py (1–140)`, `anvesh/storage/db.py (1–80 + symbol outline)`, `anvesh/storage/schemas.py (outline, enums, `SafetyEvent`, header)`, `anvesh/rendering.py (header)`, `anvesh/evaluation/scenarios.py (1–50)`, `scripts/export_ui_data.py (1–80, 449–462, 507–790)`, `scripts/analyze_video.py (1–60)`, headers (≈12 lines) of the other 8 scripts, `configs/lanes/s02_c006.toml (1–80 + grep)`, `ANVESH_C006_SPEED_VALIDATION_AUDIT.md (1–80)`, `docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md` (headings, Part 2 scope row, selected lines), `tests/perception/test_speed_evidence_safety.py (1–60)`, `tests/perception/conftest.py`, `E:\traffic dataset\ReadMe.txt` (calibration/GPS/timestamps lines) and `…\c006\calibration.txt`.
- **Searched/queried only (not read):** `ui/anvesh_dashboard.template.html` (keyword greps), `stitch/*/DESIGN.md|code.html` (greps), `tests/**` (listing, collection counts, greps for model/video/skip usage), `docs/LANE_SPEED_IMPLEMENTATION_NOTES.md` and `docs/M6_5_REAL_WORLD_VALIDATION.md` (not opened), the 7 other root `ANVESH_*.md` audits (not opened), the other `scripts/*` bodies, `feedback/update_engine.py`, `world/{corridor,corridor_state,synthetic_fixture}.py`, `evaluation/{baselines,metrics}.py`, `perception/visualize.py` (outline/grep only).
- **Programmatically analysed:** `ui/data/anvesh_s02_c006_data.json`, `ui/data/anvesh_demo_data.json`, `ui/live/anvesh_live_data.json` (headers).
- **Not opened:** `docs/*.pdf`, screenshots/JPGs, `ui/live/vendor/`, `demo.mp4`/`traffic.mp4` content (only decoded by the benchmark), `anvesh.egg-info`.
- **Executed:** the test suite; a scratch benchmark (`…\scratchpad\bench.py`, outside the repo) on `demo.mp4` and c006 `vdo.avi` (300 frames each, post-warm-up; also reproduced the `reset()` crash).

**Caveat on benchmark/throughput numbers:** they are from one laptop GPU (RTX 3050 6 GB), one run each, nano model, loop includes decode + `model.track()` + `TrackManager` only. GPU use is inferred from `torch.cuda.is_available()==True` and the latency; I did not profile utilisation.

## Important entry points discovered
`anvesh` / `python -m anvesh.main` (stub) · `scripts/run_pipeline.py` · `scripts/export_ui_data.py` · `scripts/run_s02_c006_showcase.py` · `scripts/analyze_video.py` · `scripts/run_real_world_validation.py` · `scripts/run_evaluation.py` · `scripts/run_{world_alignment,corridor_state,fusion,feedback,stopped_vehicle,end_to_end}_demo.py` · `scripts/serve_presentation.py` · `scripts/build_ui_artifact.py`, `build_offline_assets.py`, `rebuild_dashboard_live.py`, `rebuild_live_dashboard.py`, `export_s02_c006_lane_geometry.py`. The only product-relevant *library* entry points are `anvesh.pipeline.run_corridor_window` and `anvesh.perception.pipeline.run_m1_pipeline` (both batch/one-window).

## Current test command
README: `pytest`. What I ran (no cache written to the repo):
`.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q`

## Current test result
**606 passed in 5.96 s** (wall ≈ 7.6 s). No failures, skips or xfails.

## Current git status
Unchanged from the starting snapshot except for this report (see the final listing appended below).

```
$ git status --short   (branch: main @ 042ea4b; taken after writing this report)
 M anvesh/perception/speed_estimation.py
 M anvesh/rendering.py
 M scripts/run_s02_c006_showcase.py
 M tests/perception/test_speed_estimation.py
 M tests/test_rendering.py
?? ANVESH_C006_LANE_COVERAGE_ANALYSIS.md
?? ANVESH_C006_SPEED_VALIDATION_AUDIT.md
?? ANVESH_COMMIT9_INTEGRATION_AUDIT.md
?? ANVESH_LANE_SPEED_FORENSIC_AUDIT.md
?? ANVESH_PRODUCT_GAP_AUDIT.md
?? ANVESH_S02_C007_C009_VISUAL_INSPECTION.md
?? ANVESH_S02_CAMERA_PAIR_ANALYSIS.md
?? ANVESH_S02_REMAINING_CAMERA_VISUAL_INSPECTION.md
?? demo.mp4
?? docs/ANVESH_Demo_Explanation_Guide.pdf
?? docs/demo_reference/s02_c006_c008_inspection/
?? docs/demo_reference/s02_c006_lane_coverage_analysis/
?? docs/demo_reference/s02_c007_c009_inspection/
?? docs/demo_reference/screenshots/
?? scripts/analyze_video.py
?? scripts/build_offline_assets.py
?? scripts/build_ui_artifact.py
?? scripts/export_s02_c006_lane_geometry.py
?? scripts/rebuild_dashboard_live.py
?? scripts/rebuild_live_dashboard.py
?? scripts/serve_presentation.py
?? stitch/
?? tests/perception/test_speed_evidence_safety.py
?? traffic.mp4
?? ui/
```

Diff against the session-start snapshot: the **only** change is the addition of `?? ANVESH_PRODUCT_GAP_AUDIT.md` (this file). The 5 pre-existing modified files and all pre-existing untracked paths are untouched; nothing was committed.
