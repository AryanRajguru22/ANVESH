# Milestone M-A — Streaming single-camera world model: report

Scope: `ANVESH_PRODUCT_GAP_AUDIT.md` §23. Nothing here makes ANVESH production-ready. Raw outputs: `outputs/cityflow/`, `outputs/m_a/`. Environment: RTX 3050 6 GB laptop, Python 3.11.9, Ultralytics 8.4.132 (pinned in `requirements-lock.txt`). Nothing was committed.

## 1. Validated vs merely implemented

| Item | Status |
|---|---|
| Speed-safety invariant (unvalidated geometry → no m/s / km/h) | Implemented, **behaviourally tested** (data-flow test + positive control + mutation test), exercised on real c006 |
| Session-scoped vehicle identity, time-based lifecycle | Implemented; duplicate identity = 0 on unit/property tests, c006 full video, demo 240 s, demo 34-min soak |
| Streaming engine / store / sliding windows / bounded queues | Implemented; parity and soak run on real video |
| RTSP source | **Implemented and unit-tested against fake captures only. Never run against a real camera/server.** |
| Metric speed from *validated* geometry | Code path exists and is tested on synthetic survey fixtures. **No real camera has validated metric calibration**; `fit_planar_homography` still stamps VALIDATED "by construction" (see §10) |
| CityFlow GT metrics | Real, but precision/IDF1 are lower bounds (see §6) |

## 2. Speed-safety invariant
`MotionSpace.WORLD` means exactly "validated metres". A new `GeometryValidity` enum (IMAGE / WORLD_UNVALIDATED / WORLD_VALIDATED) is derived from `CalibrationProfile.geometry_validity` (never from `is_calibrated`). Structural guards:
- `VehicleTrack` rejects `WORLD` without `WORLD_VALIDATED`; `CalibrationProfile` rejects VALIDATED with non-metre units.
- `build_vehicle_track`: the centroid→homography→"m/s" path is **deleted**. Unvalidated geometry → IMAGE kinematics (px/s) tagged `WORLD_UNVALIDATED`; validated → ground-point projection through the single estimator `estimate_vehicle_speed`; unmeasurable speed → `speed_available=False`, not a fabricated 0.
- `VehicleSpeedEstimate`: km/h only for VALIDATED; unvalidated values live only in `unvalidated_world_speed` (APPROXIMATE can't hold a km/h value; enforced in the constructor). `format_speed_text` renders "speed unvalidated".
- `aggregate_traffic_state` raises if its declared `motion_space` disagrees with its tracks; `export_ui_data` derives the space from the shared helper (it previously used `is_calibrated`); `_stopped_tracks` re-checks per track; exports carry `speed_unit`.
- Dashboard template: unconditional "m/s" labels on `mean_speed` fixed to follow `motion_space` (the built `ui/*.html` and `ui/data/*.json` were **not** regenerated and are stale).

## 3. Identity / lifecycle
Key = `(camera_id, session_id, serial)`, printed `camera:<session>-t<serial>` (single colon; dashboard-compatible). Serials come from the manager and are never reused; a raw ByteTrack id maps only to its *currently live* state, so a returning id can only open a new identity. Same-frame duplicate raw ids and raw ids returning after a stall are split. Occlusion is in **seconds**; the engine lifts the horizon to ByteTrack's own `track_buffer` (30 *frames*, not fps-scaled; 3.15 s at 10 fps, 1.26 s at 25 fps), which removes the revive-after-finalize duplicates at the source. `begin_session`/epoch change fences identity on reconnect/reset. Optional `run_id` (`derive_run_id`) makes keys unique across runs; **default off** (every run restarts at `s000`). `ByteTrackYoloTracker.reset()` now rebuilds the model (old body crashed the next `update()`; regression test uses real Ultralytics and reproduces the old crash).

## 4. Streaming design
`FrameSource` (file, with optional pacing; RTSP with backoff/reconnect/epoch/gap/jitter/health) → `BoundedQueue` (BLOCK for files, DROP_OLDEST for live; every drop counted) → `StreamingPerceptionEngine` (detect+track → `StreamingTrackStore` → `SlidingWindowScheduler`) → `WindowSnapshot` → `WindowReasoner` (existing `aggregate_traffic_state` → `run_corridor_window`). Windows close on stream time (no EOF dependence), ids `cam:w000123`, configurable duration/stride/lateness; windows contain only in-window samples (fixes lifetime-mean-speed attribution). `run_m1_pipeline` and the exporter use the same engine/window builder. Overlapping windows are supported but `WindowReasoner` refuses them (hysteresis tracker assumes consecutive windows).

## 5. Ranking abstention (needed for c006 criterion)
Fixing provenance alone left c006 windows ranking "H1, belief 0.87/0.89, tier high" because in image space only H1 is evaluable (audit finding 3). Added a **generic** rule: fewer than 2 evaluable hypotheses (completeness ≠ NONE) → `INSUFFICIENT_EVIDENCE`, per-entry tiers shown as LOW. No H1 score/threshold/evidence was changed. Side effect: every image-space camera (incl. all 205 demo windows) now abstains.

## 6. CityFlow S02/c006 (2110 frames, 124 GT tracks, 5081 GT boxes; `scripts/eval_cityflow.py`)
**GT caveat:** `gt.txt` only annotates vehicles seen by ≥2 cameras → recall, IDR, ID switches, fragmentations are valid; **precision/IDP/IDF1 are lower bounds; true precision is unavailable.** Frame offset +1 confirmed empirically (peak recall). Predictions outside `roi.jpg` ignored. Metrics code cross-checked against `motmetrics` 1.4.0: IDF1/IDP/IDR identical to 4 decimals, fragmentations identical, ID switches 4 vs 4 (shipped baseline) and 42 vs 40 (ours; tie-breaking).

| Tier | Result |
|---|---|
| 1 Detector (YOLOv8n, conf 0.25, IoU 0.5) | recall **0.633**; precision lower bound **0.142** (TP 3218 / FP 19445 / FN 1863) |
| 2 Tracker (raw ByteTrack ids) | IDF1 ≥ **0.209** (IDP 0.131, IDR **0.514**), ID switches **42**, fragmentations **144** |
| 3 TrackManager before (raw-id keyed, 5-frame clock) | same as tier 2 (it echoes raw ids); per-state view IDF1 0.206, IDSW 44; **761 states / 507 keys; 97 keys duplicated; duplicate rate 33.4%**; 225/991 (22.7%) per-window entries duplicated; median tracklet 0.6 s |
| 3 TrackManager after | IDF1 0.209, IDSW 42, FM 144 (unchanged); **508 states / 508 keys; duplicate rate 0.0%**; 0/767 window entries; median tracklet 1.3 s |
| Serial identity alone (legacy 5-frame clock) | duplicates 0, but 846 tracklets, IDF1 0.206 — shows the serial guarantees uniqueness while the aligned horizon repairs fragmentation |

**The identity fix removed duplicate keys; it did not improve tracking accuracy** (IDF1/IDSW/FM are the tracker's). Shipped baselines through the same harness (sanity): `mtsc_tc_yolo3` IDF1 0.151, det_yolo3 recall 0.562 — also precision-depressed by the missing annotations. Demo (no GT): before 1320 states / 1108 keys, 196/1598 window entries duplicated (12.3%, reproduces the audit); after 1109/1109, 0/1402.

## 7. Batch / stream parity (`scripts/stream_parity.py`)
- Cached real tracker output, 22 windows: SYNC == THREADED(lossless) == PACED(8×) **exactly** (world-model content and reasoner outcome/reliability); 0 drops.
- Real YOLO+ByteTrack, first 60 s: run-to-run and sync-vs-threaded identical (GPU nondeterminism was zero here — a measurement, not a guarantee).
- **Tolerance:** exact when no frame is dropped (deterministic function of the frame sequence). With drops the windows are flagged `degraded` and equality is not claimed. Post-hoc consumers (windows cut from final states) differ **only in `occlusion_state`** (349 tracks, 21 windows) because they see the future; downstream: 0/22 outcome flips, reliability deltas up to 0.079 in 21 windows. The exporter uses this post-hoc mode.

## 8. Latency / throughput (real YOLOv8n+ByteTrack, ms, p50/p95)
Paced 1× runs are the latency figures; unpaced runs are saturation throughput (queue wait dominates their end-to-end numbers).

| Run | source → processing fps | detect+track | store | windowing | frame e2e | window e2e (post-close) | drops |
|---|---|---|---|---|---|---|---|
| c006 1080p, 1 stream, 1× (120 s) | 10 → 10.0 | 32.9 / 65.6 | 0.11 / 0.24 | 1.6 / 7.2 | 33.3 / 66.6 | 41 / 66 | 0 |
| c006 1 stream, unpaced | 10 → 33.8 (3.4×) | 25.6 / 46.2 | 0.10 / 0.16 | 1.3 / 2.1 | 251 / 425 (queue) | 252 / 395 | 0 |
| demo 640×360, 1×, 60 s | 25 → 24.8 | 25.4 / 51.8 | 0.11 / 0.24 | 4.4 / 32.9 | 27.7 / 88.2 | 57 / 444 | **13 of ~1500 (0.9%)** |
| demo unpaced | 25 → 40.3 (1.6×) | 23.3 / 37.6 | 0.12 / 0.23 | 4.1 / 5.6 | 243 / 322 (queue) | — | 0 |
| c006+c009 concurrent, 1× (120 s) | 10 → 10.0 each | 50.2–52.6 / 79–81 | 0.05–0.08 / 0.13–0.16 | 1.3–1.7 / 3.7–4.9 | 50.5–52.9 / 80–82 | 52–58 / 82–83 | 0 |
| c006+c009 concurrent, unpaced | 10 → 28.7 each (57.4 total) | 33.5–33.8 / 59 | 0.08 / 0.15 | 0.9–1.3 / 1.7–3.2 | 364–367 / 506–512 (queue) | — | 0 |

Reasoning (`run_corridor_window`) 1–4 ms per window. GPU utilisation (nvidia-smi, polled 1 Hz) mean 27–51%, max ≤ 100%; peak RSS ≈ 1.3–1.5 GiB. Window *close* additionally waits for the window to end (10 s) plus `lateness_s` (default 0). **Throughput is below the audit's tight-loop figure** (audit: 80 fps at 640×360; here 40 fps threaded, 49 fps single-thread sync): thread/GIL overhead plus run-to-run machine variance. At 25 fps this laptop does not sustain real time with margin (p95 detect+track 52 ms > 40 ms frame interval) — it dropped 0.9% of frames.

## 9. 34-minute soak (demo.mp4, 51,201 frames, `scripts/stream_soak.py`)
Unpaced (38.0 fps, 1348 s wall) through the same code path — **not a 34-minute real-time run**. 0 frames dropped, 205 windows (all abstained: image space), **9412 identities created = 9412 unique, 0 duplicates**. RSS: 540 MiB at start → 1362 MiB within 60 s (model/CUDA load) → creeping +6.6 MiB over the next 960 s (≈0.4 MiB/min, ≈25 MiB/h, cause not isolated) → one 135 MiB step down at t≈1048 s → 1234 MiB flat (+0.7 MiB over 300 s). Retained tracks 17–72, frame/queue structures flat, CUDA allocated ≈ 19–24 MiB flat. **Bounded: no growth beyond that creep.** Note: my pre-declared automated check (|Q4 vs Q2| ≤ 5%) printed `false` because of the 9.4% *decrease*, not growth; reported as is. The soak process also holds a ≈9.4k-string id set and sampler rows (≈ a few MiB).

## 10. c006 safety (`scripts/c006_safety_report.py`, 30 s, 3 windows)
| | motion space of tracks reaching reasoning | median "speed" | tracks > 55.6 "m/s" | duplicate ids / window | outcome |
|---|---|---|---|---|---|
| Before | world | 381 / 348 / 260 | 43 / 47 / 32 | 6 / 10 / 8 | ranked H1 0.85 / 0.86 **high**, insufficient |
| Speed fix only | image (px/s) | 71 / 55 / 51 px/s | 0 | 0 | ranked H1 0.87 / 0.89 **high**, insufficient *(structural, image-space)* |
| + abstention | image | same | 0 | 0 | **insufficient_evidence ×3**, tier low, candidate vehicle not_ranked |
Full 211 s c006 through the stream path: 22/22 windows `insufficient_evidence`. Congestion levels there still come from the capacity-proxy occupancy (arbitrary 20-vehicle constant) — untouched.

## 11. Tests
692 passed (was 606). New: MOT metrics toy cases (12), speed-safety data flow (9), TrackManager identity (22), streaming engine/windows/queues/sources/RTSP/pipeline (34), real-Ultralytics reset regression (4), ranking abstention (4), exporter-level unvalidated-profile test, schema/pipeline/e2e additions. **Contract changes to existing tests (deliberate, not weakened):** TrackManager API/keys (raw-id lookups → identities, frame → time horizon); APPROXIMATE speeds no longer carry km/h (`test_speed_estimation.py`, `test_rendering.py`, `test_export_ui_data.py`); rendering e2e fixture now pairs its WORLD track with a validated profile; export fake pipeline now uses the real converter; one traffic-state test used mismatched spaces. Pre-existing uncommitted user edits in `speed_estimation.py`, `rendering.py`, `run_s02_c006_showcase.py`, `test_speed_estimation.py`, `test_rendering.py` were built on, not reverted.

## 12. Known limitations / still unsafe
- `fit_planar_homography` asserts VALIDATED without a hold-out check; no real camera has validated metric speed, so metric behaviour is untested on real data. P2.
- Detector/tracker quality is poor on this GT (recall 0.63, IDF1 ≥ 0.21 lower bound); no fine-tuning attempted.
- Congestion level/occupancy still an arbitrary capacity proxy; two stopped-vehicle definitions still coexist; image-space `safety/stopped_vehicle` still emits events from a 5 px/s heuristic.
- Identity is per *session*; cross-run uniqueness needs `run_id` (opt-in). No re-ID / stitching across tracklets (median tracklet 1.3 s).
- Live drops stretch ByteTrack's *frame* buffer in wall time; the occlusion horizon assumes nominal fps. Engine flags degraded windows but doesn't correct.
- `ui/data/*.json`, built dashboards and the 130 MB HTML are stale (demo would now abstain); the dashboard's "Real-time evidential scoring" copy untouched; `serve_presentation.py` still serves the repo root.
- Throughput below the audit's number; 25 fps/640×360 not sustained with margin on this laptop; 2-stream only at 10 fps.
- GT metrics are c006 only (c009 not scored); precision unavailable; Ultralytics AGPL and dataset licence unreviewed.

## 13. Recommended next milestone
P2 — earn `WORLD_VALIDATED`: a site/known-length calibration with hold-out reprojection checks and reference speeds for ≥1 camera, a capability registry, and lane/direction attributes — then, only then, behaviour detectors. In parallel: score c009, improve tracker/detector against the harness, and decide the detector licence.
