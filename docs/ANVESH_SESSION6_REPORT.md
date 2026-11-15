# ANVESH - Session 6 report (2026-10-31): presentation product, live video session, honest capability status

Status of the product: **RESEARCH / DEMONSTRATION ONLY.** Not approved for deployment, real challans or legal decisions. Repository `D:\ANVESH`, branch `main`, HEAD `b6d5ecd` (unchanged). Nothing was staged, committed, pushed, stashed, reset, cleaned or deleted.

## 1. Bottom line

* The repository already contained a far more complete application than the task text assumed (auth, cameras, streaming engine, 10 detectors, evidence, cases, registry/challan sandbox, retention, audit, dashboard). The job was to verify, harden, integrate, calibrate, measure and package it - not to build it.
* **A full 20-minute recorded-video session now runs end to end through the real app** (YOLOv8n + ByteTrack on the GPU), at video speed, detection/tracking/events/evidence produced *while the video plays*, 0 dropped frames, flat memory. It was run twice on a fresh disposable data directory.
* **What the 20 minutes of footage honestly shows:** vehicle detection/tracking/counting, queue/stopped-vehicle and congestion episodes (27 incidents: 14 stopped-vehicle, 13 congestion). It produced **no** over-speed, sudden-braking, aggressive-driving, wrong-way or collision events. Those are shown only on clearly labelled **SIMULATED** scripted cameras.
* **Not done / not possible in this footage:** automatic plate reading (plates are about 8 px wide), a trained accident model, wrong-way on real footage, MFA, any real registry/challan.
* Several real defects were found and fixed (section 5), most importantly sudden-braking and over-speed detectors that produced hundreds of false events on real footage.

## 2. Feature matrix (verified behaviour, not existence of code)

| Feature | State before | Work this session | Test evidence | Demo status |
|---|---|---|---|---|
| Vehicle detection | real (YOLOv8n COCO default) | compared with the MIO-TCD fine-tuned checkpoint on the demo video (unlabelled proxies); default kept | 20-min runs; `demo/runs/detector_compare.json` | **real**; accuracy on this footage unmeasured (no labels) |
| Vehicle tracking / stable IDs | real (ByteTrack) | none | 20-min runs; ~5,552 session track IDs (not unique vehicles) | real; ID fragmentation exists (see 8) |
| Vehicle classification | real, coarse | none | - | car/bus/truck; lorries are sometimes "bus" |
| Speed estimation | gated, no real calibration existed | calibration from lane-marking standard; new `livespeed` estimator; km/h with +/- band in live view | 11 new tests; plausibility check | **approximate**, near field of the toward carriageway only |
| Wrong-way | implemented, simulated only | zones configured for the demo road | existing unit tests + SIMULATED camera (1 event) | **not demonstrated on real footage** (none occurs) |
| Wrong-lane | implemented, needs `allowed_classes` | not applicable to this road | unit tests | not applicable |
| Rash / aggressive driving | repeated-lane-change candidate | lane polygons configured; label "aggressive-driving candidate" | unit tests | no event in 20 min |
| Sudden braking | **defective** (199 false events / 160 s) | rewritten: smoothed, significant, quick, monotonic, edge-clip guard, zone-limited | 6 new tests | no event in 20 min; candidate only |
| Overspeeding | **defective** (inflated by jitter) | rewritten: smoothed speed, error band, 5 km/h margin, 2 s sustained, speed zone | 4 new tests + existing engine test | no event in 20 min (the 300 s plausibility run saw a max per-track speed of 116 km/h vs the 112.7 limit + 5 margin; the full 20 minutes were not separately profiled for max speed) |
| Stopped vehicle | real, heuristic | none | runs | works; queue stops rated low; 14 events in the first 160 s |
| Congestion / jam | real, heuristic | note about approximate geometry made honest | runs: 13 episodes | works; episodes flap |
| Jam-cause investigation | hypothesis ranking | none | runs | reports "no candidate cause could be ranked" (no obstruction in footage) |
| Accident detection | experimental candidate | **hardened**: closing, sudden-stop, queue, standstill gates | 6 new tests; SIMULATED fixture fires, SIMULATED queue does not; 0 candidates in 20 min | experimental; no real collision exists to test |
| Plate detection | no model | not done (see 6) | - | **unavailable** |
| Plate OCR | no engine | not installed (see 6) | - | **unavailable**; manual entry; live table says "unreadable (too small)" |
| Incident evidence | real | none | 15/15 workflow checks; every asset re-hashes; clip plays in Chrome | works |
| Live dashboard | real (polling + MJPEG) | layout (video + event feed lead), progress, source fps, queue depth, calibration label, tracked-vehicle table with speed and plate readability | 12/12 browser checks | works |
| Case investigation | real | none | workflow check | works |
| Registry verification | sandbox only, no override | none (preserved) | override -> 422; official hand-off -> 409 | **sandbox**; operational approval blocked by design |
| Sandbox challan | real | none | workflow check (TEST-CHALLAN reference) | works, labelled SANDBOX |
| Retention and audit | real | none | 77 tests; plan API on final data; audit chain ok (458 rows at end) | works |

## 3. Video selection

* **Chosen:** `demo.mp4` (640x360, 25 fps, 34 min 8 s; first 1200 s used), SHA-256 `fbbefd1b...ade5291bb`. Fixed elevated camera over a divided carriageway, daytime, congestion builds and clears, lane lines visible. Scene-cut scan of the used 1200 s: **0 cuts** (one continuous shot).
* **Rejected:** `traffic.mp4` - scene-cut scan finds 3 cuts (145, 254, 274 s); an edited montage of night and day scenes. Intel clips (30-54 s). CityFlow/AIC22 (about 3 min per camera, 10 fps, non-commercial academic licence, homography not metric). S7 gantry dataset (CC BY per a search snippet; portal blocks automated fetch; not downloaded or verified). No suitable Wikimedia/stock video found.
* **Licence caveat (important):** `demo.mp4`'s container says `produced by Google Inc.`: almost certainly a YouTube download. **Origin, uploader and licence are unknown.** It is treated as copyright-unknown: classroom presentation only, not redistributed, not committed (it is git-ignored). If a licensed video is wanted, the S7 dataset is the best lead to check manually. Full comparison table: `demo/SOURCE.md`.
* Resolution is low: median vehicle box width 28 px. That is why plates cannot be read.

## 4. Speed calibration (read `demo/CALIBRATION.md`)

* Scale from UK *Traffic Signs Manual* ch. 5 Table 2-2 / Figure 2-3 (read from the GOV.UK PDF): motorway lane line = diagram 1005.1 = 2000 mm mark + 7000 mm gap (9 m pitch; stud spacing 18 m = 2 pitches agrees). Lane width 3.65 m assumed.
* Dash end points measured on a temporal-median background; homography fit on dashes 0-2 (10 points, RMSE 0.094 m); **dash 3 held out: RMSE 0.43 m**. This checks geometric self-consistency, not the assumed 9 m pitch (a homography absorbs any uniform scale).
* New validation kind `MARKING_STANDARD_HOLDOUT` (deliberately not `SURVEY_HOLDOUT`). Dashboard/API say "APPROXIMATE metric ... site NOT surveyed"; events from it are confidence `low`; speed is shown with `~` and a +/- band, only inside the near-field `speed_zone`.
* Plausibility (not validation): first 300 s, median track speed 97 km/h, p10 87, p90 107, max 116 - the right order for a UK motorway. With a 6 m pitch the same vehicles would read ~65 km/h.
* Unverified: that the road is built to that diagram, that it is a UK road at all (inferred from the picture), lane width, planarity, that the video is not time-altered. The scale uncertainty (plausibly +/-10 % or more) is **not** inside the displayed +/-.

## 5. What was changed (files)

Modified (tracked or not, `anvesh/app/` was already untracked so `git diff` will not show these):
* `anvesh/world/calibration.py` - `MARKING_STANDARD_HOLDOUT` kind, `is_marking_standard_validated`, 4-point hold-out minimum.
* `anvesh/app/cameras.py` - `speed_zone` config key; `geometry_label` for the approximate kind.
* `anvesh/app/detectors.py` - **OverspeedDetector rewritten**; **BrakingDetector rewritten**; **CollisionCandidateDetector gates**; congestion uncertainty text; `DetectorContext.frame_size`.
* `anvesh/app/runner.py` - live speed per track, `live_tracks`, `progress`, `duration_s`, `geometry`, plate-size readability heuristic, label clamped inside the frame.
* `anvesh/app/capabilities.py` - label "Aggressive-driving candidate".
* `anvesh/app/static/views_ops.js`, `app.js`, `app.css` - live layout, progress, speed table, plate readability, candidate labels.
* `tests/app/test_detectors.py` (+15 tests), `.gitignore` (demo data dirs), `docs/ANVESH_APP_GUIDE.md` and `docs/ANVESH_RELEASE_READINESS.md` (new sections).

New: `anvesh/app/livespeed.py`, `tests/app/test_live_speed.py` (11 tests), `scripts/run_demo.py`, `docs/ANVESH_SESSION6_REPORT.md`, and the whole `demo/` kit: `README.md`, `SOURCE.md`, `CALIBRATION.md`, `camera_config.json`, `tools/` (`build_demo_config.py`, `speed_plausibility.py`, `scene_cuts.py`, `compare_detectors_demo.py`, `monitor_run.py`, `officer_workflow.py`, `ui_acceptance.py`, `ui_check.py`, `run_test_groups.sh`, `_api.py`). `demo/runs/` and the `demo/data-*` directories are outputs (git-ignored).

### Defects found on real footage and fixed
1. **Sudden braking: 199 false events in 160 s.** Cause: 0.5 s frame-segment differencing of jittery box positions. Now: 1.2 s least-squares speeds, drop >= 25 km/h from >= 40 km/h, >= 3.5 m/s^2 over 1.0-2.5 s, drop larger than the error bands, monotonic, per-track debounce.
2. **Over-speed: 38 events in 270 s**, speeds up to >132 km/h that the per-track estimator never saw (max 116). Cause: summed path length of jittery far-field positions. Now smoothed slope, error band, margin, 2 s sustained, speed zone only.
3. **A lorry leaving the frame read as "sudden braking"** (its clipped bottom edge froze). Now boxes touching the frame edge are discarded as speed sources.
4. **Collision false-positive risk** ("two vehicles stopping together is not an accident"): gates added - closing distance, stop within 2.5 s of first contact, not a stationary queue (>= 3 neighbours already stopped at contact), not a standstill (>= 60 % of other vehicles stationary). Box overlap alone never qualifies.
5. The congestion incident said "Metric geometry validated" for an approximate calibration - corrected.

## 6. What was deliberately NOT done, and why

* **Plate detector / OCR: no training, no install.** At 640x360 a plate is ~8 px (estimate: plate ~27 % of box width; median box 28 px; near-field cars ~130 px give ~35 px); OCR needs several times that. Only a Vietnamese plate dataset exists (out of domain, previous training died in epoch 1, no checkpoint). EasyOCR's pretrained weights have no stated licence, and C: has 7.7 GB free (97 % full). The product says "unreadable (too small)" and uses manual entry with the DEMO PLACEHOLDER plate. A plate study needs closer/higher-resolution footage first.
* **Accident model:** no accessible licensed dataset on disk (CADP terms unverified, raw video not shared). Nothing trained; the heuristic candidate stays experimental. No accuracy claim.
* **Fine-tuned vehicle model not promoted.** Unlabelled proxies on 180 s (60-240 s) of the demo video, pretrained vs fine-tuned: mean detections/frame 12.1 vs 12.6; distinct track IDs 808 vs 615; tracks < 1 s 361 vs 151; median track length 1.96 vs 2.84 s; mean confidence 0.598 vs 0.704. A visual side-by-side showed parity. But a 300 s run with the fine-tuned model raised the event count (19 stopped + 4 congestion vs 14 + 1 in the same period). No labels, so no accuracy claim; default unchanged. `run_demo.py --detector finetuned` is available as an opt-in. The held-out MIO-TCD test was not rerun (lock file untouched, 18 Oct mtime).
* No web downloads were made. No GIF recording. The Claude-in-Chrome extension was not connected, so the browser was driven with Playwright + the installed Chrome (headless) from the system Python.

## 7. Performance - complete 20-minute session (final run, fresh `demo/data-final`)

| Metric | Value |
|---|---|
| Source | `demo.mp4`, first 1200 s, 640x360, 25 fps, played at video speed (`realtime_factor 1.0`) |
| Frames processed / dropped | **30,001 / 0** |
| Processing fps (average) | **24.95** (real time; source 25.0) |
| Frame processing latency | mean 27.7, p50 27.7, p95 37.5, p99 47.2 ms (max 2,229 ms - likely first-frame model warm-up; not confirmed) |
| Detect+track (GPU) | p50 24.8, p95 31.8 ms; decode p50 1.2 ms |
| End-to-end frame p50/p95/p99 | 28.2 / 38.7 / 59.8 ms |
| Queue | frames capacity 8, high water 8, **0 dropped**, depth ~0 at rest; windows queue high water 2 |
| Server RSS | 1,379 - 1,624 MB (first-quarter mean 1,451, last-quarter mean 1,501: +3 %, no leak trend) |
| GPU | RTX 3050 6 GB: `nvidia-smi` reported 138 MB used and 0-57 % utilisation in every sample (sampled every 10 s; the figure looks low for a CUDA process and was not cross-checked) |
| Frame ring (evidence buffer) | <= 13.5 MB (bounded) |
| Incidents | 27: 14 stopped-vehicle, 13 congestion; 121 traffic windows; evidence failures 0 |
| Repeatability | an earlier run on the same data (with the same detector logic) gave the identical 27 incidents |

Raw data: `demo/runs/perf_final.csv`, `perf_run1.csv`. Event latency (detection to incident row) is not separately instrumented; stopped-vehicle events are by design detected after a dwell and a closed window (up to window_s = 10 s + dwell).

## 8. Known problems and limits of the 20-minute result

* **Stop detection uses the approximate metric scale over the whole frame.** Because the calibration makes the camera "metric" for the engine, stop episodes (0.75 m radius), traffic state and the "world / world_validated / metric_displacement_m" row in the incident page use the extrapolated scale outside the validated near field - and all 14 real stops are on the away-from-camera carriageway. The radius unit is correct (metres in WORLD space); the scale there is approximate. Each stopped-vehicle incident now says this in its uncertainty list; the capability note was reworded. It was not re-run on the 20 minutes (text-only change; unit tests re-run).
* 14 stopped-vehicle incidents in the first ~160 s are queue stops (rated low); the queue is one jam, but each vehicle gets an incident. 13 congestion episodes flap between free flow/building/congested; the count/occupancy proxy is not a speed measure.
* Per-stopped-vehicle "confidence: high" (dwell score) says the stop is clear in image space, not that the incident is meaningful. The motion-space row prints `world_validated` (the schema enum) even for the approximate calibration; the geometry label elsewhere is explicit.
* Tracker ID breaks exist (5,552 session track IDs for roughly 1,000-2,000 vehicles); counts are of session IDs, not unique vehicles.
* The detector's accuracy on this footage is unmeasured. Class labels are coarse; lorries are sometimes "bus".
* No speed event occurred; the over-speed and braking detectors are unit-tested with scripted tracks only. Their thresholds are policy defaults, not tuned on labelled data.
* `demo.mp4` licence unknown; calibration approximate; road identity unverified.

## 9. Tests

All run sequentially (9 pytest processes, `bash demo/tools/run_test_groups.sh`; log `demo/runs/test_groups.log`), each `exit 0`:

| Group | Passed |
|---|---|
| tests/app alerts, api, foundation, incidents | 38 |
| tests/app cases, registry_handoff, registry_policy, plates, plates_robustness | 110 |
| tests/app detectors, event_logic, session_synthetic, ingestion, **live_speed (new)** | 80 (was 54) |
| tests/app retention, security_hardening, account_security | 77 |
| tests/streaming + tests/training | 304 |
| tests/perception | 193 |
| tests/world + safety + storage | 132 |
| tests/evidence, hypotheses, propagation, feedback, fusion, evaluation | 229 |
| tests/test_*.py (17 top-level files) | 187 |
| **Total** | **1350 passed, 0 failed, 0 skipped** (previous 1324; +26 new tests) |

Other verification: `officer_workflow.py` **15/15** (final run: hashed evidence re-hashes; manual plate recorded as `manual_entry`; OCR "unavailable"; override -> 422; registry labelled SANDBOX; hand-off `TEST-CHALLAN-...`; case `challan_reference_received`; package exports; official hand-off 409; demo account cannot register an operational camera 403; audit chain ok). **Restart persistence VERIFIED** (case, 7 transitions, 8 evidence hashes, audit 317 rows). `ui_acceptance.py` **12/12** in headless Chrome (live stream decodes, evidence images load, clip plays 640x360 10.8 s, banner present, no JS errors, no overflow at 1440 and 390 px). **Presenter click-through** (`demo/tools/presenter_walkthrough.py`, fresh `run_demo.py --reset`, default `demo/data`, headless Chrome, no API shortcuts): 17/17 - Cameras -> Start (real video), live page, SIMULATED queue + collision cameras, collision candidate appears once, open case, submit, reviewer verifies evidence, investigator enters the placeholder plate, reviewer2 verifies plate, officer looks up the sandbox registry, approves, submits to the SANDBOX simulator, case shows a TEST- reference. SIMULATED scenarios: collision -> 1 candidate (all gates listed); wrong-way -> 1 event; queue (8 stopped vehicles) -> 0 collisions.
Static checks: `py_compile` of every changed Python file, `node --check` of changed JS. No linter exists in the environment.

## 10. Model and checkpoint inventory (unchanged)

| File | SHA-256 | Note |
|---|---|---|
| `models/yolov8n.pt` | `f59b3d83...ccfc83b36` | default detector (Ultralytics AGPL-3.0 / enterprise licence question open) |
| `models/finetuned/anvesh_mio5_yolov8n_best.pt` | `cdc3cf10960e58eff105fff4584c32c91f810a99c08149fc434ef5ba7f73bf21` | MIO-TCD fine-tune, best checkpoint; CC BY-NC-SA 4.0 data: non-commercial only; not default |
| `models/finetuned/anvesh_mio5_yolov8n_last.pt` | `0ca32c411134e6fed3a187ea95ce8e9de47e94aca6a7a65267aa789dad035119` | |
| `weights/yolo26n.pt` | `9b09cc8b...fad4ef` | present, not used |
Held-out MIO-TCD test lock: 1 run, untouched. `outputs/app` database and credentials: untouched. No plate or accident checkpoint exists.

## 11. Remaining blockers and recommendations (in order)

1. **Replace or license-clear the video** (or get the owner's permission) before any use beyond the classroom. Best lead: the S7 gantry set (verify licence and plate legibility by hand).
2. **Label ~200 frames of the demo footage** and score pretrained vs fine-tuned properly; only then decide promotion. Until then the default stays.
3. **Survey two known distances** (or identify the road) to turn the approximate speed into a defensible one; then tune over-speed/braking thresholds on labelled examples.
4. **Plates need different footage** (>= 60-80 px plate width): a close-range or high-resolution camera, then an Indian-plate detector/OCR with a verified licence. Until then manual entry only.
5. **Collision:** gather real collision/near-collision clips with a clear licence before any claim; keep it experimental.
6. Queue de-duplication (one episode instead of N stopped-vehicle incidents during a jam) and congestion hysteresis tuning.
7. Licence decisions (Ultralytics, MIO-TCD, OCR weights); MFA with an approved library; a real camera/RTSP test; authorised registry/e-challan - all unchanged from sections 2 and 8 of the release-readiness report.

## 12. Commands

```powershell
cd D:\ANVESH
.venv\Scripts\python.exe scripts\run_demo.py --reset                 # fresh demo; browse http://127.0.0.1:8760/ ; passwords: demo\data\initial_credentials.txt
.venv\Scripts\python.exe scripts\run_demo.py --reset --autostart     # acceptance run (starts the 20-minute session)
.venv\Scripts\python.exe demo\tools\officer_workflow.py --data-dir demo\data          # dry run of the officer workflow, 15 checks
bash demo/tools/run_test_groups.sh demo/runs/test_groups.log                         # all tests, 9 processes
```
Presentation sequence and honest capability table: `demo/README.md`. Stop with Ctrl+C. Reset = `--reset` (deletes only `demo/data`).
Leftover disposable data (safe to delete): `demo/data-final`, `data-run1`, `data-ft`, `data-test`, `probe_data` (about 400 MB). C: has 7.7 GB free; nothing was written there beyond tool caches.

## 13. Git and preservation

Branch `main`, HEAD `b6d5ecd`; nothing staged, committed, pushed, stashed, reset or cleaned; `git status` lists the earlier modified files plus the new untracked ones above. Existing databases (`outputs/app`), model checkpoints (hashes above), the held-out lock and the default detector were not modified. Processes: I started and stopped only my own servers (an attempt to stop servers by command-line pattern was blocked by the permission classifier and was redone by exact PID).
