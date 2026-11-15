# ANVESH execution report (living document)

Last updated: 2026-10-21 (fifth session - section 10; section 9 is the 2026-10-20 record, sections 1-8 the 2026-10-18 record). Nothing below is claimed unless a log or command output in this session supports it.

## 1. Status at a glance

| Track / stage | State | Evidence |
|---|---|---|
| A. Vehicle detector (MIO-TCD, YOLOv8n) | **trained (8/8 epochs), evaluated on val, held-out test and CityFlow; NOT promoted to app default** | `docs/ANVESH_MODEL_TRAINING_REPORT.md`, `outputs/training/compare_*.json` |
| B. Plate detector | dataset **present on disk and untracked at the start of this session** (`datasets/plates_hf/`, 470 MB; origin of the download not recorded in this session), not yet trained | section 5.1 |
| B. Plate OCR | **blocked** (no accessible Indian text-labelled dataset) | section 5 |
| C. Accident detector | dataset vetted, not downloaded, not trained | section 6 |
| Streaming / dashboard stages | existing MVP unchanged this session, not yet re-verified | - |

## 2. What was inspected (this session)

- Processes: no training process alive at session start. RAM 15.69 GB total, 3.45 GB free (browsers hold most of the rest). GPU RTX 3050 6 GB Laptop, 0 MiB used.
- Disk: **C: 2.8 GB free** (risk: pagefile/temp), D: 125.6 GB free, E: 82.3 GB free.
- Previous run `models/finetuned/runs/anvesh_mio5_v1`: contains only `args.yaml` and an empty `weights/`. It died during Ultralytics' label scan at 57,434 / 88,270 images. **0 epochs completed, no checkpoint.** Kept as the record of the failed attempt.
- `datasets/mio_tcd/yolo5/labels/train.cache` (972 KB, 12:41) is not a cache of the full 88,270-image list (the scan only writes it when it finishes; it is left over from the smoke subset), so Ultralytics rescans.
- Dataset integrity: image/label file counts per split match `dataset_manifest.json` (88,270 / 10,865 / 10,865). Archive SHA-256 re-computed = manifest value `0801d0ee34e7fb211945cd5ef836551efadac3a1ec876b126608ec5fbe7fab64`.
- Base checkpoint `models/yolov8n.pt` SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

## 3. Track A: vehicle detector

- Not resumed: nothing to resume (no `last.pt`). Fresh run, unique name `anvesh_mio5_v1b`.
- Command: `PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/train_detector.py --name anvesh_mio5_v1b --epochs 8 --batch 16 --imgsz 640 --patience 4 --close-mosaic 2 --workers 2`
- **Deviation from the first attempt: `--workers 4` -> `--workers 2`** because only 3.45 GB RAM was free. Nothing else changed.
- Training runs alone (no regression suite, no other GPU job).
- Label scan of 88,270 images took about 8.5 min (no usable cache existed). Epoch 1 started at about 3 it/s (5,517 iterations/epoch) -> about 32 min of training per epoch plus validation, so roughly 4.5 h for 8 epochs (the earlier 2.8 h estimate assumed 4 workers). GPU memory 2.1 GB of 6 GB. Free system RAM was 0.25-1.2 GB throughout, mostly because of open browsers; this is the main risk to the run. Ultralytics writes `last.pt` after each epoch, so a kill after epoch 1 would be resumable.

**Result (filled 2026-10-18 from run files):** completed 8/8 epochs in 2.97 h, still improving at epoch 8 (not converged). Best checkpoint SHA-256 `cdc3cf10960e58eff105fff4584c32c91f810a99c08149fc434ef5ba7f73bf21`.

| Metric (shared 5-class taxonomy) | pretrained COCO | fine-tuned |
|---|---|---|
| MIO-TCD **val** mAP50 / mAP50-95 | 0.236 / 0.150 | 0.885 / 0.667 |
| MIO-TCD **held-out test** mAP50 / mAP50-95 | 0.215 / 0.142 | 0.870 / 0.651 |
| Test P@0.25 / R@0.25 | 0.641 / 0.364 | 0.880 / 0.913 |
| **CityFlow S02** class-agnostic recall (IoU 0.5; precision not computable) | **0.748** | 0.718 |

Test per-class AP50 (fine-tuned): bicycle 0.705 (213 boxes), car 0.944, motorcycle 0.866 (269), bus 0.955, truck 0.877. Full per-class tables, bootstrap CIs, caveats and the unresolved Ultralytics cross-check gap (ours lower by 0.011-0.045 AP50): `docs/ANVESH_MODEL_TRAINING_REPORT.md`. The held-out test was run once; lock file `datasets/mio_tcd/yolo5/test_evaluation.lock.json` records it. **Decision: fine-tuned detector not promoted** (CityFlow recall not better; no downstream/latency comparison yet).

### 3.1 Verification run, 2026-10-18 (second session)

- At session start: no python processes, no test lock, GPU idle. Free RAM 4.19 GB, **C: free 0.7 GB (was 2.8 GB; risk)**.
- Held-out test: `compare_detectors.py --split test ... --crosscheck-val --out outputs/training/compare_test.json` -> exit 0, ran once.
- CityFlow: `compare_detectors_cityflow.py ... --out outputs/training/compare_cityflow.json` -> exit 0.
- `summarize_training.py --run models/finetuned/runs/anvesh_mio5_v1b ...` -> exit 0 (the script docstring's example still says `anvesh_mio5_v1`; the real run is `v1b`). Output: `outputs/training/training_summary.md`, figures in `docs/training_figures/`.
- Regression, first run (alone, `python -m pytest -q -rfEs -p no:cacheprovider --basetemp=outputs/pytest_tmp/full`): **8 failed, 1072 passed, 0 skipped**, 144 s, exit 1. All 8 had one cause: `scripts/run_real_world_validation.py:180` and the export path read `tracker.model_id` (a real `ByteTrackYoloTracker` property) but the test doubles (`_FakeTracker`, `object()` stubs) lacked it. Production code unchanged.
- Fix (tests only): `model_id` added to `_FakeTracker` in `tests/test_real_world_validation_runner.py` and a `_StubTracker` in `tests/test_export_ui_data.py` replacing 5 `object()` stubs.
- Regression, second run (same command, `--basetemp=outputs/pytest_tmp/full2`): **1080 passed, 0 failed, 0 skipped**, 245.8 s, exit 0 (`outputs/training/logs/pytest_full2.log`).
- Real-checkpoint tests, run explicitly (`pytest -v -rs tests/perception/test_class_map_and_weights.py -k real_`): 2 passed (pretrained COCO mapping; fine-tuned 6-class layout + tracker inference on a blank frame).

## 4. Code changes this session

- `tests/perception/test_class_map_and_weights.py`: the real-checkpoint test now expects the 6-class layout (`other_vehicle` present in the model, ignored by `ClassMap`).
- `scripts/train_detector.py`: docstring said "5-class"; the dataset has 6 training classes.

## 5. Track B: plate detection and OCR dataset vetting (web lookups only, nothing downloaded)

| Candidate | Licence / terms found | Verdict |
|---|---|---|
| HF `keremberke/license-plate-object-detection` (from Roboflow Universe "Vehicle Registration Plates", Augmented Startups) | CC BY 4.0 per the HF page; 6,176 / 1,765 / 882 images, 1 class `license_plate`, boxes only, region not stated | **usable for the plate *detector***; no plate text, so not for OCR; region unverified |
| Indian_LPR (arXiv 2111.06054, 16,192 images, 21,683 plates, char-level labels) | author states the dataset is **not public** ("legalities") | blocked |
| IEEE DataPort Indian plate sets (YOLO + OCR) | **subscription required**, licence not stated | blocked (credential) |
| Kaggle `kedarsai/indian-license-plates-with-labels` | page content unreadable without login, licence unverified | blocked until licence/login verified |
| Roboflow Universe Indian-plate-detection | HTTP 403, API key needed | blocked |
| CCPD (Chinese plates, 300k+ images, text in filename) | MIT per repo README; downloads from Google Drive/Baidu | only verified text-labelled option, **out-of-domain (Chinese plates)** |
| UFPR-ALPR | academic research only, request required | not pursued |

## 6. Track C: accident dataset vetting (web lookups only, nothing downloaded)

| Candidate | Finding | Verdict |
|---|---|---|
| CADP (CCTV, 1,416 YouTube-derived segments, 205 with spatio-temporal annotation) | page says download implies "non-commercial and research based purposes" use; raw video vs frames not confirmed | best fit (fixed CCTV), needs README check; non-commercial |
| DoTA (dashcam, 4,677 videos) | dashcam perspective; downloader scrapes YouTube | not suitable for CCTV |
| TUM Traffic Accid3nD | CC BY-NC-SA 4.0; accident timing not described; 3D multi-sensor, large | not suitable |
| DAD | dashcam, no licence info found | not pursued |

### 5.1 Plate detector attempt (2026-10-18, second session)

- `datasets/plates_hf/` (470 MB; `keremberke/license-plate-object-detection` Roboflow export) was already present and untracked at session start; who downloaded it is not recorded. Zip SHA-256: train `d44bf42f2a46e3df0d88f98311d6bca0eaadbaa0958aa41b629e1c101baddb14`, valid `2ec44d2f6d418a5ad30783bccd2d1883b45aca3baa02843ae5d2421796fe0c01`, test `8fbc65b333765e9883ba0a86a767c87875c6e1a420ff9eb6c411ed9e58155026` (also in `datasets/plates_yolo/dataset_manifest.json`).
- Checks (script `scripts/prepare_plates_hf.py`, new): one category (`license_plate`, no phantom class), 6,176 / 1,765 / 882 images with 6,413 / 1,840 / 902 boxes, no image without a box, **0 base-id overlap between splits**. Converted to `datasets/plates_yolo/` (new; `plates_hf` untouched).
- **Domain finding (sampled 12 train images):** plates are Vietnamese (e.g. `51A-052.27`), taken at close range at a parking-barrier camera; 82% of images are 472x303; some frames are padded/rotated although the README says no augmentation. A detector trained here does not represent wide-angle Indian CCTV, and its test score would be inflated by same-camera near-duplicates (split leakage by *vehicle/scene* was not ruled out; only filename overlap was).
- Licence: README says CC BY 4.0, "Provided by undefined"; attribution chain unverified; plate region not Indian. Not cleared for deployment.
- **Training: NOT completed.** Command `train_detector.py --data datasets/plates_yolo/data.yaml --name plates_hf_v1 --epochs 10 --batch 16 --imgsz 640 --patience 4 --close-mosaic 2 --workers 2`. Claude Code's memory-pressure guard stopped the job during epoch 1 (311/386 iterations, 5.8 it/s). No epoch finished: no `results.csv`, no weights, no manifest in `models/finetuned/runs/plates_hf_v1/`. Not restarted (per the tool's instruction). Nothing was written to `models/plate_detector.pt` (the app auto-loads that path); no plate detector exists.
- Free RAM afterwards 6.9 GB, C: 1.5 GB free.

## 6.1 OCR dataset and CADP licensing research (web lookups only; nothing downloaded; ran while plate training was in progress)

| Item | Finding | Verdict |
|---|---|---|
| Indian_LPR (arXiv 2111.06054) | abstract: 16,192 images, 21,683 plates, 4-point plate and per-character annotations; the arXiv page gives no download link; the only licence shown (CC BY 4.0) covers the paper, not stated for the data; earlier session found the author saying the data is not public | blocked / unverified |
| IEEE DataPort Indian plate sets | subscription-gated, no licence shown | blocked |
| Roboflow Indian-plate-detection | CC BY 4.0 shown, but detection only (no text) | not an OCR source |
| **CADP** (project page + README, read as text only) | Terms: "By downloading this dataset you agree to use this dataset for non-commercial and research based purposes"; README: "Available for research and non commercial use ... We prohibit from other uses of the data without prior written consent from authors." **Raw videos are NOT shared** (YouTube ID list only; "We cannot share the original YouTube videos"). Provided: extracted frames (Google Drive), annotations JSON (filedn.com), durations. Spatio-temporal boxes for Person/Car/Bus/Two-wheeler/Three-Wheeler/Other. Frame rate, resolution, count and **download size are not stated** | research-only; cannot be assumed to permit deployment or a deployed model trained on it; size unknown until the Drive file is opened |

## 7. Open items

See the ranked list in section "Remaining work" once the session closes. Nothing has been staged, committed or pushed.


## 8. Local-CCTV benchmark and detector promotion gate (third session, 2026-10-18)

Mandate: resolve the evaluator discrepancy, build a reproducible local-camera benchmark workflow, do not promote, do not touch the app default, do not retrain, no plate/OCR/accident work. Nothing staged, committed or pushed.

| Item | State | Evidence |
|---|---|---|
| Evaluator vs Ultralytics `val()` gap | **explained on val** (multi-label NMS); test closure inferred, not measured | `docs/ANVESH_MODEL_TRAINING_REPORT.md` section 7.1, `outputs/training/diag_eval_gap.json` |
| CityFlow recall gap vs ignored `other_vehicle` | **ruled out** (27 boxes, recall unchanged 0.7178) | section 7.3, `outputs/training/diag_cityflow_recall.json` |
| Pretrained vs fine-tuned fairness | checked: identical images/settings; ranking robust to NMS convention | training report 7.2 |
| Local-camera annotation/import/QA/split workflow | **built and unit-tested on a synthetic fixture** | `docs/ANVESH_LOCAL_CCTV_BENCHMARK.md`, `anvesh/training/local_cctv.py`, `scripts/local_cctv.py` |
| Local benchmark (pretrained vs fine-tuned) | **NOT RUN - no labelled local footage exists.** The benchmark exits "NOT RUN" with code 3 and writes nothing when no labelled frames exist (tested) | needs your footage and annotations (section 1 of the benchmark doc) |
| Promotion gate | defined, implemented, tested; actual output today **DO_NOT_PROMOTE** (CityFlow line fails on its -0.030 point estimate; all local lines not evaluable) - `outputs/local_benchmark/gate_2026-10-18.json` | benchmark doc section 6 |
| Latency | tool built; measurements unstable (3-6x run-to-run) so **no latency claim** | benchmark doc section 7, `outputs/local_benchmark/latency_cityflow_c006*.json` |
| App default model | unchanged (`models/yolov8n.pt`) | `anvesh/app/config.py` untouched |

### 8.1 Commands run this session (all from `D:\ANVESH`, `.venv\Scripts\python.exe`)

* `scripts/diagnose_eval_gap.py --n 1500 --out outputs/training/diag_eval_gap.json` -> exit 0 (first attempt failed with "Inference tensors do not track version counter" because a model object was reused for `predict` after `val()` had fused it; fixed by using a fresh instance, second run exit 0).
* `scripts/diagnose_cityflow_recall.py --out outputs/training/diag_cityflow_recall.json` -> exit 0; operational recall reproduced `compare_cityflow.json` exactly (TP 1,575 / 1,511).
* `scripts/benchmark_local_cctv.py latency ...` on CityFlow S02 c006 frames, four invocations (two superseded non-interleaved, two interleaved) -> exit 0, results unstable (benchmark doc section 7).
* Tests: see 8.2.
* `benchmark_local_cctv.py run --split all` on the synthetic fixture with the real `models/yolov8n.pt` (tooling check only: flat gray frames and fabricated labels; JSON written, error visuals rendered; no number from it is a result). `... gate --cityflow outputs/training/compare_cityflow.json` -> DO_NOT_PROMOTE.
* Not run: `compare_detectors.py --split test` (locked, left alone), any training, any local-footage benchmark.

### 8.2 Test results

`pytest tests/training tests/perception/test_class_map_and_weights.py -q -p no:cacheprovider --basetemp=outputs/pytest_tmp/local4`: **52 passed, 0 failed** (re-run after the last code change; 16 new tests in `tests/training/test_local_cctv.py` use only a synthetic fixture and say nothing about any detector). The full 1,080-test suite was not re-run: no existing source file was modified.

### 8.3 Files changed or added this session

Added: `anvesh/training/local_cctv.py`, `anvesh/training/local_benchmark.py`, `scripts/local_cctv.py`, `scripts/benchmark_local_cctv.py`, `scripts/diagnose_eval_gap.py`, `scripts/diagnose_cityflow_recall.py`, `tests/training/test_local_cctv.py`, `docs/ANVESH_LOCAL_CCTV_BENCHMARK.md`, `outputs/training/diag_*.json` and logs, `outputs/local_benchmark/*.json`. Edited (append-only / one line): `docs/ANVESH_MODEL_TRAINING_REPORT.md` (section 7), `docs/ANVESH_EXECUTION_REPORT.md` (this section), `.gitignore` (+`datasets/local_cctv/`). Scratch copy of 1,500 val images/labels: `outputs/training/diag/val_subset/` (safe to delete; `outputs/` is gitignored).

### 8.4 Blockers and the single next action

Blocker: no labelled footage from the intended cameras. **Next action:** record 3+ cameras x 2+ recordings (including night if in scope), run `local_cctv.py extract`, annotate ~400 frames with the rules in the benchmark doc (CVAT YOLO export), import, `qa`, `split --group-by recording`, then `benchmark_local_cctv.py run --split val` to pick settings and `--split test` once. Confirm or adjust the proposed thresholds in benchmark doc section 6 before that run.


## 9. Fourth session (2026-10-20): inventory, hardening, streaming, retention, tooling, documentation

Mandate: finish everything that does not need local CCTV footage, honestly. Rules kept: worked only in `D:\ANVESH` on `main`; **nothing staged, committed, pushed, reset or deleted**; no checkpoint overwritten; **the held-out MIO-TCD test was not re-run** (lock file SHA-256 `9850d283...f44a4` identical before and after the whole session, 1 recorded run; a new test now guards this); no dataset downloaded; no training started; no package installed; nothing was run against `outputs/app` (the acceptance evidence) - the demo used a fresh data directory `outputs/exec4/app`.
Conditions: RAM free 1.9-2.6 GB of 15.7 GB (browsers), `C:` 1.6-2.9 GB free (risk), `D:` 125 GB free, GPU idle, no python processes at start.

### 9.1 Phase 1 - current state (verified against code and tests, not copied from reports)

Starting point: `main` @ `b6d5ecd`, 96 `git status` entries (45 modified, 51 untracked) - all pre-existing M-A/M-A.1 modifications and the earlier sessions' work, left untouched except where listed in 9.Z. The reports were checked against code: the acceptance report's claims about the app (roles, four-eyes, immutable tables, evidence hashing, restart recovery via `mark_interrupted`, source redaction, sandbox isolation triggers) all held when re-read and when the demo was re-run; the `app` suite (129 tests) was green before any change (3 min 16 s).

| Class | Items |
|---|---|
| **Verified working** (re-verified this session) | login / roles / session revocation; CSRF for cookie sessions; audit chain + SQL append-only triggers; evidence capture + hashing; incident -> case -> sandbox hand-off -> package; restart persistence (`demo_e2e.py --verify-only`: VERIFIED); stream queues / drop accounting / reconnect with fake captures; sandbox isolation at SQL level; fail-closed speed gating; missing-weights fail-closed |
| **Implemented but under-tested before this session** | request framing / timeouts; camera update validation; wrong-typed JSON on every route; corrupt frames; truncated recordings; concurrent-source isolation; incremental-processing proof; OCR engine failure / invalid output; retention (report only); one-run test locks (enforced in scripts, never tested); manifest/event tooling (absent) |
| **Missing or incomplete** | evidence expiry (now built); event-class taxonomy (now added); stationary-group queue rule (now added); accident evaluation tooling (now built); training pre-flight / resume (now built); working OCR and plate detector; helmet/seatbelt; map view; audit expiry; real RTSP/webcam validation; UI for retention plan/purge |
| **Blocked: needs local CCTV footage** | detector accuracy on your cameras; promotion decision; stop-vs-queue precision and the `queue_peer_stops` value; wrong-way/lane/red-light/turn on real examples; overspeed / braking (needs survey-validated geometry); plate legibility; real latency at your resolution; false-alarm rate of `possible_collision` |
| **Blocked: external data / licences / credentials / authorisation** | Ultralytics (AGPL-3.0 vs enterprise) and MIO-TCD (CC BY-NC-SA 4.0) decisions; Indian plate OCR data (Indian_LPR not public, IEEE DataPort subscription, Kaggle unverifiable); registry and e-challan access; webhook / SMTP; camera credentials; OCR weights licence check |
| **Preserved artifacts** | `outputs/app` (acceptance data), `outputs/demo`, `outputs/training`, `outputs/local_benchmark`, `models/finetuned/*` (best 6,228,010 B SHA `cdc3cf10...bf21` unchanged), `datasets/mio_tcd` incl. the test lock, `datasets/plates_hf` / `plates_yolo`, `runs/`, `weights/`, all untracked audits and reports |

### 9.A Security and reliability (every item: failing test first, then fix)

Method: read `server.py`, `routes.py`, `auth.py`, `cameras.py`, `cases.py`, `plates.py`, `evidence.py`, `db.py`, `services.py`, `runner.py`; wrote regression tests (`tests/app/test_security_hardening.py`, 15 tests); ran them against the **unfixed** code (12 of the 13 non-fuzz tests failed as predicted; the 13th, non-numeric Content-Length, already passed); fixed; reran.

| # | Finding | Severity | Fix | Test |
|---|---|---|---|---|
| A1 | `Content-Length: -1` -> `rfile.read(-1)` blocks the handler thread until the client leaves (thread exhaustion) | high | strict `content_length()`: negative/non-numeric -> 400 and connection closed | `test_negative_content_length...` (live: HTTP 400) |
| A2 | chunked request bodies not decoded -> their bytes parsed as the next request on a keep-alive connection | medium | `Transfer-Encoding` refused with 411 + close | `test_chunked_request_bodies...` (live: HTTP 411) |
| A3 | no per-connection timeout (slow-loris holds a thread forever) | medium | `Handler.timeout` = 60 s (`request_timeout_s`) | `test_idle_and_half_open_connections...` |
| A4 | **a `file` camera could be registered on `anvesh_app.sqlite3` / `initial_credentials.txt` (the repo root is an allowed media root and no extension check existed); session start wrote the snapshot before opening, and `/sessions/<id>/video` streamed the path to any user with `view.evidence`** - password hashes, registry owner data and the audit store were downloadable | **high** | file sources must be video files, inside an allowed root, outside the data dir except `uploads/`; re-checked at session start and replay; `VIDEO_EXT` single definition | `test_file_cameras_cannot_point...`, `test_update_cannot_repoint...`, `test_replay_endpoint_refuses_a_legacy_camera...` (live: 422 for the db and the credentials file) |
| A5 | `CameraService.update` wrote name/lat/lon/offset/fps/enabled without validation (live `nominal_fps` bypass) | medium | shared validators for create and update | `test_camera_update_validates_every_field_it_writes` |
| A6 | non-dict / wrong-typed nested camera config (`lanes: [5]`, `thresholds: [1]`, `calibration: "x"`) -> HTTP 500; NaN/inf accepted | medium | structural checks, finite-number checks, parser wrapped to raise `ConfigError` | `test_malformed_nested_camera_config...` |
| A7 | `active: 0` / `"false"` bypassed the self-lockout guard (`is False`) and were coerced by truthiness; non-string password / role -> 500; unbounded username/password | medium | typed accessors (`Request.opt/need`), service-level type and length checks | `test_user_update_input_types...` |
| A8 | case-action body keys (`user`, `case_id`) collided with service arguments -> 500 | low | only 4 named parameters forwarded; unknown keys 422 | `test_case_action_cannot_be_fed...` |
| A9 | login: username of 100 KB written into the audit actor field; failure table could grow without bound | low | username capped, table bounded (oldest evicted) | `test_login_input_is_bounded...` |
| A10 | **a plate corrected after `verify_plate` kept `plate_verified_by` pointing at a person who never saw the new text; approval passed** | high (integrity) | `cases.plate_verified_reading_id` (schema v2, backfilled) - readiness requires the verification to cover the current reading; `request_more` clears it | `test_a_plate_changed_after_verification...` |
| A11 | evidence package: the manifest hashed whatever bytes were on disk, so a tampered file produced a self-consistent package; a missing/purged file crashed the export (500) | medium | every file checked against the registered SHA-256; mismatch/missing -> 409 and no package; purged files listed | `test_export_refuses_tampered_evidence...` |
| A12 | wrong-typed fields anywhere else | - | fuzz: 4 roles x every POST/PATCH/PUT route x ~250 bodies | `test_no_mutating_endpoint_returns_500...` passes; `test_fuzz_harness_detects_an_unhandled_exception` proves the harness reports a planted fault |

Not changed on purpose (policy / cannot be fixed in code): P1 registry override on operational cases (docstring and tests disagree - needs your decision), P2 admins can create officer accounts, P3 per-username in-memory lockout / no MFA. Details: `docs/ANVESH_RELEASE_READINESS.md` section 3. Existing tests I changed: `test_api.py` (schema version literal `1` -> `SCHEMA_VERSION`; `purge_implemented is False` -> the new retention report fields). Neither weakens a check. The pre-existing assertion `self_lockout[0] in (200, 409)` was left (weak); the strict version is in the new tests.
Restart recovery: `mark_interrupted` verified by the demo's restart (sessions left running become `interrupted`); malformed requests and unauthorised access are covered by the RBAC matrix test and the fuzz.

### 9.B Streaming and incremental processing

Verified by reading `pipeline.py` / `rtsp.py` / `ingestion.py` and by tests. Frames are processed one at a time through bounded queues; windows close on stream time. New proof: `test_windows_are_delivered_before_the_source_ends...` makes the source *refuse to yield frame 250 until a window has been delivered* (a pipeline that needed the whole video would deadlock).
**Dashboard mechanism (checked in `app.js`)**: the browser polls the JSON API with `setTimeout` (about 4 s on the incident views) and shows the live annotated view as an `<img>` of `multipart/x-mixed-replace` MJPEG. There is **no WebSocket or SSE**; I did not build one. Incremental update of the dashboard is therefore "poll + MJPEG", verified in the previous session's Chrome run, not re-driven in a browser this session (no JS changed).
Found and fixed: a corrupt live frame (`ok=True` with `None`/empty image) would have reached the detector and ended the whole session -> now skipped and counted (`SourceHealth.corrupt_frames`), 30 in a row = reconnect; a truncated recorded file ended "cleanly" because OpenCV returns the same False for EOF and a failed read -> now flagged when fewer frames were decoded than the container reports (session stays `completed` with `source_warning`; an MP4 whose index was cut off already failed loudly at open - tested).
New tests (`tests/streaming/test_synthetic_load.py`, 12): incremental delivery, bounded memory, stalled consumer (window drops counted, queue never above capacity), frame-drop conservation for queue sizes 1/2/8, 8 concurrent sources (7 bit-identical to their single-source run, 1 crashing alone), `stop()` of an endless source leaves no `anvesh-*` threads, corrupt frames (isolated / run / end-to-end), short file. **All sources are doubles (in-memory lists, generators, injected fake captures) and are labelled so in the file; none is real webcam / RTSP / RTMP acceptance.**
Measurements (`scripts/stream_synthetic_bench.py`, 5 repeats + 1 discarded warm-up, sequential, stub tracker, paced sources where drops are measured; `outputs/exec4/stream_synthetic_bench.json`; flagged UNSTABLE when max/min of p95 or throughput exceeds 1.5): 

| scenario | throughput fps min/med/max | e2e p95 ms min/med/max | queue drops | stable? |
|---|---|---|---|---|
| steady lossless, unpaced | 7585 / 9584 / 17518 | 0.7 / 1.2 / 1.6 | 0 | **UNSTABLE** |
| live keeps up, 100 fps paced, 5 ms stub | 50 / 99 / 100 | 8.8 / 35.1 / 107.6 | 387 over 5 runs | **UNSTABLE** |
| live overload, 400 fps paced, 5 ms stub | 142 / 146 / 155 | 19.3 / 21.9 / 23.0 | 3771 | stable |
| 4 concurrent sources, 50 fps each, 2 ms | 199.9 / 200.3 / 200.5 | 3.2 / 3.5 / 4.5 | 0 | stable |
| 8 concurrent sources, 50 fps each, 2 ms | 399.3 / 399.9 / 400.5 | 3.4 / 3.4 / 3.6 | 0 | stable |
| fake RTSP, disconnect every 200 frames | 18789 / 25557 / 26657 | n/a (virtual clock) | 0 | stable |

Reading it: the plumbing sustains 8 x 50 fps of stub work with no drops on this laptop; the "keeps up" scenario drops frames on some runs and swings 12x in p95 because Windows `time.sleep` granularity (the 5 ms "detector" sleeps ~15 ms) made capacity ~150 fps, not 200 - a property of the stub, **not** a detector result. An earlier unpaced version of the overload scenario reported ~99 % drops, an artefact of an infinitely fast source; it was replaced. **None of these numbers is detector, camera, dashboard or application latency.**

### 9.C Tracking and event logic

Existing coverage is extensive (track manager identity/expiry/occlusion, stop episodes across windows, displacement criterion, lane geometry, congestion hysteresis, collision/red-light/turn detectors); I added only what was missing. **Real finding:** with the default `max_capacity_vehicles` (20), a queue of 8 cars stationary for 40 s produced **8 separate HIGH-severity "long stop outside congested traffic" incidents** (probe on the simulated `queue` scenario) because queue context relied only on a count/occupancy proxy. Fix: a *stationary-group* rule (>= `queue_peer_stops`, default 3, vehicles stationary in the same window => severity low, uncertainty note, `indicators.queue_signal`); the incident is still raised and the head-of-queue caveat is stated. Limits: severity never downgrades once raised; the value 3 is untuned; precision on real roads is unmeasured. Added `tests/app/test_event_logic.py` (16 tests): lone stop = obstruction candidate, queue = low, configurable group size, wrong-way direction boundary (cos < -0.7: 134 deg no, 140 deg yes), identity-switch de-duplication, incident status machine (closed is terminal), dismissed incidents cannot seed cases, classification. Speed- and geometry-dependent events stay unavailable without survey-validated geometry (tested).

Event classification (`implementation_class`, now served by the capabilities API):

| Class | Event types |
|---|---|
| implemented heuristic (rules over tracked boxes; no trained event model) | stopped_vehicle, long_duration_stop, congestion, possible_collision (experimental), wrong_way*, wrong_lane*, illegal_turn*, rash_driving*, post_incident_disruption (* needs per-camera zones/lanes/rules) |
| integration-dependent | overspeed, sudden_braking (survey-validated metric geometry), red_light_violation (external or configured signal state) |
| trained model | **none** - the only learned component is the vehicle detector that supplies the boxes |
| not implemented | helmet_violation, seatbelt_violation (no detector); learned accident detection; plate OCR as a working capability |

### 9.D Dashboard, incidents, evidence, retention

`demo_e2e.py` on a fresh data directory and port (60 s of `demo.mp4`; real YOLOv8n + ByteTrack, 1501 frames in 31 s, 0 dropped): incidents `stopped_vehicle` 1 + `congestion` 1 (6 evidence assets, hashes ok), case created -> evidence verified -> plate entered (DEMO PLACEHOLDER, OCR unavailable, as stated by the API) -> plate verified -> sandbox registry -> approved -> SANDBOX hand-off `TEST-CHALLAN-...` -> `challan_reference_received`; simulated collision fixture -> `possible_collision` candidate -> TEST alert "not delivered to any external party". Audit chain ok (81 rows), then **restart: VERIFIED** (case + 7 transitions, hashes, chain ok at 84 rows). Live curl checks: retention plan 200, `Content-Length: -1` 400, chunked 411, file cameras on the database/credentials 422. Filtering, pagination, empty and invalid input paths are covered by `test_api.py` (existing), the fuzz and the retention tests; the dashboard JS was not changed and was not re-driven in a browser.
Retention (`anvesh/app/retention.py`, 10 tests): evidence **files** only; plan -> `plan_id` -> purge with a reason; tombstones (`evidence_purges`, append-only) keep hashes; open incidents and any unfinished case protect; minimum period 30 d for a purge; interrupted purges finish on the next run; paths outside the evidence root are never deleted; purged assets answer 410 `purged`; exports skip and list them. Everything is tested on temp directories only. No UI added.
Sandbox labelling re-checked: every sandbox registry / challan result still carries SANDBOX / TEST labels in the API, report and package; the SQL triggers still refuse them on operational cases. No map or live data was invented.

### 9.E Number-plate detection and OCR

Interrupted plate run `models/finetuned/runs/plates_hf_v1`: contains `args.yaml`, `labels.jpg`, three `train_batch*.jpg` and an **empty** `weights/` - no `results.csv`, no `last.pt`, no `best.pt`: **nothing to resume and no valid checkpoint** (nothing deleted; `train_detector.py --resume plates_hf_v1` now refuses with that reason, tested on the real directory). `anvesh_mio5_v1` (the first MIO attempt) is likewise empty. `models/plate_detector.pt` does not exist and was not created.
Built: `anvesh/training/yolo_dataset.py` + `scripts/validate_yolo_dataset.py` (missing vs empty label files, class ids, box ranges, orphans, unreadable images, exact and base-id leakage); `anvesh/training/preflight.py` + `train_detector.py --preflight-only/--skip-preflight/--resume` (refuses to start when free RAM / disk / VRAM are short; never predicts runtime without a measured throughput; never overwrites a manifest); `scripts/audit_split_leakage.py`. 
Run on the real `datasets/plates_yolo` (read-only): **OK** - 6176/1765/882 images, 6413/1840/902 boxes, 0 missing or empty labels, 0 identical images and 0 base ids across splits (`outputs/exec4/plates_yolo_validation.json`, 4 min 45 s). Thumbnail near-duplicate screen (`plates_leakage.json`): 2.0 % of val and 2.2 % of test images have a near-identical training image (RMS <= 6); median distance to the nearest training image 28.8 / 29.4 vs 34.0 / 36.8 to the split's own images - mild scene overlap, not the heavy leakage that was feared; the dataset's real problem remains the domain (Vietnamese plates, parking-barrier camera). This neither clears the licence nor makes it an Indian-plate detector.
Pre-flight on this machine now: **REFUSED** (free RAM 2.24 GB < 3 GB; temp drive 1.58 GB free warning). **No training was attempted this session.**
OCR code hardened and tested with doubles (`test_plates_robustness.py`, 26 tests): the 0.85 threshold boundary, malformed marks never candidates, invalid engine confidence (NaN/inf/out of range/bool/str) discarded -> `needs_review`, several plates per crop, empty/unreadable crops (cannot be verified, can be corrected), engine exception on one crop recorded as `unreadable` + audit entry while other crops continue, detector exception reported, tampered evidence frames refused, repair suggestions labelled `derived_positional` and never replacing the read text, correction chain with a different verifier. **No OCR engine or plate model is installed, so OCR is not integrated or accurate - claim nothing.** Engines reviewed (none installed; EasyOCR, PaddleOCR, Tesseract: Apache-2.0 code, weights/traineddata licences must be checked at install - release readiness section 4).

### 9.F Accident-detection research pipeline

`docs/ANVESH_ACCIDENT_RESEARCH_PIPELINE.md` (architecture, annotation schema, split protocol, evaluation protocol, candidate data). Code: `anvesh/training/temporal_events.py`, `scripts/event_manifest.py`, `tests/training/test_temporal_events.py` (37 tests, synthetic data only): schema validation (missing-vs-empty `fully_annotated`, required licence, overlaps), source-group splitting (whole groups, events in every split, refuses instead of returning an empty split, deterministic), event evaluation (recall, delay with early alarms preserved, false alarms/hour only over verified-negative video, cooldown merge, video-level bootstrap, operating curve, `claims_allowed` false below 30 events / 10 h or off the test split). **No accident model, dataset download or training.** CADP: research / non-commercial, raw video not distributed; a web search today could not re-confirm the terms.

### 9.G Vehicle model evaluation and local-benchmark readiness

Nothing re-run. Verified by test and by hash: the real MIO-TCD lock still has 1 run (`test_the_real_mio_tcd_test_lock...`), the lock-refusal path of `compare_detectors.py` (temp dataset, lock bytes unchanged), the local benchmark's `test` / `eval` one-run lock with explicit `--allow-test-rerun` (exit 4 then override), `val` not locked, and again that all-unlabelled footage cannot be scored and the gate is INCONCLUSIVE. Already covered earlier and re-read: missing label != empty label, grouped split / leakage / adjacent-recording warnings, manifest history, QA, import, gate (held-out `test` only). The NMS discrepancy diagnostics (multi-label NMS in `val()` vs single-label in `predict()`) were reviewed in the training report 7.1 and not repeated. Promotion: **unchanged - not promoted**; `anvesh/app/config.py` default untouched.

### 9.H Documentation

New: `docs/ANVESH_RELEASE_READINESS.md` (acceptance matrix, blockers, licences, authorisations, rollback, commands), `docs/ANVESH_ACCIDENT_RESEARCH_PIPELINE.md`. Updated: `docs/ANVESH_APP_GUIDE.md` (security model, retention, schema v2 and rollback, new threshold, endpoints), this section, an addendum in `docs/ANVESH_ACCEPTANCE_REPORT.md`, a note in `docs/ANVESH_MODEL_TRAINING_REPORT.md` and `docs/ANVESH_LOCAL_CCTV_BENCHMARK.md`. Earlier sections of every report are unchanged (historical).

### 9.T Tests - what ran, what did not (read this before trusting "green")

**The full suite did NOT complete.** I started it once, after the code was stable, as `python -m pytest -q -rfEs -p no:cacheprovider --basetemp=outputs/pytest_tmp/exec4_full` (output `outputs/exec4/pytest_full.log`). Claude Code's memory guard killed it at about 15 % ("system is running low on memory", free RAM 1.9-3 GB with browsers open); the log shows one `F` in the first 5 % and no summary, so it can neither confirm nor refute anything. Per the tool's instruction I did **not** restart it. The `F` was almost certainly `test_api.py::test_validation_and_pagination_and_error_shape` (found and fixed below, because I had not run `test_api.py` after my changes - my mistake).
What I ran instead (all `-p no:cacheprovider`, sequential, each after the final edit of the code it covers unless noted):

| Run | Result |
|---|---|
| baseline `tests/app` before any change | 129 passed (3 min 16 s) |
| `tests/app/test_security_hardening.py` against the UNFIXED code | 12 failed, 1 passed (the expected failures) |
| `test_api.py` + `test_security_hardening.py` + `test_ingestion.py` after the file-source fix | 35 passed, 1 failed -> the failure was a flaky helper in my chunked-body test (reads one TCP segment); helper fixed, 4 framing tests passed on 3 consecutive reruns |
| `test_api.py` alone | found 1 real regression of mine: the extension check came before the media-root check so an existing test saw a different 422 message; fixed by checking the root first (test not edited), then passes |
| `tests/perception/test_ingestion.py`, `tests/app/{test_alerts,test_registry_handoff,test_session_synthetic,test_plates,test_cases,test_incidents,test_foundation}.py` | 90 passed (1 min 40 s) |
| `test_security_hardening, test_retention, test_event_logic, test_plates_robustness, test_detectors` + all of `tests/streaming` + all of `tests/training` | 392 passed, 1 failed (plate history order flake, see below) in 2 min 57 s |
| plate history flake | `PlateService.history()` tie-broke same-millisecond readings by random id; fixed to `rowid` (insertion order); `test_plates_robustness + test_plates + test_cases` 60 passed on 4 consecutive reruns (it had failed 1 run in 3 before the fix) |
| `scripts/stream_synthetic_bench.py`, `validate_yolo_dataset.py`, `audit_split_leakage.py`, `train_detector.py --preflight-only`, `demo_e2e.py` (+ `--verify-only`) | exit 0 / preflight exit 2 (refused, RAM) / demo VERIFIED; outputs in `outputs/exec4/` |

New tests this session: security 15, retention 10, event logic 16, plate robustness 26, synthetic-stream 12, YOLO-dataset/pre-flight/resume 23, temporal-events 37, held-out locks 6 = **145**. Not run after my edits (so unverified by me): the rest of the legacy suite - `tests/{evidence,hypotheses,propagation,safety,storage,world,feedback,fusion,evaluation}`, `tests/perception` except `test_ingestion.py`, and the top-level `tests/test_*.py`. My changes that could reach them: `SourceHealth` gained a defaulted field (`corrupt_frames`; the two positional constructors in tests are unaffected), `VideoFileFrameSource.health()` now returns `last_error`, `RtspFrameSource` skips unusable frames. **Run the full suite on a machine with free RAM before relying on it** (`docs/ANVESH_RELEASE_READINESS.md` section 7). Format / lint: none installed (ruff, black, flake8, mypy absent; nothing was installed); `py_compile` passed for every changed Python file; app startup verified by the demo server (`/ready` -> schema 2). Known environmental noise: OpenCV prints `libopenh264` / `moov atom not found` lines in some tests.

### 9.R Resources and training

No training or fine-tuning was attempted. Pre-flight on the final machine state refuses a plate-detector run (free RAM 2.24 GB < 3 GB minimum; temp drive 1.58 GB free). Expected cost if it were run later: 386 iterations/epoch (6,176 images at batch 16); runtime is not predictable without a measured throughput (the previous plate run reached 5.8 it/s before being stopped; this laptop varies 3-6x between runs). Disk: `D:` 125 GB free, `C:` 1.6-2.9 GB (risk). Peak memory pressure this session killed one background job (the full pytest run).

### 9.L Tasks that can only begin once you provide local CCTV footage

1. Extract frames (`scripts/local_cctv.py extract`, one every 1-2 s), annotate ~400 frames over >= 3 cameras / >= 8 recordings incl. night if in scope (CVAT YOLO export), `import-yolo`, `qa`, `split --group-by recording`.
2. `benchmark_local_cctv.py run --split val` (choose settings), then `--split test` once; then `gate` -> promotion decision for the fine-tuned detector (needs your explicit approval).
3. Measure stop-vs-queue behaviour and set `queue_peer_stops`; measure stationary-vehicle precision on a labelled sample.
4. Per-camera zones / lanes / signal / turn rules for wrong-way, wrong-lane, red-light, illegal turn; first real examples.
5. Survey-based calibration + hold-out validation for any speed claim.
6. Plate legibility check on real frames before any OCR/plate-detector work; Indian plate data.
7. Real latency (detector + decode + evidence) at your resolution, on a quiet machine, >= 3 repeats; a live RTSP session with a deliberate disconnect.
8. Footage / incident examples for an accident dataset (CADP is research-only).

### 9.Z Git state and files

Branch `main`, HEAD `b6d5ecd` (unchanged; 15 local commits ahead of `origin/main` already at session start), **0 staged, 0 committed, 0 pushed, 0 stashed**. `git status --porcelain`: 105 entries (96 at start): 47 modified, 58 untracked. The 9 additions are 2 tracked files I modified (`anvesh/perception/ingestion.py`, `anvesh/streaming/rtsp.py`) and 7 new untracked files; other new files sit inside directories that were already untracked (`anvesh/app/`, `tests/app/`, `tests/training/`, `anvesh/training/`).
I once overwrote a tracked file by mistake (`scripts/stream_bench.py`, via a Write that treated it as new); it was unmodified at HEAD, I restored it from `git show HEAD:scripts/stream_bench.py` and `git status` shows it clean; my benchmark lives in `scripts/stream_synthetic_bench.py`.
Hashes unchanged: MIO-TCD test lock `9850d283...f44a4`; fine-tuned best checkpoint `cdc3cf10...bf21`.
Changed (existing): `anvesh/app/{server,routes,auth,cameras,cases,db,evidence,plates,reports,runner,services,detectors,capabilities}.py`, `anvesh/perception/ingestion.py`, `anvesh/streaming/rtsp.py`, `scripts/train_detector.py`, `tests/app/test_api.py` (2 assertions, see 9.A), docs listed in 9.H.
New: `anvesh/app/retention.py`; `anvesh/training/{yolo_dataset,preflight,temporal_events}.py`; `scripts/{stream_synthetic_bench,validate_yolo_dataset,audit_split_leakage,event_manifest}.py`; tests `tests/app/{test_security_hardening,test_retention,test_event_logic,test_plates_robustness}.py`, `tests/streaming/test_synthetic_load.py`, `tests/training/{test_yolo_dataset,test_temporal_events,test_held_out_locks}.py`; docs `ANVESH_RELEASE_READINESS.md`, `ANVESH_ACCIDENT_RESEARCH_PIPELINE.md`; outputs in `outputs/exec4/` (gitignored: bench, validation, leakage, demo report/data dir, logs).

### 9.N Prioritised plan for the next session

1. Run the full suite on a quiet machine (close browsers; >= 4 GB free RAM) and fix whatever the untested legacy groups show.
2. Decide P1 (registry override on operational cases) and the Ultralytics / MIO-TCD licence route; these gate everything external.
3. Supply footage -> 9.L steps 1-3 (benchmark + queue rule).
4. A real RTSP test window.
5. Retention UI (plan/purge screens) and the audit-expiry procedure, if wanted.
6. Only then: OCR engine approval + Indian plate data; accident data collection per the research pipeline.


## 10. Fifth session (2026-10-21): registry approval invariant, account security, retention screen, acceptance run

Owner decisions applied (requirements, not questions): (1) an operational case needs an authorised registry result before approval - no override; (2) ANVESH stays research/demo-only, no deployment authorisation, nothing represented as licence-cleared; (3) safest practical account design; (4) no staging/committing/pushing.
Start state verified: `D:\ANVESH`, branch `main`, HEAD `b6d5ecd`, 0 staged, 47 modified, 58 untracked; MIO-TCD test lock `9850d283...f44a4` and fine-tuned checkpoint `cdc3cf10...bf21` unchanged (re-hashed at the end: identical); default detector `models/yolov8n.pt` unchanged; free RAM 1.5-2.3 GB (Edge/WSL hold the rest), `C:` 1.7 GB free.

### 10.1 How approval worked before (found by reading `cases.py`, `routes.py`, `registry.py`, `handoff.py`, then confirmed by tests)

- `CaseService.readiness()` item `registry` was `bool(found) or bool(case["registry_override_reason"])`, where `found` was **any** lookup of the case with status `found` - for any plate text, any reading, any age, any adapter kind.
- The override entered through `POST /cases/{id}/actions` (`registry_override_reason` was whitelisted in `CASE_ACTION_PARAMS`) -> `transition(... registry_override_reason=...)`, was stored on the case and satisfied the item **for operational cases too**. The schema trigger only stopped a *sandbox row* attaching to an operational case; it did not stop the override.
- Further gaps found while tracing it: a lookup was not tied to the plate reading that was verified (a result for an earlier plate, or for a manually entered unverified plate, still counted); `retry_submission` put a case back into `approved_for_submission` with no re-check; `handoff.submit` trusted the state alone; a malformed registry row crashed readiness (500) instead of failing closed (found by the new test).
- Tests that encoded the old behaviour: `test_cases.py` (7 call sites, incl. `test_approval_requires_registry_or_override...`), `test_registry_handoff.py` (2), `test_security_hardening.py` (1, incidental). Each was **replaced** with an assertion of the new rule (e.g. the override test now asserts the override is refused with 422 and the case unchanged); none deleted.

### 10.2 The invariant and where it is enforced

> An operational case cannot be approved - nor retried, nor handed off - unless its current plate reading is verified (by someone other than who entered it) and a registry result that came from the configured **authorised, non-sandbox** gateway, is fresh, well-formed, audit-confirmed and **bound to exactly that reading** says `found`.

1. **Service layer** (`CaseService.registry_support`, used by `readiness()`, so by `approve`, `retry_submission`, the UI checklist): checks reading binding (`registry_lookups.reading_id` = `cases.plate_verified_reading_id` = current head), plate text equality, `adapter_kind`/`is_simulated` (operational: production only), adapter name equals the configured gateway and it is connected, data well-formed with at least one vehicle field and a source label, `retrieved_at` within `registry_result_max_age_s` (24 h, `ANVESH_REGISTRY_MAX_AGE_S`) and not in the future, and a matching `registry.lookup` row in the append-only audit chain (a hand-inserted row does not count). A plate correction invalidates both the verification and the registry binding. Operational registry lookups are only made for the case's verified reading (data minimisation).
2. **Second gate**: `HandoffService.submit` re-checks plate+registry for operational cases at submission time (`handoff.blocked` audited, 409).
3. **Third gate, database**: schema v3 trigger `cases_operational_approval_needs_registry` refuses any `UPDATE ... state = 'approved_for_submission'` of an operational case without a production, non-simulated `found` lookup for the verified reading (tested with raw SQL and with the internal `apply_handoff_result`).
4. **API**: `registry_override_reason` and every other unknown body field -> 422 (`there is no registry override`), recorded as `case.approve_blocked`; a registry lookup request cannot choose its adapter, status or simulation flag (tested). Alternative entry points covered: `approve`, `retry_submission`, `handoff.submit`, `apply_handoff_result`, direct SQL, `PATCH /cases/{id}`.
5. **Audit**: blocked attempts (`case.approve_blocked`, `case.retry_submission_blocked`, `handoff.blocked`) carry status and failing items only (no owner data, no credentials); successful approvals audit `case.approve`; the chain verifies after all of it.
6. **No authorised registry exists here**, so operational approval is **blocked by design**. Enabling it needs a departmental authorisation and a gateway implementing `registry.py`'s contract (`ANVESH_REGISTRY_URL`/`_API_KEY`); nothing else will. Demo/simulated cases keep the same shape but accept the labelled SANDBOX result. A sandbox/synthetic result can never satisfy an operational case (service rejects it; the schema forbids the sandbox row).
Tests: `tests/app/test_registry_policy.py`, 31 tests - no result; manual/verified plate is not a result; unconfigured gateway; sandbox/simulated; unauthorised provider; unaudited hand-inserted row; malformed/empty/no-source/future results; different plate/reading; correction after verification and lookup (and the proper route back); stale vs fresh; valid path with audit assertions; raw SQL; expiry after approval blocks submit and retry; demo cases; 10 spoofed-field cases; HTTP end-to-end; schema v3 migration with backup. A `AuthorisedRegistryDouble` in `tests/app/conftest.py` stands in for a gateway **in tests only**; it is not in the product.
Schema: **v3** adds `registry_lookups.reading_id` (old lookups keep NULL and therefore never support an operational approval) and the trigger; the existing backup-before-migrate copies the database first (tested).

### 10.3 Account security (details in `docs/ANVESH_ACCOUNT_SECURITY.md`)

Verified first: throttling was per username in memory (confirmed), MFA absent (confirmed; no `pyotp`/`cryptography` installed), scrypt cost N=2^14 (below current guidance), no idle timeout, administrators could create/reset any account including officers.
Implemented and tested (`tests/app/test_account_security.py`, 49 tests incl. demo isolation): scrypt N=2^17 r=8 p=1 with stored parameters, upgrade-on-login, tamper-safe parsing, two concurrent hashes max; password policy; self-service change (re-verify, no reuse, all sessions revoked); **throttling** per (user, source) 5 / per source 30 / per username 25, 5-minute lock on a monotonic clock, bounded tables, identical responses for unknown users, `auth.lockout` audited once per scope without attacker-chosen strings, `POST /users/{id}/unlock`; **provisioning**: investigator/reviewer immediate with temporary password, administrator/officer creation, role moves, re-enable and password reset need a different administrator (self-approval refused and audited, 72 h expiry, immutable decisions, requester removed -> not applied), exactly one change per call, last active administrator protected, no self role change/deactivation, unauthenticated/unauthorised creation refused; **temporary passwords** force a change before any other API call and the dashboard shows only that screen; **sessions**: 30-minute idle timeout, revocation on role change / deactivation / password change, purge of old rows, malformed tokens/headers/cookies never authenticate; **demo accounts** (`--demo-users`) are flagged, labelled, in a banner-headed credentials file and cannot act on operational cases/cameras/registry; **offline recovery CLI** `scripts/manage_users.py` (mandatory `--data-dir`, audited `cli:<user>`); schema **v4** (`users.must_change_password/is_demo/password_changed_at`, `auth_sessions.last_seen_at`, `account_requests`) with backup and a test that existing users are not retro-flagged.
**Not implemented: MFA.** No vetted crypto library is available and none was hand-rolled or installed; plaintext TOTP secrets beside the password hashes would defeat the point. The secure design and exact remaining work are in section 5 of the account-security document. Lockout is in memory (lost on restart, not shared across processes, per source address - a proxy hides clients); this is stated in the code, the status endpoint and the documents.
Fixed in passing: `POST /auth/password` no longer creates a session and an `auth.login` audit entry as a side effect.

### 10.4 Research/demo-only enforcement and licensing

`/system/status` and `/capabilities` now carry `product` (`mode: research_demo`, `operational_use_authorised: false`, the reasons) and `models` (vehicle detector available but "accuracy on your cameras: not measured" and `licence_status: NOT CLEARED`; plate detector and OCR "not installed - plates are manual entry only"; accident model "not implemented; possible_collision is an experimental rule heuristic"). The dashboard shows a permanent "RESEARCH / DEMO USE ONLY" banner (plus "DEMO account" when applicable); `run_app.py` prints the mode, warns about demo accounts and about non-loopback binding without secure cookies. Sandbox registry/challan results, simulated alerts and synthetic cameras keep their labels (re-run demo: "SANDBOX", "TEST-CHALLAN", "TEST ADAPTER: not delivered to any external party"); no police notification is claimed anywhere. Detector default and checkpoint untouched. Licensing questions still open (nothing was decided or inferred): Ultralytics AGPL-3.0 vs a paid enterprise licence (a paid licence for one dependency does not by itself grant rights to MIO-TCD-derived weights); MIO-TCD CC BY-NC-SA 4.0 for the fine-tuned checkpoint; OCR weight licences; CADP research-only. See `docs/ANVESH_RELEASE_READINESS.md` section 4.

### 10.5 Retention screen

Absent before (the tab showed a stale "report only" label). Added: Administration -> Retention shows the report, a **purge-plan preview** (files that would be deleted with kind, path, size and hash prefix; eligible incidents; **protected incidents with reasons**; files pending removal; count already purged), and for `retention.manage` an execution form whose button stays disabled until a reason of >= 10 characters, the exact `plan_id` and an acknowledgement are entered. The API enforces the same (permission, 409 for a stale plan, protected incidents untouched, 30-day floor, audit rows never deleted - covered by the existing 10 retention tests plus 3 new ones for the plan contract, a stale-preview refusal, and the UI gating strings). **The screen was syntax-checked with `node --check` but not driven in a browser this session** (free RAM 1.5-2 GB made a browser run unsafe).

### 10.6 Tests: commands, exit codes, results

All run from `D:\ANVESH` with `.venv\Scripts\python.exe -m pytest ... -q -p no:cacheprovider`, sequentially, in separate processes per group (the memory guard killed the single full-suite run last session; I did not attempt one). Every command exited 0.

| Group (directories/files) | Result |
|---|---|
| `tests/app`: alerts, api, foundation, incidents | 38 passed |
| `tests/app`: cases, registry_handoff, registry_policy, plates, plates_robustness | 110 passed |
| `tests/app`: detectors, event_logic, session_synthetic, ingestion | 54 passed |
| `tests/app`: retention, security_hardening, account_security | 77 passed |
| `tests/streaming` + `tests/training` | 304 passed |
| `tests/perception` | 193 passed |
| `tests/world` `tests/safety` `tests/storage` | 132 passed |
| `tests/evidence` `hypotheses` `propagation` `feedback` `fusion` `evaluation` | 229 passed |
| `tests/test_*.py` (17 top-level files) | 187 passed |
| **Total** | **1324 passed, 0 failed, 0 skipped** (final rerun after the review fixes) (every test directory covered; it is 8 pytest processes, not one invocation) |

New tests this session: 83 (registry policy 31, account security 49, retention screen 3). Intermediate failures, all fixed in code or in my tests (none by weakening): two migration tests that hard-coded schema versions (replaced by a `downgrade_to` helper), and the malformed-registry-data crash above.
Static checks: `py_compile` of every `anvesh/app/*.py` and `scripts/*.py` passed; `node --check` of the five dashboard scripts passed; no linter is installed (ruff/black/flake8/mypy absent). Application start and the end-to-end demo: fresh data dir `outputs/exec5/app`, port 8762, `--demo-users`: real YOLOv8n + ByteTrack on 40 s of `demo.mp4` (1001 frames, 0 dropped), 1 stop + 1 congestion incident, 6 evidence assets with intact hashes, sandbox case through `challan_reference_received`, simulated collision with TEST alert, audit chain ok (81 rows); after stopping and restarting the server `--verify-only` -> **PERSISTENCE AFTER RESTART: VERIFIED** (95 audit rows, hashes intact). Live checks on the running server: `/ready` schema 4; `/system/status` shows scrypt N=2^17, `mfa: not implemented`, `product.mode research_demo`; creating an officer -> 202 pending; self-approval -> 403; creating an investigator -> 201 with `must_change_password`; 7 wrong logins -> `[401 x5, 429, 429]` while another user from the same address still logged in. The servers were stopped afterwards (only the processes I started).
Resource evidence: free RAM stayed at 1.5-2.3 GB throughout (Edge ~2 GB, WSL 0.7 GB, Defender 0.5 GB); no process of mine was killed this time. No training was started.

### 10.6a Review findings fixed, and the migration run on copies of the real databases

An independent review of this session's work found (and I fixed, each with tests, now 4 more tests in `test_account_security.py`): demo accounts could reclassify a camera to `operational` (only creation was guarded - now guarded in the route **and** `CameraService`); accounts created by a demo administrator were real accounts (demo status is now inherited, including through approved requests, and a demo administrator can manage/approve/reject/unlock only demo accounts); plate, incident, session, alert-retry, settings and purge routes had no demo guard (now closed for writes; **reads are not restricted** and the documents say so); the document promised a way to flag pre-v4 accounts that did not exist (`manage_users.py set-demo` added).
**Migration on copies** (originals only read; sha256 of both databases and their WAL/SHM files identical before and after): `outputs/app` (schema v1, 2026-10-18) -> v4 in one start, backup `anvesh_app.pre-v2-migration.sqlite3.bak` written, audit chain intact (177 rows), the 1 case and 17 incidents load without error, 114 evidence assets verify; `outputs/exec4/app` (v2) -> v4, backup `...pre-v3-migration...` written, chain intact (85 rows), 1 case, 38 assets verify. Legacy registry lookups have `reading_id` NULL (so a legacy case now shows its registry checklist item as failing - it is already completed; nothing is re-opened), and the v2 backfill set the plate binding on both cases. **Consequence for you: after migration every existing account in `outputs/app` is non-demo, and its plaintext passwords are in `outputs/app/initial_credentials.txt`; retire or flag them before any non-demo use. I did not change `outputs/app`.**
(One of my own helper scripts mangled a path and briefly created `D:\ANVESH\ANVESHoutputsexec5migr` holding two copies; I verified its contents were only my copies and removed it. `C:` was at 0 GB free for a while during this work - mostly another session's 4.8 GB scratch venv under `%TEMP%\claude`, which I left alone - and later showed 10.7 GB.)

### 10.7 Remaining blockers and licensing concerns

- **Operational approval is blocked by design**: no authorised registry (needs departmental authorisation + gateway); no e-challan integration; no live camera ever connected; detector accuracy on your cameras unmeasured (needs labelled local footage); plate OCR/detector not installed (no Indian plate data); MFA not implemented; licences uncleared (Ultralytics, MIO-TCD, OCR weights); TLS/trusted-proxy handling for exposure.
- Throttling is per process and per source address; two colluding administrators can still mint privileged accounts (audited).
- **Not run / not verified, explicitly**: the three changed dashboard screens (users tab with pending requests, forced password-change screen, retention screen) were syntax-checked with `node --check` but **never opened in a browser**; the full suite has **not** been run as a single invocation (only as sequential groups); **no independent security review or penetration test**; the operational approval path was exercised only against a **test double**, never a real registry; the offline CLI was tested on temporary databases, not on `outputs/app`.

### 10.8 Files

Modified: `anvesh/app/{auth,cases,db,config,registry,handoff,routes,server,services,bootstrap,reports,retention}.py`, `anvesh/app/static/{app.js,views_admin.js,views_cases.js}`, `scripts/run_app.py`, tests `tests/app/{conftest,test_cases,test_registry_handoff,test_retention,test_security_hardening}.py`, docs (`ANVESH_APP_GUIDE`, `ANVESH_RELEASE_READINESS`, `ANVESH_ACCEPTANCE_REPORT`, this report).
New: `anvesh/app/accounts.py`, `scripts/manage_users.py`, `tests/app/{test_registry_policy,test_account_security}.py`, `docs/ANVESH_ACCOUNT_SECURITY.md`; outputs in `outputs/exec5/` (gitignored).
Git at the end: branch `main`, HEAD `b6d5ecd`, **0 staged**, 47 modified, 60 untracked (107 entries), no commit, no push, no stash, nothing discarded.
