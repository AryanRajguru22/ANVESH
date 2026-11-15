# ANVESH local-CCTV benchmark and detector promotion gate

Date: 2026-10-18. Status: **tooling built and tested on a synthetic fixture only. No local CCTV footage or annotation has been supplied, so no local benchmark has been run and no local accuracy number exists.** Nothing in this document is a measurement of a real camera.

Related: `docs/ANVESH_MODEL_TRAINING_REPORT.md` (section 7 explains the metric discrepancy), `docs/ANVESH_EXECUTION_REPORT.md` (section 8).

## 1. What you need to supply

| Item | Minimum to be able to *run* | Minimum to be able to *promote* (section 6) |
|---|---|---|
| Source video from the intended cameras | 1 recording | >= 3 cameras, >= 8 recordings, day plus at least one of night/dusk/dawn (and rain if it is in scope) |
| Labelled frames | any (a smoke run) | >= 400 labelled frames, >= 1,500 boxes, >= 50 boxes for every class you want a per-class claim about |
| Annotation | one annotator, YOLO export (CVAT / Label Studio / labelImg all export it) | plus a second annotator on >= 10% of frames to measure agreement |
| Per recording | `camera_id`, `recording_id` | plus `recording_start_utc`, lighting/weather tags |

Footage must be lawfully held and usable for this purpose; frames are copied into `datasets/local_cctv/<name>/` (gitignore it: do not commit footage). The two clips in `datasets/recorded/` are public sample videos (Intel IoT DevKit), not the intended cameras, and are **not** a benchmark.

## 2. Dataset layout (`datasets/local_cctv/<name>/`)

```
manifest.json            single source of truth, one record per frame (history kept in manifest_history/)
frames/<frame_id>.jpg    extracted frames; never overwritten
labels/<frame_id>.txt    YOLO rows `cls cx cy w h` (normalised)
ignore/<frame_id>.txt    don't-care regions (difficult / unclassifiable)
labels_second/<annotator>/<frame_id>.txt   second-annotator labels, only for agreement
annotation_log.jsonl     append-only record of each import
qa_report.json           output of `local_cctv.py qa`
```

`frame_id = <camera_id>__<recording_id>__<source frame index, 7 digits>`; `camera_id` and `recording_id` may not contain spaces, slashes or `__`.

Manifest fields per frame. Required: `frame_id, image, width, height, image_sha256, camera_id, recording_id, labelled, split`. Provenance (recorded by the tools): `recording_start_utc, frame_index, frame_time_s, source_video, source_video_sha256, source_fps, location, conditions[], annotator, annotation_tool, annotation_tool_version, annotated_utc, reviewed`.

## 3. Annotation rules

* **Classes (ids fixed):** `0 bicycle, 1 car, 2 motorcycle, 3 bus, 4 truck, 5 auto_rickshaw`. Ids 0-4 are the shared taxonomy both candidate checkpoints output. `auto_rickshaw` is labelled because it occurs on Indian roads, but **neither checkpoint can output it** (MIO-TCD has no such class; COCO neither). It is therefore never scored as a per-class AP; it is excluded from per-class false-positive counting (a "car" box on a rickshaw is not penalised) **and** its class-agnostic recall (did *any* box land on it) is reported separately so it cannot hide misses. Seeing zero detections on rickshaws is the expected, reportable result that tells you a rickshaw class must be trained before the app can claim to see them.
* **Pedestrians, riders as separate objects, trailers/carts:** not labelled. A motorcycle box includes its rider.
* **Box convention:** tight axis-aligned box around the *visible* extent of the vehicle, including mirrors but excluding cast shadow; `cx cy w h` normalised to the frame. Truncated vehicles at the image edge are boxed to the edge.
* **Which vehicles to box:** every vehicle that is >= ~35% visible, >= 8 px on its shortest side in the stored frame, and whose class you can name with reasonable confidence.
* **Difficult / occluded:** vehicles that are visible but too small (< 8 px), < ~35% visible, or unclassifiable are drawn and labelled with the export class `difficult` (or `ignore` / `other_vehicle`). The importer writes them to `ignore/`. A detector box on a difficult region is neither a hit nor a false alarm (same don't-care rule as the MIO-TCD evaluation), **and** the number of ignore regions is reported so heavy use of "difficult" is visible, not a way to dodge hard cases.
* **Missing vs empty label file (important):** *no* label file = frame is unlabelled and excluded from every metric. An *empty* file = a human verified there are no vehicles; it counts as a negative frame for false-positive rates.
* **Sampling:** one frame every >= 1-2 s per recording (`extract --stride-seconds`); do not label every consecutive frame (near-duplicates inflate the box count without adding information).
* **Quality checks (`local_cctv.py qa`):** boxes inside the image, no degenerate boxes, minimum size, near-duplicate boxes (IoU >= 0.9), class ids in range, image SHA-256 unchanged since import, label/`labelled` consistency, group leakage, adjacent-recording warnings, and, if `labels_second/` exists, inter-annotator box agreement (F1 at IoU 0.5) and class agreement given a match. Errors make `qa` and the benchmark exit non-zero; warnings are reported.

## 4. Workflow

```
.venv\Scripts\python.exe scripts\local_cctv.py extract    --dataset datasets\local_cctv\v1 --video <clip> --camera-id cam3 --recording-id r0800 --recording-start-utc 2026-11-02T02:30:00Z --conditions day,dry
# annotate datasets\local_cctv\v1\frames in CVAT / Label Studio / labelImg, export YOLO (obj_train_data + obj.names)
.venv\Scripts\python.exe scripts\local_cctv.py import-yolo --dataset datasets\local_cctv\v1 --labels <export>\obj_train_data --classes <export>\obj.names --annotator a --tool CVAT --tool-version <v>
.venv\Scripts\python.exe scripts\local_cctv.py qa         --dataset datasets\local_cctv\v1 --agreement
.venv\Scripts\python.exe scripts\local_cctv.py split      --dataset datasets\local_cctv\v1 --group-by recording --seed 0     # or --eval-only for benchmark-only footage
.venv\Scripts\python.exe scripts\benchmark_local_cctv.py run --dataset datasets\local_cctv\v1 --split test --model pretrained=models\yolov8n.pt --model finetuned=models\finetuned\anvesh_mio5_yolov8n_best.pt --out outputs\local_benchmark\v1_test.json --visuals 24
.venv\Scripts\python.exe scripts\benchmark_local_cctv.py latency --images <dir-or-video> --model pretrained=... --model finetuned=... --out outputs\local_benchmark\latency.json
.venv\Scripts\python.exe scripts\benchmark_local_cctv.py gate --report outputs\local_benchmark\v1_test.json --latency outputs\local_benchmark\latency.json --cityflow outputs\training\compare_cityflow.json [--downstream tracking_diff.json] --out outputs\local_benchmark\gate.json
```

* **No overwrite anywhere:** `extract` refuses an existing frame/recording; `import-yolo` refuses an existing label file and an unmapped export class (it would otherwise become background); every manifest write keeps the previous revision.
* **Leakage control:** splitting assigns whole source groups (`--group-by recording|camera|camera_day`) to a split; `qa` fails if any recording appears in two splits and warns when two recordings of the same camera lie within 10 min of each other across a split boundary. Use `--group-by camera` for the strictest test (unseen cameras). If you only want a benchmark (no training on local data), use `--eval-only`.
* **Test discipline:** `--split test` and `--split eval` are locked per dataset after one run (`<split>_evaluation.lock.json`), like the MIO-TCD test; `--allow-test-rerun` overrides and records the rerun. Pick settings on `val`, then run `test` once. With `--eval-only` footage the split is `eval`, and the gate currently accepts only a `test` report, so use a real train/val/test split (or change the gate deliberately) if you intend to promote.
* **If no frame in the chosen split is labelled the benchmark exits with code 3 ("NOT RUN") and writes no results file.**

## 5. What the benchmark reports

Same prediction settings for every checkpoint (conf 0.001 for AP then thresholded, NMS IoU 0.7, max 300 detections, single-label NMS, FP16, imgsz 640, batch 8 for accuracy; batch 1 for latency). Per checkpoint: mAP50 / mAP50-95 and per-class P / R / AP (classes with < 30 boxes are reported as "insufficient", never as a number); class-aware TP/FP/FN and P/R/F1 at conf 0.10 / 0.25 / 0.40 / 0.50; class-agnostic recall, false positives and FP per frame at the same thresholds and per camera; auto-rickshaw class-agnostic recall; recording-level bootstrap CIs (recordings are resampled, not frames) for recall, FP/frame and mAP50, including paired candidate-minus-baseline differences; the N worst frames drawn with GT (green hit / red miss), matched predictions (blue) and FPs (orange); environment (device, torch, ultralytics, precision, imgsz, batch). Latency: batch 1, 20 warm-up frames discarded, wall clock with CUDA synchronisation, median/p95/max plus Ultralytics' preprocess/inference/postprocess split, models interleaved frame by frame. Every report states that it describes *that labelled sample* and is not a general real-world claim.

## 6. Promotion criteria (proposed; confirm before treating as binding)

`anvesh/training/local_benchmark.py::PROMOTION_CRITERIA`; evaluated by `benchmark_local_cctv.py gate`. The candidate (fine-tuned) is promotable only if **every** line passes on the held-out local *test* split (group-disjoint from anything used to choose settings or to train). Any `fail` => DO_NOT_PROMOTE; no fail but something not evaluable => INCONCLUSIVE. The gate only reports; it never edits `anvesh/app/config.py`.

| # | Criterion | Threshold (proposal) |
|---|---|---|
| 1 | Data sufficiency | >= 3 cameras, >= 8 recordings, >= 400 labelled frames, >= 1,500 boxes, >= 2 distinct lighting conditions. If unmet, criteria 2-6 are "not evaluable", not passed |
| 2 | Absolute recall (class-agnostic, IoU 0.5, conf 0.25) | point >= 0.85 and recording-bootstrap 95% CI lower >= 0.80 |
| 3 | Recall non-inferiority vs pretrained | paired difference CI lower >= -0.02 |
| 4 | False positives | <= 0.5 FP per frame; and not worse than pretrained by more than 0.10 FP/frame (CI upper) |
| 5 | Better at something | class-aware mAP50 paired-difference CI lower > 0 (no reason to switch otherwise) |
| 6 | No per-camera regression | on every camera with >= 100 GT boxes, recall not lower than pretrained by more than 0.05 |
| 7 | Latency (batch 1, target hardware, quiet machine) | p95 <= 33 ms and <= 1.25x pretrained p95 |
| 8 | CityFlow non-inferiority (existing) | pooled class-agnostic recall difference >= -0.02 |
| 9 | Downstream regression | tracking/speed/incident re-evaluation with `ANVESH_FINETUNED_MODEL` set: IDF1 and MOTA each not worse than -0.01 |
| 0 | Held-out split | the report must come from the locked `test` split (`run --split test`); a report from `all`/`train`/`val`/`eval` makes lines 1-6 not evaluable (the gate checks `report["split"]`) |
| 10 | Regression suite | full pytest suite passes (checked by a human run, not by the gate) |

The numeric thresholds are my proposals for a traffic-police operating point; they have not been agreed with you. Recall 0.85 at 0.25 is deliberately not a safety claim: set it from the cost of a missed vehicle in your use case.

**Current status of each line (2026-10-18):** 1-6 not evaluable (no labelled local data); 7 not evaluable (see section 7); 8 **fails on the point estimate** (-0.030 against a -0.02 margin; no CI exists for it); 9 not run; 10 last verified 1080 passed before this session's additions. Actual output of `benchmark_local_cctv.py gate --cityflow outputs/training/compare_cityflow.json` (saved as `outputs/local_benchmark/gate_2026-10-18.json`; latency JSONs deliberately not passed because they are noise): **verdict DO_NOT_PROMOTE** - line 8 `fail` (pooled CityFlow recall difference -0.030, a point estimate with no CI, against a -0.02 margin); lines 1-7 and 9 `not_evaluable`. The gate treats a failed line as decisive even without a CI (conservative); it would become INCONCLUSIVE only if that line were not evaluable.

## 7. Measurements taken this session (not local footage)

Latency was measured on CityFlow S02 c006 frames purely to exercise and sanity-check the timing tool (RTX 3050 6 GB Laptop, torch 2.13.0+cu130, Ultralytics 8.4.132, FP16, imgsz 640, batch 1). It is **not** a local-camera result, and it is **not reliable**: the same command gave pretrained/fine-tuned median 59.8/76.4 ms, then (reversed order) 15.9/20.7 ms, then interleaved 104.0/99.2 ms and 45.2/33.5 ms (`outputs/local_benchmark/latency_cityflow_c006*.json`). A 3-6x run-to-run swing on identical code means this laptop's clocks/background load dominate (GPU was at 990 of 2100 MHz and 57 C when checked). The two checkpoints share a backbone (different class heads, 6 vs 80 classes) and showed no consistent ordering across runs (interleaved medians 104.0 vs 99.2 ms, then 45.2 vs 33.5 ms), but no absolute latency, relative latency or pass/fail against 33 ms can be claimed. Needed: a quiet machine (browsers closed, fixed power plan, GPU clocks stable) and >= 3 repeated runs.

## 8. Limitations and the minimum data that would change them

* One camera, one hour of day footage cannot support a promotion; the gate's data-sufficiency line is the floor, not a target. Rain, night and glare change detector behaviour qualitatively; if they matter operationally they must be in the labelled set.
* Per-class claims need >= 50 (ideally >= 200) boxes per class; bicycle/motorcycle are rare on most carriageways, so expect to report them as insufficient.
* Auto-rickshaw, e-rickshaw, tempo, tractor and cycle-rickshaw are not in any current checkpoint; labelling them now (as `auto_rickshaw`, others need new class ids) is cheap and tells you the size of the gap before any training.
* Recall at IoU 0.5 says nothing about box tightness, which matters for speed estimation and plate cropping; `mAP50-95` is reported for that reason.
* Annotation quality is capped by the annotator; without a second annotator on a sample there is no estimate of label noise.
* This benchmark measures the detector in isolation. Whether a better detector improves tracking, speed or incident evidence is criterion 9 and requires the existing tracking evaluation to be re-run with the candidate.


## 9. Addendum 2026-10-20

Still **no local footage and no local benchmark**. Added tests (`tests/training/test_held_out_locks.py`): the `test` and `eval` splits run once then exit 4 until `--allow-test-rerun` (recorded as a second run); `val` is never locked; all-unlabelled footage raises NOT RUN and an empty gate input is INCONCLUSIVE, never PROMOTE; the real MIO-TCD test lock still shows one run. Everything that waits for footage is listed in `docs/ANVESH_EXECUTION_REPORT.md` section 9.L. A related suspicion to measure on your footage: the stop-vs-queue rule (`queue_peer_stops`) is an untuned policy value.
