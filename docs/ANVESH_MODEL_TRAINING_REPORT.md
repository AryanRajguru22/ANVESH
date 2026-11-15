# ANVESH detector training and evaluation report

Date: 2026-10-18. Every number below is copied from a file produced in this session (paths given). Nothing is typed in from memory.
Generated tables: `outputs/training/training_summary.md` (from `scripts/summarize_training.py`).

## 1. Verdict (read this first)

| Question | Answer |
|---|---|
| Did fine-tuning work on its own dataset? | Yes. On the MIO-TCD held-out test split, mAP@50 rose from 0.215 to 0.870 (shared 5-class taxonomy). |
| Does that prove better accuracy on ANVESH's real camera footage? | **No.** On CityFlowV2 S02 CCTV (class-agnostic recall, IoU 0.5) the fine-tuned model is **slightly worse**: pooled recall 0.718 vs 0.748 for the pretrained COCO model. |
| Is the fine-tuned detector the app default? | **No. Not promoted.** Default stays `models/yolov8n.pt` (`anvesh/app/config.py:19`); the fine-tuned model is only used when `ANVESH_FINETUNED_MODEL` is set explicitly. |

mAP on MIO-TCD measures agreement with MIO-TCD labels on MIO-TCD-style images. It is not a statement about Indian traffic, night, rain, or the ANVESH cameras.

## 2. Artefacts

| Item | Value |
|---|---|
| Base checkpoint | `models/yolov8n.pt`, SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Fine-tuned checkpoint | `models/finetuned/anvesh_mio5_yolov8n_best.pt`, SHA-256 `cdc3cf10960e58eff105fff4584c32c91f810a99c08149fc434ef5ba7f73bf21` (byte-identical to `runs/anvesh_mio5_v1b/weights/best.pt`) |
| Last-epoch checkpoint | `anvesh_mio5_yolov8n_last.pt`, SHA-256 `0ca32c411134e6fed3a187ea95ce8e9de47e94aca6a7a65267aa789dad035119` (differs from best) |
| Training run | `models/finetuned/runs/anvesh_mio5_v1b`; 8 of 8 epochs completed, 2.97 h, `--batch 16 --imgsz 640 --patience 4 --close-mosaic 2 --workers 2` |
| Dataset | MIO-TCD localization, converted by `scripts/prepare_mio_tcd.py`: 88,270 train / 10,865 val / 10,865 test images. Training classes: bicycle, car, motorcycle, bus, truck, other_vehicle (6). Pedestrian / non-motorized are ignore regions in evaluation. |
| Taxonomy used for all comparisons | bicycle, car, motorcycle, bus, truck (`other_vehicle` = don't-care). Models are mapped by class **name**, so COCO and fine-tuned weights are scored identically. |

Training was not converged: validation mAP@50 was still rising at epoch 8 (0.801 -> 0.808 -> 0.812 over epochs 6-8, `results.csv`). More epochs would likely help; not run.

## 3. Three different kinds of metric (do not mix them)

### 3a. Validation split (MIO-TCD val, 10,865 images). Used during training to monitor and to pick best.pt. Not independent of model selection.

| model | mAP50 | mAP50-95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---|---|---|---|---|
| pretrained COCO yolov8n | 0.236 | 0.150 | 0.633 | 0.405 | 0.494 |
| fine-tuned | 0.885 | 0.667 | 0.875 | 0.918 | 0.896 |

(`outputs/training/compare_val.json`. Ultralytics' own per-epoch 6-class val mAP50 at epoch 8: 0.812, mAP50-95 0.603 — a different protocol, see 4.)

### 3b. Held-out test split (MIO-TCD test, 10,865 images). Evaluated exactly once (`datasets/mio_tcd/yolo5/test_evaluation.lock.json` holds 1 run). Same checkpoint hashes as above.

| model | P (max-F1) | R (max-F1) | mAP50 | mAP50-95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---|---|---|---|---|---|---|
| pretrained COCO yolov8n | 0.337 | 0.239 | 0.215 | 0.142 | 0.641 | 0.364 | 0.464 |
| fine-tuned | 0.880 | 0.830 | 0.870 | 0.651 | 0.880 | 0.913 | 0.896 |

Paired image-level bootstrap (n=100, seed 0), fine-tuned minus pretrained: mAP50 +0.656 (95% CI 0.639 to 0.674); mAP50-95 +0.510 (0.496 to 0.523).

Per class on the test split (P and R at the max-F1 point, conf 0.001 sweep):

| class | GT boxes | pretrained P / R / AP50 / AP50-95 | fine-tuned P / R / AP50 / AP50-95 |
|---|---|---|---|
| bicycle | 213 | 0.198 / 0.033 / 0.023 / 0.007 | 0.719 / 0.696 / 0.705 / 0.432 |
| car | 26,041 | 0.588 / 0.478 / 0.485 / 0.271 | 0.916 / 0.899 / 0.944 / 0.693 |
| motorcycle | 269 | 0.081 / 0.056 / 0.018 / 0.009 | 0.931 / 0.762 / 0.866 / 0.528 |
| bus | 1,164 | 0.447 / 0.406 / 0.353 / 0.289 | 0.940 / 0.936 / 0.955 / 0.876 |
| truck | 5,924 | 0.370 / 0.221 / 0.193 / 0.136 | 0.893 / 0.856 / 0.877 / 0.726 |

Weak classes: bicycle (AP50 0.705) and motorcycle (0.866) have only 213 and 269 test boxes, so their estimates are noisy (no per-class CI was computed). Confusion at conf 0.25 / IoU 0.45 (test, `compare_test.json` `rows_pred_cols_true`; figure `docs/training_figures/confusion_test_finetuned.png`): the largest error source is **missed ground-truth boxes** (2,703 of 26,041 cars, 519 of 5,924 trucks, 51 of 213 bicycles unmatched), followed by car/truck confusion (560 true trucks predicted car, 349 true cars predicted truck) and 1,683 predicted cars with no matching ground truth. Bicycle/motorcycle confusion is small in count (17 true motorcycles predicted bicycle, 5 the reverse).

### 3c. CityFlowV2 S02 validation cameras c006-c009 (real traffic CCTV, never used for training). Class-agnostic recall at IoU 0.5, conf 0.25, imgsz 640, every 10th frame.

Ground truth lists only vehicles seen by >= 2 cameras and has **no class labels**, so precision and per-class accuracy are **not computable**. (`outputs/training/compare_cityflow.json`)

| camera | GT boxes | pretrained recall | fine-tuned recall |
|---|---|---|---|
| c006 | 507 | 0.627 | 0.600 |
| c007 | 428 | 0.827 | 0.857 |
| c008 | 347 | 0.902 | 0.859 |
| c009 | 823 | 0.717 | 0.659 |
| pooled | 2,105 | **0.748** (1575 TP / 530 FN) | **0.718** (1511 TP / 594 FN) |

Fine-tuned wins one of four cameras and loses three. Sampled frames per camera are only 193-211 (not 600; the clips are shorter), so the pooled difference (-0.030) is small and has no confidence interval; treat it as "no evidence of improvement on this footage", not as proof of regression. Descriptive class mix on c006 differs sharply (fine-tuned labels 1,260 detections as truck vs 234 for pretrained, with similar totals) — with no class ground truth this may be car/truck confusion on this camera, but that is **unverified**.

## 4. Verification caveats

1. **Cross-check against Ultralytics `val()` does not agree exactly.** Per-class AP50 on the test split, Ultralytics vs ours-without-ignore-regions: bicycle 0.654 vs 0.643, car 0.947 vs 0.938, motorcycle 0.876 vs 0.855, bus 0.968 vs 0.954, truck 0.914 vs 0.869. Our implementation is lower in every class (largest gap truck, 0.045). The script's own comment says these should agree; the cause was **not investigated** (AP interpolation, matching rule and `other_vehicle` handling are candidates). Our figures are therefore the more conservative ones, but this remains an open verification item. Ultralytics' all-class test mAP50 0.797 / mAP50-95 0.588 (6 classes including `other_vehicle`, no ignore regions, 36,469 instances) is a different protocol from the 0.870 / 0.651 above and should not be compared with it directly.
2. **Latency.** Fine-tuned model only (the cross-check `val()` is not run for the COCO model): Ultralytics' validation pipeline reported 1.3 ms preprocess + 3.1 ms inference + 1.2 ms postprocess per image on the RTX 3050 Laptop at batch 16 (`compare_test.log`; precision mode not confirmed). This is not ANVESH streaming latency. `inference_seconds` in the comparison JSONs (test: 613 s pretrained vs 105 s fine-tuned) is **not** a valid latency comparison (run order, disk cache, and different filtering). Both models are the same YOLOv8n architecture. No end-to-end ANVESH pipeline latency benchmark was run for the fine-tuned model.
3. In-domain advantage: MIO-TCD labels define "truck" and "car" differently from COCO, so the pretrained baseline is penalised for taxonomy mismatch as well as for visual error. The gap overstates the pure "learning" effect.
4. Checkpoint loading and class map: `tests/perception/test_class_map_and_weights.py::test_real_finetuned_checkpoint_class_layout_and_inference_through_the_tracker` passes — class names resolve by name, `other_vehicle` is trained but ignored by `ClassMap`, the tracker runs on a blank frame.
5. Readiness reporting: the app reports `detector.source` (pretrained / finetuned) and weights file name (`anvesh/app/routes.py:63`) and stamps evidence with `finetuned:sha256-<12 hex>` (`anvesh/perception/weights.py`). Missing weights fail closed instead of downloading.

## 5. Promotion decision

Not promoted. Reasons: (a) CityFlow recall does not improve and is lower on 3 of 4 cameras; (b) no class-level validity on representative footage; (c) cross-check gap unexplained; (d) no downstream (tracking / speed / incident) regression comparison of the two detectors; (e) no end-to-end latency measurement. To revisit: hand-label a few hundred frames from ANVESH's own cameras (class + box), re-run `compare_detectors` style scoring on them, and rerun the downstream tracking evaluation with `ANVESH_FINETUNED_MODEL` set.

## 6. Reproduce

```
.venv\Scripts\python.exe scripts\compare_detectors.py --split test --model pretrained=models\yolov8n.pt --model finetuned=models\finetuned\anvesh_mio5_yolov8n_best.pt --crosscheck-val --out outputs\training\compare_test.json
.venv\Scripts\python.exe scripts\compare_detectors_cityflow.py --model pretrained=models\yolov8n.pt --model finetuned=models\finetuned\anvesh_mio5_yolov8n_best.pt --out outputs\training\compare_cityflow.json
.venv\Scripts\python.exe scripts\summarize_training.py --run models\finetuned\runs\anvesh_mio5_v1b --compare-val outputs\training\compare_val.json --compare-test outputs\training\compare_test.json --cityflow outputs\training\compare_cityflow.json --figures docs\training_figures
```
The test command is refused on a second run by design (lock file). Note the lock is written **before** scoring, so a crash mid-run would consume the held-out test; run it only when memory is free.


## 7. Metric discrepancy audit and comparability (added 2026-10-18, third session; sections 1-6 above are unchanged)

Scope: no model was retrained, the protected test split was **not** re-run and its lock (`datasets/mio_tcd/yolo5/test_evaluation.lock.json`, 1 run) is untouched; no existing result file was modified. The diagnostics below ran on the **validation** split and on CityFlow only. Code: `scripts/diagnose_eval_gap.py`, `scripts/diagnose_cityflow_recall.py`. Outputs: `outputs/training/diag_eval_gap.json`, `outputs/training/diag_cityflow_recall.json`. (Disclosure: to check class-id conventions I counted class frequencies in the first 2,000 test *label* files; no model output was produced on test and nothing was selected from it.)

### 7.1 Why our evaluator is lower than Ultralytics `val()` (caveat 4.1, now explained on val)

Fine-tuned checkpoint, 1,500 validation images (every 7th, copied to a scratch dir so the real label cache is untouched), 0 duplicate label rows in the subset. Per-class AP50, our scorer **without** don't-care regions (what `val()` effectively does):

| variant | bicycle | car | motorcycle | bus | truck | mean (5 cls) |
|---|---|---|---|---|---|---|
| U: Ultralytics `val()` | 0.729 | 0.943 | 0.953 | 0.977 | 0.916 | 0.9038 |
| B: our path as in `compare_detectors.py` (single-label NMS, `classes=` filter, FP16) | 0.752 | 0.932 | 0.950 | 0.964 | **0.856** | 0.8907 |
| C: B without the `classes=` filter (class 5 dropped afterwards) | identical to B to 16 digits | | | | | 0.8907 |
| D: C + **multi-label NMS** | 0.748 | 0.943 | 0.952 | 0.980 | 0.915 | **0.9076** |
| E: D in FP32 | identical to D | | | | | 0.9076 |

* **Cause (closed by ablation): `val()` calls NMS with `multi_label=True` (`ultralytics/models/yolo/detect/val.py:124`; `predict.py:54` passes no `multi_label`, so it is off).** With multi-label NMS one box may enter the ranking once per plausible class, so a car/truck-ambiguous vehicle can still contribute a correct low-confidence prediction in the other class. Turning it on in our path moves the 5-class mean from 0.8907 to 0.9076 (U = 0.9038) and truck from 0.856 to 0.915 (U = 0.916), i.e. the bulk of the truck-sized gap disappears. After D our value is 0.004 above U on this subset (bicycle 0.748 vs 0.729, truck 0.9153 vs 0.9164); that residual is within what letterbox/padding differences and a 1,500-image subset (bicycle has few boxes) can produce and was **not** investigated further.
* **Not causes:** the `classes=` filter (C = B exactly: the filter is applied after the per-box best class is chosen, so it only removes class-5 boxes, which are then absent from the score anyway); FP16 vs FP32 (E = D exactly at AP50; the flag really does change precision: verified `fp16 True/False` and parameter dtype `float16/float32`); duplicate label rows (0); class-name mapping (by name, identical to training ids 0-4); IoU matching and AP interpolation (same Ultralytics `ap_per_class`, and variant D reproduces U).
* **Inference to the test split (inferred, not measured there):** the test-split gaps in 4.1 (ours lower by 0.011-0.045, largest for truck) have the same sign and ordering as B-vs-U here, so multi-label NMS is the expected explanation; because the test split may not be re-run, this stays an inference. The ground-truth counts are consistent: our 33,611 five-class boxes plus the 2,858 `other_vehicle` boxes Ultralytics also counts (derived by subtraction, not independently counted) equal the 36,469 instances in `compare_test.log`.
* **Which side is right for ANVESH:** the app's pipeline uses Ultralytics `track()`, which goes through the same single-label predictor post-processing (conf 0.25), so our single-label numbers describe the deployed behaviour better; the `val()` numbers are the standard benchmark convention and are about 0.013-0.015 mAP50 higher for the same weights. Neither is wrong; they must not be mixed.

### 7.2 Is the pretrained-vs-fine-tuned comparison fair? (checked)

* Same images and settings: `compare_detectors.py` builds one sorted image list and calls the same `run_model` for both checkpoints (n_images 10,865 and n_gt 33,611 identical for both in `compare_test.json` and `compare_val.json`; checkpoint SHA-256s recorded and equal to section 2). Same conf 0.001, IoU 0.7, max_det 300, imgsz 640, class mapping by name, same matcher and AP code.
* The unfair part is not the evaluator but the data: MIO-TCD labels define "truck"/"car" differently from COCO (caveat 4.3).
* Ranking is robust to the NMS setting: on the same 1,500 val images the pretrained model scores 0.222 (single-label) and 0.243 (multi-label) against 0.891 / 0.908 for the fine-tuned model, so the order and the size of the gap (~0.67) do not depend on which NMS convention is used.
* CityFlow: both models ran on the same frames in the same loop with the same settings; re-running reproduces the stored result exactly (pretrained TP 1,575, fine-tuned TP 1,511). `diag_cityflow_recall.json`.

### 7.3 CityFlow: is the fine-tuned model losing recall to its ignored class? (no)

`ClassMap` ignores `other_vehicle`, so boxes with that label are invisible to the app. Counting them anyway (class-agnostic recall, same frames): fine-tuned pooled recall **0.7178 -> 0.7178** (27 `other_vehicle` boxes in ROI, none matched a missing GT box). With multi-label NMS 0.7188 (pretrained 0.7482 -> 0.7492). So the -0.030 CityFlow gap (operational figures 0.718 vs 0.748, unchanged) is **not** an artefact of the ignored class or the NMS convention. (The pretrained "incl_other" column also counts COCO traffic lights/stop signs and is only meaningful for the fine-tuned model.) Whether the gap is real is still undecided: it has no confidence interval, rests on 4 cameras, and the CityFlow GT only contains vehicles seen by >= 2 cameras.

### 7.4 Which numbers are comparable

| Comparison | Comparable? | Note |
|---|---|---|
| pretrained vs fine-tuned within one `compare_*.json` (val or test) | **Yes** | identical images, settings and scorer; single-label NMS, so absolute values are ~0.015 conservative vs `val()` |
| fine-tuned `compare_val.json` (0.885) vs `compare_test.json` (0.870) | Yes, roughly | same scorer; val was used for checkpoint choice, so val is mildly optimistic |
| our 5-class mAP50 (0.870 test) vs Ultralytics 6-class `val()` (0.797 test) or per-epoch `results.csv` (0.812 val) | **No** | 6 classes incl. `other_vehicle`, no don't-care regions, multi-label NMS |
| per-class AP50 ours-no-ignore vs `val()` | **Yes after matching NMS** | gap closed on val (7.1); test-split closure inferred |
| our P/R at conf 0.25 vs Ultralytics P/R | No | different threshold semantics (we threshold before matching; Ultralytics reports at the max-F1 point) |
| MIO-TCD mAP vs CityFlow recall | No | different data, labels, metric; CityFlow has no class labels and no precision |
| `inference_seconds` in the compare JSONs, latency from `compare_test.log` | No | see caveat 4.2 and 7.5 |

**Still unresolved:** (a) the +0.004 residual after matching NMS; (b) test-split closure was not measured; (c) CityFlow recall difference has no CI; (d) no class-level truth on CityFlow; (e) no valid latency number (7.5); (f) no downstream tracking comparison; (g) anything about the ANVESH cameras (no local labelled data exists yet: `docs/ANVESH_LOCAL_CCTV_BENCHMARK.md`).

### 7.5 Latency

The only existing latency figure (1.3/3.1/1.2 ms, fine-tuned only, batch 16) is Ultralytics' validation pipeline and is not ANVESH streaming latency. A batch-1 end-to-end tool was added (`scripts/benchmark_local_cctv.py latency`) and run on CityFlow c006 frames (not local footage): the same command produced median 59.8/76.4 ms (pretrained/fine-tuned), 15.9/20.7 (reversed order), 104.0/99.2 and 45.2/33.5 (interleaved), i.e. a 3-6x run-to-run swing from machine state. **No latency conclusion is drawn**; the two checkpoints share a backbone (different class heads) and showed no consistent ordering across runs. Details and method: `docs/ANVESH_LOCAL_CCTV_BENCHMARK.md` section 7.

### 7.6 Promotion decision (unchanged: not promoted) and what replaces the vague "to revisit" in section 5

The explicit gate is in `docs/ANVESH_LOCAL_CCTV_BENCHMARK.md` section 6 (recall floor and non-inferiority with recording-level bootstrap CIs, FP/frame ceiling, per-camera regression, latency, CityFlow and downstream non-inferiority). Actual gate output on the inputs that exist (`benchmark_local_cctv.py gate --cityflow outputs/training/compare_cityflow.json`, saved `outputs/local_benchmark/gate_2026-10-18.json`): **DO_NOT_PROMOTE** - the CityFlow line fails on its point estimate (-0.030 vs a -0.02 margin; no CI), every local line and the downstream line are not evaluable (no labelled local footage; downstream not run); latency was not passed because the measurements are noise. `anvesh/app/config.py` and the default model are untouched.


## 8. Addendum 2026-10-20 (no re-evaluation; sections 1-7 unchanged)

- The held-out MIO-TCD test split was **not** re-run. Its lock file (1 run) is byte-identical to 2026-10-18 and a test now fails if a second run is ever recorded.
- **Licence consequence for the fine-tuned checkpoint**: MIO-TCD is CC BY-NC-SA 4.0 (dataset page, read 2026-10-20), which does not address trained models. The checkpoint is derived from NonCommercial, ShareAlike data, so it must not be deployed commercially and its use by a department needs a legal decision. It remains non-default. Ultralytics YOLOv8 is AGPL-3.0 or a paid Enterprise License. See `docs/ANVESH_RELEASE_READINESS.md` section 4.
- Plate detector: the only attempt (`plates_hf_v1`) left no checkpoint; nothing to resume. The plate dataset validates clean (no exact or base-id leakage; 2 % near-duplicate overlap with training by a thumbnail screen) but is Vietnamese and its licence chain is unverified. Execution report 9.E.
- Training script now has a resource pre-flight, `--resume RUN` (refuses without a real `last.pt`) and never overwrites a manifest. On the current machine state the pre-flight refuses to start (free RAM 2.24 GB).
