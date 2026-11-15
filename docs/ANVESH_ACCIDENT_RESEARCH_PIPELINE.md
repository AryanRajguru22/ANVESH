# ANVESH accident / incident detection - research pipeline (design and tooling)

Date: 2026-10-20. Status: **design + validated tooling only. No accident model exists, none has been trained, no dataset has been downloaded.**
The only accident-related detector in the application is the `possible_collision` rule heuristic (`anvesh/app/detectors.py`), which is
labelled experimental and has never seen a real collision. Nothing in this document is a claim about accuracy.

## 1. Why a separate pipeline

A rule over 2-D boxes (approach, sustained contact, joint stop) cannot separate a collision from two cars queueing bumper to bumper, a
vehicle occluding another, or a tracker identity swap. A learned temporal model could, but only if it is trained and judged on the right
data with the right protocol. The failure modes of getting the protocol wrong are well known and silent: adjacent-frame leakage inflates
held-out scores; counting alarms in unlabelled footage as "correct" hides false alarms; reporting per-frame accuracy hides that an
operator is paged 40 times a shift.

## 2. Architecture (proposed; none of stages 3-4 exists)

```
camera/recording
  1. vehicle detector + tracker           EXISTS (YOLOv8n + ByteTrack, tracked boxes with session-scoped ids)
  2. per-window temporal features         EXISTS in part (speed/step displacement, stop episodes, concurrency, box IoU); to add:
                                          pairwise closing speed, IoU history, heading change, abrupt group deceleration, occupancy jump
  3. temporal event model                 NOT BUILT. Options, in order of data appetite:
                                          (a) gradient-boosted trees / logistic regression over window features (needs hundreds of events),
                                          (b) 1-D CNN / GRU over the feature time series (needs thousands),
                                          (c) clip-level video model (VideoMAE/SlowFast class; needs thousands + GPU time; licences differ).
  4. alert logic                          cooldown, minimum duration, hysteresis, a score threshold chosen on VALIDATION at a stated
                                          false-alarms-per-hour budget; every alert is a candidate for a human, never an enforcement
  5. evaluation (event level)             EXISTS (anvesh/training/temporal_events.py): recall, detection delay, false alarms / hour
```

Recommendation: start with (a) on stage-2 features. It needs the least data, is auditable (feature importances), reuses the tracker that is
already deployed, and gives a baseline any heavier model must beat on the same split.

## 3. Annotation schema (implemented and validated)

JSON, `schema_version` 1 (`anvesh/training/temporal_events.py`, CLI `scripts/event_manifest.py validate`):

- **video**: `video_id`, `source_group`, `camera_id?`, `fps`, `duration_s`, `license` (required, free text), `fully_annotated` (bool, required), `split?`
- **event**: `event_id`, `video_id`, `type`, `start_s`, `end_s`, `annotator`, `certainty` (`certain` | `probable`); events in one video may not overlap.

Two fields carry the important semantics:

- `fully_annotated` is the **missing-vs-empty** flag. `true` = a person reviewed the whole video and listed every event, so time outside the
  events is verified negative and an alarm there is a false alarm. `false` = unlabelled: alarms there are ignored (neither hit nor false alarm)
  and the video contributes **zero** negative hours. The validator refuses a manifest where the flag is missing or not a boolean.
- `source_group` is the unit that may not straddle a split (next section).

Annotation rules to adopt (to be confirmed with the annotators): the event **start** is the first frame at which contact/impact is visible (not
the stop afterwards); `end_s` is when the involved vehicles stop moving relative to each other; ambiguous cases are `probable`, never dropped; a
second annotator labels >= 10 % of videos and event-level agreement (IoU of intervals >= 0.5) is reported before training starts.

## 4. Splitting: by source recording, never by frame or window

`split_by_source_group` assigns **whole source groups** to train/val/test. All clips cut from one recording/camera session share a group.
Windows and frames from one recording are near-identical, so any split finer than the recording leaks. Details implemented and tested:

- groups that contain events are placed first, into the split furthest below its event target, so small datasets still get events in every split;
- it **refuses** (raises) rather than return a split with no events;
- `validate_manifest` fails if a `source_group` appears in two splits or if only some videos have a split;
- the same manifest + seed always gives the same split; `scripts/event_manifest.py split` refuses to overwrite and refuses an already-split manifest.

For still-image detector datasets the equivalents are `scripts/validate_yolo_dataset.py` (exact-duplicate and base-id leakage, missing-vs-empty labels)
and `scripts/audit_split_leakage.py` (thumbnail near-duplicate screening). Those tools do **not** make a video-derived frame set safe: frames
cut from video must still be split by source video.

## 5. Evaluation protocol (implemented)

`evaluate_event_detector` / `scripts/event_manifest.py evaluate`, on the **test** split only for any claim:

- **Event recall**: events with >= 1 alarm in `[start - early_tol, end + late_tol]` (default 2 s / 2 s) / all events. Several alarms for one event are one detection.
- **Detection delay**: alarm time minus event start for the first matching alarm; median / mean / p90; early alarms are reported as early (negative), not clipped to zero.
- **False alarms per hour**: alarms (after a cooldown merge, default 5 s) outside every event window in `fully_annotated` videos / hours of those videos.
- **Uncertainty**: video-level bootstrap 95 % intervals for recall and false alarms/hour (videos are resampled, not alarms).
- **Operating curve** over score thresholds, so a threshold is chosen on validation from a stated false-alarm budget, not on test.
- **`claims_allowed`** is false unless the evaluated split is `test`, has >= 30 events and >= 10 h of verified-negative video. Otherwise the report says `NOT_EVALUABLE_FOR_CLAIMS` and why.

The unit tests use hand-checkable synthetic manifests and a throw-away synthetic "speed-drop" rule purely to exercise the code. They say nothing about accidents.

## 6. Candidate data (availability and licence reviewed; nothing downloaded)

| Source | Fit | Access / licence (as found) | Verdict |
|---|---|---|---|
| **CADP** (CMU, CCTV traffic-camera accidents, 1,416 YouTube-derived segments, 205 with spatio-temporal annotation) | fixed CCTV view - the closest to ANVESH | project page + README (read 2026-10-18): use **"for non-commercial and research based purposes"**, other uses need the authors' written consent; **raw videos are not distributed** (YouTube id list; extracted frames via Google Drive; annotations JSON); frame rate / size not stated. A web search on 2026-10-20 could not re-confirm the terms or find a current download page | research only; check the Drive content and size first; a model trained on it cannot be assumed deployable by a department |
| DoTA / DAD | dashcam | not CCTV; DoTA downloader scrapes YouTube | not suitable |
| TUM Traffic Accid3nD | multi-sensor roadside | CC BY-NC-SA 4.0 | not suitable (non-commercial, sensor type) |
| **Your own CCTV recordings** | exact domain | yours to license | **the only data that can support a deployment claim**; needs real incidents, which are rare - plan on months of footage plus staged/near-miss data being clearly labelled as such |

Licence consequence worth deciding early: if the department deploys, anything trained on NC-licensed data is a legal question to settle with counsel before it is relied on.

## 7. What can only start once there is data

1. Pick a labelled source (own footage and/or CADP frames after reading its README) and write the manifest (`license`, `fully_annotated`, `source_group` per recording).
2. Second-annotator agreement on >= 10 %.
3. `event_manifest.py validate` -> `split` (seeded) -> freeze the test split.
4. Baseline (a), threshold chosen on validation at a false-alarm budget the control room agrees to, then **one** evaluation on test.
5. Shadow-mode run on live cameras before any alert reaches an operator.
