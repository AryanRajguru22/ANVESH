# M6.5 — Real-World Perception Hardening & Validation

## 1. Purpose

M6 ran the M1–M4 pipeline end to end against two independent real-video
sequences and found every window classified `insufficient_evidence`. M6.5
existed to answer one question honestly, without touching the algorithms
themselves: **is that a genuine limitation of light real traffic and V1's
documented evidence gaps, or a fixable configuration/tuning problem?**

M6.5 was split into two phases:

- **Phase 1 (inspection only):** trace the exact causes of track
  fragmentation and evidence starvation using small, single-variable
  diagnostics, without changing any default configuration or code.
- **Phase 2 (configuration plumbing):** add an *optional*, clearly
  separate real-camera threshold configuration, changing only thresholds
  that could be independently derived (i.e. derivable without reference to
  how much traffic the two clips happened to contain), and compare it
  against the frozen M6 default.

M6.5 never modified M1–M5 detection, tracking, calibration, fusion, or
propagation logic, and never fabricated calibration or corridor geometry.

## 2. Default configuration result (frozen M6 baseline)

Running `scripts/run_real_world_validation.py` with no arguments (byte-for-byte
the same behavior as the original M6 run):

| sequence | tracks | max vehicle_count/window | congestion level | outcome |
|---|---|---|---|---|
| seq1_car_detection | 7 | 4 | free_flow (all windows) | insufficient_evidence (4/4 windows) |
| seq2_person_bicycle_car | 5 | 3 | free_flow (all windows) | insufficient_evidence (6/6 windows) |

Propagation (M5) correctly reports `UNRESOLVABLE` for both sequences: both
cameras are uncalibrated (`motion_space=IMAGE`), and
`flow_density_from_traffic_state` raises before any distance is used.

## 3. Real-camera configuration result

Running with `--real-camera-config configs/experiment/m6_real_camera_thresholds.toml`
(only `max_capacity_vehicles` overridden, per sequence: 18 for seq1, 15 for
seq2) produced **no observable change**:

- Same vehicle counts, same congestion levels (`free_flow` throughout).
- Same outcome distribution: still `insufficient_evidence` on every window,
  both sequences.
- Occupancy shifted numerically (e.g. seq2 window `[40,50)`: 3/20=0.15 →
  3/15=0.20) but never crossed `CongestionThresholds.low_occupancy=0.3`, so
  no congestion-state or evidence change followed.

This is reported as a genuine **null result**, not reframed as a partial
success.

## 4. Tracking fragmentation finding (Phase 1)

Both real videos produce short-lived, fragmented tracks (mean duration
~0.7–2.1s) despite each showing continuous vehicle motion on visual
inspection. Two single-variable diagnostics were run and both were
**null results**:

- `max_missed_frames` 5 vs. 30: no change in track count or duration.
- Detection confidence threshold 0.25 vs. 0.15: no change in track count
  or duration (only more raw per-frame detections).

Frame-by-frame tracing of the raw ByteTrack ID stream, cross-checked
against extracted frames, found the actual cause: for `car-detection.mp4`,
the detector produced **zero detections for 21 consecutive frames**
while a visually-continuous vehicle's bounding box shrank and migrated
toward the frame's top edge — a YOLOv8n/COCO domain mismatch with this
camera's steep top-down geometry near the frame edge, not a
configuration or tracker-lifecycle issue. This is classified as a **data/
model-domain limitation**, out of scope to fix in M6.5 (fixing it would
require retraining or fine-tuning the detector, which is explicitly
forbidden).

## 5. Evidence-starvation finding (Phase 1)

Traced exactly (not guessed) for a representative window
(`car-detection.mp4`, `[10,20)`, vehicle_count=4, `motion_space=IMAGE`):

- H1: NEUTRAL — occupancy 4/20=0.2 never reaches `low_occupancy=0.3` or
  `high_occupancy=0.6`; congestion state never leaves `free_flow`.
- H2/H3: NONE — unconditionally gated off for any `motion_space != WORLD`.
- H4/H6: NONE — unconditionally unavailable in V1 (no lane geometry, no
  multi-window cyclical history).
- H5: NEUTRAL — vehicle_count 4 < `elevated_vehicle_count=8`.

Result: total ignorance mass (theta) = 1.0 exactly →
deterministic `INSUFFICIENT_EVIDENCE`. This is not a bug or a borderline
case; it is the documented, correct behavior of the evidence rules given
this input.

## 6. Independent derivation of the 18 / 15 capacity values

Both validation videos are 768×432 px (331,776 px² frame area). Median
high-confidence (`conf >= 0.4`) car/bus/truck bounding-box area was
measured directly from each camera's own detections:

- seq1_car_detection: 17,777 px² (n=44 high-confidence boxes)
- seq2_person_bicycle_car: 21,399 px² (n=70 high-confidence boxes)

`max_capacity_vehicles = frame_area / median_vehicle_area`, floored:

- seq1: 331,776 / 17,777 = 18.66 → 18
- seq2: 331,776 / 21,399 = 15.50 → 15

This derivation depends only on camera field-of-view geometry and typical
vehicle footprint size in that camera's own frame — it would produce the
same number regardless of whether the clip showed light or heavy traffic,
which is the independence property M6.5 required before any threshold
could be changed. Notably, both derived values land close to the existing
default of 20, which undermines rather than supports an initial Phase 1
hypothesis that the default was "wildly oversized" for these cameras.

## 7. Why `elevated_vehicle_count` was NOT changed

`elevated_vehicle_count` (H5, default 8) has no independent derivation
available: there is no historical or reference traffic-volume baseline for
either camera to calibrate an "elevated" cutoff against. Any value chosen
would either restate the geometric capacity above (too close to
reverse-engineering a wanted outcome) or be an unsupported guess with no
traceable justification. It was left at the project default, exactly as
instructed: if no defensible basis exists, leave the value unchanged.

The same reasoning applied to all four `CongestionThresholds` fields (no
verifiable traffic-engineering standard was available to cite for these
specific cameras) and to `EvidenceThresholds.stopped_speed_m_per_s` /
`min_dwell_seconds` (both are gated by `motion_space == WORLD`, and both
real-camera sequences are `motion_space = IMAGE`, so these values are
inert here regardless of what they are set to).

## 8. Calibration/propagation limitation

No calibration data exists for either sequence: no surveyed segment
length, no real reference points, no inter-camera distance. M6.5 did not
fabricate any of these. Propagation continues to correctly refuse to run
(`UNRESOLVABLE`, via the existing `ValueError` safety check in
`flow_density_from_traffic_state`) for both the default and real-camera
configurations. This is unchanged, correct behavior, not a limitation
introduced or worked around by M6.5.

## 9. Important negative/null results

- Neither `max_missed_frames` nor detection confidence explains track
  fragmentation (Phase 1).
- The only independently-derivable threshold change (`max_capacity_vehicles`)
  produces **zero change** in congestion classification, evidence
  availability, or ranking outcomes (Phase 2).
- Real footage on these two clips never produces a `ranked` candidate-cause
  outcome under either configuration — every window across both sequences
  and both configs is `insufficient_evidence`.

These results are reported as-is. No threshold was tuned to make a
`ranked` outcome appear.

## 10. What M6.5 does NOT prove

- It does not prove ANVESH produces accurate candidate-cause rankings on
  real traffic — no window in either sequence ever reached a `ranked`
  outcome, so no ranking accuracy claim can be made either way.
- It does not validate detection or tracking precision/recall — no
  frame-level human annotation was performed on either video.
- It does not validate cross-camera fusion, propagation, or corridor-level
  behavior on real data — the two sequences are not two views of one
  corridor (see `datasets/recorded/validation/SOURCE.md`), and propagation
  never ran (correctly refused, uncalibrated).
- It does not establish that `max_capacity_vehicles=18/15` are "correct"
  in any general sense — they are one honest geometric estimate for these
  two specific cameras, not a calibrated or validated capacity figure.

## 11. What remains unresolved

- The detector/model domain-mismatch near frame edges (Phase 1's traced
  root cause of track fragmentation) is unresolved and out of scope for
  M6.5 (would require retraining/fine-tuning, which is forbidden).
- No independent basis was found for `elevated_vehicle_count` or any
  `CongestionThresholds` field for these cameras; if a defensible source
  is ever identified, that would be a distinct future undertaking with its
  own honesty checks, not an M6.5 change.
- No real calibration or corridor geometry exists for either sequence;
  cross-camera/propagation validation on real data remains entirely
  unattempted.

## 12. What M6.5 is

M6.5 is a **validation and hardening checkpoint**: it inspects, diagnoses,
and (where independently justifiable) configures the existing M1–M5
pipeline against real video, and reports what happens — honestly,
including when the answer is "nothing changed." It is **not** evidence of
real-world accuracy, not a production-readiness milestone, and not a
claim that ANVESH's candidate-cause reasoning has been validated against
any real ground truth.
