# ANVESH V1 Speed Validation Audit — c006 Outlier Speeds

## Symptom

The V1 Dataset Showcase (AICity/CityFlow S02 c006) displayed box-level
speed values such as `~2084 km/h (approx.)`, `~2492 km/h (approx.)`,
`~3217 km/h (approx.)` — physically impossible for a road vehicle.

## Investigation

Traced the complete path: detection → track → ground-contact point →
image/world position → displacement → elapsed time → speed → formatting
→ UI, using the real repository code and the real c006
`calibration.txt`/detection data (not a synthetic stand-in), reproducing
the exact `~2492 km/h` value end-to-end.

### What was checked and ruled out

- **Track gaps.** `anvesh.perception.motion.compute_robust_speed`
  already excludes any consecutive step whose `dt` exceeds `max_gap_s`
  (default 0.6s, `anvesh.perception.speed_estimation.
  DEFAULT_MAX_GAP_S`) from the distance/time totals, and already has
  dedicated tests (`tests/perception/test_motion.py::
  test_robust_large_temporal_gap_with_no_remaining_steps`,
  `test_robust_large_gap_excluded_but_enough_steps_remain`). Not the
  cause.
- **Unit conversion arithmetic.** `MPS_TO_KMH = 3.6` is correct m/s→km/h
  arithmetic. Not the cause by itself.
- **Numerical instability / invalid trajectory.** Every reproduced
  outlier had `RobustSpeedStatus.OK` — the math completed cleanly on
  finite, well-sampled input. Not the cause.
- **Insufficient temporal baseline.** All reproduced cases had
  `sample_count` well above `DEFAULT_MIN_SAMPLES` (3) and elapsed time
  well above `DEFAULT_MIN_ELAPSED_S` (0.3s). Not the cause.

### Root cause

`anvesh.world.calibration.from_cityflow_homography` already documents,
correctly, that a CityFlow-supplied homography's output units are
**unvalidated** (`units="unknown"`, `metric_validation=UNVALIDATED`).
Probing the real c006 homography (`E:\traffic dataset\validation\S02\
c006\calibration.txt`) via `CalibrationProfile.image_to_world` at a grid
of image rows shows the implied **world-scale is 2–28 "units" per IMAGE
PIXEL**, growing sharply toward the frame's vanishing line (ordinary
perspective foreshortening):

| image y | world-units / pixel (x) | world-units / pixel (y) |
|---|---|---|
| 100 | 5.90 | 28.32 |
| 300 | 4.77 | 11.19 |
| 460 | 4.00 | 6.65 |
| 1079 | 2.39 | 1.87 |

For comparison, the c006 lane polygons' own top edge (`L1`, y=210,
x=769→846) spans 77px — if 1px ≈ 5–6 "units", that lane would be ~450
"units" wide, roughly two orders of magnitude too large to be metres for
a real ~3.5m-wide traffic lane. **The homography's output units are not
metres**, or at minimum are not consistently metres across this frame —
exactly what `metric_validation=UNVALIDATED` already warned, now
confirmed numerically rather than merely flagged.

Reproducing track `3`'s real per-frame ground points (bottom-center of
its YOLOv8 bounding box, exactly as `estimate_vehicle_speed` uses)
through this homography and `compute_robust_speed` gives
`speed = 692.2 m/s → 2491.9 km/h`, matching the observed `~2492 km/h`
exactly. The per-step contributions (830–1940 km/h each, sustained
across many consecutive frames, not one noisy step) show this is **not**
primarily single-frame bounding-box jitter — it is *ordinary,
real, on-screen vehicle motion* (a few pixels per 0.1s frame — normal for
a moving car) amplified by the oversized per-pixel world-scale above.
Jitter (a stationary vehicle's few-pixel detector noise) makes the same
defect worse still, confirmed separately (see Test 4 below): even a
purely jitter-only, near-zero-real-motion trajectory produces an
implausible derived speed under this homography.

**Conclusion:** the estimator was multiplying a projected world
displacement by `MPS_TO_KMH` and presenting the result as an ordinary
"approximate" observation without ever checking whether the number was
physically possible for a road vehicle — the missing piece was a
**physical-plausibility gate**, not a displacement/gap/timing bug.

### What was NOT the (primary) cause, but is real and documented as a limitation

Ground-point/detector jitter does independently contribute (Test 4), and
is worse for objects farther from the camera (higher per-pixel scale),
but the dominant, reproducible effect is the oversized/unvalidated
homography scale itself, which amplifies *any* image-space displacement
— real motion or noise alike.

## Fix

**Scope: `anvesh/perception/speed_estimation.py` only** (plus its direct
UI-facing formatter, `anvesh/rendering.py::format_speed_text`, and the
two showcase-generation scripts that write this data to disk). No
perception/tracking/evidence/DS-fusion/ranking/propagation/lane module
was modified.

1. New `SpeedConfidence.OUTLIER` value, alongside the existing
   `VALIDATED`/`APPROXIMATE`/`UNAVAILABLE`.
2. New `MAX_PLAUSIBLE_SPEED_KMH = 200.0` constant — an explicit,
   documented, generous **physical** ceiling for a road vehicle in
   ordinary traffic-camera footage. Not a measured accuracy claim, not
   this camera's real (much lower) legal speed limit, not
   vehicle-class-specific. Chosen to sit comfortably above any real
   vehicle's sustained speed while sitting comfortably below every
   outlier this investigation actually observed (~450–3200 km/h) — see
   the table above and the c006 sample's own `speed_confidence_counts`
   below.
3. `estimate_vehicle_speed`: after computing `speed_kmh`, if it exceeds
   `MAX_PLAUSIBLE_SPEED_KMH` the result is `SpeedConfidence.OUTLIER`
   instead of `APPROXIMATE`/`VALIDATED` — even for a metric-VALIDATED
   calibration, since a physically implausible value cannot
   simultaneously be a confirmed metric observation. `RobustSpeedStatus`
   stays `OK` (the math is fine; this is a plausibility judgment, a
   separate axis) and the `Measurement` is still attached (never
   discarded — available for diagnostics), matching this module's
   existing "never fabricate, never silently drop" discipline.
4. `format_speed_text`: `OUTLIER` renders as `"UNRELIABLE — OUTLIER"` —
   never the raw number, never a fabricated substitute.
5. **No clipping.** `2084 km/h` never becomes `150 km/h` or any other
   substituted value anywhere in this fix — it becomes an explicit
   OUTLIER marker, or is available unchanged in the raw
   `VehicleSpeedEstimate.measurement` for diagnostics.

## Track gaps (explicit re-verification)

`compute_robust_speed`'s existing gap exclusion was re-verified
end-to-end through `estimate_vehicle_speed` (not just at the
lower-level, unit-agnostic layer it already had tests for):
`test_track_gap_does_not_produce_a_huge_speed` and
`test_track_gap_leaving_no_usable_steps_is_unavailable_not_huge`
(`tests/perception/test_speed_estimation.py`) construct a trajectory with
a deliberate 4.8s gap to a point 500 pixel-units away and confirm the
gap step is excluded (small in-cluster motion is measured instead of a
fabricated cross-gap speed), and that a gap leaving no usable steps
correctly reports `UNAVAILABLE`, never a huge number. No behavior change
was needed here — this was already correct; the tests close a coverage
gap, not a bug.

## Speed evidence safety

Traced every current caller of `estimate_vehicle_speed`/
`VehicleSpeedEstimate`: only `scripts/export_ui_data.py` (dashboard box
labels) and `anvesh/rendering.py` (video-overlay labels, itself not
wired into any pipeline caller). **Neither `evidence_builder.py`,
`candidate_vehicle.py`, `ranking.py`, `ds_fusion.py`, nor
`propagation/shockwave.py` import this estimator at all** — confirmed by
static import analysis, pinned as an automated regression test
(`tests/perception/test_speed_evidence_safety.py`), which fails loudly
if any of those modules ever start importing it without first excluding
`OUTLIER`/`UNAVAILABLE`. An OUTLIER speed was therefore already
structurally incapable of contaminating H2/H3/H4 evidence, candidate
resolution, ranking, or propagation, before this fix — this fix closes
the *display* bug and adds a standing guarantee against the *evidence*
risk.

(Separately noted, out of scope for this pass: `evidence_builder.py`'s
own H2/H3 "stopped vehicle" threshold reads `VehicleTrack.speed_estimate`
— a DIFFERENT, non-robust, centroid-based computation
(`schema_conversion.build_vehicle_track` → `motion.
compute_world_space_speed`, no gap-exclusion) that is subject to the
same underlying homography-scale characteristic. Its failure direction is
under-detection of genuinely stopped vehicles — not fabrication of false
"stopped" evidence — so it is not a "Test 5"-type contamination risk, but
is flagged here as a legitimate follow-up outside "this specific
speed-estimation correction"'s scope, since fixing it would mean changing
`schema_conversion.build_vehicle_track`'s general-purpose, widely-shared
behavior.)

## Numeric effect on the c006 showcase (regenerated via
`scripts/run_s02_c006_showcase.py`, no manual JSON edits)

| speed_confidence | before fix | after fix |
|---|---|---|
| approximate | 74 | 7 |
| outlier | — | 66 |
| unavailable | 28 | 34 |

The 7 tracks that remain `approximate` read 95–193 km/h — still
unvalidated-metric, but no longer physically impossible.

## Lane geometry

Untouched. `configs/lanes/s02_c006.toml` was not modified; L1/L2 vertices
are byte-identical to before this pass. `tests/world/`,
`tests/perception/test_lane_assignment.py` (128 tests) re-run clean.

## Candidate-vehicle resolution

Untouched (`anvesh/hypotheses/candidate_vehicle.py` not modified). The
regenerated c006 sample's 3 windows still resolve to
`no_vehicle_for_hypothesis` (×2) / `not_ranked` (×1, insufficient
evidence) — no candidate is manufactured; this was true before and after
this fix, and the showcase UI continues to render that honestly (never a
fabricated red highlight).
