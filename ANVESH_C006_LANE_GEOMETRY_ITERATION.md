# ANVESH Commit 9A.1 — c006 Lane Geometry Correction

Geometry/configuration iteration only. **No source code, test code,
rendering code, speed-estimation code, candidate-vehicle code, or
pipeline behavior was modified.** The only file changed under `anvesh/`,
`scripts/`, or `tests/` is none — the single file changed anywhere in
the repository is `configs/lanes/s02_c006.toml`. No ANVESH-V0 file and
no CityFlow dataset file was touched. Not committed (per instructions).

## 1. Objective

Correct the c006 near-side lane geometry using real trajectory evidence
(not repeated visual pixel-picking), per the findings of
`ANVESH_C006_LANE_COVERAGE_ANALYSIS.md`, while explicitly avoiding
"optimize for assigned %" — every proposed polygon must correspond to an
actual, physically real, drivable lane, validated against real vehicle
ground-point trajectories and cross-checked visually against multiple
real frames.

## 2. Previous geometry (Commit 3, now superseded)

One 4-vertex quadrilateral, `L1`, hand-measured from pavement paint in a
single reference still:

```
L1 (old): (660,220) -> (1010,220) -> (946,440) -> (622,440)
```

## 3. Evidence from the previous lane-coverage analysis

`ANVESH_C006_LANE_COVERAGE_ANALYSIS.md` established, from a 30-second
bounded real-video sample (`ui/data/anvesh_s02_c006_data.json`, 102
unique tracks, 3,926 detection-level ground points):

- The near-side carriageway has **2 physically distinct through-traffic
  lanes** (12 tracks in one, 10 in the other), not the 1 lane `L1`
  modeled.
- The old `L1` polygon was **not lane-pure**: only 35.8% of its own real
  lane's observations fell inside it (too narrow), while 46.3% of the
  *other* lane's observations also fell inside it (too wide) — a
  geometric defect, not merely under-coverage.
- The far-side carriageway's S-curve exclusion and the left-turn
  pocket's exclusion were both confirmed correct.
- The second (right) lane is pavement-marked with an exclusive
  right-turn arrow + "ONLY" text; despite that, every one of its 10
  sampled tracks drove straight through the entire visible corridor
  without turning — real, but only provably stable *within camera view*.

## 4. Proposed geometry — derivation

Per this task's explicit instruction, vertices were derived from real
**bottom-center ground points** (`bbox_xyxy` bottom-center — the
vehicle's actual roadway footprint, never the bounding-box centroid;
matches `anvesh.perception.types.RawDetection.ground_point`'s own
convention, unchanged from the coverage analysis), never re-measured
from pavement paint alone.

**Step 1 — identify each lane's own stable tracks.** From the 12
"near-side LEFT" tracks identified previously, one (`track 180`) was
excluded from the fitting set: its own trajectory data shows it curving
substantially further left than the other 11 as it nears the camera
(consistent with entering the left-turn pocket, per the prior analysis's
own note), and including it would contaminate the fitted lane boundary
with turn-pocket behavior. The remaining 11 "LEFT-core" tracks, and all
10 "RIGHT" tracks, show a tight, consistent, monotonically-shifting
ground-point band at every sampled row — no further exclusions needed.

**Step 2 — measure real per-row extent.** At three sampled rows
(y=210, y=330, y=460 — three rows instead of the original's two, see §12
for why), the min/max x extent of each lane's own real ground points was
measured directly:

| Row (y) | LEFT-core min/max | RIGHT min/max | Raw gap between lanes |
|---|---|---|---|
| 210 | 777 / 838 | 883 / 932 | 45px |
| 330 | 644 / 703 | 822 / 846 | 119px |
| 460 | 541 / 598 | 773 / 806 | 175px |

**Step 3 — add a conservative safety margin.** A flat **8px** margin was
added to every edge (outward for the lane's own boundary). This is
small relative to the smallest observed gap (45px at y=210) and leaves a
verified, deliberate buffer between the two lanes at every row (see §10)
— a point near the true, unmeasured painted lane-divider is reported
`unknown_outside_regions`, never guessed into either lane.

**Step 4 — validate the fit's curvature error.** A straight line fit
between just the top and bottom rows was checked against the measured
middle-row values and found to deviate by **up to ~29px** at the
midpoint (a real, consistent road-curvature effect, present in both
lanes) — large enough, relative to a ~60-70px lane width, to matter. Adding
the third (middle) sampled row and using a 6-vertex, 2-segment polygon
per lane reduces this deviation to **under 10px** at every checked
intermediate row. `anvesh.world.lane_geometry.LaneRegion` already
supports any polygon with ≥3 vertices — no code change was needed to use
6 instead of 4.

## 5. Proposed geometry — exact vertex coordinates

```
L1 (new, corrected):
  (769, 210) -> (846, 210) -> (711, 330) -> (606, 460) -> (533, 460) -> (636, 330)
  [top-left, top-right, mid-right, bottom-right, bottom-left, mid-left]

L2 (new, PROVISIONAL):
  (875, 210) -> (940, 210) -> (854, 330) -> (814, 460) -> (765, 460) -> (814, 330)
  [same vertex-order convention as L1]
```

Both constructed and confirmed loadable via the real, unmodified
`anvesh.world.lane_config.load_lane_config` /
`anvesh.world.lane_geometry.LaneRegion` / `assign_lane` — no degenerate-
polygon error, no code path touched.

## 6. Visual validation

All images under
`docs/demo_reference/s02_c006_lane_geometry_iteration/`:

- **`00_old_vs_new_with_trajectories.jpg`** — the primary comparison:
  old `L1` (green) straddling both real trajectory streams (orange =
  physical lane 1, red = physical lane 2), new `L1` (yellow) and new
  `L2` (magenta) each tightly and cleanly bounding one real stream, with
  a visible gap between them landing on real, unassigned pavement.
- **`01`–`05` (`validation_*.jpg`)** — the proposed polygons overlaid on
  **five different frames spanning the entire video timeline** (start,
  25%, 50%, 75%, end — not a single still), each showing real, differently
  -queued traffic. In every one, real vehicles' fronts/ground contact
  points sit inside the correct proposed polygon; the polygons visibly
  track the pavement's real, gently-curving lane lines rather than
  cutting across them.

Item 10's "validate against multiple representative frames, not a single
still" is satisfied directly by images `01`–`05`.

## 7. Unique-track validation

**A. Total unique tracks: 102** (unchanged from the previous analysis —
same video, same model, same tracker, deterministic).

**B. Relevant near-side tracks: 22** (12 physical-lane-1, including the
excluded-from-fitting `track 180`; 10 physical-lane-2).

**C. Per-relevant-track detail** (observations, old %L1, new %L1, new
%L2, new %unknown, lane-switching):

| track_id | class | physical lane | n_obs | old %L1 | new %L1 | new %L2 | new %unknown | switches lane? |
|---|---|---|---|---|---|---|---|---|
| 254 | truck | 1 | 39 | 30.8 | 43.6 | 0.0 | 56.4 | no |
| 429 | car | 1 | 77 | 32.5 | 48.1 | 0.0 | 51.9 | no |
| 310 | truck | 1 | 45 | 46.7 | 62.2 | 0.0 | 37.8 | no |
| 16 | car | 1 | 119 | 26.1 | 37.0 | 0.0 | 63.0 | no |
| 11 | car | 1 | 98 | 55.1 | 71.4 | 0.0 | 28.6 | no |
| 543 | truck | 1 | 71 | 31.0 | 46.5 | 0.0 | 53.5 | no |
| 147 | car | 1 | 84 | 33.3 | 47.6 | 0.0 | 52.4 | no |
| 496 | car | 1 | 104 | 21.2 | 30.8 | 0.0 | 69.2 | no |
| 633 | car | 1 | 47 | 53.2 | 63.8 | 0.0 | 36.2 | no |
| 271 | truck | 1 | 57 | 52.6 | 63.2 | 0.0 | 36.8 | no |
| 5 | car | 1 | 75 | 46.7 | 62.7 | 0.0 | 37.3 | no |
| 180 *(excluded from fitting; turn-pocket suspect)* | car | 1 | 49 | 10.2 | 0.0 | 0.0 | 100.0 | n/a (never assigned either lane) |
| 7 | car | 2 | 89 | 51.7 *(old leakage)* | 0.0 | 59.6 | 40.4 | no |
| 473 | car | 2 | 64 | 37.5 *(old leakage)* | 0.0 | 43.8 | 56.2 | no |
| 437 | car | 2 | 55 | 43.6 *(old leakage)* | 0.0 | 50.9 | 49.1 | no |
| 298 | car | 2 | 39 | 53.8 *(old leakage)* | 0.0 | 56.4 | 43.6 | no |
| 134 | car | 2 | 84 | 34.5 *(old leakage)* | 0.0 | 40.5 | 59.5 | no |
| 278 | truck | 2 | 56 | 50.0 *(old leakage)* | 0.0 | 55.4 | 44.6 | no |
| 333 | car | 2 | 45 | 60.0 *(old leakage)* | 0.0 | 64.4 | 35.6 | no |
| 388 | car | 2 | 48 | 56.2 *(old leakage)* | 0.0 | 62.5 | 37.5 | no |
| 285 | truck | 2 | 56 | 46.4 *(old leakage)* | 0.0 | 48.2 | 51.8 | no |
| 190 | truck | 2 | 69 | 40.6 *(old leakage)* | 0.0 | 46.4 | 53.6 | no |

Every "old %L1" value for a physical-lane-2 track was pure leakage (`L2`
did not exist under the old geometry) — the new geometry reduces every
one of those to exactly **0.0%**. Every physical-lane-1 track's genuine
`L1` percentage **increased** (old 21.2–55.1% → new 30.8–71.4%) because
the corrected polygon now actually bounds its own real lane, and `track
180` — previously leaking 10.2% into the old, over-wide `L1` — now
correctly gets **0%** in either lane (it is a turn-pocket-bound vehicle;
excluding it entirely is the correct outcome, not a regression).

**No track ever gets assigned to both `L1` and `L2`** — confirmed for
all 22 relevant tracks by inspecting each one's full per-frame
lane-assignment sequence: every track shows either 1 transition (already
inside a lane at first sight, exits once near the bottom) or 2
(enters once, exits once) — never more, and never a distinct-lane set
containing both `L1` and `L2`. This is the same zero-flicker,
single-entry/single-exit pattern for all 22 tracks (see §11 for detail).

One near-miss worth recording: **`track 10`** (not one of the 22 "core"
tracks; previously bucketed as "other/ambiguous") starts at
`(818, 329)` — right at `L2`'s left edge — and smoothly, continuously
drifts left over 65 frames to `(629, 1080)` by the bottom of frame, a
plausible real lane change from lane 2 into lane 1. It is correctly
handled: assigned `L2` for 9 observations while genuinely inside that
polygon, then `unknown_outside_regions` for the remainder of its
transit through the gap zone — **never** incorrectly picked up by `L1`
(verified: 0/66). This is exactly the intended, safe behavior of
per-detection (not per-track-cached) lane assignment for a real
lane-change vehicle.

## 8. Old vs. new coverage

All numbers cross-validated two independent ways: (1) direct
`anvesh.world.lane_geometry.assign_lane` calls against the existing 30s
sample's real ground points, and (2) a full, fresh end-to-end re-run of
`scripts/run_s02_c006_showcase.py --sample-seconds 30` against the real
dataset video with the new config (§9) — both agree exactly (414/865 and
314/605 in both cases).

| | Old (`L1` only) | New (`L1`+`L2`) |
|---|---|---|
| Physical lane 1 (865 obs) → correct lane | 310 (35.8%) | 414 (**47.9%**) |
| Physical lane 1 → wrong lane | n/a (only 1 lane existed) | 0 (0.0%) |
| Physical lane 2 (605 obs) → correct lane | 0 (0.0%, lane didn't exist) | 314 (**51.9%**) |
| Physical lane 2 → wrong lane (leakage into `L1`) | 280 (**46.3%**) | 0 (**0.0%**) |
| Combined near-side, correct-lane coverage (of 1,470 obs) | 310 (21.1%) *(excluding the 280 that were wrong-lane leakage counted as "assigned" but not genuinely useful)* | 728 (**49.5%**) |
| All 3,926 observations, any `assigned` status (pipeline's own raw total) | 652 (16.6%) | 783 (**19.9%**) |

The honest framing: raw "% assigned" only moved from 16.6% to 19.9%
across the *whole* 3,926-observation sample (most of which is
background/far-side/other traffic neither geometry should ever cover —
see §10) — but restricted to the 1,470 observations that are actually
this camera's own near-side through-traffic, **genuine correct-lane
coverage more than doubled** (21.1% → 49.5%), while **wrong-lane leakage
was eliminated** (46.3% → 0.0%). This is the intended trade and matches
the task's explicit instruction not to chase raw percentage.

## 9. Lane purity

(Restating §8's per-physical-lane rows in the requested form.)

- Physical lane 1 observations assigned to `L1`: **47.9%** (414/865)
- Physical lane 1 observations assigned to `L2`: **0.0%** (0/865)
- Physical lane 2 observations assigned to `L2`: **51.9%** (314/605)
- Physical lane 2 observations assigned to `L1`: **0.0%** (0/605)

## 10. Cross-lane leakage

- **Old** (one-directional, since only `L1` existed): physical lane 2 →
  `L1` = **46.3%** (280/605).
- **New** (both directions checked): physical lane 1 → `L2` = **0.0%**;
  physical lane 2 → `L1` = **0.0%**.
- The deliberate gap between the two polygons (§4, Step 3) is **29px at
  y=210, 103px at y=330, 159px at y=460** after margins — confirmed
  never crossed by either lane's real trajectory data in this sample.

## 11. Exclusion integrity

Checked directly against every one of the 102 tracks' real ground
points, using the actual `assign_lane` function with the new regions:

| Excluded category | n tracks | n obs | → `L1` | → `L2` | → unknown |
|---|---|---|---|---|---|
| Far-side carriageway (S-curve) | 17 | 812 | 0 | 0 | **812 (100%)** |
| Static/background (parked/parking-lot/distant) | 46 | 1,251 | 6 | 0 | 1,245 (99.5%) |
| Other/ambiguous/partial | 17 | 393 | 40 | 9 | 344 (87.5%) |

The far-side carriageway shows **zero** leakage into either new polygon
— fully preserved.

The small `L1`/`L2` counts in the other two rows were individually
traced back to specific tracks, not a systemic exclusion failure:

- **3 "static/background" tracks** (`627`, `468`, `175` — 6 observations
  total) are very short (1–3 observations each), which is exactly why my
  span-based static/background heuristic (bounding-box movement < 30px)
  miscategorized them — on inspection, their actual (x, y) coordinates
  sit squarely inside physical lane 1's real path (e.g., `627`'s
  `(702, 289)` matches lane-1's expected position at that row almost
  exactly). These are real, briefly-glimpsed lane-1 vehicles, not
  parked/background objects; the new geometry classifying them as `L1`
  is **correct**, not a leak.
- **6 "other" tracks** (`627` already counted above; `3`, `676`, `556`,
  `319`, `684` → `L1`; `10` → `L2`, discussed in §7) are, on the same
  inspection, real near-side through-traffic that simply didn't get
  folded into the "core" 22-track groups in the original analysis
  (mostly because they are shorter/partial tracks, or — for `10` —
  because it appears to execute a real lane change). None of them are
  turn-pocket, far-side, or background traffic.

**No genuine exclusion violation was found** — every excluded category
(far-side, turn pocket, background/parking) that should stay excluded,
does.

## 12. Whether lane assignments remain stable through the visible corridor

Checked for all 22 relevant tracks by inspecting each one's full
per-frame lane-assignment sequence (not just aggregate percentages):
every track shows either 1 or 2 total status transitions across its
entire lifetime (enter-once/exit-once, or already-inside/exit-once), and
**every track's set of assigned lanes contains exactly one lane_id, never
both**. There is no frame-to-frame flicker (e.g., `L1, unknown, L1,
unknown, L1...`) anywhere in the sample. This is the expected, correct
behavior of a per-detection (never per-track-cached) lane assignment
applied to a geometrically accurate polygon.

## 13. Is a polygonal image-space model still appropriate?

**Yes — for this camera's short, curving corridor, a hand-measured,
multi-row polygonal image-space model remains appropriate, provided
enough rows are sampled to track the real curvature.** The previous
analysis (`ANVESH_C006_LANE_COVERAGE_ANALYSIS.md` §7) had already
flagged that a straight-edged, 2-row quadrilateral degrades over a long
or curving span; this iteration quantifies that directly (§4, Step 4:
~29px deviation with 2 rows vs. under ~10px with 3), confirming the
diagnosis and validating the fix within the same modeling family — no
world-space/calibrated redesign was necessary to correct the specific
defect found. A longer-term move to a calibrated world-space lane
definition (this camera already has a CityFlow homography) remains a
reasonable future direction if this camera's geometry needs to extend
much further than the current y=210–460 window, but is explicitly out
of scope for this iteration.

## 14. Limitations

- Both polygons are still a piecewise-linear (2-segment) approximation
  of a continuously curving lane boundary, accurate at the three sampled
  rows and linearly interpolated between them — not an exact curve fit.
- Derived from one 30-second sample (300 frames); a longer or
  differently-timed sample could reveal slightly different real
  boundary extents, though the 8px margin is intended to absorb
  reasonable sample-to-sample variation.
- Neither lane extends to the stop line/crosswalk (both stop at y=460,
  matching the original design's stop-bar caution) — a vehicle queued
  at the intersection beyond that point is honestly reported
  `unknown_outside_regions`, not a coverage gap to "fix" by extending
  further into the apron/crosswalk area.
- `track 180`'s exclusion from the fitting set, and its resulting 0%
  assignment, rests on a single track's trajectory (it exits the frame
  early rather than reaching the bottom) — a reasonable, evidence-backed
  inference, but not independently confirmed by a second turning
  vehicle in this sample.

## 15. Is `L2` provisional or defensible?

**Provisional — carried forward unchanged from the previous analysis's
own conclusion, and the new geometry's own validation does not resolve
the open question.** `L2` is:
- **Defensible as real, physical pavement**: 10 independent tracks, 605
  real observations, a tight and consistent trajectory band, visually
  confirmed against a real analyzed video frame showing simultaneous
  queued traffic in both lanes.
- **Defensible as internally correct within this iteration**: 0%
  cross-lane leakage in either direction, 0% leakage from any excluded
  category, stable (non-flickering) assignment.
- **Not defensible as an unconditional "general lane" the way `L1`
  is**: its pavement carries an exclusive-turn marking whose legal/
  physical effect just beyond this camera's field of view cannot be
  observed from this data. The config file marks it PROVISIONAL
  explicitly, with a caveat any downstream consumer should read before
  treating its speed/lane data with `L1`-equivalent confidence.

## 16. Files changed / intentionally untouched

**Changed:**
- `configs/lanes/s02_c006.toml` (the only intended change — `L1`
  corrected, `L2` added as provisional).
- `ANVESH_C006_LANE_GEOMETRY_ITERATION.md` (this document, new).
- `docs/demo_reference/s02_c006_lane_geometry_iteration/` (new folder, 6
  validation images).
- `ui/data/anvesh_s02_c006_data.json` (regenerated by the required
  post-correction 30s smoke test, §17 below — this file was already
  untracked, pre-existing Commit-9A output; regenerating it is the
  explicit smoke-test deliverable this iteration asked for, not an
  incidental change).

**Intentionally untouched (verified via `git status`):**
- All 7 pre-existing unrelated dirty files (`anvesh/evidence/
  evidence_builder.py`, `anvesh/perception/traffic_state.py`,
  `configs/experiment/m6_real_camera_thresholds.toml`, and their 4 test
  counterparts) — unchanged, not staged, not reverted.
- Every `anvesh/` source module, `anvesh/rendering.py`,
  `anvesh/perception/speed_estimation.py`,
  `anvesh/hypotheses/candidate_vehicle.py`, `anvesh/pipeline.py`,
  `anvesh/perception/lane_assignment.py`, `anvesh/world/lane_geometry.py`
  — no generic lane-assignment/rendering/speed/candidate/pipeline code
  was modified.
- `configs/lanes/s02_c009.toml` and every other camera's lane config.
- ANVESH-V0 and the CityFlow dataset.
- All other untracked pre-existing artifacts (`demo.mp4`, `traffic.mp4`,
  the other `ANVESH_*.md` documents, `docs/ANVESH_Demo_Explanation_
  Guide.pdf`, `docs/demo_reference/` subfolders from prior iterations,
  `scripts/analyze_video.py` and the other pre-existing untracked
  scripts, `stitch/`).

**⚠️ One important, unresolved finding — flagged, not fixed:**
running the full test suite after the config change surfaces **5 failing
tests** (`tests/perception/test_lane_assignment.py::
test_vehicle_clearly_inside_c006_l1_is_assigned`,
`test_lane_assignment_survives_schema_conversion_path`,
`test_integration_detection_through_track_manager_to_lane_history`;
`tests/world/test_lane_config.py::test_c006_expected_lane_ids`,
`test_loaded_regions_are_usable_with_assign_lane`). Every one of these
hardcodes an assumption from the OLD, now-corrected `L1` geometry (a
literal "point known to be inside the old L1 polygon" fixture, or an
assertion that `s02_c006.toml` defines exactly `["L1"]`) — this is the
direct, expected consequence of intentionally changing that polygon,
not a bug introduced elsewhere (confirmed: `python -m pytest -q` on the
unmodified tree before this change showed 590/590 passing; after only
the config edit, 585 passed / 5 failed, all five in exactly the two
files that hardcode c006's old geometry). **These test files were not
modified**, per this task's explicit scope restriction ("Only modify:
`configs/lanes/s02_c006.toml` and create/update documentation/artifacts
... Do NOT modify source code"). They will need a follow-up update
(new interior-point fixtures for the corrected `L1`, and an updated
expected lane-id list of `["L1", "L2"]`) before this change is safe to
commit.

## 17. Post-correction smoke test

`python scripts/run_s02_c006_showcase.py --sample-seconds 30` re-run
against the real dataset video, using the corrected config, after the
geometry was finalized (per item 12's explicit ordering requirement):

- Frames processed: 300 (same 30.0s / 10fps sample as before —
  deterministic given the same video/model).
- Tracks: 102 (unchanged).
- `lane_status_counts`: `unknown_outside_regions=3143`, `assigned=783`
  (up from the pre-correction `unknown=3274`, `assigned=652`).
- `speed_confidence_counts`: `approximate=74`, `unavailable=28`
  (unchanged — lane geometry does not affect speed estimation, as
  expected; speed math was not touched).
- `candidate_status_counts`: `no_vehicle_for_hypothesis=2`,
  `not_ranked=1` (unchanged — lane geometry does not feed
  evidence/ranking today, as expected; evidence/candidate logic was not
  touched).
- Errors/warnings: none.
- Wrote `ui/data/anvesh_s02_c006_data.json` (2,225,135 bytes).

Cross-validated independently two ways (direct `assign_lane` calls
against the pre-existing sample, and this fresh full pipeline re-run):
both agree exactly on every number in §7–§11.

## Final recommendation

**B) Corrected `L1` + `L2`.**

- `L1`'s correction is unconditionally defensible: it fixes a real
  geometric defect (not just a coverage gap), is validated against five
  different real frames spanning the full video, shows zero leakage in
  either direction, and stable, non-flickering assignment.
- `L2` is included as an explicitly-documented **provisional** lane
  (§15) — real, physically distinct, internally validated pavement, but
  carrying an unresolved legal/turn-restriction caveat this data cannot
  settle. It is proposed, not silently promoted to `L1`-equivalent
  status, per the acceptance criteria's explicit allowance for this
  outcome.
- **Before this can be committed**, the 5 tests identified in §16 need
  to be updated to reflect the corrected geometry (out of this
  iteration's scope, since it is restricted to config + documentation
  only) — this is the one remaining blocker, not a limitation of the
  geometry itself.
