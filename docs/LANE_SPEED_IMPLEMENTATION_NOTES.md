# Lane Assignment & Approximate Speed — Implementation Notes

Running implementation log for the plan in `ANVESH_LANE_SPEED_FORENSIC_AUDIT.md`
§11. Each commit gets one dated section here: what changed, why, and what it
deliberately does not yet do. This is a companion to that audit, not a
replacement for it — see the audit for full background/rationale on the
overall lane/speed/candidate-vehicle design.

---

## Commit 1 — Vehicle ground/contact-point accessor (`RawDetection.ground_point`)

**What changed:** Added `RawDetection.ground_point` (`anvesh/perception/
types.py`) — a read-only property returning the bounding box's bottom-center
point: `((x1 + x2) / 2.0, y2)`, equivalently `bbox.x + bbox.width / 2`,
`bbox.y + bbox.height` in `(x, y, width, height)` terms. Purely additive: no
existing method, property, caller, or test was modified.

### Why bottom-center is used for road/lane reasoning

`RawDetection.centroid` (the box center) sits above the road plane for any
vehicle taller than it is wide in screen space — a bus or truck's box center
can be well above the pavement, and how far above scales with the vehicle's
apparent height, which itself varies with distance from the camera. This is
a documented, verified defect in both ANVESH-V0's lane/speed logic
(`modules/lane_manager.py`, `modules/speed_estimator.py`, both keyed to box
center) and ANVESH-V1's current trajectory storage (`TrackManager`/
`schema_conversion.py`, also box-center via `centroid`) — see the forensic
audit §1.3/§3.2. Bottom-center is the point on the box closest to where the
vehicle's wheels actually meet the road, which is the physically meaningful
point for anything that reasons about *where on the road* a vehicle is:
future lane-polygon tests (lane boundaries exist on the road plane) and any
homography/world-space projection (which assumes the projected point lies on
the road plane — see `anvesh.world.calibration.CalibrationProfile`).

### Why `centroid` remains available, unchanged

`centroid` is still what `TrackManager.update()` and `schema_conversion.
build_vehicle_track()` use today for trajectory accumulation and speed
estimation. This commit does not touch either of those call sites, and does
not change what track trajectories or speed estimates are built from —
that is a separate, not-yet-made decision, explicitly out of scope for this
commit (see the audit's Commit 1 scope: "no lane detection yet, no speed
changes yet"). Keeping `centroid` untouched means every existing test,
pipeline stage, and exported artifact behaves identically to before this
change.

### Explicit limitations

- **Still 2D, still image-space.** `ground_point` is derived purely from the
  2D bounding box; it is not a measured 3D contact point. Occlusion (the
  vehicle's true bottom edge hidden behind another object) biases this point
  upward, the same way it would bias any other box-derived proxy — this is
  not solved by this commit and is not claimed to be.
- **Frame-edge boxes.** For a box that touches the bottom edge of the frame,
  `ground_point` is the frame boundary, not necessarily the vehicle's true
  contact point (the vehicle may extend further out of frame).
- **No orientation awareness.** The property does not know vehicle
  orientation; for a vehicle photographed at a steep side angle, the
  bottom-center of its box is not necessarily under its true centerline.
- **Does nothing beyond computing a point.** This commit performs no lane
  assignment, no calibration, no coordinate transform, and does not change
  speed estimation or the candidate-ranking/dashboard pipeline in any way —
  those remain future commits in the audit's plan (§11, Commits 2–10).
- **Degenerate (zero-area) boxes.** `RawDetection.__post_init__` currently
  accepts `x1 == x2` and/or `y1 == y2` (it only rejects `x2 < x1` / `y2 <
  y1`). `ground_point` handles this case without error (it just returns a
  single point), but a zero-area box is itself already a pre-existing
  modeling edge case in `RawDetection`, not something this commit
  introduces or resolves.

### Tests added

`tests/perception/test_types.py` (new file):
- normal box → correct bottom-center;
- a tall box, to explicitly demonstrate `ground_point` differs from
  `centroid` (the bias this property exists to avoid);
- fractional/sub-pixel coordinates;
- a degenerate zero-area box (valid per existing `RawDetection` convention);
- confirmation that an invalid box (`x2 < x1` or `y2 < y1`) is still
  rejected at construction time, per existing convention, before
  `ground_point` is ever reachable;
- a regression check that `centroid` itself is unchanged.

---

## Commit 2 — Deterministic lane geometry primitives (`anvesh/world/lane_geometry.py`)

**What changed:** Added a new, standalone module: `LaneRegion` (a polygonal
lane region, in image/pixel coordinates), `LaneAssignmentStatus` (an explicit
five-state enum), `LaneAssignmentResult`, and `assign_lane(point, regions,
boundary_tolerance=...) -> LaneAssignmentResult`. Nothing outside this module
was touched — no vehicle, track, or detection is assigned a lane yet, no
c006/c009 coordinates were added, and nothing was wired into
`schema_conversion.py`, the dashboard, speed estimation, candidate ranking,
ANVESH-V0, or the CityFlow dataset (per this commit's explicit scope).

### Why polygonal regions were chosen

A lane is, physically, an area of pavement — an arbitrary-vertex-count
polygon is the direct representation of "this area of the image is lane N,"
and is exactly what V0's `modules/lane_manager.py` also reached for (two
hand-typed quadrilaterals), confirming polygons are the natural fit for this
problem even though everything about V0's *implementation* of that idea was
rejected (see the audit SS1.6): hardcoded to exactly two lanes, fitted by eye
to one frame, tested with box-center via first-match-wins linear scan,
returning `0` identically for "off-road," "unmapped lane," and "we don't
know." This module keeps the representation (a polygon per lane) but rebuilds
the API around explicit lane IDs, arbitrary vertex counts (not just
quadrilaterals), and an all-regions-tested, order-independent membership
test — see "how ambiguity is represented" below.

### Why image-space geometry is appropriate initially

Every camera the showcase corridor uses (c006, c009) is fixed and unmoving,
with stable, human-visible painted lane markings — image-space polygons
drawn against that fixed frame are directly authorable and auditable by a
human looking at one representative frame, need no calibration, and are
available even for an `UNCALIBRATED` camera (`anvesh.world.calibration`).
The audit (SS8.1) also identifies a world-space alternative (rectangles in a
calibrated camera's metric plane) as viable for later; this commit only
builds the geometry primitive itself, not either camera's config, so that
choice is deliberately still open — `LaneRegion` takes plain `(x, y)` pairs
and has no opinion on which space they came from.

### How ambiguity is represented

`assign_lane` tests a point against *every* supplied region independently
(never V0's first-match-wins scan) and collects the full, sorted set of
matching lane IDs:

- zero matches → `UNKNOWN_OUTSIDE_REGIONS`;
- exactly one match → `ASSIGNED`, with `lane_id` set;
- two or more matches → `AMBIGUOUS` (covers both overlapping regions and a
  point on/near a boundary shared by two adjacent, non-overlapping lanes) —
  `lane_id` stays `None`, and the sorted `candidate_lane_ids` tuple lists
  every match;
- no regions supplied at all → `UNKNOWN_NO_GEOMETRY`;
- the supplied region collection itself is inconsistent (duplicate
  `lane_id`s) → `INVALID_GEOMETRY`, since a duplicate ID structurally
  violates "lane IDs must be explicit and stable" in a way no single
  region's own construction can catch.

A single-polygon structural defect (fewer than 3 vertices, or
zero-area/collinear vertices) is instead rejected at `LaneRegion`
construction time with `ValueError` — the same fail-fast convention
`anvesh.world.calibration.fit_planar_homography` already uses for a
degenerate/collinear reference-point fit, so a malformed polygon can never
reach `assign_lane` in the first place.

Boundary membership itself (needed for the "on/near a shared boundary"
case) is computed as an actual perpendicular point-to-segment distance
(projection parameter clamped to `[0, 1]`, not a raw cross/dot-product
comparison), tested against one `boundary_tolerance` (default `1e-6`
coordinate units — effectively floating-point equality, overridable by the
caller for looser-precision geometry) — this is what makes the result
independent of the order `regions` are passed in: reversing a two-lane
tuple's order produces a byte-for-byte identical `LaneAssignmentResult`,
which is asserted directly in
`test_boundary_ambiguity_is_independent_of_region_order`, not just implied
by a repeated-call determinism check.

### Why this is not yet a claim of automatic lane detection

`assign_lane` performs zero perception, segmentation, or learning — it is a
pure geometry function over caller-supplied polygons and a caller-supplied
point. No lane region for any real camera exists yet (no config file, no
c006/c009 coordinates); no vehicle detection or track is tested against
this module anywhere in the codebase yet; and the module imports nothing
from `perception` or `world.calibration`, so it makes no claim about where a
vehicle's ground-contact point comes from or which coordinate space it is
in. Those are Commit 3+ concerns per the audit's plan (SS11).

### Tests added

`tests/world/test_lane_geometry.py` (new file, 13 tests): a point clearly
inside a lane; a point clearly outside all lanes; a point on a shared lane
boundary (`AMBIGUOUS`); overlapping lane regions (`AMBIGUOUS`); an invalid
polygon (too few vertices; degenerate/collinear) rejected at construction;
duplicate lane IDs (`INVALID_GEOMETRY`) at query time; no regions at all
(`UNKNOWN_NO_GEOMETRY`); multiple lanes with deterministic, stable IDs;
fractional/sub-pixel coordinates; repeated evaluation gives an identical
result; and — the determinism case a same-order repeat can't catch —
reversed region order gives a byte-for-byte identical result, for both the
`AMBIGUOUS` boundary case and a plain `ASSIGNED` case.

---

## Commit 3 — c006/c009 lane geometry configs (`configs/lanes/s02_c006.toml`, `configs/lanes/s02_c009.toml`)

**What changed:** Added the first real, camera-specific lane configuration
for the showcase corridor (`E:\traffic dataset\validation\S02\c006\vdo.avi`
upstream, `...\c009\vdo.avi` downstream), plus a small loader
(`anvesh/world/lane_config.py`, `load_lane_config` / `LaneCameraConfig`)
that turns a TOML file into `LaneRegion` objects from Commit 2. This is
configuration + loading only: nothing calls `load_lane_config` from
`anvesh/perception/`, `anvesh/pipeline.py`, `scripts/export_ui_data.py`, or
the dashboard. No vehicle, track, or detection is assigned a lane anywhere
in the codebase after this commit.

### Method

Vertices were not typed in by eye against a raw screenshot (that is exactly
V0's rejected method — audit §1.1/§1.6). Each candidate boundary line was
picked by overlaying a 50px pixel-coordinate grid on a representative,
lightly-trafficked frame (`docs/demo_reference/.../c006_75pct_f1581.jpg`,
`.../c009_25pct_f527.jpg` — chosen specifically because queued traffic
obscures lane paint in the busier sampled frames), reading candidate
divider/edge positions off the grid, then rendering the candidate polygon
back onto the frame and a second frame with real traffic to confirm it (a)
stays on drivable pavement and (b) matches the actual path a real vehicle
column drives through the corridor. Several early candidates failed this
check and were corrected — most notably an initial c009 boundary that
turned out to trace the left-turn pocket instead of the through lane, and
a c006 attempt that extended too far toward the stop line and clipped both
flanking turn-arrow markings. This back-and-forth is the reason the method
is documented in this much detail: pixel-grid tracing on a curved,
perspective road is easy to get subtly wrong in a way that still *looks*
plausible without a rendered-overlay check.

### Why only one lane per camera

Direct pixel inspection (this commit) found a different lane layout than
the earlier, single-pass visual-inspection documents estimated:

- **c006**: 3 lanes in the near-side carriageway, but only the *center*
  lane carries no turn-restriction paint. Both flanking lanes are marked
  "ONLY" (left-turn pocket, right-turn pocket) with paint starting well up
  the visible corridor, not confined to a short stub near the stop line.
- **c009**: only 2 lanes in the near-side carriageway (one less than
  `ANVESH_S02_C007_C009_VISUAL_INSPECTION.md`'s preliminary "2 through + 1
  turn pocket" estimate) — a single dashed/solid divider was found, not
  two, confirmed against the path of a real sedan-then-SUV-then-pickup
  vehicle column. Only the one non-pocket lane carries no turn-restriction
  paint.

Given both cameras have exactly one lane whose entire visible length is
free of turn-restriction paint, both configs define exactly one lane,
`L1`. This is a smaller lane count than the audit's preliminary "2 through
+ 1 turn pocket" estimate for either camera — the earlier estimate was a
quick visual scan without a pixel grid or a drawn-overlay check, and is
superseded for lane geometry purposes by this commit's direct, verified
measurement. Nothing about this changes the audit's downstream design
(§8.1's `LANE_UNKNOWN` / multi-lane branching logic is unaffected by how
many lanes a given camera happens to have).

### Turn pockets: excluded, not truncated

Every lane marked "ONLY" with a turn arrow (c006's left and right pockets,
c009's left pocket) is excluded outright from both configs, per design
requirement G/§7's instruction to make an explicit choice here. Two
reasons, not one: (1) a vehicle in a turn pocket is, by definition, about
to leave the through corridor, which conflicts with the "stable lane a
vehicle travels along" assumption lane assignment and future speed
estimation both need; (2) an earlier attempt to keep the *unmarked upper
portion* of each turn pocket (truncating the polygon before the "ONLY"
paint begins, the same way each lane's polygon already stops short of the
stop-bar/crosswalk) was abandoned once direct pixel inspection showed the
turn-restriction paint starts well up each pocket's visible length, not
in a short stub near the intersection — there was no honestly-defensible
"clearly general-purpose" sub-region left to carve out. Excluding the
whole marked lane is the conservative choice per requirement 8, over
pretending a partial polygon is still safely general-purpose.

### Regions deliberately excluded (both cameras)

- All turn-only pockets (see above).
- The far-side (opposing-direction) carriageway, beyond the grass median —
  visibly present at both cameras but smaller/more distant (matching the
  audit's own §3.6/§6.4 note about weaker contact-point precision at
  range), and for c006 specifically its lane boundary follows a
  pronounced S-curve around the median that a straight-edged quadrilateral
  cannot represent without either clipping real pavement or bleeding into
  the median at some point along the curve.
- The intersection apron/crosswalk area beyond each config's near
  (bottom) boundary (c006: y=440; c009: y=600) — `L1`'s polygon stops
  short of the stop line by construction, so a queued vehicle right at the
  intersection tests as `UNKNOWN_OUTSIDE_REGIONS`, never a false match.

### Coordinate convention

Image/pixel space for both files: x right from 0, y down from 0, origin at
the source frame's top-left corner — matching
`anvesh.world.lane_geometry.LaneRegion.vertices` and
`RawDetection.bbox_xyxy` (`anvesh/perception/types.py`), not calibrated
world-space metres. Both cameras are 1920×1080 (confirmed via OpenCV
metadata in `ANVESH_S02_CAMERA_PAIR_ANALYSIS.md`).

### Known limitations

- Hand-measured visual approximation from one sampled frame per camera,
  not a calibrated or surveyed measurement — expect several pixels of
  error at any point along an edge, worst between the two sampled boundary
  rows.
- Valid only for each camera's current fixed framing; any change in
  camera position, angle, zoom, or resolution silently invalidates these
  coordinates (same caveat the audit raised against V0's hand-typed
  polygons, §1.5) — neither the config nor its loader detects this.
- Only one lane per camera is defined (see above) — deliberately
  conservative, not a claim that these roads have only one usable lane.

### Tests added

`tests/world/test_lane_config.py` (new file, 17 tests): both configs load
successfully; expected lane IDs (`["L1"]` for each camera); loaded regions
are valid `LaneRegion` instances (construction-time polygon validation from
Commit 2 already enforces this — a successful load is itself proof); every
vertex is within the documented image bounds; documented resolution is
1920×1080 for both; repeated loading is deterministic (same camera_id,
resolution, lane IDs, and vertices across five reloads); the conventional
`configs/lanes/<camera_id>.toml` path helper; a missing file raises
`FileNotFoundError`; a config with a duplicate `lane_id` raises
`ValueError`; a config with an invalid polygon raises `ValueError`
(delegated to `LaneRegion`); and one composition test confirming a loaded
config's regions plug directly into Commit 2's `assign_lane` (an interior
point resolves `ASSIGNED`/`L1`, a far-outside point resolves
`UNKNOWN_OUTSIDE_REGIONS`) — still no pipeline wiring, just proof the two
commits work together.

---

## Commit 4 — Wire vehicle lane assignment (`anvesh/perception/lane_assignment.py`)

**What changed:** Added a new module, `anvesh/perception/lane_assignment.py`,
that composes Commit 1 (`RawDetection.ground_point`), Commit 2
(`world.lane_geometry.assign_lane`), and Commit 3 (`world.lane_config.
load_lane_config`) against real perception/tracking objects:
`assign_detection_lane(detection, regions)` for a single observation, and
`build_track_lane_history(camera_id, state, regions)` for a whole track's
per-observation history (`TrackLaneHistory`, a tuple of
`LaneAssignmentSample`). `anvesh/perception/track_manager.py` gained one
additive field, `TrackState.ground_point_trajectory` — the same per-frame
samples as the existing `trajectory`, but from each detection's
`ground_point` instead of `centroid` — populated alongside (never
replacing) the existing trajectory in `TrackManager.update()`. Nothing
else changed: `schema_conversion.py`, `storage/schemas.py`,
`perception/pipeline.py`, `anvesh/pipeline.py`, and every export/dashboard
script are untouched by this commit.

### How lane assignment is computed

Per observation, not per track: `assign_detection_lane` takes one
`RawDetection` and a camera's lane `regions` and returns exactly what
`assign_lane(detection.ground_point, regions)` returns — a pure, stateless
call with no caching. `build_track_lane_history` applies this
independently to every sample in a track's `ground_point_trajectory`,
producing one `LaneAssignmentSample` (timestamp + `LaneAssignmentResult`)
per frame the track was observed in. This directly answers design question
6 (audit-adjacent, this commit's own instructions): lane assignment is
calculated **per raw detection and propagated into the track's history**,
never computed once (e.g. from the track's first or "current" observation)
and carried forward — a vehicle that changes lanes mid-track shows that
transition across `samples`, rather than presenting a stale first-seen
lane as if it still applied. No lane-*change* detection (comparing
consecutive samples to flag a transition) is implemented here — this
commit only makes the raw per-frame history available; a future commit can
compare consecutive `samples` to detect a change.

### Why bottom-center (ground_point), not centroid

Unchanged from the Commit 1 rationale, now actually exercised for the
first time: `centroid` (box center) sits above the road plane for any
vehicle taller than it is wide in screen space, drifting toward the
camera relative to where the vehicle's wheels actually meet the road —
exactly the wrong point to test against lane paint that exists *on* the
road surface. `assign_detection_lane` uses `detection.ground_point`
exclusively; `tests/perception/test_lane_assignment.py::
test_ground_point_not_centroid_determines_assignment` constructs a tall
box whose `ground_point` falls inside a lane region while its `centroid`
falls outside it, and asserts the two source points give different
`LaneAssignmentResult`s — a regression guard against ever swapping the
attribute this module reads.

### Meaning of UNKNOWN / AMBIGUOUS (unchanged from Commit 2, now reachable end-to-end)

`assign_detection_lane`/`build_track_lane_history` return exactly the
`LaneAssignmentStatus` Commit 2 defined — `ASSIGNED`, `AMBIGUOUS`,
`UNKNOWN_NO_GEOMETRY`, `UNKNOWN_OUTSIDE_REGIONS`, `INVALID_GEOMETRY` — with
no translation, defaulting, or collapsing step in between. Nothing in this
commit ever substitutes a guessed lane ID for a non-`ASSIGNED` status: a
vehicle outside every configured region, on/near a shared boundary, in a
camera with no lane config, or facing an internally inconsistent region
set all surface as their own explicit status, exactly as Commit 2
guarantees, all the way out to `LaneAssignmentSample.result`.

### Where this attaches (and where it deliberately doesn't)

Lane assignment is a new, non-schema companion representation
(`TrackLaneHistory`), the same pattern `EvidenceItem`,
`StoppedVehicleClassification`, and `CorridorStateAssembly` already use —
per the audit §9's explicit recommendation not to fold a derived
diagnostic into a frozen Part 5 schema. `storage.schemas.VehicleTrack` is
untouched: it carries no `lane_id` or `lane_assignment` field, and
`schema_conversion.build_vehicle_track`'s output is asserted identical
whether or not `TrackState.ground_point_trajectory` is populated
(`test_existing_schema_conversion_behavior_is_unchanged_by_ground_point_
trajectory`). `build_track_lane_history` mirrors `build_vehicle_track`'s
`f"{camera_id}:{state.track_id}"` track-ID convention so the two outputs
describing the same track are trivially joinable by ID later, without
either one depending on the other.

This commit does not wire either function into `perception/pipeline.py`'s
`run_m1_pipeline` loop or the top-level `anvesh/pipeline.py` orchestrator
— both call sites are left for a later commit (and, independently, both
files already carry unrelated in-progress changes in the working tree
that this commit does not touch). `TrackManager.update()`'s new
`ground_point_trajectory` bookkeeping is unconditional (always populated,
no lane-config dependency), keeping `TrackManager` itself lane-agnostic —
exactly the property the audit's own §3.6 discussion wanted preserved.

### Limitations of the current one-lane-per-camera showcase configuration

Carried over from Commit 3, worth restating now that lane assignment is
actually reachable: both `configs/lanes/s02_c006.toml` and `s02_c009.toml`
define exactly one lane (`L1`) each. This means, for the showcase corridor
as currently configured:

- Every `ASSIGNED` result for either camera will always name `L1` — there
  is no second lane to be `AMBIGUOUS` against in practice, so
  `AMBIGUOUS`/multi-lane behavior is only exercised by synthetic
  fixtures in tests, not by anything the real c006/c009 configs can
  currently produce.
- A vehicle in either flanking turn pocket, the far-side carriageway, or
  the intersection apron beyond each `L1` polygon's near boundary will
  resolve to `UNKNOWN_OUTSIDE_REGIONS` — an honest "not on our one mapped
  lane," not a claim that the vehicle is off-road.
- Ground-point-based assignment inherits every limitation `RawDetection.
  ground_point` already documents (2D image-space proxy only, biased by
  occlusion, no vehicle-orientation awareness) and every limitation the
  Commit 3 configs document (hand-measured single-frame approximation,
  invalidated silently by any camera reframing).

### Tests added

`tests/perception/test_lane_assignment.py` (new file, 8 tests): a vehicle
clearly inside c006 `L1` and c009 `L1` (using the real Commit 3 configs)
both resolve `ASSIGNED`; a vehicle outside all regions resolves
`UNKNOWN_OUTSIDE_REGIONS`; a ground point exactly on a shared boundary
between two synthetic lanes resolves `AMBIGUOUS`; a tall box whose
`ground_point` and `centroid` disagree confirms `ground_point` (not
`centroid`) drives the result; lane assignment composes with
`schema_conversion.build_vehicle_track` on the same `TrackState` without
altering the frozen `VehicleTrack` it produces (schema is asserted
unchanged, both structurally and value-for-value, with and without
`ground_point_trajectory` populated); and one full integration test
(requirement 14) driving a `RawDetection` through `TrackedDetection` →
`TrackManager.update()` → `build_track_lane_history`, against the real
c006 config, asserting the resulting `TrackLaneHistory` names `L1`.

`tests/perception/test_track_manager.py` (extended, +1 test): confirms
`ground_point_trajectory` accumulates the bottom-center point per frame,
independently of and alongside the unchanged centroid `trajectory` —
every pre-existing test in this file is unmodified and still passes,
demonstrating `TrackManager`'s existing tracking behavior (trajectory
accumulation, occlusion-state transitions, track IDs, finalization) is
unaffected by this commit.

---

## Commit 5 — Ingest CityFlow homography calibration (`anvesh/world/calibration.py`)

**What changed:** Calibration ingestion only — no speed estimation, no
lane-assignment change, no candidate-vehicle work, no dashboard change.
Added `parse_cityflow_calibration_text`, `from_cityflow_homography`, and
`load_cityflow_calibration` to `anvesh/world/calibration.py`, plus two new
explicit-state enums (`CalibrationProvenance`, `MetricValidationStatus`)
and two new `CalibrationProfile` fields (`provenance`, `metric_validation`,
both with backward-compatible defaults). `fit_planar_homography` and
`uncalibrated_profile` are unchanged in behavior — they now just tag
themselves explicitly (`MANUAL_SURVEY`/`VALIDATED` and `UNKNOWN`/
`UNVALIDATED` respectively) instead of leaving that implicit. Nothing
outside `anvesh/world/calibration.py` was touched.

### CityFlow calibration format

Confirmed directly from the real files (not assumed from documentation):

```
Homography matrix: a b c;d e f;g h i
Intrinsic parameter matrix: a b c;d e f;g h i    (optional)
Distortion coefficients: c0 c1 c2 c3             (optional)
Reprojection error: <float>
```

`E:\traffic dataset\validation\S02\c006\calibration.txt` and `...\c009\
calibration.txt` (the showcase cameras) each carry only the "Homography
matrix" and "Reprojection error" lines — no intrinsic matrix, no
distortion coefficients. Scanning the rest of the validation set found
exactly one camera, `S05\c035`, whose file carries the optional intrinsic
-matrix/distortion-coefficients lines, confirming the parser needs to
handle them even though neither showcase camera's file does.
`parse_cityflow_calibration_text` parses all four fields (two required,
two optional) from a plain string — no dependency on any specific file
path, so it composes with `load_cityflow_calibration` (real file) or a
fixture string (tests) identically.

### How it enters ANVESH

`load_cityflow_calibration(calibration_profile_id, camera_id, path)` reads
a `calibration.txt` file, parses it, and calls `from_cityflow_homography`,
which builds a `CALIBRATED` `CalibrationProfile` — the SAME type
`fit_planar_homography` produces, so `is_calibrated`, `image_to_world`,
and `world_to_image` all work identically regardless of which
constructor built the profile. This was a deliberate choice: CityFlow's
homography is not routed through `fit_planar_homography` itself (that
function fits its own homography from correspondence points; CityFlow
supplies neither points nor anything to fit — only an already-computed
matrix), and no reference points are fabricated to satisfy the pre-
existing `>=4 reference_points` invariant. That invariant
(`CalibrationProfile.__post_init__`) now only applies when
`provenance == CalibrationProvenance.MANUAL_SURVEY` — a CityFlow-supplied
profile honestly carries `reference_points=()`.

### What the homography does and does not establish

**Does establish:** a genuinely usable image↔"world" projective transform
— `from_cityflow_homography` preserves the 9 supplied values exactly (no
rescaling, no re-fitting; verified byte-for-byte against the real c006/
c009 files in `test_c006_homography_values_preserved_exactly`/
`test_c009_...`), validates only shape (3×3) and numeric type (finite),
computes the matrix inverse so `world_to_image` works, and stores
CityFlow's own reported reprojection error unmodified.

**Does NOT establish:** that the output space is metres, GPS coordinates,
or any other physical unit. CityFlow's own documentation does not state
this, and the forensic audit's own numerical check
(`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` §4.3) found the projected values
are plausible-but-unconfirmed as some local metric frame (values in the
low thousands, not GPS-decimal-degree scale; reprojection errors of
~7–10 are plausible as metres-scale residuals but not proven) — "weakly
supports... but is not proof." `from_cityflow_homography` therefore
always sets `units="unknown"`, overriding `CalibrationProfile`'s own
`"meters"` default (which is only correct for the `MANUAL_SURVEY` path,
where a human supplied real measured-metre correspondences). The
`reprojection_error_m` field's "_m" suffix is inherited naming from that
same MANUAL_SURVEY path — for a CityFlow-supplied profile it is NOT a
units claim; that residual's real units are exactly as unvalidated as
everything else about this homography.

### Provenance

`CalibrationProvenance` (new enum): `MANUAL_SURVEY` (site-survey
correspondences, `fit_planar_homography`), `CITYFLOW_SUPPLIED`
(precomputed external matrix, `from_cityflow_homography`), `UNKNOWN` (no
meaningful provenance — `uncalibrated_profile`). Every existing
`CalibrationProfile` constructor now sets this explicitly; the dataclass
field itself defaults to `MANUAL_SURVEY` only for backward compatibility
with direct construction (existing tests construct `CalibrationProfile`
by hand without passing it). `method` (the existing free-text field) is
unchanged and complementary: `provenance` is the coarse, programmatically
-branchable category; `method` stays the more detailed algorithm/version
tag (`"cityflow_precomputed_v1"` vs `"manual_planar_homography_v1"`).

### Why speed remains approximate until independently validated

`MetricValidationStatus` (new enum): `VALIDATED` / `UNVALIDATED` — a
SEPARATE question from `CalibrationStatus`/`is_calibrated` ("does a usable
transform exist") and from `provenance` ("who produced it"). It answers:
has the claimed output-unit interpretation been independently confirmed
against a known real-world reference distance (e.g. a measured lane
width)? `fit_planar_homography` sets `VALIDATED` — a real survey's
correspondence points already are measured-metre distances, so the metric
claim is validated by construction. `from_cityflow_homography` always
sets `UNVALIDATED`, unconditionally — there is no parameter to override
this at ingestion time, deliberately: requirement 6 is that a precomputed
homography existing must never, by itself, upgrade ANVESH's confidence to
"metric speed is calibrated." No independent validation step (checking a
known lane width or similar against this homography's own output) exists
anywhere in the codebase yet — that remains explicitly future work. A
later speed-estimation commit must read `metric_validation`, not just
`is_calibrated`, before presenting a speed as confirmed-metric rather than
approximate (audit §8.2's `is_approximate`/`SpeedConfidence` design).

### Tests added

`tests/world/test_calibration.py` (extended, +24 tests, all 11
pre-existing tests unmodified and still passing): parsing the real c006
and c009 calibration text (embedded as fixture strings, not read from the
16GB dataset — requirement 9) including exact byte-for-byte homography
preservation and reprojection-error parsing; parsing the one file in the
validation set that has the optional intrinsic-matrix/distortion
-coefficients lines; six malformed-input cases (missing homography line,
missing reprojection-error line, wrong row count, wrong column count, a
non-numeric homography value, a non-numeric reprojection error) each
raising `ValueError`; provenance/units/metric_validation correctly marked
for a CityFlow-supplied profile (and explicitly asserted `!= "meters"` /
`!= VALIDATED`, not just checked for a new value); the ingested
homography actually projects and round-trips via the existing
`image_to_world`/`world_to_image` methods; malformed-homography rejection
(non-3×3 shape, non-finite values, singular matrix); `load_cityflow_
calibration` reading a real file (via `tmp_path`, still no dependency on
the mounted dataset) and raising `FileNotFoundError` for a missing one; a
CityFlow-provenance profile with zero reference points constructing
without error; and three "existing workflows unchanged" tests —
`fit_planar_homography` is now explicitly `MANUAL_SURVEY`/`VALIDATED`,
`uncalibrated_profile` is explicitly `UNKNOWN`, and the original `>=4
reference points` invariant still rejects a `MANUAL_SURVEY` profile with
too few points.

---

## Commit 6 — Provenance-aware approximate speed estimation (`anvesh/perception/speed_estimation.py`)

**What changed:** Speed estimation only — no candidate-vehicle work, no
evidence/ranking change, no lane-change detection, no dashboard change.
`anvesh/perception/motion.py` gained a third, unit-agnostic computation —
`RobustSpeedStatus`, `RobustSpeedResult`, `compute_robust_speed` —
extending (not duplicating) its existing `_step_speeds`/`_mean_and_error`
machinery via a new shared `_consecutive_steps` helper.
`anvesh/perception/speed_estimation.py` (new) composes that with Commit 4's
`TrackState.ground_point_trajectory` and Commit 5's `CalibrationProfile`
(`provenance`/`metric_validation`) to produce a `VehicleSpeedEstimate` —
a km/h `Measurement` plus an explicit `SpeedConfidence`
(`VALIDATED`/`APPROXIMATE`/`UNAVAILABLE`). The existing
`ImageSpaceSpeedEstimate`/`WorldSpaceSpeedEstimate`/
`compute_image_space_speed`/`compute_world_space_speed`, and everything
in `schema_conversion.py` that already uses them, are completely
unchanged.

### Mathematical approach

`compute_robust_speed` computes `total_distance / total_elapsed_time`
across every valid consecutive trajectory step — each step's own
duration naturally weights its contribution, rather than either (a) a
naive first-vs-last-point measurement or (b) an unweighted arithmetic
mean of per-step instantaneous speeds (`compute_image_space_speed`'s
existing approach, left untouched). (b) matters because a single
near-duplicate-timestamp jitter step can produce an enormous
instantaneous speed that then counts as "one vote out of N" in a plain
mean; under time-weighting, that same step contributes a small increment
to *both* the distance and time sums (jitter is small-magnitude by
nature), so it cannot dominate the result —
`tests/perception/test_motion.py::test_robust_noisy_outlier_step_does_not_dominate_speed`
demonstrates this directly: one 1000-units/s instantaneous outlier among
10-units/s motion still resolves to ≈10, not the ≈340 a naive mean would
give. The reported `error` reuses the *existing* `_mean_and_error`
variance machinery over the same per-step speeds, so genuine noise still
shows up as a large uncertainty — honestly reported, not hidden by the
robust central estimate (requirement 10: value + error, never a falsely
precise single number).

### Temporal sampling

Timestamp-based throughout — `dt` is always a real elapsed-time
difference between two samples, never "number of detections between
them" (the V0 defect the audit's §2.3 identified and the reason
`TrackState`/`Frame` are timestamp-keyed in the first place).
`estimate_vehicle_speed`'s defaults (`DEFAULT_MIN_SAMPLES=3`,
`DEFAULT_MIN_ELAPSED_S=0.3s`, `DEFAULT_MAX_GAP_S=0.6s`) are derived, not
arbitrary: CityFlowV2's confirmed 10 fps frame rate for c006/c009
(`ANVESH_S02_CAMERA_PAIR_ANALYSIS.md`) gives a 0.1s frame interval, so
`min_elapsed_s` requires spanning at least 3 frame-intervals (not one or
two adjacent, highly-correlated frames — requirement 7's "robust
window", never just first-vs-last), and `max_gap_s` sits comfortably
above `TrackManager`'s own default `max_missed_frames=5` (0.5s at
10fps) — the longest a track can go while still `PARTIAL` rather than
finalized `LOST`, so a gap at or below that is ordinary occlusion, not a
discontinuity worth excluding. All three are plain parameters, freely
overridable for a different camera's own frame rate.

### Coordinate semantics

`estimate_vehicle_speed` projects `state.ground_point_trajectory`
(bottom-center, requirement 1 — **never** `state.trajectory`, which
remains centroid-based and untouched) through
`calibration_profile.image_to_world`, then runs `compute_robust_speed` on
the result, then multiplies by `MPS_TO_KMH = 3.6` — **always** under the
assumption that the projected distance unit is metres. That assumption
is applied identically regardless of provenance, because both
provenances a `CalibrationProfile` can currently carry give it *some*
basis: `MANUAL_SURVEY`'s correspondence points are real measured metres;
`CITYFLOW_SUPPLIED`'s scale is merely plausible-as-metres per the
forensic audit's own numerical check (§4.3), never proven. If a future
provenance existed with no such basis at all, this module would need a
third branch — none exists yet, so none was speculatively added.

### Approximation / provenance / failure states

Three independent axes, none collapsed into the others:

1. **`RobustSpeedStatus`** (motion.py) — did the *math* succeed?
   `OK`, or one of `INSUFFICIENT_OBSERVATIONS` /
   `INSUFFICIENT_ELAPSED_TIME` / `INVALID_TRAJECTORY` /
   `LARGE_TEMPORAL_GAP` (all four producible by `compute_robust_speed`
   itself) / `CALIBRATION_UNAVAILABLE` / `INVALID_CALIBRATION` (reserved
   for `estimate_vehicle_speed`'s own composition logic — `compute_
   robust_speed` never returns these two, since it has no calibration
   knowledge at all, matching `motion.py`'s existing strict separation
   of concerns).
2. **`CalibrationProvenance`** (Commit 5) — who produced the
   homography: `MANUAL_SURVEY` / `CITYFLOW_SUPPLIED` / `UNKNOWN`.
3. **`SpeedConfidence`** (this commit) — is the km/h number's physical
   -unit claim trustworthy? `VALIDATED` only when the math reached `OK`
   *and* `calibration_profile.metric_validation ==
   MetricValidationStatus.VALIDATED` (today: `MANUAL_SURVEY` only).
   `APPROXIMATE` when the math reached `OK` but `metric_validation ==
   UNVALIDATED` (today: `CITYFLOW_SUPPLIED` always lands here — there is
   no parameter anywhere in `estimate_vehicle_speed` that can override
   this). `UNAVAILABLE` for every non-`OK` status, with `measurement`
   forced to `None` by `VehicleSpeedEstimate.__post_init__` — never a
   fabricated placeholder number.

Requirement 9 (no silent zero) holds throughout: every degenerate case
returns its own named status; a genuinely stationary vehicle still
returns `OK`/`speed=0.0` (a real, distinct measurement, not an error).

### Limitations

- Ground-point-based, so it inherits every limitation `RawDetection.
  ground_point` already documents (2D proxy, occlusion bias, no vehicle
  -orientation awareness).
- The `MPS_TO_KMH` conversion for a `CITYFLOW_SUPPLIED` profile is a
  provisional assumption, not a proven fact — `SpeedConfidence.
  APPROXIMATE` exists specifically to carry that caveat forward; nothing
  downstream of `VehicleSpeedEstimate` should present it as a confirmed
  metric value bare, without checking `confidence` first.
- No independent validation step (checking a known lane width or
  reference distance against a `CITYFLOW_SUPPLIED` homography's own
  output) exists anywhere in the codebase yet — `metric_validation` can
  only ever be `UNVALIDATED` for such a profile until that is built.
- `max_gap_s`-excluded steps are simply dropped from the totals, not
  separately reported in `VehicleSpeedEstimate` — a caller cannot
  currently tell "OK, computed from a full clean window" apart from "OK,
  computed after silently dropping one gap step" without inspecting
  `RobustSpeedResult.sample_count` itself (not currently threaded
  through to `VehicleSpeedEstimate`). Acceptable for this commit's scope;
  worth revisiting if gap frequency turns out to matter for the
  showcase.
- `estimate_vehicle_speed` is not wired into `perception/pipeline.py` or
  `anvesh/pipeline.py` — same deliberate stopping point Commit 4 left for
  lane assignment, for the same reason (both orchestration files already
  carry unrelated in-progress changes in the working tree that this
  commit does not touch).

### Why this is NOT yet validated physical speed

A `CITYFLOW_SUPPLIED` `CalibrationProfile` can never reach
`SpeedConfidence.VALIDATED` through this module, by construction — there
is no code path, parameter, or default that flips
`metric_validation` from `UNVALIDATED` to `VALIDATED`; that would require
an actual independent check (e.g. a known real-world reference distance
measured against this homography's own output), which does not exist
anywhere in the codebase. Every km/h number this module produces for a
CityFlow-sourced camera therefore carries `confidence=APPROXIMATE`
permanently, until such a validation step is built and explicitly run —
this commit deliberately stops short of building it, per its own scope.

### Tests added

`tests/perception/test_motion.py` (extended, +14 tests, all 11
pre-existing tests unmodified and passing): straight-line known speed;
zero movement is `OK` not an error; timestamp-based (not detection
-count-based) — the same 3 samples compressed into a shorter real-time
window produce a proportionally higher speed; irregular-timestamp
weighting; insufficient observations; insufficient elapsed time; a large
gap that leaves zero usable steps (`LARGE_TEMPORAL_GAP`) versus one that
leaves a valid remaining segment (`OK`, using only that segment);
non-finite coordinates (NaN and infinity, separately); the noisy-outlier
-does-not-dominate case described above; `RobustSpeedResult`'s own
all-or-nothing field validation (both directions); and the default
`min_samples=2` matching the existing two-point convention.

`tests/perception/test_speed_estimation.py` (new file, 11 tests, no
dependency on the real dataset — a synthetic 2× -scale homography fixture
stands in for a real CityFlow matrix, requirement 14): missing/explicitly
-uncalibrated calibration is `UNAVAILABLE`; a synthetic CityFlow-style
homography transformation produces the hand-computed expected km/h value;
`CalibrationProvenance.CITYFLOW_SUPPLIED` and `MetricValidationStatus.
UNVALIDATED` both hold for that profile; the core rule — a mathematically
clean CityFlow-derived speed is `APPROXIMATE`, never `VALIDATED` — plus
the contrasting case where a real `fit_planar_homography` survey *does*
reach `VALIDATED`, proving `APPROXIMATE` isn't just a hardcoded universal
default; ground-point trajectory drives the result even when the
(separately supplied, stationary) centroid trajectory would give zero;
`VehicleSpeedEstimate`'s own all-or-nothing field validation; and one full
integration test (requirement 13) driving a `RawDetection` through
`TrackedDetection` → `TrackManager.update()` →
`estimate_vehicle_speed`, against a CityFlow-style profile, asserting the
resulting estimate is `OK`/`APPROXIMATE` with the expected km/h value.
