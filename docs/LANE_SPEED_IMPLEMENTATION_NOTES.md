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
