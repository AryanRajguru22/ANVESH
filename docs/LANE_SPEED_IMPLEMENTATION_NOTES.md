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
