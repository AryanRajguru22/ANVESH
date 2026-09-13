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
