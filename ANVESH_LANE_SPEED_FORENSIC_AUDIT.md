# ANVESH — Lane Assignment & Approximate-Speed Forensic Audit

**Scope:** Static, read-only forensic inspection of two repositories, ahead of
adding lane assignment + approximate speed + candidate-vehicle highlighting to
the ANVESH V1 showcase corridor (upstream `c006` / downstream `c009`, CityFlowV2
S02). **No code was changed. No files were modified in either repository. No
dataset files were modified. `C:\Users\Aryan\ANVESH-V0` was only read, never
written.**

- **V1 / current:** `D:\ANVESH`
- **V0 / old prototype:** `C:\Users\Aryan\ANVESH-V0` (already carries its own
  prior forensic audit, `V0_AUDIT.md`, which this document cross-checks
  against the actual source rather than restating uncritically)

---

## 1. V0 Lane Detection

**Files:** `lane_detection.py` (repo root) is **0 bytes** — dead from the
first commit, never implemented. The real, active implementation is
`modules/lane_manager.py` (31 lines total), wired into `main.py`.

### 1.1 Representation

```python
# modules/lane_manager.py:5-18
# Adjusted to your screenshot geometry
LANE_1 = np.array([[450, 220], [820, 220], [960, 720], [380, 720]])
LANE_2 = np.array([[820, 220], [1250, 220], [1400, 720], [960, 720]])
```

Two hand-typed 4-point pixel polygons (quadrilaterals), fitted by eye to one
specific 1280×720 frame from one specific video. The in-code comment
("Adjusted to your screenshot geometry") is explicit that this was manual,
one-off, per-scene work — not a general or reusable representation.

### 1.2 Lane numbering

`LaneManager.detect_lane(x, y)` (lines 23-31): tests `LANE_1` first, then
`LANE_2`, first match wins; returns integer `1`, `2`, or `0` (undefined/no
match). No lane ever exceeds ID `2` because only two polygons exist —
lane count is a hardcoded fact of the source file, not a property read from
config or geometry.

### 1.3 Vehicle → lane assignment

`main.py:70`: `lane_id = lane_manager.detect_lane(cx, cy)`, where `(cx, cy)`
is the **bounding-box center** (`main.py:51-52`: `cx=(x1+x2)//2`,
`cy=(y1+y2)//2`), tested via `cv2.pointPolygonTest(LANE_N, (x, y), False)`.
**This is box-center, not bottom-center/contact-point** — for a tall vehicle
viewed at an angle, the box center sits above the road plane and drifts
toward the camera relative to where the vehicle actually contacts the road,
which is exactly the failure mode design requirement A in this task is meant
to avoid.

### 1.4 Perspective / ROI assumptions

The two polygons are trapezoids that narrow toward the top of the frame,
informally encoding the camera's perspective narrowing for *that one frame*.
There is no explicit ROI concept separate from the polygons themselves, and
there is no relationship enforced between these lane polygons and
`config.py`'s separate `SRC_POINTS` homography quad (used only by
`speed_estimator.py`) — **two independently hand-typed geometries describe
"the road," with nothing checking that a lane polygon lies inside the
calibrated homography region or represents the same physical pavement**
(confirmed directly in `config.py:26-31` vs. `modules/lane_manager.py:6-18`).

### 1.5 Limitations and failure modes (verified against source, not just V0_AUDIT.md's prior claims)

- **Non-portable:** any change of camera position, angle, zoom, or resolution
  silently invalidates both polygons; there is no detection of this — a
  vehicle just gets silently misassigned or returns lane `0`.
- **Exactly two lanes, hardcoded:** cannot represent 1, 3, 4+ lanes, a
  merge/turn pocket, or a curved corridor without hand-editing source.
- **No lane-change history:** lane ID is recomputed independently every
  frame; nothing tracks a transition from `LANE_1` to `LANE_2` for the same
  track.
- **Box-center bias** (§1.3): systematically wrong exactly where lane
  assignment matters most — near lane boundaries and for tall vehicles.
- **Silent failure on out-of-polygon points:** lane `0` is returned
  identically whether the vehicle is off-road, in an un-mapped third lane,
  or simply outside the small mapped region — there is no distinction
  between "confidently no lane" and "we don't know."

### 1.6 Verdict: what (if anything) is recoverable

**Nothing in `lane_manager.py` is worth carrying forward as code** — this
matches the prior `V0_AUDIT.md` §7/§22 assessment, and direct inspection of
the 31-line file confirms it: two magic-number polygons and a linear
`pointPolygonTest` scan. The only transferable *idea* is the general shape of
the interface (`detect_lane(x, y) -> lane_id`), which is such a thin contract
it doesn't meaningfully constrain a new design. Per design requirement B,
this file must not be copied, adapted, or used as a starting point beyond
that one-line interface shape.

---

## 2. V0 Speed Estimation

**File:** `modules/speed_estimator.py` (58 lines), the only speed logic in
the repo (`unusual_behavior.py`, `violation_logger.py` are 0 bytes; no other
file computes speed).

### 2.1 Exact method (verified line-by-line against source)

1. **Homography setup** (`__init__`, lines 10-24): `cv2.findHomography` maps
   `config.py`'s hand-typed `SRC_POINTS` (4 pixel points) to a synthetic
   `REAL_WIDTH_METERS × REAL_LENGTH_METERS` = `12m × 30m` rectangle
   (`config.py:23-24`). This size and the four pixel points are typed
   constants with **no documented survey/measurement method**.
2. **Projection** (`to_world`, lines 29-32): `cv2.perspectiveTransform`
   projects a pixel `(x, y)` onto this flat-plane homography — assumes a
   perfectly flat road, no lens distortion correction.
3. **History buffer** (line 26): `deque(maxlen=10)` of `(world_x, world_y,
   time.time())` per `track_id`.
4. **Speed = (oldest vs. newest in the current buffer)** (lines 42-50): NOT a
   fixed wall-clock window — the buffer holds up to 10 *detections*, so the
   effective time span depends on how fast the outer loop runs (i.e., on
   runtime FPS/inference speed), which is a self-referential coupling:
   slower inference → longer smoothing window → smoother but laggier speed.
5. **Unit conversion:** Euclidean distance / elapsed time × 3.6 → km/h
   (line 50).
6. **Sanity clip:** `if speed_kmh > 150: return 0` (lines 53-54) — a single
   noisy detection makes a genuinely fast vehicle render as `0 km/h`
   ("stopped"), silently, with no distinction from an actually-stopped
   vehicle. This directly feeds V0's jam detector (`jam_detector.py`)'s
   slow-vehicle count.
7. **Smoothing:** a plain moving average over the last 5 *computed* speed
   values (lines 27, 56-58) — no Kalman filter, no median/outlier-robust
   statistic.

### 2.2 Assumptions / units

- Units: km/h, derived from an assumed metric homography target plane
  (12m × 30m).
- Flat ground plane, no camber/grade handling.
- Box-center (not contact-point) is the position fed into the homography
  (`main.py:51-52,63`) — same bias discussed in §1.3, here affecting speed
  scale/accuracy rather than lane assignment: taller vehicles project to a
  slightly different world position than their true ground contact point,
  and this bias is not constant across the frame (worse near the camera).
- No lens-distortion correction.
- No documented calibration procedure — `SRC_POINTS` and the 12×30m size are
  simply typed into `config.py`.

### 2.3 What is actually being computed

Not an instantaneous frame-to-frame speed and not a fixed-time-window
average — a **detection-count-window average velocity** whose real-world
time span varies with runtime performance. This is the single most important
thing to not inherit: any new speed estimator must key its averaging window
to wall-clock time (or a fixed frame-count *at a known, fixed camera FPS*,
which V1 already has since CityFlow video is a fixed 10 fps file, not a live
inference loop), never to "however many detections happened to arrive."

### 2.4 Worth recovering

- **The core idea (homography → world-plane distance/time → speed) is
  directionally correct** and is, in fact, already what V1's
  `perception/motion.py` + `world/calibration.py` implement, more rigorously
  (see §4). There is nothing left to "port" — V1 already has a better
  version of this exact concept.
- **Nothing in the code itself is worth recovering.** The specific defects
  (detection-count window, hard clip-to-zero instead of rejection/flagging,
  box-center proxy, unvalidated 12×30m constant) are exactly the failure
  modes V1's own design discipline (frozen `Measurement(value, error)`,
  explicit `MotionSpace` typing, `CalibrationStatus.UNCALIBRATED` as a first-
  class state) was built to avoid, per V0_AUDIT.md §21/§26 and confirmed here
  by direct reading.

---

## 3. V1 Perception/Tracking

### 3.1 Detection + tracking

`anvesh/perception/tracking.py`: `ByteTrackYoloTracker` wraps
`ultralytics.YOLO(...).track(persist=True, tracker="bytetrack.yaml", ...)`
(lines 100-109) — same library/tracker choice as V0, but isolated behind a
`Tracker` `Protocol` (lines 17-20) with a pure, independently-testable parser
(`parse_tracked_results`, lines 23-68) that converts raw Ultralytics
`Results` into `(raw_detections, tracked_detections)` — a real seam V0 never
had (V0's parsing was inline in `main.py`'s loop).

### 3.2 Bounding-box representation

`RawDetection.bbox_xyxy` (`perception/types.py:31-57`) is a validated,
frozen `(x1, y1, x2, y2)` tuple in **source-frame pixel coordinates**, with
`__post_init__` enforcing `x2 >= x1`, `y2 >= y1`, and confidence in `[0, 1]`.
A `centroid` property (lines 54-57) exists — **box center, not
bottom-center** — and this is what actually gets used downstream:
`TrackManager.update` (`track_manager.py:64`) reads `td.detection.centroid`
directly into the stored trajectory. **There is currently no bottom-center /
ground-contact-point accessor anywhere in the codebase.** This must be added,
not assumed to already exist.

### 3.3 Track IDs

`TrackedDetection.track_id` (`perception/types.py:60-69`) is a validated
non-empty string, sourced from ByteTrack's integer ID cast to `str`
(`tracking.py:66`). `TrackManager` (`track_manager.py`) owns the actual
per-track lifecycle: `TrackState` (dataclass, lines 22-35) holds
`trajectory: list` of `(x, y, timestamp)` triples, `occlusion_state`
(`VISIBLE`/`PARTIAL`/`LOST`, driven by `consecutive_missed_frames` vs.
`max_missed_frames`/`partial_after_missed_frames`), and `age_seconds`.
Finished tracks move from `_active` into `_finished`
(`_finalize_track`, lines 99-103) — **the trajectory is timestamp-keyed
throughout**, not detection-count-keyed, which is exactly the property V0's
speed estimator lacked (§2.3).

### 3.4 Frame timestamps / FPS

`Frame` (`perception/types.py:16-27`) carries `index` and `timestamp`
explicitly (not inferred from wall-clock at render time, as V0 did via
`time.time()` deltas in its render loop — `main.py:153-155`). For a
pre-recorded file source (`VideoFileFrameSource`, `perception/ingestion.py`,
not read in full here but referenced throughout `pipeline.py`), this
timestamp is derived from the video's own known FPS, not from how fast the
host machine happens to process frames — this is the fix for V0's §2.3
self-referential-window bug, and it is already in place architecturally.

### 3.5 Where trajectories are available

`TrackState.trajectory` (image-space) during a run, and
`VehicleTrack.position_history` (`storage/schemas.py:266-287`) in the frozen,
persisted form after `schema_conversion.build_vehicle_track` runs
(`perception/schema_conversion.py:68-104`). The frozen form already branches
on calibration (§4): `position_history` is `WorldPosition` instances (metres)
if a `CalibrationProfile` is supplied and calibrated, or plain pixel tuples
otherwise, with `motion_space` (`MotionSpace.WORLD` / `MotionSpace.IMAGE`)
recording which.

### 3.6 Where a lane assignment could cleanly attach

Two candidate points, both real, both viable — this audit does not assume
one is "correct" without weighing them:

- **A. At `TrackManager.update()` time (image-space, per-frame):** the
  natural point to compute a bottom-center point per detection and test it
  against a lane polygon (image-space or world-space) *before* it's folded
  into `TrackState.trajectory`. Pro: available for a live overlay
  (`visualize.py`) frame-by-frame; con: `TrackManager` currently knows
  nothing about calibration or lane geometry, and today only stores
  `(x, y, t)`, not per-sample metadata like lane ID — adding this changes
  `TrackState`'s stored shape.
- **B. At `schema_conversion.build_vehicle_track()` time (world-space, once
  calibrated):** lane assignment done *after* projecting to world
  coordinates, alongside the existing calibration branch (lines 80-92) —
  this is where V1 already distinguishes calibrated/uncalibrated cameras, so
  it is the natural place to also distinguish "lane geometry known" /
  "lane geometry unavailable" using the exact same branching discipline the
  file already uses for `MotionSpace`.

**Recommendation (elaborated in §8):** compute the bottom-center point at
detection time (a new `RawDetection.ground_point` property, image-space,
always available, no calibration required), but perform the actual
image-space-polygon **or** world-space-polygon lane test at `schema_conversion`
time, mirroring the existing WORLD/IMAGE branch exactly — this keeps
`TrackManager` calibration-agnostic (as it is today) and keeps the "which
geometry mode was used" decision co-located with the existing analogous
decision for speed.

---

## 4. V1 Calibration / World Geometry

### 4.1 What exists today

`anvesh/world/calibration.py` — `CalibrationProfile` (frozen dataclass,
lines 57-118):

- `status: CalibrationStatus` (`UNCALIBRATED` / `CALIBRATED`) — an
  uncalibrated camera is a **first-class, explicit state**, not an implicit
  null; `image_to_world()`/`world_to_image()` both raise `RuntimeError` if
  called on an uncalibrated profile (lines 100-104, 108-109) rather than
  silently returning pixels.
- `reference_points: tuple` of `CalibrationPoint` (`image_xy`, `world_xy`
  pairs, lines 38-54) — **raw manually-surveyed correspondences**, not a
  precomputed matrix.
- `homography` / `inverse_homography`: 3×3 numpy arrays, `None` unless
  calibrated.
- **Validation invariant** (`__post_init__`, lines 71-87): a `CALIBRATED`
  profile **must** carry `>= 4 reference_points`
  (`"a CALIBRATED profile requires at least 4 reference points"`, line 82);
  an `UNCALIBRATED` profile must carry neither a homography nor
  `reprojection_error_m`.
- The only public constructor that produces a `CALIBRATED` profile is
  `fit_planar_homography(calibration_profile_id, camera_id,
  reference_points)` (lines 121-162), which calls `cv2.findHomography` on
  the supplied points itself and computes its **own**
  `reprojection_error_m` from the fit (lines 152, 165-170).
- `world_xy` convention (lines 41-45, docstring): **"x runs along the
  corridor's direction of travel, y runs across it"** — an explicitly
  **local, per-corridor** convention, not a georeferenced/GPS system
  (blueprint explicitly excludes GIS/city-scale modeling).

### 4.2 What CityFlowV2's `calibration.txt` actually provides

Read directly (not assumed) for both cameras in the chosen corridor:

```
c006: Homography matrix: -39.660220243283760 15.592282221103229 3099.834691537730578;
                          -1.073038463974085 0.181163893667474 62.034492542754101;
                           0.001914912485368 0.011919314669326 1.000000000000000
      Reprojection error: 10.254867218805751

c009: Homography matrix: -39.931762346875395 15.097050319357958 3066.432457736477318;
                           1.564708487880704 1.493191280606009 68.977560416958141;
                           0.001539740258468 0.011743673340779 1.000000000000000
      Reprojection error: 7.105744643288634
```

This is **one already-fitted 3×3 matrix and a single scalar reprojection
error per camera — no correspondence points at all.** This was verified by
reading both files directly, not inferred from documentation.

### 4.3 What the matrix actually projects to (empirically tested, not assumed)

Per design requirement D's instruction to inspect before assuming
interchangeability, the c006/c009 matrices were numerically applied to
sample pixel coordinates (frame corners and center) using plain
`numpy`/`cv2`-equivalent homogeneous-coordinate math (no ANVESH or CityFlow
code was run — this is a standalone, throwaway calculation):

| Camera | pixel (0,0) | pixel (960,540) | pixel (1920,1080) | pixel (1920,0) |
|---|---|---|---|---|
| c006 | (3099.8, 62.0) | (−2863.1, −93.8) | (−3202.8, −102.7) | (−15619.7, −427.3) |
| c009 | (3066.4, 69.0) | (−3074.4, 269.6) | (−3443.5, 281.6) | (−18603.9, 776.8) |

**Findings from this test:**

- The output is **not** GPS latitude/longitude (values are in the thousands
  and go negative; real GPS decimal degrees for any real location are in
  roughly the ±180 range).
- The output is **not** a small, near-zero, per-camera-local frame either —
  magnitudes in the low thousands, in a coordinate space whose origin is
  evidently *not* anchored near either camera's own field of view. This is
  consistent with (but not confirmed as) a shared, scenario-wide local
  tangent-plane / equirectangular-style projection used by the AI City
  Challenge evaluation harness to make multiple cameras' world coordinates
  comparable — **this must be verified against the dataset's own
  documentation or a known reference distance (e.g., a lane width of
  ~3–3.7 m) before being trusted**, not assumed from this calculation alone.
- Pixels far from the actual road/ROI (e.g., `(1920, 0)`, which is above the
  horizon in both cameras per the earlier visual inspection) project to
  wildly different-scale outputs (−15,619 / −18,603) — expected behavior for
  a flat-road-plane homography applied outside the plane it was fit to, and
  a concrete reminder that **any lane polygon or speed measurement must stay
  inside the same region the homography was actually fit against.**
- The reprojection errors (10.25, 7.11) are **plausible as meters-scale
  residuals**, not plausible as GPS-degree residuals (a 7–10 *degree* error
  would be absurd) — this weakly supports "meters in some local/shared
  frame" over "raw GPS," but is not proof.

### 4.4 Concrete incompatibilities with `CalibrationProfile` (the actual audit finding)

1. **Structural:** `CalibrationProfile.__post_init__` requires `len(
   reference_points) >= 4` for any `CALIBRATED` status (line 82). CityFlow's
   `calibration.txt` supplies **zero** correspondence points — only the
   already-fitted matrix and its own reprojection error. The existing public
   constructor, `fit_planar_homography`, **cannot** be used to load this data
   (it fits its own homography from points; it does not accept a precomputed
   one). There is currently **no code path** that can turn a
   `calibration.txt` file into a valid `CalibrationProfile` without either
   (a) fabricating ≥4 fake "reference points" by back-projecting pixels
   through the given matrix and its inverse — which would trivially
   "validate" while adding zero real survey information and misrepresenting
   `reprojection_error_m`'s provenance — or (b) adding a new, explicit
   constructor for externally-supplied homographies that does not pretend to
   have raw correspondence points.
2. **Semantic (unverified, flagged not assumed):** `CalibrationProfile`'s own
   documented convention (`world_xy`: local, per-corridor, "x along travel,
   y across") is not confirmed to match whatever coordinate system
   CityFlow's matrix actually targets (§4.3). Wiring the raw matrix in
   directly, unexamined, risks silently mislabeling that output space as
   ANVESH's local corridor-meters convention when it may be a different
   (if also metric) frame.
3. **Units:** `CalibrationProfile.units` defaults to the string `"meters"`
   but is never independently verified anywhere in the codebase — it is
   trusted at face value by every downstream consumer
   (`perception/motion.py`'s `compute_world_space_speed`, `safety/
   stopped_vehicle.py`'s `world_speed_threshold_m_per_s` branch,
   `evidence_builder.py`'s `stopped_speed_m_per_s` check). If CityFlow's
   output space is not actually 1:1 with meters, every one of these
   consumers would silently compute a wrong-but-confidently-labeled speed.
4. **Cross-camera framing (V1's `world_xy` docstring says "local, per-
   corridor" — i.e., not meant to be shared across cameras — yet
   `world/corridor.py`'s whole reason to exist is a 2-camera corridor with a
   *known distance between cameras*, which strongly implies some cameras'
   world frames *do* need to relate to each other for anything beyond
   per-camera speed).** This is a design tension already latent in V1
   (not introduced by CityFlow): `CorridorTopology.distance_between_m` is a
   **separately operator-supplied** number (`distance_from_upstream_m`,
   `world/corridor.py:23`), not derived from either camera's own
   `CalibrationProfile` — so V1 already does not assume the two cameras'
   `world_xy` frames are the same frame; it only needs one shared, trusted
   corridor distance. This actually **reduces** the blast radius of
   CityFlow's coordinate-system ambiguity: even if c006's and c009's
   projected coordinates are not directly comparable, per-camera world-space
   speed only needs each camera's own homography to be internally
   consistent (correct scale, whatever its origin/axis convention), which is
   independently checkable via a known real-world reference distance in each
   camera's own footage.

### 4.5 Recommendation direction (elaborated in §8)

Do not silently wrap CityFlow's matrix in a `CalibrationProfile` as if it
came from `fit_planar_homography`. Add an explicit, separately-named
constructor (e.g. `from_precomputed_homography(...)`) that:
- accepts the raw homography + the dataset's own reported reprojection error
  directly, tagging `method="cityflow_precomputed_v1"` (not
  `"manual_planar_homography_v1"`) so provenance is never ambiguous,
- stores `reference_points=()` (honestly empty — no survey points exist) and
  either relaxes the dataclass's `>= 4` invariant specifically for this
  provenance tag, or introduces a small `CalibrationProvenance` enum so the
  invariant can stay meaningful for manually-surveyed profiles while still
  admitting an externally-supplied one,
- is only trusted for **relative** measurements (speed magnitude in
  whatever this space's true units are) until a real-world reference
  distance (e.g., known lane width, or the dataset's own documented units)
  is used to confirm the meters claim — until then, per design requirement
  E, any speed derived from it must be labeled **approximate** and the scale
  assumption stated, not silently presented as validated metric truth.

---

## 5. V1 Evidence / Ranking

### 5.1 What "ranking" currently ranks

`hypotheses/cause_model.py` defines a **closed set of six categorical
congestion-cause hypotheses**, H1–H6 (`downstream_bottleneck`,
`stopped_disabled_vehicle`, `collision_like_obstruction`,
`lane_blockage_non_incident`, `upstream_demand_surge`,
`signal_intersection_restriction`), plus H7 (`insufficient_evidence`,
represented as Dempster-Shafer ignorance mass on Θ, never a scored
singleton — lines 180-200). **`CandidateCauseHypothesis.ranked_list`
(`storage/schemas.py:353-391`) ranks these six categories, not individual
vehicles.** This is the single most important structural fact for this
task: there is no existing "candidate vehicle ranking" data structure — only
a "candidate cause-of-congestion category ranking."

### 5.2 Where an actual vehicle track re-enters the picture

`evidence/evidence_builder.py`'s `Evidence.supporting_track_refs`
(`storage/schemas.py:318-336`, populated at
`evidence_builder.py:137-155`/`219-258`/`261-298`) **is** a list of real
`track_id` strings — but only for H2 (`_evidence_h2`, populated when
exactly one stopped track exists, line 252: `stopped_refs if len(stopped)
== 1 else []`) and H3 (`_evidence_h3`, populated when ≥2 stopped tracks
exist, line 292). **H1, H4, H5, H6 never populate `supporting_track_refs`**
(H1's evidence is about global occupancy, not a specific vehicle; H4/H6 are
hardcoded `NONE`-completeness/`INSUFFICIENT` always, per the module's own
documented V1 limitations, lines 20-29; H5 is about aggregate flow rate).

**This means a "highest-ranked candidate vehicle" only exists when the
winning hypothesis is H2 or H3** — for H1/H4/H5/H6, there is no single
vehicle to blame by design, and any implementation must represent that as an
explicit "no vehicle candidate for this cause" state (design requirement G),
never fall back to guessing one.

### 5.3 The missing hop: ranking → evidence → track

`RankedHypothesisEntry.supporting_evidence_refs` (`storage/schemas.py:354-
370`) stores **evidence_id strings**, not `Evidence` objects and not
`track_id`s directly (confirmed at `hypotheses/ranking.py:101`:
`_evidence_refs(...)` extracts `.evidence.evidence_id`, and at
`export_ui_data.py:144`: `"supporting_evidence_refs": list(
e.supporting_evidence_refs)` — plain ID strings all the way to the
dashboard JSON today). **No function anywhere currently resolves an
evidence_id back to its `Evidence.supporting_track_refs`.**

However, the raw material for that resolution already exists in one place at
runtime: `anvesh/pipeline.py`'s `AnveshWindowResult` (lines 79-99) carries
**both** `evidence_by_camera: dict` (the original `EvidenceItem`s, each with
`.evidence.evidence_id` and `.evidence.supporting_track_refs`) **and**
`candidate_ranking` (the `CandidateCauseHypothesis`) side by side, for the
same window, in the same object. `scripts/export_ui_data.py`'s
`_window_result()` (lines 215-233) already exports both
`evidence_by_camera` and `candidate_ranking` into the same per-window
dashboard JSON block — **the two pieces needed to resolve "top hypothesis →
supporting vehicle track" are already sitting next to each other in the
exported data; nothing currently joins them.**

### 5.4 Persistence caveat

`storage/db.py`'s `RankingStore` persists `CandidateCauseHypothesis` (which
only carries evidence_id strings) and `CorridorStateStore` persists
`CorridorState`; **there is no `EvidenceStore`** — raw `Evidence`/
`EvidenceItem` records (with `supporting_track_refs`) are never durably
persisted anywhere in the current codebase. This is fine for the immediate
dashboard/demo use case (§5.3's in-memory join is sufficient for one
pipeline run), but means a *later*, historical/audit "why was this vehicle
highlighted three days ago" query would require either persisting
`Evidence` records too, or persisting the resolved `track_id` alongside the
ranking at build time. This audit flags it; it is out of scope to fix for
the immediate showcase per this task's instructions.

### 5.5 What would need to be built (elaborated in §8)

A small, new, explicitly-named resolver (e.g.
`hypotheses/candidate_vehicle.py`) that:
- takes the top `RankedHypothesisEntry` (`ranked_list[0]`, already
  deterministically sorted by `(belief desc, plausibility desc,
  hypothesis_id asc)` per `ranking.py:106`) and the `evidence_by_camera`
  dict from the same `AnveshWindowResult`,
- looks up each of the top entry's `supporting_evidence_refs` against the
  `EvidenceItem`s' `.evidence.evidence_id`, and collects the union of their
  `.evidence.supporting_track_refs`,
- returns a new, explicit, non-schema companion result (matching the
  project's established pattern — `EvidenceItem`, `TrackState`,
  `StoppedVehicleClassification`, `CorridorStateAssembly` are all this kind
  of "diagnostic wrapper around/adjacent to a frozen schema object," never a
  frozen schema mutation) with **an explicit status**, never a silently
  guessed vehicle:
  - `RESOLVED` (exactly one track_id) — the outcome the red highlight uses,
  - `AMBIGUOUS` (>1 track_id, e.g., H3's multi-vehicle case) — a list, not a
    single pick; the UI/task requirement is "highest-ranked candidate
    vehicle," and where the evidence itself names several vehicles, picking
    one of them arbitrarily would itself be a fabrication,
  - `NO_VEHICLE_FOR_HYPOTHESIS` (top hypothesis is H1/H4/H5/H6, or has no
    track refs) — the honest "this cause isn't about one vehicle" case,
  - `NOT_RANKED` (`outcome != RANKED`, i.e. `HIGH_CONFLICT` or
    `INSUFFICIENT_EVIDENCE`) — no candidate should ever be highlighted when
    the ranking engine itself abstained.

This satisfies design requirement F directly: the resolver only ever
surfaces a track_id that is already present in the evidence model's own
`supporting_track_refs`, never invents or hardcodes one, and it is
structurally incapable of presenting "highest-ranked candidate" as
"confirmed responsibility" — it is one more derived, explicitly-labeled
diagnostic, exactly like every other non-schema object this codebase already
uses.

---

## 6. Dashboard

### 6.1 What renders today

The real UI is the static, data-driven `ui/live/anvesh_live_dashboard.html`
(and its non-live sibling `ui/anvesh_dashboard.html`), fed by
`scripts/export_ui_data.py`'s JSON — `perception/visualize.py` is a
separate, plain OpenCV debug-video overlay (bounding box + track ID +
trajectory trail, `draw_frame()`, lines 18-44), used only to produce an
annotated `.mp4` for `scripts/run_pipeline.py`, not the interactive
dashboard.

### 6.2 Bounding boxes / track IDs

`scripts/export_ui_data.py:_box_entry()` (lines 435-441) exports
`{track_id, bbox_xyxy, vehicle_class, confidence}` per detection per frame
for the single-camera "real video" demo mode. In the dashboard JS
(`ui/live/anvesh_live_dashboard.html:449-457`), the canvas render loop reads
`b.bbox_xyxy` directly and draws a rectangle; **there is currently no lane
or speed field in this per-frame box object at all** — confirmed by reading
`_box_entry` in full; it has exactly four keys.

### 6.3 Existing red-highlight convention (already present, not new)

Two independent "red" mechanisms already exist and must not be confused:

1. **User-click selection** (`anvesh_live_dashboard.html:450-451`):
   `const flagged = state.selectedTrack === ... + b.track_id; ctx.strokeStyle
   = flagged ? "#ef4444" : "#38bdf8";` — purely an interactive UI selection,
   not evidence-driven.
2. **Safety-event flagging** (lines 390, 503-504, 646): any track whose ID
   appears in `win.safety_events` (i.e., has a `StoppedVehicleClassification`
   → `SafetyEvent`, `safety/stopped_vehicle.py:317-331`) is rendered with
   `border-error`/`text-error` (red) styling in the track-chip list and
   track inspector — **this is already an evidence-driven, track_id-keyed
   red-highlight mechanism**, just not currently wired to the canvas bbox
   itself, and not the same thing as "highest-ranked cause-hypothesis
   candidate."

**The "candidate red" this task wants is a third, new concept** —
track_id-keyed like mechanism 2, reusing the exact same visual language
(`#ef4444` / `border-error`, not a new color), but driven by the §5.5
resolver's output rather than by `safety_events` membership.

### 6.4 Where lane/speed labels could attach

`_box_entry()` (§6.2) is the natural place to add two new, always-optional
keys once the corresponding data exists: a `lane` field (string/enum, never
silently defaulting to a guessed lane per design requirement G — e.g.
`"lane_2"` / `"unknown"` / `"no_geometry"`) and a `speed_estimate` field that
carries the existing `Measurement(value, error)` shape already used
everywhere else in the schema (`storage/schemas.py:104-114`), **plus an
explicit `is_approximate: true` / `motion_space` tag** so the dashboard can
render an "≈" prefix or a muted style rather than a bare number (design
requirement E — "must not display fake precision"). The dashboard's existing
label-drawing code (`anvesh_live_dashboard.html:453`:
`` `#${b.track_id} ${b.vehicle_class}` ``) is a one-line template string —
trivial to extend with `lane`/`speed` conditionally, but the underlying
Python export currently has nothing to extend it with (no lane assignment
exists at all yet, per §1/§3.6; per-frame world speed at the individual-
detection level does not currently flow into `_box_entry` either — only
the track-lifetime `speed_estimate` on `VehicleTrack` does).

### 6.5 Where the highest-ranked candidate would render

Two integration points, not mutually exclusive:

- **Canvas overlay** (`anvesh_live_dashboard.html:449-457`): extend the
  existing `flagged` boolean to also be true when
  `b.track_id === candidate_vehicle.track_id` (from §5.5's resolver output,
  exported alongside `candidate_ranking` in `_window_result()`), using the
  same `#ef4444` stroke — this is the most literal reading of "highlighted
  RED" in the task.
- **Ranking panel:** wherever `candidate_ranking`/`ranked_list[0]` is
  already rendered in the dashboard (the "top hypothesis" summary implied by
  `_top_entry()`, `export_ui_data.py:198-202`, used today for feedback
  prior/revised comparisons) — adding the resolved vehicle reference (or its
  explicit "no vehicle" / "ambiguous" state, §5.5) right next to the
  hypothesis name, so the causal story and the visual highlight are never
  presented independently of each other.

---

## 7. Design-requirement mapping (A–H)

| Req | Requirement | How the audit informs it |
|---|---|---|
| A | bottom-center/contact-point → lane region, deterministic/auditable | Neither V0 (`main.py:51-52`, box-center) nor current V1 (`types.py:54-57` `centroid`, box-center) has a contact-point accessor at all — this is new code, not a fix to existing code (§1.3, §3.2). |
| B | do not blindly copy V0's `lane_detection.py` | Moot as literally stated — that file is 0 bytes; the real V0 lane logic is `modules/lane_manager.py`, and §1.6 confirms nothing in it should be reused beyond the one-line interface shape. |
| C | no large ML lane-segmentation model unless demonstrated necessary | Nothing inspected here demonstrates a need for one: c006/c009 are fixed, unmoving cameras with stable, human-visible painted lane markings (per the prior visual inspection reports) — a deterministic polygon test (§8) is sufficient and matches design requirement A's own instruction. |
| D | investigate whether perspective-aware polygonal lane regions are appropriate for fixed cameras | §4.3's homography test confirms the road plane assumption is already load-bearing everywhere in V1 (speed, calibration); polygons defined either in image-space (perspective-narrowing quads, as V0 did, but per-camera and derived from actually-observed lane paint rather than eyeballed) or in world-space (rectangular lane strips post-projection, only for a calibrated camera) are both consistent with "fixed camera" — recommendation in §8. |
| E | speed must be explicitly APPROXIMATE until real calibration is validated | V1 already has the right *shape* for this (`MotionSpace.IMAGE` vs `WORLD`, `CalibrationStatus`) but §4.4 shows the CityFlow homography's unit/frame claims are unverified — any speed computed from it must carry an explicit approximate/unvalidated marker end-to-end (schema → export → dashboard), which does not exist today even for the current `WORLD` path (a calibrated `WorldSpaceSpeedEstimate` is presented with no separate "provenance confidence" of its own). |
| F | red candidate = highest-ranked ANVESH-evidence vehicle; never hardcode; never equal confirmed responsibility | §5 shows the ranking system ranks hypothesis *categories*, not vehicles; a track_id only exists for H2/H3 via `supporting_track_refs`; §5.5's resolver is required and must expose explicit non-RESOLVED states rather than ever guessing. |
| G | missing/ambiguous lane geometry → explicit unavailable/uncertain state, never a guessed lane | Matches the project's own existing, repeatedly-used pattern (`CalibrationStatus.UNCALIBRATED`, `EvidenceCompleteness.NONE`, `UNRANKED_PLACEHOLDER`, `StoppedVehicleStatus.INSUFFICIENT_EVIDENCE`) — §8 proposes a `LaneAssignmentStatus`-style enum following the exact same convention, not a new pattern. |
| H | V0 must not be modified | Confirmed: only `Read`/`Bash cat`/`Glob`/`Grep` were used against `C:\Users\Aryan\ANVESH-V0` in this audit; no `Write`/`Edit` call targeted that path at any point. |

---

## 8. Recommended architecture (design only — not implemented)

### 8.1 Lane representation

**Perspective-aware polygonal lane regions, defined per-camera in config
(TOML), not in source code** — directly following the existing
`anvesh/perception/config.py` / `configs/experiment/*.toml` pattern already
used for `PipelineConfig` and the M6.5 real-camera threshold overrides
(`configs/experiment/m6_real_camera_thresholds.toml`). Two viable coordinate
choices, both perspective-aware, differing only in *when* perspective is
resolved:

- **Image-space quads** (available for every camera, calibrated or not) —
  closest analog to V0's polygons, but per-camera-configured, not hardcoded,
  and tested against a **bottom-center point**, not box-center.
- **World-space rectangles** (only for a `CALIBRATED` camera) — once a
  detection's bottom-center is projected via `CalibrationProfile.
  image_to_world`, a lane can be a simple axis-aligned or oriented rectangle
  in the corridor's local meters frame, which is simpler to author and
  reason about than a perspective quad, at the cost of only being available
  for calibrated cameras.

Recommend supporting **both**, selected by the same `is_calibrated` branch
`schema_conversion.py` already uses (§3.6, option B) — an uncalibrated
camera gets image-space lane polygons if configured, a calibrated one
prefers world-space rectangles if configured, and a camera with **no lane
config at all** (or a bottom-center point outside every configured region)
reports an explicit `LANE_UNKNOWN`/`LANE_UNAVAILABLE` status — never a
guessed lane (design requirement G).

### 8.2 Approximate-speed architecture

No new speed *math* is needed — `perception/motion.py`'s
`compute_world_space_speed`/`compute_image_space_speed` are already
sound and already keyed to wall-clock timestamps (§3.3/§3.4), unlike V0.
What's missing is **provenance/confidence labeling that survives to the
UI**:

- Add an explicit calibration-provenance tag (§4.5's
  `method="cityflow_precomputed_v1"` vs. `"manual_planar_homography_v1"`) on
  `CalibrationProfile`.
- Add an `is_approximate: bool` (or a small `SpeedConfidence` enum) alongside
  `Measurement` wherever a world-space speed derived from an
  externally-supplied (not independently re-validated) homography is
  reported — default `True`/"approximate" for any CityFlow-sourced
  calibration until a real-world reference distance check is performed and
  recorded.
- Propagate that flag through `schema_conversion.py` → `export_ui_data.py`
  → the dashboard label, so the UI can render `≈42 km/h` (or similar),
  never a bare, confident-looking number, per design requirement E.

### 8.3 Candidate-red architecture

- New module `anvesh/hypotheses/candidate_vehicle.py` implementing the §5.5
  resolver: `(CandidateCauseHypothesis, evidence_by_camera) ->
  CandidateVehicleResolution`, with an explicit status enum
  (`RESOLVED` / `AMBIGUOUS` / `NO_VEHICLE_FOR_HYPOTHESIS` / `NOT_RANKED`) —
  never a bare `track_id` with no status, so no caller can accidentally treat
  "no candidate" as "candidate is track None."
- `AnveshWindowResult` (or `export_ui_data.py`'s `_window_result`) gains a
  `candidate_vehicle` field carrying this resolution, exported into the same
  per-window JSON block that already carries `candidate_ranking` and
  `evidence_by_camera`.
- Dashboard: extend the existing canvas `flagged` boolean (§6.5) to also
  match `candidate_vehicle.track_id` when `status == RESOLVED`, using the
  existing `#ef4444` red — reusing, not duplicating, the visual language
  already established for `safety_events`.
- Never hardcode a track_id anywhere in this chain — the resolver only ever
  returns what `Evidence.supporting_track_refs` already contains.

---

## 9. Files that would need modification (new + changed)

**New files:**
- `anvesh/hypotheses/candidate_vehicle.py` — the ranking→track resolver (§5.5/§8.3).
- `anvesh/world/lane_geometry.py` (or similar) — lane-region types (image-space quad / world-space rectangle), a `LaneAssignmentStatus` enum, and the assignment function(s) (§8.1).
- `configs/experiment/<camera>_lanes.toml` (per camera, e.g. `c006_lanes.toml`, `c009_lanes.toml`) — operator-authored lane geometry, following the existing TOML config pattern (§8.1), populated from the already-completed visual inspection of c006/c009's real lane markings.
- `tests/world/test_lane_geometry.py`, `tests/hypotheses/test_candidate_vehicle.py` — see §10.

**Changed files:**
- `anvesh/perception/types.py` — add a `RawDetection.ground_point` (bottom-center) property (§3.2/§3.6).
- `anvesh/perception/schema_conversion.py` — attach lane assignment at `build_vehicle_track` time, mirroring the existing `MotionSpace` branch (§3.6 option B, §8.1).
- `anvesh/world/calibration.py` — new `from_precomputed_homography(...)` constructor (or equivalent) for externally-supplied matrices without raw reference points; a provenance field/tag (§4.5).
- `anvesh/pipeline.py` — thread the §8.3 resolver into `run_corridor_window`/`AnveshWindowResult` (or leave this to the export layer if a lighter touch is preferred — a real design decision to make explicitly during implementation, not silently defaulted).
- `scripts/export_ui_data.py` — `_box_entry()` gains optional `lane`/`speed_estimate` keys (§6.4); `_window_result()`/`_ranking()` gain the resolved `candidate_vehicle` (§6.5).
- `ui/live/anvesh_live_dashboard.html` (and non-live sibling if kept in sync) — render lane/speed labels; extend canvas `flagged` to include the resolved candidate (§6.5).
- `anvesh/storage/schemas.py` — **only if** the team decides lane assignment or the resolved candidate belongs in the frozen Part-5 schema rather than staying a non-schema companion object (the project's dominant existing pattern for exactly this kind of derived diagnostic is to *not* touch frozen schemas — see `EvidenceItem`, `StoppedVehicleClassification`, `CorridorStateAssembly` — so the default recommendation is **not** to touch this file).

---

## 10. New tests required

Following the existing per-module test convention (`tests/<package>/test_<module>.py`, §"test directory conventions" observed directly):

- `tests/perception/test_types.py` (or extend an existing file) — `ground_point` correctness for boxes of varying aspect ratio; confirm it differs from `centroid` for a non-square box.
- `tests/world/test_lane_geometry.py` — point-in-polygon correctness for both image-space and world-space regions; explicit-unavailable-state test for a point outside all configured regions; explicit-unavailable-state test for a camera with no lane config at all; a determinism test (same point, same polygon set, same answer every call — design requirement A).
- `tests/world/test_calibration.py` (extend) — `from_precomputed_homography` construction and validation (rejects missing homography/error, does not fabricate reference points); explicit provenance-tag test distinguishing it from `fit_planar_homography`.
- `tests/hypotheses/test_candidate_vehicle.py` — one test per resolver outcome: `RESOLVED` (H2, single track ref), `AMBIGUOUS` (H3, multiple refs), `NO_VEHICLE_FOR_HYPOTHESIS` (top hypothesis H1/H5), `NOT_RANKED` (outcome is `HIGH_CONFLICT`/`INSUFFICIENT_EVIDENCE`) — each asserting the resolver never fabricates a track_id absent from the input evidence.
- `tests/perception/test_schema_conversion.py` (extend) — lane assignment attached correctly for both calibrated and uncalibrated tracks; explicit-unknown-lane path when no lane config is supplied.
- `tests/test_export_ui_data.py` (extend) — `_box_entry` optional lane/speed keys present only when the corresponding data exists; `candidate_vehicle` exported and its non-`RESOLVED` states round-trip correctly to JSON (i.e., "no candidate" must be distinguishable in the JSON from "candidate is absent from the payload").
- An end-to-end smoke test analogous to the existing `tests/test_m4_end_to_end.py`/`test_m8_end_to_end.py` pattern, exercising c006/c009 lane config + candidate resolution together on a short synthetic fixture (not the real CityFlow video, to keep tests fast and offline, consistent with how existing M4/M5/M8 end-to-end tests use hand-authored fixtures rather than real footage).

---

## 11. Proposed implementation plan (small commits — not implemented in this session)

1. **`ground_point` accessor.** Add `RawDetection.ground_point` (bottom-center) to `perception/types.py` + unit test. No behavior change to any existing consumer (additive only).
2. **Lane geometry primitives.** Add `world/lane_geometry.py`: region types (image-space quad, world-space rectangle), `LaneAssignmentStatus` enum (`ASSIGNED` / `UNKNOWN_NO_GEOMETRY` / `UNKNOWN_OUTSIDE_REGIONS`), and a pure `assign_lane(point, regions) -> (lane_id | None, status)` function + tests. No wiring into the pipeline yet.
3. **Per-camera lane config.** Author `configs/experiment/c006_lanes.toml` / `c009_lanes.toml` from the already-completed visual inspection of these two cameras' real lane markings, plus a TOML loader mirroring `perception/config.py`'s pattern.
4. **Wire lane assignment into `schema_conversion.build_vehicle_track`.** Attach lane status/id to the track's world (or image) trajectory, mirroring the existing `MotionSpace` branch. Extend `tests/perception/test_schema_conversion.py`.
5. **CityFlow calibration ingestion.** Add `CalibrationProfile.from_precomputed_homography(...)` + provenance tag + tests; a small loader for CityFlow's `calibration.txt` format; **do not** yet mark any resulting speed as validated-metric — default to the approximate/unvalidated tag from §8.2.
6. **Approximate-speed labeling.** Thread an `is_approximate`/provenance flag from `CalibrationProfile` through `schema_conversion.py`'s `speed_estimate` construction, ending at whatever `export_ui_data.py` emits.
7. **Candidate-vehicle resolver.** Add `hypotheses/candidate_vehicle.py` + full test suite (§10) against hand-built `CandidateCauseHypothesis`/`evidence_by_camera` fixtures — no pipeline wiring yet, so this is independently reviewable.
8. **Wire the resolver into `pipeline.run_corridor_window`/`AnveshWindowResult`.** Add `candidate_vehicle` to the result bundle; extend the relevant end-to-end test.
9. **Dashboard export.** Extend `scripts/export_ui_data.py`'s `_box_entry`, `_ranking`/`_window_result` with the new optional fields; extend `tests/test_export_ui_data.py`.
10. **Dashboard rendering.** Extend `ui/live/anvesh_live_dashboard.html`'s canvas draw loop and label templates for lane/speed text and the resolved-candidate red highlight, reusing the existing `#ef4444` convention. Manual visual verification against a short c006/c009 clip (per this task's "no full-video processing yet" instruction, deferred to whenever implementation is actually authorized).

Each commit is independently testable and reviewable, and steps 1–3 and 5–7 have no dependency on each other and could be reordered or parallelized if desired.

---

## 12. Confirmations

- `C:\Users\Aryan\ANVESH-V0` was only read (`Read`, `Bash cat/find/ls`) — no `Write`/`Edit` call was made against any path under it, at any point in this audit.
- No file under `D:\ANVESH` was modified — this audit only added the present document.
- No CityFlow dataset file (`E:\traffic dataset\...`) was modified — `calibration.txt` files were only read; the homography-projection check in §4.3 was a standalone, throwaway numerical calculation, not a run of any ANVESH or CityFlow code.
- No ANVESH pipeline, YOLO, or tracking code was executed as part of this audit.
