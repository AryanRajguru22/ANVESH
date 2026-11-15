# ANVESH Commit 9 Integration Audit

Read-only audit. No source file, test, config, dataset, or ANVESH-V0 file
was modified while producing this document — only `Read`/`Grep`/`Glob`/
`Bash` (read-only: `git diff`, `git status`, `cat`, `grep`, `wc`) were
used. This document itself is the only file written.

Companion to `ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` (the Commit 1–8 design
audit) and `docs/LANE_SPEED_IMPLEMENTATION_NOTES.md` (the Commit 1–8
implementation log). Where this document repeats a fact already
established there, it is restated only because Commit 9 planning depends
on it.

---

## 1. Actual current call graph (traced directly, not assumed)

There is **no single existing call graph** from video to visualization —
there are three, independent, already-working chains, none of which
touches c006/c009, `calibration.txt`, or lane geometry today. Confirmed
by `grep -rl "c006\|c009\|S02" scripts/ configs/ anvesh/`: the only hits
are `anvesh/world/calibration.py`, `anvesh/world/lane_config.py`,
`anvesh/perception/speed_estimation.py` (comments/docstrings only, no
executable path), and `configs/lanes/s02_c006.toml` / `s02_c009.toml`
themselves. **Commit 9 is not "extending an existing c006/c009 pipeline"
— it is building the first one**, by composing primitives Commits 1–8
already finished.

### Chain 1 — `scripts/run_pipeline.py` (M1-only debug video)

```
run_pipeline.main()
  -> perception.config.load_pipeline_config()          (configs/experiment/m1_pipeline.toml)
  -> perception.pipeline.run_m1_pipeline(...)
       per frame: tracker.update() -> track_manager.update()
                  -> [on_frame_tracked hook, if given]   (NEW, uncommitted — see §2)
                  -> draw_frame() -> VideoWriter.write()  (perception/visualize.py, UNTOUCHED by Commits 1-8)
       after all frames: track_manager.finalize() -> schema_conversion.build_vehicle_track() per track
  -> writes one annotated .mp4
```

Single camera (`cam-dev-01`, `datasets/recorded/dev/car-detection.mp4`).
No windowing, no evidence, no ranking, no calibration profile ever
passed. `draw_frame` is called **live, inline, per frame**, before the
whole video (and therefore any window) is finished.

### Chain 2 — `scripts/run_real_world_validation.py` (M1→M3→M4 text report)

```
main() -> _run_all_sequences() -> _run_sequence() per sequence
  -> run_m1_pipeline(...)                                   (no visualization_output_path)
  -> perception.windowing.generate_windows/build_traffic_state_timeline
  -> per window: tracks_in_window -> compute_reliability -> build_evidence_for_camera
                 -> evaluation.baselines.run_baseline(BaselineType.A, ...)
  -> _print_sequence_report() / _print_two_sequence_integration()  (text only)
```

Operates on `seq1_car_detection` / `seq2_person_bicycle_car` — generic,
**uncalibrated** validation clips (`configs/experiment/
m6_real_camera_thresholds.toml` states this explicitly: "both real-camera
sequences are `motion_space = IMAGE`"). No rendering at all. Uses
`evaluation.baselines.run_baseline`, not `anvesh.pipeline.
run_corridor_window` — this is an evaluation harness, not the production
path.

### Chain 3 — `scripts/export_ui_data.py` (M1→M8 dashboard JSON) — the most mature real-camera chain that exists

```
main() -> real_video_limited_chain(video_path=demo.mp4, camera_id="cam-demo", ...)
  -> run_m1_pipeline(..., on_frame_tracked=_capture_frame)   # captures {frame_index, timestamp, boxes:[_box_entry(td)]}
  -> windowing.generate_windows -> per window:
       aggregate_traffic_state(..., MotionSpace.IMAGE, ...)   # HARD-CODED IMAGE space
       CameraWindowInput(role=UPSTREAM, ...)
       anvesh.pipeline.run_corridor_window(window, (camera_input,))     # <- the REAL M8 orchestrator
       _window_result(result, {camera_id: window_tracks})     # serializes candidate_ranking, evidence_by_camera, ...
  -> writes ui/data/anvesh_demo_data.json
```

Also used by `scripts/analyze_video.py` (a thin, already-parameterized
CLI wrapper around `real_video_limited_chain`, built for "any judge
video"). `ui/live/anvesh_live_dashboard.html` (per the Commit 8 audit's
own §6) then reads this JSON client-side: a `<canvas>` render loop draws
`b.bbox_xyxy` per box and colors a track red when it is
`state.selectedTrack` (user click) or appears in `safety_events` — this
is a **second, independent rendering surface** from `anvesh/rendering.py`
(Commit 8's OpenCV overlay), and it is the one already wired to real
video + the real M8 engine end to end.

**Neither Chain 1 nor Chain 3 has ever been given a `calibration_profile`
for a real camera.** `run_m1_pipeline(calibration_profile=...)` and
`schema_conversion.build_vehicle_track`'s WORLD-space branch exist and
are tested (Commit 2 of M2, unrelated to this project's Commits 1–8) but
are only exercised by synthetic fixtures today. Wiring calibration into
Chain 3 for c006/c009 will be the **first time** a real YOLO+ByteTrack
track feeds the WORLD-space evidence path (H2/H3 stopped-vehicle
detection, `aggregate_traffic_state`'s WORLD density branch) — flagged as
a regression-risk item in §5.

---

## 2. Pre-existing dirty files (confirmed via `git diff`, contents read in full)

All four are **unrelated to lane/speed/candidate/rendering** and must not
be touched by Commit 9 beyond what's noted:

| File | What's actually in the diff | Relation to Commit 9 |
|---|---|---|
| `anvesh/perception/pipeline.py` | Additive `on_frame_tracked` callback param on `run_m1_pipeline`, invoked once per frame right after `tracked_detections` are computed. Nothing else changed. | **Directly useful** — this is exactly the hook Commit 9 needs to capture per-frame detections without a second tracking pass (see §4, §6). Must be preserved and built on, not reverted. |
| `anvesh/evidence/evidence_builder.py` | H5 ("elevated inflow") rewritten from an absolute `elevated_vehicle_count` cutoff to a `capacity_flow_veh_per_s`-relative one; new `EvidenceThresholds` fields/validation. Evidence **math** change (H5 only). | **Must not be touched.** Commit 9 is explicitly forbidden from changing evidence scoring; this file is mid-edit for an unrelated reason. Commit 9 only *calls* `build_evidence_for_camera`/`EvidenceThresholds()` with their current (dirty, in-progress) signatures — already compatible, no coupling. |
| `anvesh/perception/traffic_state.py` | New `mean_concurrency` field/computation, occupancy numerator switched from `vehicle_count` to `mean_concurrency`. Traffic-state **math** change. | **Must not be touched.** Commit 9 only calls `aggregate_traffic_state(...)` positionally/by-keyword exactly as `export_ui_data.py` already does; the signature Commit 9 needs (`segment_length_m`, `max_capacity_vehicles`, `motion_space`) is untouched by this diff. |
| `configs/experiment/m6_real_camera_thresholds.toml` | Comment/doc update matching the H5 rename above (`elevated_vehicle_count` → `capacity_flow_veh_per_s`, left unset). | **Unrelated** — this file configures `seq1`/`seq2` (Chain 2), never c006/c009. Commit 9 should add a **new** config file, never edit this one. |
| `tests/evidence/test_evidence_builder.py`, `tests/fusion/test_ds_fusion.py`, `tests/perception/test_pipeline_end_to_end.py`, `tests/perception/test_traffic_state.py`, `tests/test_real_world_validation_runner.py` | Test-side counterparts of the three diffs above (H5 rework, `mean_concurrency`, `on_frame_tracked`). | **Unrelated**, must not be touched or reformatted incidentally. |

Also confirmed: `git grep AnveshWindowResult(` finds **exactly one**
construction site (`anvesh/pipeline.py`'s own `return` statement) — no
test or script constructs it directly. Adding a field to
`AnveshWindowResult` is therefore a safe, purely-additive change with
exactly one call site to update.

---

## 3. The three-way granularity mismatch (the central finding — drives every design choice below)

**Question A's answer, stated once, used everywhere else in this
document:**

| Quantity | Correct granularity | Why |
|---|---|---|
| **Lane assignment** | Per **detection** (per frame, per vehicle) | `perception.lane_assignment.assign_detection_lane(detection, regions)` (Commit 4) is already a pure, stateless, per-observation function — it needs zero window or track-lifetime context. Its own module docstring is explicit: lane assignment is "calculated per raw detection and propagated into the track's history... never computed once... and carried forward," specifically so a lane change mid-track is visible. Recomputing it live, per frame, costs one point-in-polygon test — negligible. |
| **Approximate speed** | Per **track** (whole finalized lifetime), computed once | `perception.speed_estimation.estimate_vehicle_speed(camera_id, state, calibration_profile)` (Commit 6) takes a track's **entire** `ground_point_trajectory` and runs one robust (`compute_robust_speed`) estimate over it — there is no sliding-window/incremental variant. Recomputing it every frame from a growing trajectory would be O(frames²) per track over a video and is not what the function was built for (extending it to be incremental would be a **speed-math change**, explicitly forbidden for Commit 9). The one number is computed after `track_manager.finalize()`, once per track, and displayed identically for every frame that track appears in. |
| **Candidate-vehicle resolution** | Per **window** | `hypotheses.candidate_vehicle.resolve_candidate_vehicle` (Commit 7) takes a `CandidateCauseHypothesis` + `evidence_by_camera` — both window-scoped by construction (`ranking.py`'s own `build_ranking` takes one `window` tuple). There is no finer granularity available; a red highlight can only change at window boundaries. |

This mismatch is not a defect to fix — each granularity is the correct
one for what the quantity actually measures — but it is why rendering
cannot be a single live pass over the video the way `draw_frame` is
today (§4).

---

## 4. Two-pass architecture this implies (answers A, B together)

Because candidate resolution needs a whole window's evidence, and a
window's evidence needs the tracks that fell in it, and `run_m1_pipeline`
only produces the final track list *after* the whole video has been
decoded, a **single forward pass cannot render frame N with candidate
data for the window frame N belongs to** — that window may not close
until long after frame N was decoded.

The existing, already-uncommitted `on_frame_tracked` hook (§2) already
solves the expensive half of this: `scripts/export_ui_data.py`'s
`_capture_frame` closure (lines 604–613) already captures every frame's
`tracked_detections` into an in-memory list **during the single YOLO
+ByteTrack pass**, so detection/tracking inference never needs to run
twice. What Commit 9 adds on top:

```
PASS 1 (existing shape, unchanged inference cost):
  run_m1_pipeline(video_path, camera_id, tracker, ...,
                  calibration_profile=<loaded profile>,      # NEW argument, already supported by run_m1_pipeline
                  on_frame_tracked=_capture_frame)            # existing hook — captures per-frame tracked_detections
    -> M1PipelineResult.vehicle_tracks   (schema-level, WORLD if calibrated)
    -> M1PipelineResult.track_states     (NEW field — raw TrackState list, needed for ground_point_trajectory)

  windowing.generate_windows / build_traffic_state_timeline   (existing, unchanged)
  per window:
    tracks_in_window -> compute_reliability -> CameraWindowInput
    anvesh.pipeline.run_corridor_window(...)                  # NEW: also returns candidate_vehicle (§5)
    -> AnveshWindowResult with candidate_ranking + candidate_vehicle, both window-scoped, both from ONE call

  per track_state in M1PipelineResult.track_states:
    estimate_vehicle_speed(camera_id, track_state, calibration_profile)   # ONE speed estimate per track, Commit 6, untouched
    -> speed_by_raw_track_id: dict[str, VehicleSpeedEstimate]

PASS 2 (data-only — reuses PASS 1's already-captured frames, no re-inference):
  for each captured {frame_index, timestamp, tracked_detections}:
    for each detection:
      lane_result = assign_lane(detection.ground_point, lane_regions)     # LIVE, per frame — cheap geometry (§3)
      speed_estimate = speed_by_raw_track_id.get(track_id)                # LOOKUP, not recomputed (§3)
      window = the window whose [start, end) contains `timestamp`
      candidate_resolution = per_window_result[window].candidate_vehicle  # LOOKUP, not recomputed
    -> emit structured per-box data (see §6 for exactly what "emit" means)
```

This answers **B** directly: the renderer (whichever surface — see §6)
never computes anything; it only receives, per frame per vehicle,
already-computed values (`LaneAssignmentResult`, `VehicleSpeedEstimate`,
one `CandidateVehicleResolution` shared by every vehicle in that window),
matching `anvesh/rendering.py`'s design requirement 10 verbatim (already
built, Commit 8) — Pass 2 is exactly the shape `render_annotated_frame`
already expects.

This also answers **C** precisely: candidate resolution must be called
**inside `run_corridor_window`**, immediately after `build_ranking`,
using the *same* `evidence_by_camera` and `candidate_ranking` local
variables already in scope in that function (see §5) — never rebuilt or
re-fetched by a separate caller. This makes "the exact evidence/ranking
generated by the same window" true by construction, not by convention.

---

## 5. File-by-file: what must change, why, and the risk

### 5.1 `anvesh/pipeline.py` — NOT currently dirty

**Why:** This is the one function (`run_corridor_window`) that already
has `candidate_ranking` and `evidence_by_camera` in scope, together, for
one window (lines 196–206 of the current file). It is the single correct
place to call `resolve_candidate_vehicle` — anywhere else risks a caller
rebuilding or mismatching the evidence that produced the ranking.

**Change:** Two additions, both purely additive:
1. `from anvesh.hypotheses.candidate_vehicle import resolve_candidate_vehicle`
2. Immediately after `candidate_ranking = build_ranking(...)`:
   `candidate_vehicle = resolve_candidate_vehicle(candidate_ranking, evidence_by_camera)`
3. Add `candidate_vehicle: object` to `AnveshWindowResult` (its own
   dataclass, listed alphabetically/logically next to `candidate_ranking`)
   and pass it in the final `return AnveshWindowResult(...)`.

**Pre-existing dirty changes here:** none — this file is clean.

**Layering:** N/A (nothing to layer around).

**Regression risk:** Very low. `AnveshWindowResult` has exactly one
construction site (§2), so no other file breaks. `resolve_candidate_vehicle`
is a pure, already-tested (17 unit + 2 integration tests, Commit 7),
side-effect-free function — calling it unconditionally, every window,
costs one dict flatten + a few string comparisons over already-in-memory
data. The module's own docstring ("pure composition... reimplements
none of... ") should gain one line noting it now also composes Commit
7's resolver, to keep that docstring accurate.

### 5.2 `anvesh/perception/pipeline.py` — ALREADY DIRTY (§2)

**Why:** `M1PipelineResult` currently discards every `TrackState` the
moment it's converted to a schema `VehicleTrack` (`track_manager.
all_tracks()` is a local variable, never returned). Lane assignment
needs nothing from here (it works directly off live `tracked_detections`,
§3–4) but **speed estimation does**: `estimate_vehicle_speed` requires a
`TrackState.ground_point_trajectory`, which only exists on the raw
`TrackState`, never on the frozen `VehicleTrack` schema object.

**Change:** One additive field + one additive line, layered on top of
the existing (uncommitted) `on_frame_tracked` diff, in the same function:
- `M1PipelineResult` gains `track_states: list = field(default_factory=list)`.
- After `track_manager.finalize()`, alongside the existing
  `result.vehicle_tracks = [...]` comprehension: `result.track_states =
  list(track_manager.all_tracks())`.

**Pre-existing dirty changes here:** the `on_frame_tracked` hook (§2) —
must be preserved exactly. The new field/line is independent of it (one
touches the per-frame loop, the other touches the post-`finalize()`
block) — no line-level conflict expected, but the edit must be applied
*with* the current working-tree content (not a checkout of the last
commit), since the function's current line numbers already differ from
`git show HEAD:...`.

**Regression risk:** Very low — a new, default-empty-list field nothing
existing reads. `scripts/run_pipeline.py`, `scripts/
run_real_world_validation.py`, and `scripts/export_ui_data.py`'s current
call sites are all keyword-argument calls that don't enumerate
`M1PipelineResult`'s fields, so they are unaffected.

### 5.3 `scripts/export_ui_data.py` — UNTRACKED (whole file is new/uncommitted; treat its *content* as must-preserve, same as a dirty file)

**Why:** This is Chain 3 (§1) — already the most complete real-camera→
dashboard-JSON path that exists, already parameterized generically
(`real_video_limited_chain`'s docstring: "every parameter defaults to
the exact original DEMO_* constant... purely additive"), and already
consumed by a second caller (`scripts/analyze_video.py`). Building a
parallel, separate script instead of extending this one would duplicate
substantial working logic (`_write_trimmed_copy`, GPU detection, the
`on_frame_tracked` capture closure, the window loop) for no benefit.

**Changes, all additive, all defaulting to today's exact behavior:**

1. `real_video_limited_chain(...)` gains two new optional parameters:
   `calibration_profile=None`, `lane_regions=()`.
   - Threads `calibration_profile` into the existing
     `run_m1_pipeline(..., calibration_profile=calibration_profile)`
     parameter (already accepted, never previously used here).
   - The window loop's hard-coded `MotionSpace.IMAGE` (line 645) becomes
     conditional: `MotionSpace.WORLD if calibration_profile is not None
     and calibration_profile.is_calibrated else MotionSpace.IMAGE` —
     mirrors `schema_conversion.build_vehicle_track`'s own existing
     branch (§1), not new logic.
   - Inside `_capture_frame`, when `lane_regions` is non-empty, each
     box additionally computes `assign_lane(td.detection.ground_point,
     lane_regions)` and passes the result to `_box_entry`.
   - After `run_m1_pipeline` returns (now also carrying `track_states`,
     §5.2), one `estimate_vehicle_speed(camera_id, state,
     calibration_profile)` call per track builds a `speed_by_track_id`
     dict, threaded into `_box_entry` for every box of that track.
2. `_box_entry(td, lane_result=None, speed_estimate=None)` gains two
   optional keys, present **only when supplied** — preserving
   `tests/test_export_ui_data.py::test_box_entry_serializes_real_fields_only`'s
   "real fields only" contract for demo.mp4's existing zero-argument call
   (where both stay `None`/absent, exactly today's 4-key shape). Reuses
   `anvesh.rendering.format_lane_text`/`format_speed_text` (Commit 8,
   already built, already tested) to decide the *displayed string* —
   never reimplements that decision here (see §6 for why this matters).
3. `_window_result(result, tracks_by_camera)` gains one new key:
   `"candidate_vehicle": _candidate_vehicle(result.candidate_vehicle)`
   — a new, small serializer (same style as `_ranking`/`_safety_event`),
   reading the field added in §5.1. Also reuses `anvesh.rendering.
   is_resolved_candidate`/`ambiguity_banner_text` where a display flag is
   needed (§6), rather than re-deriving "is this track the highlighted
   one" independently in this file.

**Pre-existing dirty changes here:** none in the git sense (untracked,
not modified-tracked), but the *entire current content* — three
synthetic scenarios, `real_video_limited_chain`, `_write_trimmed_copy`,
GPU detection — is prior work that must be preserved. None of it is
touched except the specific additive changes above; `main()`'s
zero-argument `real_video_limited_chain()` call and the three
`scenario_*()` functions are byte-for-byte unaffected (new parameters
default to values that reproduce current behavior exactly).

**Regression risk:** Low-to-moderate. The MotionSpace branch change is
the one edit that changes *computed values* — but only on the new code
path (a caller that supplies `calibration_profile`); nobody currently
does, so demo.mp4's exported JSON is unaffected. The genuinely new
real-camera risk (first WORLD-space evidence run on real, noisy
YOLO+ByteTrack tracks — §1) surfaces here, not as a code defect but as
an open empirical question: real track jitter could make H2/H3's
`stopped_speed_m_per_s`/`min_dwell_seconds` thresholds behave differently
on real footage than on the clean synthetic fixtures they were tuned
against. This is explicitly **not** something Commit 9 should "fix"
(evidence math is out of scope) — it should be *observed* via the test
strategy in §7, not assumed away.

### 5.4 New file(s) — camera-specific configuration and entry point

**Why (answers D, I together):** `configs/lanes/s02_c006.toml` /
`s02_c009.toml` already exist (Commit 3) and already declare
`camera_id = "s02_c006"` / `"s02_c009"` and `source_video = 'E:\traffic
dataset\validation\S02\c00X\vdo.avi'` — an **external, absolute,
machine-specific path outside the repo**, not a repo-relative one like
`demo.mp4`/`traffic.mp4`. By CityFlow's own on-disk layout, the matching
`calibration.txt` is the sibling file
`E:\traffic dataset\validation\S02\c00X\calibration.txt`. Nothing in the
repo currently expresses "for this camera_id, here is the calibration
file" as configuration — hard-coding that path resolution into
`export_ui_data.py` or `anvesh/pipeline.py` would be exactly the
S02-coupling requirement I forbids.

**Recommended new files (Commit 9a, single-camera scope — see §8):**
- **No new config-loader module needed yet.** `world.lane_config.
  load_lane_config`/`default_lane_config_path` (Commit 3) and
  `world.calibration.load_cityflow_calibration` (Commit 5) already do
  everything required; a caller composes them in ~4 lines:
  ```python
  lane_config = load_lane_config(default_lane_config_path("s02_c006"))
  calibration_path = Path(lane_config.source_video).parent / "calibration.txt"
  calibration = load_cityflow_calibration("calib-s02-c006", "s02_c006", calibration_path)
  ```
- **One new script**, e.g. `scripts/run_s02_camera_demo.py`, that does
  exactly this composition for one camera, then calls
  `export_ui_data.real_video_limited_chain(video_path=lane_config.
  source_video, camera_id="s02_c006", calibration_profile=calibration,
  lane_regions=lane_config.regions, pre_trim_to_sample=True,
  sample_seconds=<bounded>, ...)`, and writes to a **separate** output
  path (e.g. `ui/data/anvesh_s02_c006_data.json`) — never overwriting
  `ui/data/anvesh_demo_data.json`. This is the ONLY file in this list
  that ever mentions `"s02_c006"`, `"c006"`, or the `E:\traffic dataset`
  path — every other file stays generic, taking camera_id/lane
  regions/calibration profile as plain arguments (satisfies I directly).
  `pre_trim_to_sample=True` (already an existing, tested parameter,
  §5.3's docstring) is how this script honors "do not run the full
  CityFlow showcase yet" — it bounds the very first real run to a short
  sample, exactly the mechanism `analyze_video.py` already uses for
  arbitrary long videos.

**Regression risk:** None — new file, zero existing callers.

### 5.5 Tests (new/extended, additive only)

- `tests/perception/test_pipeline_end_to_end.py` (already dirty, §2):
  one new test asserting `M1PipelineResult.track_states` is populated
  and its `track_id`s match `vehicle_tracks`' `state.track_id` half
  (before the `camera_id:` qualification) one-to-one.
- `tests/test_m8_end_to_end.py` (existing, exercises `run_corridor_window`
  already): extend with assertions that `result.candidate_vehicle` is
  present, is a `CandidateVehicleResolution`, and its `status`/
  `hypothesis_id` are consistent with `result.candidate_ranking` for
  both a `RANKED` and an `INSUFFICIENT_EVIDENCE` fixture (the file
  already has both, per §2's diff/`CandidateOutcome.INSUFFICIENT_EVIDENCE`
  assertion at line 245).
- `tests/test_export_ui_data.py` (existing, 5 tests, §5.3's risk):
  extend with (a) `_box_entry` keeps its 4-key shape when `lane_result`/
  `speed_estimate` are omitted (regression guard for the existing "real
  fields only" test's intent), (b) `_box_entry` gains the two new keys
  with values matching `anvesh.rendering.format_lane_text`/
  `format_speed_text` when supplied, (c) `_window_result` includes
  `candidate_vehicle` and it round-trips through `json.dumps`, (d) a
  small synthetic-video, no-calibration call to `real_video_limited_chain`
  confirms it is byte-for-byte unaffected when the two new parameters
  are omitted (regression guard for demo.mp4 and `analyze_video.py`).
- New: a focused test for whichever new script is added (§5.4) —
  camera_id/lane-config-path/calibration-path resolution logic only
  (mocking or skipping the actual `E:\traffic dataset` read, since that
  path will not exist on CI or most dev machines — see §5.6/G).

### 5.6 Explicitly NOT changed by Commit 9

`anvesh/evidence/evidence_builder.py`, `anvesh/hypotheses/ranking.py`,
`anvesh/fusion/ds_fusion.py`, `anvesh/hypotheses/candidate_vehicle.py`,
`anvesh/world/lane_geometry.py`, `anvesh/world/lane_config.py`,
`anvesh/world/calibration.py`, `anvesh/perception/lane_assignment.py`,
`anvesh/perception/speed_estimation.py`, `anvesh/rendering.py`,
`anvesh/perception/visualize.py`, `anvesh/perception/traffic_state.py`,
ANVESH-V0, the CityFlow dataset. Every one of these is *called*, none is
*edited*.

---

## 6. Where "rendering" actually happens (answers H)

Commit 8 built `anvesh/rendering.py`'s `render_annotated_frame` — a
server-side OpenCV function that bakes lane polygons, per-vehicle
labels, and the red candidate highlight directly into output video
pixels. **It has zero callers today.** Separately, and already fully
wired end-to-end for real video (§1, Chain 3), `ui/live/
anvesh_live_dashboard.html` is a client-side `<canvas>` renderer reading
JSON that `scripts/export_ui_data.py` produces.

These are not competing choices — they are two different deliverables,
and Commit 9 does not need to build both:

- **The dashboard JSON path (§5.3) is the lower-risk, higher-leverage
  target for Commit 9.** It reuses a chain that is already tested,
  already real-video-capable, and already serves the judge-facing
  `analyze_video.py` demo. No new video-writing/two-pass-decode
  machinery is needed — Pass 2 (§4) becomes "attach lane/speed/candidate
  data to the boxes already captured in Pass 1," not "re-decode the
  video and draw on it."
- **`anvesh/rendering.py` stays unused by Commit 9**, but its **pure**
  helpers (`format_lane_text`, `format_speed_text`, `is_resolved_candidate`,
  `ambiguity_banner_text` — none of which touch `cv2`) are exactly the
  right thing to call from `_box_entry`/`_window_result` (§5.3) so the
  *decision logic* ("is this the highlighted candidate," "what string
  does this lane status render as") lives in exactly one place and is
  exactly the place Commit 8 already tested it — the dashboard JSON
  should ship pre-decided display strings/flags (`"lane": "L1"`,
  `"is_candidate": true`), not raw enums the JS would have to
  re-interpret, so no highlighting *decision* is ever duplicated in
  JavaScript.
- A genuine standalone annotated-MP4 output (actually calling
  `render_annotated_frame`, i.e. Pass 2 becomes a real second video
  decode + `cv2.VideoWriter`) is a reasonable **later** commit, once the
  dashboard path has validated that the lane/speed/candidate values look
  correct against real S02 footage — recommended, not scoped into
  Commit 9 (see §8).

---

## 7. E, F, G — availability, backward compatibility, synthetic scenarios

**E — cameras without lane config or calibration:** Already fully
handled by existing, tested code, requiring zero new fallback logic:
`assign_lane` returns `UNKNOWN_NO_GEOMETRY` for `lane_regions=()`
(Commit 2); `estimate_vehicle_speed` returns `confidence=UNAVAILABLE`
for `calibration_profile=None`/uncalibrated (Commit 6);
`anvesh.rendering.format_lane_text`/`format_speed_text` already render
both states explicitly (Commit 8, tested). `real_video_limited_chain`'s
new parameters (§5.3) both default to "no lane/no calibration" —
i.e. every existing and future non-S02 camera automatically gets exactly
today's behavior with no branch added anywhere for "does this camera
have configuration."

**F — single- and two-camera backward compatibility:** `run_corridor_window`
already accepts 1 or 2 `CameraWindowInput`s (unchanged by §5.1's edit);
`real_video_limited_chain` remains single-camera-shaped in Commit 9a
(§8) — a two-camera corridor is a materially larger extension (two
independent M1 passes over two videos of possibly different duration/fps,
a shared window schedule, `fuse_two_cameras` instead of
`single_camera_result`) and is scoped to Commit 9b, not squeezed into
9a's single-camera slice. No existing single-camera caller's behavior
changes.

**G — existing synthetic scenarios:** `export_ui_data.py`'s
`scenario_cross_camera_agreement`/`scenario_camera_disagreement`/
`scenario_reliability_degradation` call `run_corridor_window` exactly as
they do today; §5.1's change only *adds* a field to the result they
already receive, and their `_window_result(...)` calls automatically
gain the new `candidate_vehicle` key (computed from their existing
synthetic evidence — this is a **feature**, not a regression: `tests/
test_export_ui_data.py` should assert these synthetic windows now also
carry a sensible `candidate_vehicle`, e.g. the `_stopped_track` fixture's
H2/H3 windows should resolve `RESOLVED`). No synthetic scenario's
*existing* asserted output changes.

---

## 8. Recommendation: two commits, not one

**Commit 9a — single real S02 camera, dashboard JSON path only.**
Touches: `anvesh/pipeline.py` (§5.1), `anvesh/perception/pipeline.py`
(§5.2, layered on the existing dirty diff), `scripts/export_ui_data.py`
(§5.3), one new script (§5.4), plus tests (§5.5). Produces one real,
lane/speed/candidate-annotated dashboard dataset for c006 (or c009)
alone — no fusion, no corridor topology, no propagation (none is
available for a single camera anyway), no OpenCV standalone video.
Small, independently reviewable, and it is exactly the vertical slice
needed to discover whether the WORLD-space evidence path behaves
sensibly on real footage (§5.3's flagged open question) *before*
doubling the surface area with a second camera.

**Commit 9b — two-camera c006+c009 corridor.** Extends 9a's script to
run both cameras, build a shared window schedule across two
independently-decoded videos, and call `run_corridor_window` with both
`CameraWindowInput`s (`fuse_two_cameras` path). No real
`segment_length_m`/`CorridorTopology` exists for this pair (confirmed —
no distance between c006 and c009 is documented anywhere in this repo),
so `corridor_topology`/`segment_length_m` stay unset and
`corridor_state`/`propagation_prediction` are correctly reported
unavailable via `run_corridor_window`'s own existing, already-tested
"reason" mechanism — not a gap Commit 9b needs to fill.

**Why not one commit:** 9a and 9b have genuinely different risk
profiles and genuinely different new-code shapes (single video/window
schedule vs. two independently-decoded videos reconciled into one shared
schedule). Splitting them matches this project's own established
convention (eight prior commits, each one concern) and lets 9a's
single-camera real-data results inform whether 9b's fusion step needs
anything adjusted before it's attempted — exactly the same reasoning
that already produced eight small commits instead of one large one for
Commits 1–8.

A follow-on (not scheduled here) — wiring `anvesh/rendering.py` into an
actual standalone annotated MP4 for c006/c009 — is a reasonable Commit
9c or later, deferred until 9a/9b's dashboard numbers are visually
sanity-checked against real footage.
