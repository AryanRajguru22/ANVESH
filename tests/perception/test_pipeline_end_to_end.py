from pathlib import Path

from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.track_manager import derive_run_id
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import CameraObservation, MotionSpace, OcclusionState, VehicleClass, VehicleTrack
from anvesh.world.calibration import fit_planar_homography
from tests.world.support import make_reference_points


def _rid(camera_id, video_path):
    """The deterministic per-recording run prefix run_m1_pipeline applies by default (M-A.1)."""
    p = Path(video_path)
    return derive_run_id(camera_id, p.name, p.stat().st_size)


class ScriptedTracker:
    """A deterministic, scripted Tracker (no YOLO/ByteTrack) for end-to-end
    pipeline testing, keyed only by frame_index -- ignores image content."""

    # This scripted tracker keeps a lost track for 0 frames (it never revives
    # an id), so the pipeline's occlusion horizon is just the requested
    # `max_missed_frames`-derived floor. (A real ByteTrack declares 30.)
    lost_buffer_frames = 0

    def __init__(self, script: dict):
        self._script = script

    def reset(self) -> None:
        pass

    def update(self, image, frame_index: int, timestamp: float):
        entries = self._script.get(frame_index, [])
        raw = []
        tracked = []
        for track_id, (x, y) in entries:
            detection = RawDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                class_name="car",
                vehicle_class=VehicleClass.CAR,
                confidence=0.9,
                bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
            )
            raw.append(detection)
            tracked.append(TrackedDetection(detection=detection, track_id=track_id))
        return raw, tracked


def test_end_to_end_pipeline_produces_expected_schema_objects(tiny_video_path: Path):
    # Track "1" appears in frames 0-2 then disappears (finalized as lost,
    # given max_missed_frames=1 below); track "2" appears only in frame 4.
    script = {
        0: [("1", (10.0, 10.0))],
        1: [("1", (12.0, 10.0))],
        2: [("1", (14.0, 10.0))],
        4: [("2", (50.0, 50.0))],
    }
    tracker = ScriptedTracker(script)

    result = run_m1_pipeline(
        video_path=tiny_video_path,
        camera_id="cam-e2e",
        tracker=tracker,
        model_id="scripted-test",
        model_version="0.0.0",
        max_missed_frames=1,
        partial_after_missed_frames=1,
    )

    assert result.frame_count == 5
    assert len(result.camera_observations) == 5
    assert all(isinstance(o, CameraObservation) for o in result.camera_observations)

    # frame 3 has no detections -> that CameraObservation must still exist, empty
    assert result.camera_observations[3].detections == []

    assert len(result.vehicle_tracks) == 2
    assert all(isinstance(t, VehicleTrack) for t in result.vehicle_tracks)

    # Identity is the manager-assigned session-scoped id (serials in creation
    # order), NOT the raw tracker id -- the raw ids stay on the TrackState.
    tracks_by_id = {t.track_id: t for t in result.vehicle_tracks}
    rid = _rid("cam-e2e", tiny_video_path)
    assert set(tracks_by_id) == {f"cam-e2e:{rid}-s000-t00000", f"cam-e2e:{rid}-s000-t00001"}
    assert [s.raw_tracker_id for s in result.track_states] == ["1", "2"]
    track_1 = tracks_by_id[f"cam-e2e:{rid}-s000-t00000"]
    track_2 = tracks_by_id[f"cam-e2e:{rid}-s000-t00001"]

    assert track_1.occlusion_state == OcclusionState.LOST  # unseen 0.2 s > the 1.5-frame (0.15 s) horizon
    assert len(track_1.position_history) == 3
    assert track_1.speed_estimate.value > 0.0

    # track "2" only ever seen once, still active at end-of-stream -> finalized by pipeline
    assert len(track_2.position_history) == 1
    assert track_2.speed_estimate.value == 0.0  # single sample: no measurable motion

    assert track_1.motion_space == MotionSpace.IMAGE
    assert track_2.motion_space == MotionSpace.IMAGE


def test_track_states_retains_raw_track_state_per_vehicle_track(tiny_video_path: Path):
    """`M1PipelineResult.track_states` (Commit 9A, single-camera lane/speed
    integration) must carry the raw `TrackState` for every track
    `vehicle_tracks` was built from, one-to-one by (unqualified) track_id
    -- needed because `perception.speed_estimation.estimate_vehicle_speed`
    requires `TrackState.ground_point_trajectory`, which the frozen
    `VehicleTrack` schema object never carries."""
    script = {
        0: [("1", (10.0, 10.0))],
        1: [("1", (12.0, 10.0))],
        2: [("1", (14.0, 10.0))],
        4: [("2", (50.0, 50.0))],
    }
    tracker = ScriptedTracker(script)

    result = run_m1_pipeline(
        video_path=tiny_video_path,
        camera_id="cam-e2e",
        tracker=tracker,
        model_id="scripted-test",
        model_version="0.0.0",
        max_missed_frames=1,
        partial_after_missed_frames=1,
    )

    assert len(result.track_states) == len(result.vehicle_tracks) == 2
    qualified_ids = {t.track_id for t in result.vehicle_tracks}
    raw_ids_qualified = {f"cam-e2e:{s.track_id}" for s in result.track_states}
    assert raw_ids_qualified == qualified_ids

    state_1 = next(s for s in result.track_states if s.raw_tracker_id == "1")
    assert state_1.track_id == f"{_rid('cam-e2e', tiny_video_path)}-s000-t00000"
    assert len(state_1.ground_point_trajectory) == 3


def test_end_to_end_pipeline_produces_world_space_tracks_when_calibrated(tiny_video_path: Path):
    """Same scripted scenario as above, but with a calibrated profile
    supplied -- the pipeline must thread it through to produce metric,
    WORLD-space VehicleTracks instead of the M1 pixel-space ones."""
    # 5 frames, 20 px/frame inside the fitted homography's well-conditioned
    # region (~0.05 m/px -> ~10 m/s): enough samples/elapsed time for the
    # single metric-speed estimator (>= 3 samples, >= 0.3 s).
    script = {i: [("1", (100.0 + 20.0 * i, 200.0))] for i in range(5)}
    tracker = ScriptedTracker(script)
    profile = fit_planar_homography("calib-e2e", "cam-e2e", make_reference_points())
    assert profile.supports_metric_speed

    result = run_m1_pipeline(
        video_path=tiny_video_path,
        camera_id="cam-e2e",
        tracker=tracker,
        model_id="scripted-test",
        model_version="0.0.0",
        max_missed_frames=1,
        partial_after_missed_frames=1,
        calibration_profile=profile,
    )

    assert len(result.vehicle_tracks) == 1
    track = result.vehicle_tracks[0]
    assert track.motion_space == MotionSpace.WORLD
    assert track.speed_available is True
    assert track.speed_estimate.value > 0.0
    for sample in track.position_history:
        assert hasattr(sample, "world_x") and hasattr(sample, "world_y")


def test_on_frame_tracked_hook_is_called_once_per_frame_with_matching_data(tiny_video_path: Path):
    """Additive, optional hook (video-demo integration): must fire once per
    frame with exactly the same (frame_index, timestamp, tracked_detections)
    the pipeline itself uses for track_manager.update() -- and must not
    change anything else about the pipeline's return value."""
    script = {
        0: [("1", (10.0, 10.0))],
        1: [("1", (12.0, 10.0))],
        4: [("2", (50.0, 50.0))],
    }
    calls = []

    result_with_hook = run_m1_pipeline(
        video_path=tiny_video_path,
        camera_id="cam-hook",
        tracker=ScriptedTracker(script),
        model_id="scripted-test",
        model_version="0.0.0",
        max_missed_frames=1,
        partial_after_missed_frames=1,
        on_frame_tracked=lambda frame_index, timestamp, tracked: calls.append((frame_index, timestamp, tracked)),
    )
    result_without_hook = run_m1_pipeline(
        video_path=tiny_video_path,
        camera_id="cam-hook",
        tracker=ScriptedTracker(script),
        model_id="scripted-test",
        model_version="0.0.0",
        max_missed_frames=1,
        partial_after_missed_frames=1,
    )

    assert len(calls) == 5  # once per frame, including frames with zero detections
    assert [c[0] for c in calls] == [0, 1, 2, 3, 4]
    # the hook receives identity-labelled detections, never raw tracker ids
    rid = _rid("cam-hook", tiny_video_path)
    assert calls[0][2][0].track_id == f"{rid}-s000-t00000"
    assert calls[0][2][0].detection.bbox_xyxy == (5.0, 5.0, 15.0, 15.0)
    assert calls[2][2] == []  # frame 2: no scripted detections -> empty tracked list
    assert calls[4][2][0].track_id == f"{rid}-s000-t00001"

    assert result_with_hook.frame_count == result_without_hook.frame_count
    assert len(result_with_hook.vehicle_tracks) == len(result_without_hook.vehicle_tracks)


def test_end_to_end_pipeline_writes_annotated_video(tiny_video_path: Path, tmp_path: Path):
    script = {0: [("1", (8.0, 6.0))], 1: [("1", (8.0, 6.0))]}
    tracker = ScriptedTracker(script)
    output_path = tmp_path / "debug_out.mp4"

    result = run_m1_pipeline(
        video_path=tiny_video_path,
        camera_id="cam-viz",
        tracker=tracker,
        model_id="scripted-test",
        model_version="0.0.0",
        visualization_output_path=output_path,
    )

    assert result.frame_count == 5
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_unvalidated_calibration_through_the_full_pipeline_yields_image_space_tracks(tiny_video_path: Path):
    """Pipeline-level speed-safety check: a CityFlow-style (calibrated but
    metric-UNVALIDATED) profile threaded through `run_m1_pipeline` must NOT
    produce WORLD tracks, whatever the trajectory looks like."""
    import numpy as np

    from anvesh.world.calibration import from_cityflow_homography

    script = {i: [("1", (100.0 + 20.0 * i, 200.0))] for i in range(5)}
    profile = from_cityflow_homography("calib-cf", "cam-cf", np.diag([40.0, 40.0, 1.0]), reprojection_error=5.0)

    result = run_m1_pipeline(
        video_path=tiny_video_path, camera_id="cam-cf", tracker=ScriptedTracker(script),
        model_id="scripted-test", model_version="0.0.0", calibration_profile=profile,
    )

    (track,) = result.vehicle_tracks
    assert track.motion_space == MotionSpace.IMAGE
    assert track.geometry_validity.value == "world_unvalidated"
    assert all(isinstance(p, tuple) for p in track.position_history)
    assert track.speed_estimate.value < 400.0  # px/s (200 px/s), not 40x-scaled "m/s"
