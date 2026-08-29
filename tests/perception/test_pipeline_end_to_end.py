from pathlib import Path

from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import CameraObservation, MotionSpace, OcclusionState, VehicleClass, VehicleTrack
from anvesh.world.calibration import fit_planar_homography
from tests.world.support import make_reference_points


class ScriptedTracker:
    """A deterministic, scripted Tracker (no YOLO/ByteTrack) for end-to-end
    pipeline testing, keyed only by frame_index -- ignores image content."""

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

    tracks_by_id = {t.track_id: t for t in result.vehicle_tracks}
    track_1 = tracks_by_id["cam-e2e:1"]
    track_2 = tracks_by_id["cam-e2e:2"]

    assert track_1.occlusion_state == OcclusionState.LOST  # missed 2 frames > max_missed_frames=1
    assert len(track_1.position_history) == 3
    assert track_1.speed_estimate.value > 0.0

    # track "2" only ever seen once, still active at end-of-stream -> finalized by pipeline
    assert len(track_2.position_history) == 1
    assert track_2.speed_estimate.value == 0.0  # single sample: no measurable motion

    assert track_1.motion_space == MotionSpace.IMAGE
    assert track_2.motion_space == MotionSpace.IMAGE


def test_end_to_end_pipeline_produces_world_space_tracks_when_calibrated(tiny_video_path: Path):
    """Same scripted scenario as above, but with a calibrated profile
    supplied -- the pipeline must thread it through to produce metric,
    WORLD-space VehicleTracks instead of the M1 pixel-space ones."""
    script = {
        0: [("1", (10.0, 10.0))],
        1: [("1", (250.0, 200.0))],
    }
    tracker = ScriptedTracker(script)
    profile = fit_planar_homography("calib-e2e", "cam-e2e", make_reference_points())

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
    assert track.speed_estimate.value > 0.0
    for sample in track.position_history:
        assert hasattr(sample, "world_x") and hasattr(sample, "world_y")


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
