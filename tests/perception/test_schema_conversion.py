from anvesh.perception.schema_conversion import build_camera_observation, build_vehicle_track
from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import (
    CameraObservation,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
    WorldPosition,
)
from anvesh.world.calibration import fit_planar_homography, uncalibrated_profile
from tests.world.support import make_reference_points


def _raw_detection(frame_index=0, timestamp=0.0):
    return RawDetection(
        frame_index=frame_index,
        timestamp=timestamp,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.87,
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
    )


def test_build_camera_observation_is_valid_schema_instance():
    observation = build_camera_observation(
        camera_id="cam-01",
        frame_index=5,
        timestamp=0.5,
        detections=[_raw_detection(frame_index=5, timestamp=0.5)],
        model_id="yolov8n.pt",
        model_version="pretrained",
    )

    assert isinstance(observation, CameraObservation)
    assert observation.observation_id == "cam-01:5"
    assert observation.camera_id == "cam-01"
    assert len(observation.detections) == 1
    assert observation.detections[0]["vehicle_class"] == "car"
    assert observation.detections[0]["bbox_xyxy"] == (1.0, 2.0, 3.0, 4.0)


def test_build_camera_observation_with_no_detections_is_valid():
    observation = build_camera_observation(
        camera_id="cam-01",
        frame_index=0,
        timestamp=0.0,
        detections=[],
        model_id="yolov8n.pt",
        model_version="pretrained",
    )
    assert observation.detections == []


def test_build_vehicle_track_is_valid_schema_instance():
    state = TrackState(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=1.0,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=[(0.0, 0.0, 0.0), (10.0, 0.0, 1.0)],
    )

    track = build_vehicle_track("cam-01", state)

    assert isinstance(track, VehicleTrack)
    assert track.track_id == "cam-01:1"
    assert track.camera_id == "cam-01"
    assert track.vehicle_class == VehicleClass.CAR
    assert track.occlusion_state == OcclusionState.VISIBLE
    assert track.motion_space == MotionSpace.IMAGE


def test_vehicle_track_position_history_is_image_space_when_no_calibration_given():
    """No calibration_profile supplied (or an UNCALIBRATED one) ->
    position_history must stay plain (x, y, t) pixel tuples, never
    WorldPosition instances, and motion_space must say so explicitly."""
    state = TrackState(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=1.0,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=[(0.0, 0.0, 0.0), (10.0, 0.0, 1.0)],
    )

    track = build_vehicle_track("cam-01", state)
    assert track.motion_space == MotionSpace.IMAGE
    for sample in track.position_history:
        assert isinstance(sample, tuple)
        assert len(sample) == 3

    track_explicit_uncalibrated = build_vehicle_track(
        "cam-01", state, calibration_profile=uncalibrated_profile("calib-none", "cam-01")
    )
    assert track_explicit_uncalibrated.motion_space == MotionSpace.IMAGE


def test_vehicle_track_is_world_space_when_calibrated():
    """A CALIBRATED profile must produce real WorldPosition samples (metres)
    and a WORLD motion_space -- the M1 pixel-space convention must not be
    used once valid calibration exists."""
    profile = fit_planar_homography("calib-01", "cam-01", make_reference_points())

    trajectory = [(10.0, 10.0, 0.0), (500.0, 10.0, 1.0)]
    state = TrackState(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=1.0,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=trajectory,
    )

    track = build_vehicle_track("cam-01", state, calibration_profile=profile)

    assert track.motion_space == MotionSpace.WORLD
    assert len(track.position_history) == 2
    for sample in track.position_history:
        assert isinstance(sample, WorldPosition)

    import math

    expected_world_0 = profile.image_to_world((10.0, 10.0))
    expected_world_1 = profile.image_to_world((500.0, 10.0))
    assert track.position_history[0].world_x == expected_world_0[0]
    assert track.position_history[1].world_x == expected_world_1[0]

    expected_speed_m_per_s = math.hypot(
        expected_world_1[0] - expected_world_0[0], expected_world_1[1] - expected_world_0[1]
    )
    # speed must be metric (m/s), not the raw pixel displacement
    assert track.speed_estimate.value != 490.0  # raw pixel distance / 1s would be 490 px/s
    assert track.speed_estimate.value == expected_speed_m_per_s


def test_vehicle_track_speed_estimate_matches_motion_module():
    from anvesh.perception.motion import compute_image_space_speed

    trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 2.0)]
    state = TrackState(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=2.0,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=trajectory,
    )

    track = build_vehicle_track("cam-01", state)
    expected = compute_image_space_speed(trajectory)

    assert track.speed_estimate.value == expected.speed_px_per_s
    assert track.speed_estimate.error == expected.error_px_per_s
