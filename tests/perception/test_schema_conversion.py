from anvesh.perception.schema_conversion import build_camera_observation, build_vehicle_track
from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import CameraObservation, OcclusionState, VehicleClass, VehicleTrack


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


def test_vehicle_track_position_history_is_image_space_not_world_position():
    """Guards the documented M1/schema incompatibility: position_history
    must stay plain (x, y, t) pixel tuples, never WorldPosition instances,
    since no CalibrationProfile exists until M2."""
    state = TrackState(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=1.0,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=[(0.0, 0.0, 0.0), (10.0, 0.0, 1.0)],
    )

    track = build_vehicle_track("cam-01", state)

    for sample in track.position_history:
        assert isinstance(sample, tuple)
        assert len(sample) == 3


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
