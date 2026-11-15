import pytest

from anvesh.perception.schema_conversion import build_camera_observation, build_vehicle_track
from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import (
    CameraObservation,
    GeometryValidity,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
    WorldPosition,
)
from anvesh.world.calibration import uncalibrated_profile
from anvesh.world.synthetic_fixture import fit_synthetic_validated_homography
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


def _ground_state(points, centroid_points=None, track_id="s000-t00001"):
    return TrackState(
        track_id=track_id,
        vehicle_class=VehicleClass.CAR,
        first_seen=points[0][2],
        last_seen=points[-1][2],
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=centroid_points if centroid_points is not None else [(0.0, 0.0, t) for _x, _y, t in points],
        ground_point_trajectory=points,
    )


# 5 ground points, 0.1 s apart, 5 px/step in x, inside the fitted
# homography's well-conditioned region (see tests/world/support.py).
_MOVING_GROUND_POINTS = [(100.0 + 5.0 * i, 200.0, 0.1 * i) for i in range(5)]


def test_vehicle_track_is_world_space_when_validated():
    """VALIDATED metric geometry (fit from surveyed metre correspondences)
    produces WorldPosition samples (metres) from the GROUND point, and a
    metric m/s speed from the single provenance-aware estimator -- the
    centroid history (deliberately all-zero here) must play no part."""
    import math

    profile = fit_synthetic_validated_homography("calib-01", "cam-01", make_reference_points())
    assert profile.geometry_validity == GeometryValidity.WORLD_VALIDATED
    state = _ground_state(_MOVING_GROUND_POINTS)

    track = build_vehicle_track("cam-01", state, calibration_profile=profile)

    assert track.motion_space == MotionSpace.WORLD
    assert track.geometry_validity == GeometryValidity.WORLD_VALIDATED
    assert track.speed_available is True
    assert len(track.position_history) == 5
    for sample in track.position_history:
        assert isinstance(sample, WorldPosition)
    expected_first = profile.image_to_world((100.0, 200.0))  # the ground point, not the (0, 0) centroid
    assert track.position_history[0].world_x == expected_first[0]

    projected = [profile.image_to_world((x, y)) for (x, y, _t) in _MOVING_GROUND_POINTS]
    distance = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(projected, projected[1:]))
    assert track.speed_estimate.value == pytest.approx(distance / 0.4)
    assert track.speed_estimate.value != 25.0  # not the raw pixel speed (5 px / 0.1 s = 50 px/s)


def test_unvalidated_calibration_never_yields_world_space_or_metres():
    """THE deleted path: a CALIBRATED but metric-UNVALIDATED profile (CityFlow
    supplied) used to project the centroid trajectory and label it m/s. It
    must now produce an IMAGE-space track carrying px/s, record
    WORLD_UNVALIDATED, and never apply the transform to the track."""
    import numpy as np

    from anvesh.perception.motion import compute_image_space_speed
    from anvesh.world.calibration import from_cityflow_homography

    profile = from_cityflow_homography("calib-cf", "cam-cf", np.diag([40.0, 40.0, 1.0]), reprojection_error=5.0)
    assert profile.is_calibrated  # a transform exists...
    assert profile.geometry_validity == GeometryValidity.WORLD_UNVALIDATED  # ...but is not trusted
    centroid = [(10.0 + 5.0 * i, 20.0, 0.1 * i) for i in range(5)]
    state = _ground_state(_MOVING_GROUND_POINTS, centroid_points=centroid)

    track = build_vehicle_track("cam-cf", state, calibration_profile=profile)

    assert track.motion_space == MotionSpace.IMAGE
    assert track.geometry_validity == GeometryValidity.WORLD_UNVALIDATED
    assert all(isinstance(p, tuple) and not isinstance(p, WorldPosition) for p in track.position_history)
    assert track.position_history == centroid  # untouched pixel samples
    expected_px = compute_image_space_speed(centroid)
    assert track.speed_estimate.value == expected_px.speed_px_per_s  # 50 px/s, NOT 40x that
    assert track.speed_estimate.value < 100.0


def test_validated_geometry_with_too_few_samples_has_no_speed_not_zero():
    """A 2-sample track has no measurable speed; it must say so
    (`speed_available=False`) rather than present a fabricated 0 m/s that a
    consumer would read as "stopped"."""
    profile = fit_synthetic_validated_homography("calib-01", "cam-01", make_reference_points())
    state = _ground_state(_MOVING_GROUND_POINTS[:2])

    track = build_vehicle_track("cam-01", state, calibration_profile=profile)

    assert track.motion_space == MotionSpace.WORLD
    assert track.speed_available is False


def test_validated_geometry_with_implausible_speed_has_no_speed():
    profile = fit_synthetic_validated_homography("calib-01", "cam-01", make_reference_points())
    wild = [(10.0 + 300.0 * i, 200.0, 0.1 * i) for i in range(5)]  # ~300 px/step * ~0.05 m/px / 0.1 s ~ 150 m/s
    state = _ground_state(wild)

    track = build_vehicle_track("cam-01", state, calibration_profile=profile)

    assert track.speed_available is False


def test_track_id_is_camera_prefixed_session_scoped_identity_not_raw_id():
    state = _ground_state(_MOVING_GROUND_POINTS, track_id="s002-t00041")
    state.raw_tracker_id = "7"
    track = build_vehicle_track("cam-01", state)
    assert track.track_id == "cam-01:s002-t00041"
    assert track.track_id.count(":") == 1  # the dashboard splits on ':' and takes the tail


def test_geometry_helpers_never_read_is_calibrated_as_metric():
    import numpy as np

    from anvesh.perception.schema_conversion import geometry_validity_for, motion_space_for
    from anvesh.world.calibration import from_cityflow_homography

    assert geometry_validity_for(None) == GeometryValidity.IMAGE
    assert motion_space_for(None) == MotionSpace.IMAGE
    cf = from_cityflow_homography("c", "cam", np.eye(3), reprojection_error=1.0)
    assert motion_space_for(cf) == MotionSpace.IMAGE
    assert geometry_validity_for(cf) == GeometryValidity.WORLD_UNVALIDATED
    survey = fit_synthetic_validated_homography("c", "cam", make_reference_points())
    assert motion_space_for(survey) == MotionSpace.WORLD


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
