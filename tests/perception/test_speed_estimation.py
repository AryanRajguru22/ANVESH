import numpy as np
import pytest

from anvesh.perception.motion import RobustSpeedStatus
from anvesh.perception.speed_estimation import (
    DEFAULT_MAX_GAP_S,
    DEFAULT_MIN_ELAPSED_S,
    DEFAULT_MIN_SAMPLES,
    SpeedConfidence,
    VehicleSpeedEstimate,
    estimate_vehicle_speed,
)
from anvesh.perception.track_manager import TrackManager, TrackState
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import Measurement, OcclusionState, VehicleClass
from anvesh.world.calibration import (
    CalibrationProvenance,
    MetricValidationStatus,
    fit_planar_homography,
    from_cityflow_homography,
    uncalibrated_profile,
)
from tests.world.support import make_reference_points

assert DEFAULT_MIN_SAMPLES == 3
assert DEFAULT_MIN_ELAPSED_S == pytest.approx(0.3)
assert DEFAULT_MAX_GAP_S == pytest.approx(0.6)

# A simple, hand-verifiable "CityFlow-style" homography: scales (x, y) by
# 2 with no rotation/translation. Deliberately NOT the real c006/c009
# matrix (requirement 14 -- no dependency on the real dataset; a synthetic
# fixture with easy-to-predict math is preferable for these tests).
SCALE_2X_HOMOGRAPHY = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])


def _cityflow_profile():
    return from_cityflow_homography("calib-cf", "cam-cf", SCALE_2X_HOMOGRAPHY, reprojection_error=5.0)


def _manual_survey_profile():
    return fit_planar_homography("calib-ms", "cam-ms", make_reference_points())


def _state_with_ground_points(points, track_id="1", centroid_points=None):
    """A TrackState with an explicit ground_point_trajectory, and a
    (usually different) centroid trajectory -- so tests can prove which
    one estimate_vehicle_speed actually reads."""
    centroid_points = centroid_points if centroid_points is not None else [(0.0, 0.0, t) for _, _, t in points]
    return TrackState(
        track_id=track_id,
        vehicle_class=VehicleClass.CAR,
        first_seen=points[0][2],
        last_seen=points[-1][2],
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=centroid_points,
        ground_point_trajectory=points,
    )


# 5 ground points, 0.1s apart (matches CityFlow's 10fps), 0.5 pixel-unit
# steps -> 5 units/s in pixel space. Scaled 2x by SCALE_2X_HOMOGRAPHY ->
# 10 "world" units/s -> 36.0 km/h. Spans ~0.4s total -- comfortably above
# DEFAULT_MIN_ELAPSED_S (0.3s) even accounting for float summation error
# (0.1 + 0.1 + 0.1 alone lands a hair below 3*0.1 in IEEE 754).
STRAIGHT_LINE_GROUND_POINTS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.1),
    (1.0, 0.0, 0.2),
    (1.5, 0.0, 0.3),
    (2.0, 0.0, 0.4),
]
EXPECTED_KMH = 36.0


# --- requirement 12i: missing calibration ---


def test_missing_calibration_is_unavailable():
    state = _state_with_ground_points(STRAIGHT_LINE_GROUND_POINTS)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=None)

    assert estimate.confidence == SpeedConfidence.UNAVAILABLE
    assert estimate.status == RobustSpeedStatus.CALIBRATION_UNAVAILABLE
    assert estimate.measurement is None


def test_explicitly_uncalibrated_profile_is_unavailable():
    state = _state_with_ground_points(STRAIGHT_LINE_GROUND_POINTS)
    profile = uncalibrated_profile("calib-none", "cam-cf")
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=profile)

    assert estimate.confidence == SpeedConfidence.UNAVAILABLE
    assert estimate.status == RobustSpeedStatus.CALIBRATION_UNAVAILABLE


# --- requirement 12j: CityFlow homography transformation ---


def test_cityflow_homography_transformation_produces_expected_speed():
    state = _state_with_ground_points(STRAIGHT_LINE_GROUND_POINTS)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK
    assert isinstance(estimate.measurement, Measurement)
    assert estimate.measurement.value == pytest.approx(EXPECTED_KMH)


# --- requirements 12k/12l/12m: CityFlow provenance/validation/confidence ---


def test_cityflow_profile_provenance_remains_cityflow_supplied():
    profile = _cityflow_profile()
    assert profile.provenance == CalibrationProvenance.CITYFLOW_SUPPLIED


def test_cityflow_profile_metric_validation_remains_unvalidated():
    profile = _cityflow_profile()
    assert profile.metric_validation == MetricValidationStatus.UNVALIDATED


def test_speed_cannot_become_validated_merely_because_homography_exists():
    """The core rule this commit exists to enforce: a mathematically
    clean CityFlow-derived speed must still be APPROXIMATE, never
    VALIDATED -- requirements 5/6."""
    state = _state_with_ground_points(STRAIGHT_LINE_GROUND_POINTS)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.APPROXIMATE
    assert estimate.confidence != SpeedConfidence.VALIDATED


def test_manual_survey_calibration_produces_validated_confidence():
    """Contrast case: a real, independently-measured survey DOES reach
    VALIDATED -- proving APPROXIMATE isn't just a hardcoded universal
    default, it specifically tracks metric_validation."""
    profile = _manual_survey_profile()
    # Build a straight-line ground_point_trajectory inside the fitted
    # homography's well-conditioned region (see tests/world/support.py).
    points = [(10.0, 10.0, 0.0), (60.0, 10.0, 1.0), (110.0, 10.0, 2.0)]
    state = _state_with_ground_points(points)

    estimate = estimate_vehicle_speed("cam-ms", state, calibration_profile=profile, min_elapsed_s=0.0, max_gap_s=None)

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.VALIDATED


# --- requirement 12n: ground-point trajectory used, not centroid ---


def test_ground_point_trajectory_used_not_centroid():
    """centroid `trajectory` is all-zero (stationary); ground_point_
    trajectory moves -- the estimate must reflect movement, proving it
    was computed from ground_point_trajectory."""
    centroid_timestamps = (0.0, 0.1, 0.2, 0.3, 0.4)
    state = _state_with_ground_points(
        STRAIGHT_LINE_GROUND_POINTS, centroid_points=[(0.0, 0.0, t) for t in centroid_timestamps]
    )
    assert state.trajectory == [(0.0, 0.0, t) for t in centroid_timestamps]

    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.measurement.value == pytest.approx(EXPECTED_KMH)
    assert estimate.measurement.value != pytest.approx(0.0)


# --- VehicleSpeedEstimate validation ---


def test_vehicle_speed_estimate_unavailable_must_not_carry_measurement():
    with pytest.raises(ValueError):
        VehicleSpeedEstimate(
            track_id="cam:1",
            camera_id="cam",
            status=RobustSpeedStatus.CALIBRATION_UNAVAILABLE,
            confidence=SpeedConfidence.UNAVAILABLE,
            measurement=Measurement(value=10.0, error=1.0),
        )


def test_vehicle_speed_estimate_available_must_carry_measurement():
    with pytest.raises(ValueError):
        VehicleSpeedEstimate(
            track_id="cam:1",
            camera_id="cam",
            status=RobustSpeedStatus.OK,
            confidence=SpeedConfidence.APPROXIMATE,
            measurement=None,
        )


# --- requirement 13: integration test ---


def test_integration_detection_through_track_manager_to_approximate_speed():
    """Full path: RawDetection -> ground_point -> TrackedDetection ->
    TrackManager.update() (accumulating ground_point_trajectory) ->
    estimate_vehicle_speed with a CityFlow-style calibration -> an
    APPROXIMATE km/h measurement."""
    manager = TrackManager(max_missed_frames=5, partial_after_missed_frames=1)

    # Ground point is bottom-center: ((x1+x2)/2, y2). Four frames, 0.1s
    # apart, moving 0.5 px/frame in x -> ground_point x sequence
    # 0.0, 0.5, 1.0, 1.5, matching STRAIGHT_LINE_GROUND_POINTS above.
    for i, (gx, _gy, t) in enumerate(STRAIGHT_LINE_GROUND_POINTS):
        detection = RawDetection(
            frame_index=i,
            timestamp=t,
            class_name="car",
            vehicle_class=VehicleClass.CAR,
            confidence=0.9,
            bbox_xyxy=(gx - 5.0, 0.0, gx + 5.0, 10.0),  # ground_point = ((x1+x2)/2, y2) = (gx, 10.0)
        )
        manager.update(t, [TrackedDetection(detection=detection, track_id="42")])

    manager.finalize()
    state = manager.finished_tracks[0]

    # ground_point y is fixed at 10.0 for every sample (constant offset,
    # doesn't affect x-direction speed math); only x actually moves.
    assert [round(x, 3) for x, _y, _t in state.ground_point_trajectory] == [0.0, 0.5, 1.0, 1.5, 2.0]

    estimate = estimate_vehicle_speed("s02_c006", state, calibration_profile=_cityflow_profile())

    assert estimate.track_id == "s02_c006:42"
    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.APPROXIMATE
    assert estimate.measurement.value == pytest.approx(EXPECTED_KMH)
    assert estimate.measurement.error >= 0.0
