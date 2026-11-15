import numpy as np
import pytest

from anvesh.perception.motion import RobustSpeedStatus
from anvesh.perception.speed_estimation import (
    DEFAULT_MAX_GAP_S,
    DEFAULT_MIN_ELAPSED_S,
    DEFAULT_MIN_SAMPLES,
    MAX_PLAUSIBLE_SPEED_KMH,
    SpeedConfidence,
    VehicleSpeedEstimate,
    estimate_vehicle_speed,
)
from anvesh.perception.track_manager import TrackManager, TrackState
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import GeometryValidity, Measurement, OcclusionState, VehicleClass
from anvesh.world.calibration import (
    CalibrationProvenance,
    MetricValidationStatus,
    from_cityflow_homography,
    uncalibrated_profile,
)
from anvesh.world.synthetic_fixture import fit_synthetic_validated_homography
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
    """Metric geometry for these tests: a SYNTHETIC-validated profile (the explicit opt-in). A plain
    `fit_planar_homography` is UNVALIDATED since hardening A1; no real survey exists in this repo."""
    return fit_synthetic_validated_homography("calib-ms", "cam-ms", make_reference_points())


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
    # M-A contract change: an UNVALIDATED (CityFlow) transform yields its
    # displacement/time ONLY as unvalidated world-frame units/s -- never a
    # km/h `measurement`. (Previously asserted `measurement.value == 36 km/h`.)
    assert estimate.measurement is None
    assert estimate.metric_speed_kmh is None
    assert isinstance(estimate.unvalidated_world_speed, Measurement)
    assert estimate.unvalidated_world_speed.value == pytest.approx(EXPECTED_KMH / 3.6)
    assert estimate.geometry_validity == GeometryValidity.WORLD_UNVALIDATED


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
    """Contrast case: a profile that carries a validation record (here the SYNTHETIC test opt-in --
    no real survey exists in this repo) DOES reach VALIDATED -- proving APPROXIMATE isn't just a
    hardcoded universal default, it specifically tracks metric_validation."""
    profile = _manual_survey_profile()
    # Build a straight-line ground_point_trajectory inside the fitted
    # homography's well-conditioned region (see tests/world/support.py).
    points = [(10.0, 10.0, 0.0), (60.0, 10.0, 1.0), (110.0, 10.0, 2.0)]
    state = _state_with_ground_points(points)

    estimate = estimate_vehicle_speed("cam-ms", state, calibration_profile=profile, min_elapsed_s=0.0, max_gap_s=None)

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.VALIDATED
    assert estimate.geometry_validity == GeometryValidity.WORLD_VALIDATED
    assert estimate.metric_speed_kmh is not None  # the one sanctioned km/h accessor


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
    assert estimate.unvalidated_world_speed.value == pytest.approx(EXPECTED_KMH / 3.6)
    assert estimate.unvalidated_world_speed.value != pytest.approx(0.0)


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


def test_vehicle_speed_estimate_available_must_carry_a_value():
    with pytest.raises(ValueError):
        VehicleSpeedEstimate(
            track_id="cam:1",
            camera_id="cam",
            status=RobustSpeedStatus.OK,
            confidence=SpeedConfidence.APPROXIMATE,
            measurement=None,
        )


def test_approximate_estimate_can_never_carry_a_kmh_measurement():
    """Structural M-A guard: an APPROXIMATE (unvalidated-geometry) estimate
    is non-metric by definition, so a km/h `measurement` is rejected at
    construction -- a bug elsewhere cannot smuggle one in."""
    with pytest.raises(ValueError):
        VehicleSpeedEstimate(
            track_id="cam:1", camera_id="cam", status=RobustSpeedStatus.OK,
            confidence=SpeedConfidence.APPROXIMATE, measurement=Measurement(value=34.0, error=1.0),
        )
    with pytest.raises(ValueError):  # a km/h value needs validated geometry
        VehicleSpeedEstimate(
            track_id="cam:1", camera_id="cam", status=RobustSpeedStatus.OK,
            confidence=SpeedConfidence.OUTLIER, measurement=Measurement(value=3400.0, error=1.0),
            geometry_validity=GeometryValidity.WORLD_UNVALIDATED,
        )


def test_metric_speed_kmh_is_none_unless_validated():
    approx_est = VehicleSpeedEstimate(
        track_id="cam:1", camera_id="cam", status=RobustSpeedStatus.OK,
        confidence=SpeedConfidence.APPROXIMATE, measurement=None,
        unvalidated_world_speed=Measurement(value=10.0, error=1.0),
    )
    assert approx_est.metric_speed_kmh is None
    validated = VehicleSpeedEstimate(
        track_id="cam:1", camera_id="cam", status=RobustSpeedStatus.OK,
        confidence=SpeedConfidence.VALIDATED, measurement=Measurement(value=40.0, error=1.0), geometry_validity=GeometryValidity.WORLD_VALIDATED,
    )
    assert validated.metric_speed_kmh.value == 40.0
    outlier_validated_geometry = VehicleSpeedEstimate(
        track_id="cam:1", camera_id="cam", status=RobustSpeedStatus.OK,
        confidence=SpeedConfidence.OUTLIER, measurement=Measurement(value=4000.0, error=1.0), geometry_validity=GeometryValidity.WORLD_VALIDATED,
    )
    assert outlier_validated_geometry.metric_speed_kmh is None  # implausible -> never consumable


# --- requirement 13: integration test ---


def test_integration_detection_through_track_manager_to_approximate_speed():
    """Full path: RawDetection -> ground_point -> TrackedDetection ->
    TrackManager.update() (accumulating ground_point_trajectory) ->
    estimate_vehicle_speed with a CityFlow-style calibration -> an
    APPROXIMATE, NON-metric unvalidated world-unit speed."""
    manager = TrackManager(camera_id="s02_c006")

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

    # identity is the manager-assigned session-scoped id, not the raw "42"
    assert estimate.track_id == f"s02_c006:{state.track_id}"
    assert state.raw_tracker_id == "42"
    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.APPROXIMATE
    assert estimate.measurement is None  # unvalidated geometry: no km/h
    assert estimate.unvalidated_world_speed.value == pytest.approx(EXPECTED_KMH / 3.6)
    assert estimate.unvalidated_world_speed.error >= 0.0


# ---------------------------------------------------------------------------
# c006 speed-validation pass: physical-plausibility gate
# (ANVESH_C006_SPEED_VALIDATION_AUDIT.md) -- Tests 1-6 of that audit's
# regression requirements. Test 7 (lane assignment unchanged), Test 8
# (candidate resolution unchanged), Test 9 (real-video pipeline), and
# Test 10 (synthetic propagation/fusion) are covered by the existing,
# untouched suites for those modules -- nothing in this file's scope
# changes any of them.
# ---------------------------------------------------------------------------


# --- Test 1: normal, plausible movement produces a valid APPROXIMATE speed ---


def test_plausible_moderate_speed_stays_approximate_not_outlier():
    """A realistic ~50 km/h reading, comfortably under
    MAX_PLAUSIBLE_SPEED_KMH, must stay APPROXIMATE -- the gate must never
    downgrade an ordinary plausible speed."""
    # 2.0 pixel-unit steps, 0.1s apart -> 20 units/s in pixel space ->
    # scaled 2x -> 40 world-units/s -> 144.0 km/h (still < 200).
    points = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.1),
        (4.0, 0.0, 0.2),
        (6.0, 0.0, 0.3),
        (8.0, 0.0, 0.4),
    ]
    state = _state_with_ground_points(points)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.measurement is None
    assert estimate.unvalidated_world_speed.value * 3.6 == pytest.approx(144.0)  # 40 units/s read-as-m/s = 144
    assert estimate.unvalidated_world_speed.value * 3.6 < MAX_PLAUSIBLE_SPEED_KMH
    assert estimate.confidence == SpeedConfidence.APPROXIMATE


# --- Test 2: extreme displacement is rejected as OUTLIER, not shown as a number ---


def test_extreme_displacement_is_rejected_as_outlier():
    """The exact failure mode behind the observed c006 '~2084 km/h'
    value: a mathematically clean (RobustSpeedStatus.OK), well-sampled
    trajectory whose derived speed is nonetheless far beyond anything a
    road vehicle can do. Must become OUTLIER, never APPROXIMATE -- and
    the implausible value must still be carried for diagnostics, never
    discarded."""
    # 200 pixel-unit steps, 0.1s apart -> 2000 units/s in pixel space ->
    # scaled 2x -> 4000 world-units/s -> 14400.0 km/h.
    points = [
        (0.0, 0.0, 0.0),
        (200.0, 0.0, 0.1),
        (400.0, 0.0, 0.2),
        (600.0, 0.0, 0.3),
        (800.0, 0.0, 0.4),
    ]
    state = _state_with_ground_points(points)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK  # the math itself is fine -- this is a plausibility failure
    assert estimate.confidence == SpeedConfidence.OUTLIER
    assert estimate.confidence != SpeedConfidence.APPROXIMATE
    assert estimate.confidence != SpeedConfidence.VALIDATED
    # diagnostics: the rejected value is not discarded -- but under unvalidated
    # geometry it lives ONLY in the non-metric field, never as km/h
    assert estimate.measurement is None
    assert estimate.unvalidated_world_speed.value * 3.6 > MAX_PLAUSIBLE_SPEED_KMH
    assert estimate.metric_speed_kmh is None


def test_outlier_gate_applies_even_to_a_metric_validated_calibration():
    """A physically implausible value cannot simultaneously be an
    independently-confirmed metric observation -- OUTLIER must win over
    VALIDATED, not just over APPROXIMATE."""
    profile = _manual_survey_profile()
    # Reuses the well-conditioned region from
    # test_manual_survey_calibration_produces_validated_confidence, but
    # with a 1000x larger step so the derived speed is implausible.
    points = [(10.0, 10.0, 0.0), (50010.0, 10.0, 1.0), (100010.0, 10.0, 2.0)]
    state = _state_with_ground_points(points)

    estimate = estimate_vehicle_speed(
        "cam-ms", state, calibration_profile=profile, min_elapsed_s=0.0, max_gap_s=None
    )

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.OUTLIER
    assert estimate.confidence != SpeedConfidence.VALIDATED
    assert estimate.measurement is not None  # validated geometry: the (rejected) km/h is carried
    assert estimate.metric_speed_kmh is None  # ...but is never consumable


# --- Test 3: a track gap must not be read as continuous high-speed motion ---


def test_track_gap_does_not_produce_a_huge_speed():
    """Two short, closely-observed clusters separated by one large
    temporal gap. The gap step (well beyond DEFAULT_MAX_GAP_S) must be
    excluded from the distance/time totals -- exactly like
    `motion.compute_robust_speed`'s own gap tests already prove at the
    unit-agnostic layer -- so the SAME behavior must hold end-to-end
    through `estimate_vehicle_speed`'s km/h-labeled output: a naive
    "first point to last point" measurement here would report an
    enormous cross-gap speed; the correct, gap-aware measurement must
    not."""
    points = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.1),
        (1.0, 0.0, 0.2),
        # a 4.8s gap -- far beyond DEFAULT_MAX_GAP_S (0.6s) -- to a point
        # 500 pixel-units away; if this step were NOT excluded, the
        # resulting speed would be wildly implausible.
        (500.0, 0.0, 5.0),
        (500.5, 0.0, 5.1),
        (501.0, 0.0, 5.2),
    ]
    state = _state_with_ground_points(points)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK
    # Only the two small, 0.5-pixel-unit/0.1s clusters contribute -- same
    # per-step math as EXPECTED_KMH's 0.5-unit/0.1s steps.
    assert estimate.unvalidated_world_speed.value * 3.6 == pytest.approx(EXPECTED_KMH)
    assert estimate.confidence == SpeedConfidence.APPROXIMATE
    assert estimate.unvalidated_world_speed.value * 3.6 < MAX_PLAUSIBLE_SPEED_KMH


def test_track_gap_leaving_no_usable_steps_is_unavailable_not_huge():
    """When excluding the gap leaves an insufficient sample, the result
    must be UNAVAILABLE -- never a fabricated speed from the excluded
    long-distance step."""
    points = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.1),
        # a single enormous gap step -- excluded, leaving only 1 usable
        # step, below DEFAULT_MIN_SAMPLES worth of trajectory.
        (10000.0, 0.0, 10.0),
    ]
    state = _state_with_ground_points(points)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.confidence == SpeedConfidence.UNAVAILABLE
    assert estimate.measurement is None


# --- Test 4: ground-point jitter must not produce an absurd speed ---


def test_ground_point_jitter_under_a_steep_scale_is_rejected_as_outlier():
    """Reproduces the actual c006 root cause directly: an essentially
    STATIONARY vehicle (real displacement ~0) whose ground-point
    detections jitter by only 1-2 pixels frame-to-frame (ordinary
    bounding-box detector noise, not a tracking defect), observed through
    a homography whose per-pixel world-scale is steep -- exactly the
    measured c006 characteristic (several world-units per image pixel,
    growing sharply toward the vanishing line; see the module docstring's
    "PHYSICAL PLAUSIBILITY GATE" section). Must be gated as OUTLIER, never
    silently reported as a normal APPROXIMATE speed."""
    steep_homography = np.array([[80.0, 0.0, 0.0], [0.0, 80.0, 0.0], [0.0, 0.0, 1.0]])
    steep_profile = from_cityflow_homography("calib-steep", "cam-steep", steep_homography, reprojection_error=5.0)

    # Alternating +/-1.5 pixel-unit jitter around a fixed point -- a
    # believable detector-noise pattern for a parked/stationary vehicle,
    # never a deliberate large displacement.
    jitter_points = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.1),
        (0.0, 1.5, 0.2),
        (-1.5, 0.0, 0.3),
        (0.0, -1.5, 0.4),
    ]
    state = _state_with_ground_points(jitter_points)
    estimate = estimate_vehicle_speed("cam-steep", state, calibration_profile=steep_profile)

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.OUTLIER
    assert estimate.unvalidated_world_speed.value * 3.6 > MAX_PLAUSIBLE_SPEED_KMH


# --- Test 6: the gate leaves previously-existing plausible-speed behavior unchanged ---


def test_gate_does_not_change_the_original_well_established_case():
    """Regression pin for the exact fixture
    `test_cityflow_homography_transformation_produces_expected_speed` used
    before this gate existed -- the physical-plausibility check must be a
    pure addition, never altering a speed that was already well within
    plausible bounds."""
    state = _state_with_ground_points(STRAIGHT_LINE_GROUND_POINTS)
    estimate = estimate_vehicle_speed("cam-cf", state, calibration_profile=_cityflow_profile())

    assert estimate.status == RobustSpeedStatus.OK
    assert estimate.confidence == SpeedConfidence.APPROXIMATE
    assert estimate.unvalidated_world_speed.value * 3.6 == pytest.approx(EXPECTED_KMH)
