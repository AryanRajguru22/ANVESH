import pytest

from anvesh.evidence.reliability import ReliabilityScore, ReliabilityWeights, compute_reliability
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.schemas import Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack


def _track(track_id, first_seen, last_seen, occlusion_state=OcclusionState.VISIBLE, motion_space=MotionSpace.WORLD):
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-A",
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=occlusion_state,
        position_history=[],
        speed_estimate=Measurement(value=5.0, error=0.0),
        motion_space=motion_space,
    )


def test_full_coverage_calibrated_scores_high():
    tracks = [_track("1", 0.0, 10.0), _track("2", 0.0, 10.0), _track("3", 0.0, 10.0)]
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())

    score = compute_reliability("cam-A", 0.0, 10.0, tracks, aggregation)

    assert isinstance(score, ReliabilityScore)
    assert score.score == pytest.approx(1.0)
    assert score.factors["calibration"] == 1.0
    assert score.factors["tracking_continuity"] == 1.0
    assert score.factors["temporal_coverage"] == 1.0


def test_no_tracks_scores_only_the_calibration_factor():
    """With zero tracks, observation/continuity/coverage all bottom out at
    0 -- but calibration is a fact about the camera, not the window, so it
    still contributes its full weight even with no evidence this window."""
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, [], MotionSpace.WORLD, CongestionStateTracker())
    score = compute_reliability("cam-A", 0.0, 10.0, [], aggregation)
    assert score.factors["observation_count"] == 0.0
    assert score.factors["tracking_continuity"] == 0.0
    assert score.factors["temporal_coverage"] == 0.0
    assert score.factors["calibration"] == 1.0
    assert score.score == pytest.approx(0.25)


def test_partial_occlusion_lowers_continuity_factor():
    tracks = [
        _track("1", 0.0, 10.0, occlusion_state=OcclusionState.VISIBLE),
        _track("2", 0.0, 10.0, occlusion_state=OcclusionState.PARTIAL),
    ]
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    score = compute_reliability("cam-A", 0.0, 10.0, tracks, aggregation)
    assert score.factors["tracking_continuity"] == 0.5


def test_uncalibrated_camera_scores_lower_calibration_factor():
    tracks = [_track("1", 0.0, 10.0, motion_space=MotionSpace.IMAGE)]
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.IMAGE, CongestionStateTracker())
    score = compute_reliability("cam-A", 0.0, 10.0, tracks, aggregation)
    assert score.factors["calibration"] == 0.5


def test_sparse_temporal_coverage_scores_lower():
    # one track only covers 1 of the 10-second window -> low coverage
    tracks = [_track("1", 4.0, 5.0)]
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    score = compute_reliability("cam-A", 0.0, 10.0, tracks, aggregation)
    assert score.factors["temporal_coverage"] == pytest.approx(0.1)


def test_observation_count_saturates():
    tracks = [_track(str(i), 0.0, 10.0) for i in range(10)]
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    score = compute_reliability("cam-A", 0.0, 10.0, tracks, aggregation, saturating_observation_count=3)
    assert score.factors["observation_count"] == 1.0  # capped, not 10/3


def test_score_is_never_presented_as_a_probability_type():
    """Guards against accidentally renaming/reinterpreting `score` as a
    probability elsewhere -- it must stay a plain heuristic float."""
    tracks = [_track("1", 0.0, 10.0)]
    aggregation = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    score = compute_reliability("cam-A", 0.0, 10.0, tracks, aggregation)
    assert isinstance(score.score, float)
    assert score.method == "weighted_heuristic_v1"
    assert not hasattr(score, "probability")


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ReliabilityWeights(observation_count=0.5, tracking_continuity=0.5, calibration=0.5, temporal_coverage=0.5)


def test_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        ReliabilityScore(camera_id="cam-A", window_start=0.0, window_end=1.0, score=1.5, factors={})


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        ReliabilityScore(camera_id="cam-A", window_start=5.0, window_end=1.0, score=0.5, factors={})
