import pytest

from anvesh.safety.stopped_vehicle import (
    StoppedVehicleClassification,
    StoppedVehicleStatus,
    StoppedVehicleThresholds,
    build_safety_event,
    classify_stopped_vehicle,
)
from anvesh.storage.schemas import (
    ConfidenceTier,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
    WorldPosition,
)


def _image_track(positions, occlusion_state=OcclusionState.VISIBLE, track_id="cam-A:1"):
    """`positions`: list of (x, y, t) pixel tuples. `speed_estimate` is
    deliberately set to an unrelated value -- the detector must never
    read it (see stopped_vehicle.py's module docstring)."""
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-A",
        first_seen=positions[0][2],
        last_seen=positions[-1][2],
        vehicle_class=VehicleClass.CAR,
        occlusion_state=occlusion_state,
        position_history=list(positions),
        speed_estimate=Measurement(value=12345.0, error=0.0),
        motion_space=MotionSpace.IMAGE,
    )


def _world_track(positions, occlusion_state=OcclusionState.VISIBLE, track_id="cam-B:1"):
    """`positions`: list of (world_x, world_y, t) metre tuples."""
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-B",
        first_seen=positions[0][2],
        last_seen=positions[-1][2],
        vehicle_class=VehicleClass.CAR,
        occlusion_state=occlusion_state,
        position_history=[WorldPosition(world_x=x, world_y=y, timestamp=t) for (x, y, t) in positions],
        speed_estimate=Measurement(value=12345.0, error=0.0),
        motion_space=MotionSpace.WORLD,
    )


# ---------------------------------------------------------------------------
# 1. clearly moving
# ---------------------------------------------------------------------------


def test_clearly_moving_vehicle_is_moving():
    positions = [(20.0 * i, 0.0, float(i)) for i in range(6)]  # 20 px/s, well above the 5.0 px/s threshold
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.MOVING
    assert result.factors["dwell_seconds_observed"] == 0.0
    assert result.dwell_window is None


# ---------------------------------------------------------------------------
# 2. clearly stopped
# ---------------------------------------------------------------------------


def test_clearly_stopped_vehicle_is_stopped():
    positions = [(100.0, 100.0, i * 0.5) for i in range(8)]  # 8 samples, 0.0..3.5s, no movement
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.STOPPED
    assert result.dwell_window == (0.0, 3.5)
    assert result.factors["dwell_seconds_observed"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 3. short stop below dwell threshold
# ---------------------------------------------------------------------------


def test_short_stop_below_dwell_threshold_is_moving():
    positions = [
        (0.0, 0.0, 0.0),
        (20.0, 0.0, 1.0),   # fast: 20 px/s
        (40.0, 0.0, 2.0),   # fast: 20 px/s
        (41.0, 0.0, 2.75),  # slow: 1.33 px/s
        (42.0, 0.0, 3.5),   # slow: 1.33 px/s -- dwell run is only 1.5s, below dwell_seconds=3.0
        (62.0, 0.0, 4.5),   # fast again: 20 px/s
        (82.0, 0.0, 5.5),   # fast again: 20 px/s
    ]
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.MOVING
    assert result.factors["dwell_seconds_observed"] == pytest.approx(1.5)
    assert result.dwell_window is None


# ---------------------------------------------------------------------------
# 4. low-speed but moving
# ---------------------------------------------------------------------------


def test_low_speed_but_moving_is_moving_not_stopped():
    # 6 px/s throughout -- just above the default 5.0 px/s heuristic threshold.
    positions = [(6.0 * i, 0.0, float(i)) for i in range(6)]
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.MOVING
    assert result.factors["dwell_seconds_observed"] == 0.0


# ---------------------------------------------------------------------------
# 5. missing observations / large gap
# ---------------------------------------------------------------------------


def test_large_gap_drops_temporal_coverage_to_insufficient_evidence():
    positions = [
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.5),
        (10.0, 0.0, 1.0),
        # 8-second gap -- far larger than max_gap_seconds=1.0
        (10.0, 0.0, 9.0),
        (15.0, 0.0, 9.5),
        (20.0, 0.0, 10.0),
    ]
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.INSUFFICIENT_EVIDENCE
    assert "insufficient_temporal_coverage" in result.factors["reasons"]
    assert result.factors["temporal_coverage"] < 0.7
    assert result.heuristic_score == 0.0


# ---------------------------------------------------------------------------
# 6. track disappears after motion
# ---------------------------------------------------------------------------


def test_track_disappearing_after_motion_is_not_stopped():
    """A track ending in LOST right after clear motion must not be
    classified STOPPED merely because it disappeared."""
    positions = [(20.0 * i, 0.0, float(i)) for i in range(6)]
    track = _image_track(positions, occlusion_state=OcclusionState.LOST)
    result = classify_stopped_vehicle(track)
    assert result.status != StoppedVehicleStatus.STOPPED
    assert result.factors["occlusion_state"] == "lost"


def test_stopped_track_that_ends_lost_still_reports_stopped_with_explicit_caveat():
    """Unlike the case above, a track that genuinely satisfies the dwell
    requirement IS reported STOPPED even if it later goes LOST -- but the
    LOST ending must be surfaced as an explicit caveat, never silently
    dropped or treated as extra confirmation."""
    positions = [(100.0, 100.0, i * 0.5) for i in range(8)]
    track = _image_track(positions, occlusion_state=OcclusionState.LOST)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.STOPPED
    assert any("LOST" in reason for reason in result.factors["reasons"])


# ---------------------------------------------------------------------------
# 7. IMAGE-space stop (explicitly labeled heuristic)
# ---------------------------------------------------------------------------


def test_image_space_stop_uses_pixel_heuristic_threshold_and_is_labeled():
    positions = [(50.0, 50.0, i * 0.5) for i in range(8)]
    track = _image_track(positions)
    thresholds = StoppedVehicleThresholds(image_speed_threshold_px_per_s=2.0, world_speed_threshold_m_per_s=999.0)
    result = classify_stopped_vehicle(track, thresholds)
    assert result.motion_space == MotionSpace.IMAGE
    assert result.factors["motion_threshold_used"] == 2.0  # the IMAGE threshold, not the WORLD one
    assert result.status == StoppedVehicleStatus.STOPPED


# ---------------------------------------------------------------------------
# 8. WORLD-space stop (metric)
# ---------------------------------------------------------------------------


def test_world_space_stop_uses_metric_threshold():
    positions = [(10.0 + 0.05 * i, 20.0, i * 0.5) for i in range(8)]  # ~0.1 m/s, well under 0.5 m/s
    track = _world_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.motion_space == MotionSpace.WORLD
    assert result.factors["motion_threshold_used"] == 0.5
    assert result.status == StoppedVehicleStatus.STOPPED


# ---------------------------------------------------------------------------
# 9. insufficient observations
# ---------------------------------------------------------------------------


def test_insufficient_observations_yields_insufficient_evidence():
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)]  # 3 < default min_observations=5
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.INSUFFICIENT_EVIDENCE
    assert "insufficient_observations" in result.factors["reasons"]
    assert result.heuristic_score == 0.0
    assert result.confidence_tier == ConfidenceTier.LOW


# ---------------------------------------------------------------------------
# 10. insufficient temporal coverage (sparse sampling across the whole span)
# ---------------------------------------------------------------------------


def test_sparse_sampling_across_whole_span_yields_insufficient_evidence():
    positions = [(0.0, 0.0, t) for t in (0.0, 2.0, 4.0, 6.0, 8.0)]  # every gap > max_gap_seconds=1.0
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.INSUFFICIENT_EVIDENCE
    assert result.factors["temporal_coverage"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. deterministic repeated execution
# ---------------------------------------------------------------------------


def test_classification_is_deterministic_across_repeated_calls():
    positions = [(100.0, 100.0, i * 0.5) for i in range(8)]
    track = _image_track(positions)
    first = classify_stopped_vehicle(track)
    second = classify_stopped_vehicle(track)
    assert first == second


# ---------------------------------------------------------------------------
# 12. invalid configuration
# ---------------------------------------------------------------------------


def test_invalid_thresholds_raise_value_error():
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(min_observations=1)
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(min_temporal_coverage=1.5)
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(min_temporal_coverage=-0.1)
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(dwell_seconds=0.0)
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(max_gap_seconds=-1.0)
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(image_speed_threshold_px_per_s=-1.0)
    with pytest.raises(ValueError):
        StoppedVehicleThresholds(world_speed_threshold_m_per_s=-1.0)


# ---------------------------------------------------------------------------
# SafetyEvent creation
# ---------------------------------------------------------------------------


def test_safety_event_created_only_for_stopped():
    stopped_track = _image_track([(100.0, 100.0, i * 0.5) for i in range(8)])
    moving_track = _image_track([(20.0 * i, 0.0, float(i)) for i in range(6)])
    insufficient_track = _image_track([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)])

    stopped_result = classify_stopped_vehicle(stopped_track)
    moving_result = classify_stopped_vehicle(moving_track)
    insufficient_result = classify_stopped_vehicle(insufficient_track)

    assert build_safety_event(stopped_result) is not None
    assert build_safety_event(moving_result) is None
    assert build_safety_event(insufficient_result) is None


def test_safety_event_window_is_the_dwell_interval_not_the_full_track_lifetime():
    positions = [
        (0.0, 0.0, 0.0),
        (20.0, 0.0, 1.0),  # fast lead-in, NOT part of the dwell
        (20.0, 0.0, 1.5),
        (20.0, 0.0, 2.0),
        (20.0, 0.0, 2.5),
        (20.0, 0.0, 3.0),
        (20.0, 0.0, 3.5),
        (20.0, 0.0, 4.0),
        (20.0, 0.0, 4.5),
    ]
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    assert result.status == StoppedVehicleStatus.STOPPED
    event = build_safety_event(result)
    assert event is not None
    assert event.window == result.dwell_window
    assert event.window[0] > track.first_seen  # the fast lead-in must be excluded
    assert event.window[1] == track.last_seen
    assert event.event_type == "stopped_vehicle"
    assert event.track_id == track.track_id
    assert event.camera_id == track.camera_id


def test_safety_event_preserves_motion_space():
    world_positions = [(10.0 + 0.05 * i, 20.0, i * 0.5) for i in range(8)]
    track = _world_track(world_positions)
    result = classify_stopped_vehicle(track)
    event = build_safety_event(result)
    assert event.motion_space == MotionSpace.WORLD


# ---------------------------------------------------------------------------
# heuristic_score is explicitly non-probabilistic; factors/provenance preserved
# ---------------------------------------------------------------------------


def test_heuristic_score_is_documented_as_not_a_probability():
    assert "NOT a probability" in StoppedVehicleClassification.__doc__


def test_factors_expose_full_provenance():
    positions = [(100.0, 100.0, i * 0.5) for i in range(8)]
    track = _image_track(positions)
    result = classify_stopped_vehicle(track)
    expected_keys = {
        "observed_duration",
        "observation_count",
        "temporal_coverage",
        "min_temporal_coverage_required",
        "dwell_seconds_observed",
        "dwell_seconds_required",
        "mean_step_speed",
        "min_step_speed",
        "motion_threshold_used",
        "motion_space",
        "occlusion_state",
        "gap_count_beyond_max_gap_seconds",
        "gap_time_beyond_max_gap_seconds",
        "reasons",
    }
    assert expected_keys <= set(result.factors.keys())
