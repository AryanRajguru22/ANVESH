import statistics

import pytest

from anvesh.perception.traffic_state import (
    CongestionStateTracker,
    CongestionThresholds,
    DataQuality,
    aggregate_traffic_state,
)
from anvesh.storage.schemas import GeometryValidity
from anvesh.storage.schemas import (
    CongestionLevel,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
)


def _track(
    track_id,
    first_seen,
    last_seen,
    speed_value,
    vehicle_class=VehicleClass.CAR,
    occlusion_state=OcclusionState.VISIBLE,
    motion_space=MotionSpace.WORLD,
    speed_error=0.0,
):
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-A",
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=vehicle_class,
        occlusion_state=occlusion_state,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=speed_error),
        motion_space=motion_space, geometry_validity=(GeometryValidity.WORLD_VALIDATED if motion_space == MotionSpace.WORLD else GeometryValidity.IMAGE),
    )


def _tracker():
    return CongestionStateTracker()


# ---------------------------------------------------------------------------
# Aggregation: counts, speed, class counts
# ---------------------------------------------------------------------------


def test_aggregate_basic_counts_and_speed():
    tracks = [
        _track("1", 0.0, 5.0, speed_value=10.0),
        _track("2", 1.0, 6.0, speed_value=12.0, vehicle_class=VehicleClass.TRUCK),
    ]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())

    ts = result.traffic_state
    assert ts.camera_id == "cam-A"
    assert ts.vehicle_count == 2
    assert ts.mean_speed.value == pytest.approx(11.0)
    assert result.class_counts == {VehicleClass.CAR: 1, VehicleClass.TRUCK: 1}
    assert set(result.contributing_track_ids) == {"1", "2"}
    assert result.speed_quality == DataQuality.OBSERVED


def test_speed_aggregation_matches_statistics_module():
    speeds = [10.0, 12.0, 14.0]
    tracks = [_track(str(i), 0.0, 5.0, speed_value=s) for i, s in enumerate(speeds)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())

    assert result.traffic_state.mean_speed.value == pytest.approx(statistics.fmean(speeds))
    assert result.traffic_state.mean_speed.error == pytest.approx(statistics.pstdev(speeds))


def test_single_track_uses_its_own_speed_error():
    tracks = [_track("1", 0.0, 5.0, speed_value=9.0, speed_error=1.5)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())

    assert result.traffic_state.vehicle_count == 1
    assert result.traffic_state.mean_speed.value == 9.0
    assert result.traffic_state.mean_speed.error == 1.5
    assert result.speed_quality == DataQuality.OBSERVED


# ---------------------------------------------------------------------------
# Vehicle counting: window overlap
# ---------------------------------------------------------------------------


def test_vehicle_counting_excludes_tracks_outside_window():
    tracks = [
        _track("in", 2.0, 4.0, speed_value=10.0),
        _track("before", -5.0, -1.0, speed_value=10.0),
        _track("after", 20.0, 25.0, speed_value=10.0),
    ]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())

    assert result.traffic_state.vehicle_count == 1
    assert result.contributing_track_ids == ("in",)


def test_vehicle_counting_includes_partially_overlapping_tracks():
    # track spans [-2, 3], window is [0, 10] -- partial overlap still counts
    tracks = [_track("partial", -2.0, 3.0, speed_value=10.0)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())
    assert result.traffic_state.vehicle_count == 1


# ---------------------------------------------------------------------------
# Sparse / missing observations
# ---------------------------------------------------------------------------


def test_sparse_observations_single_vehicle():
    tracks = [_track("1", 0.0, 1.0, speed_value=5.0)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())
    assert result.traffic_state.vehicle_count == 1
    assert result.traffic_state.occupancy > 0.0


def test_missing_observations_zero_tracks_marks_speed_unavailable():
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, [], MotionSpace.WORLD, _tracker())

    ts = result.traffic_state
    assert ts.vehicle_count == 0
    assert ts.mean_speed.value == 0.0
    assert ts.occupancy == 0.0
    assert ts.density == 0.0
    assert ts.flow_rate == 0.0
    assert result.speed_quality == DataQuality.UNAVAILABLE
    # zero vehicles is a genuine observation (empty road), not "unavailable" data
    assert result.occupancy_quality == DataQuality.OBSERVED
    assert result.density_quality == DataQuality.OBSERVED


def test_empty_road_is_never_misread_as_congested():
    """Regression: a 0.0 m/s *placeholder* over zero vehicles must not be
    fed through the same speed threshold as a genuine 0.0 m/s over present
    (gridlocked) vehicles -- an empty road is free-flowing, not congested."""
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, [], MotionSpace.WORLD, _tracker())
    assert result.traffic_state.congestion_level == CongestionLevel.FREE_FLOW


def test_empty_windows_do_not_accumulate_false_congestion_over_time():
    tracker = _tracker()
    for _ in range(5):
        result = aggregate_traffic_state("cam-A", 0.0, 10.0, [], MotionSpace.WORLD, tracker)
        assert result.traffic_state.congestion_level == CongestionLevel.FREE_FLOW


def test_invalid_window_raises_value_error():
    with pytest.raises(ValueError):
        aggregate_traffic_state("cam-A", 10.0, 0.0, [], MotionSpace.WORLD, _tracker())


# ---------------------------------------------------------------------------
# Density: proxy vs real, per motion_space and segment_length_m
# ---------------------------------------------------------------------------


def test_density_is_real_when_calibrated_and_segment_length_given():
    tracks = [_track(str(i), 0.0, 5.0, speed_value=10.0) for i in range(4)]
    result = aggregate_traffic_state(
        "cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker(), segment_length_m=20.0
    )
    assert result.traffic_state.density == pytest.approx(4 / 20.0)
    assert result.density_quality == DataQuality.OBSERVED


def test_density_is_a_labeled_proxy_without_segment_length():
    tracks = [_track(str(i), 0.0, 5.0, speed_value=10.0) for i in range(4)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, _tracker())
    assert result.traffic_state.density == 4.0  # raw count proxy, not vehicles/metre
    assert result.density_quality == DataQuality.ESTIMATED


def test_density_is_a_proxy_when_uncalibrated_even_with_segment_length():
    tracks = [_track(str(i), 0.0, 5.0, speed_value=10.0, motion_space=MotionSpace.IMAGE) for i in range(4)]
    result = aggregate_traffic_state(
        "cam-A", 0.0, 10.0, tracks, MotionSpace.IMAGE, _tracker(), segment_length_m=20.0
    )
    assert result.density_quality == DataQuality.ESTIMATED


# ---------------------------------------------------------------------------
# Congestion state machine / hysteresis
# ---------------------------------------------------------------------------


def test_free_flow_stays_free_under_light_traffic():
    tracker = CongestionStateTracker()
    level = tracker.update(mean_speed_value=10.0, occupancy=0.1, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.FREE_FLOW


def test_single_noisy_window_does_not_flip_straight_to_congested():
    """The 'one noisy frame shouldn't flip state' regression."""
    tracker = CongestionStateTracker()
    assert tracker.state == CongestionLevel.FREE_FLOW

    # one bad window: low speed, high occupancy
    level = tracker.update(mean_speed_value=1.0, occupancy=0.9, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.BUILDING  # not straight to CONGESTED

    # immediately clears -- must revert, not have "locked in" congestion
    level = tracker.update(mean_speed_value=10.0, occupancy=0.1, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.FREE_FLOW


def test_sustained_pressure_confirms_congested():
    tracker = CongestionStateTracker()
    tracker.update(mean_speed_value=1.0, occupancy=0.9, motion_space=MotionSpace.WORLD)  # -> BUILDING
    level = tracker.update(mean_speed_value=1.0, occupancy=0.9, motion_space=MotionSpace.WORLD)  # confirmed
    assert level == CongestionLevel.CONGESTED


def test_congestion_clearing_requires_confirmation():
    tracker = CongestionStateTracker()
    tracker.update(1.0, 0.9, MotionSpace.WORLD)  # BUILDING
    tracker.update(1.0, 0.9, MotionSpace.WORLD)  # CONGESTED
    assert tracker.state == CongestionLevel.CONGESTED

    level = tracker.update(mean_speed_value=9.0, occupancy=0.2, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.DISSIPATING  # not immediately FREE_FLOW

    level = tracker.update(mean_speed_value=9.0, occupancy=0.2, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.FREE_FLOW  # confirmed clearing


def test_dissipating_can_reconfirm_back_to_congested():
    tracker = CongestionStateTracker()
    tracker.update(1.0, 0.9, MotionSpace.WORLD)  # BUILDING
    tracker.update(1.0, 0.9, MotionSpace.WORLD)  # CONGESTED
    tracker.update(9.0, 0.2, MotionSpace.WORLD)  # DISSIPATING
    level = tracker.update(mean_speed_value=1.0, occupancy=0.9, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.CONGESTED


def test_image_space_classification_uses_occupancy_only():
    """No universal pixel-speed threshold exists, so an uncalibrated camera
    must classify on occupancy alone, regardless of any 'speed' number."""
    tracker = CongestionStateTracker()
    # absurdly large px/s "speed" must not matter for IMAGE space
    level = tracker.update(mean_speed_value=99999.0, occupancy=0.9, motion_space=MotionSpace.IMAGE)
    assert level == CongestionLevel.BUILDING

    tracker2 = CongestionStateTracker()
    level2 = tracker2.update(mean_speed_value=0.0001, occupancy=0.1, motion_space=MotionSpace.IMAGE)
    assert level2 == CongestionLevel.FREE_FLOW


def test_custom_thresholds_are_respected():
    strict_thresholds = CongestionThresholds(free_speed_m_per_s=20.0, congested_speed_m_per_s=15.0, high_occupancy=0.1)
    tracker = CongestionStateTracker(strict_thresholds)
    # 10 m/s would be "free" under defaults but is "congested" under this strict config
    level = tracker.update(mean_speed_value=10.0, occupancy=0.05, motion_space=MotionSpace.WORLD)
    assert level == CongestionLevel.BUILDING


def test_aggregate_traffic_state_drives_the_tracker():
    tracker = CongestionStateTracker()
    # spans both windows below so the same heavy traffic is "observed" twice
    heavy_tracks = [_track(str(i), 0.0, 20.0, speed_value=1.0) for i in range(15)]

    result1 = aggregate_traffic_state("cam-A", 0.0, 10.0, heavy_tracks, MotionSpace.WORLD, tracker)
    assert result1.traffic_state.congestion_level == CongestionLevel.BUILDING

    result2 = aggregate_traffic_state("cam-A", 10.0, 20.0, heavy_tracks, MotionSpace.WORLD, tracker)
    assert result2.traffic_state.congestion_level == CongestionLevel.CONGESTED


# ---------------------------------------------------------------------------
# Occupancy semantics: the numerator is presence (mean concurrency), not
# throughput (vehicle_count). See traffic_state.py's OCCUPANCY docstring.
# ---------------------------------------------------------------------------


def test_occupancy_numerator_is_mean_concurrency_not_overlap_count():
    """20 vehicles each present for 1s of a 10s window are never more than a
    couple of vehicles on the road at once -- occupancy must reflect that, not
    report a full road because 20 distinct vehicles were counted."""
    # 20 staggered 1.0s tracks, each entirely inside the 10s window
    tracks = [_track(str(i), float(i) * 0.45, float(i) * 0.45 + 1.0, speed_value=10.0) for i in range(20)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())

    assert result.traffic_state.vehicle_count == 20  # throughput is unchanged and still honest
    assert result.mean_concurrency == pytest.approx(2.0)  # 20 tracks x 1.0s dwell / 10s
    assert result.traffic_state.occupancy == pytest.approx(0.1)  # 2.0 / capacity 20, NOT 20/20


def test_occupancy_is_invariant_to_window_length_for_steady_traffic():
    """The same steady traffic measured over 10s and over 20s must give the
    same occupancy. The old vehicle_count numerator doubled instead."""
    tracks = [_track(str(i), float(i), float(i) + 2.0, speed_value=10.0) for i in range(20)]
    short = aggregate_traffic_state("cam-A", 2.0, 12.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    long = aggregate_traffic_state("cam-A", 2.0, 22.0, tracks, MotionSpace.WORLD, CongestionStateTracker())

    assert short.traffic_state.vehicle_count != long.traffic_state.vehicle_count  # throughput does scale
    assert short.traffic_state.occupancy == pytest.approx(long.traffic_state.occupancy, abs=0.02)


def test_equal_throughput_different_dwell_gives_different_occupancy():
    """Same number of vehicles through the window, but one set lingers 4x as
    long -- the slower, denser case must read as more occupied."""
    quick = [_track(f"q{i}", float(i), float(i) + 1.0, speed_value=10.0) for i in range(8)]
    slow = [_track(f"s{i}", float(i), float(i) + 4.0, speed_value=2.0) for i in range(8)]
    q = aggregate_traffic_state("cam-A", 0.0, 10.0, quick, MotionSpace.WORLD, CongestionStateTracker())
    s = aggregate_traffic_state("cam-A", 0.0, 10.0, slow, MotionSpace.WORLD, CongestionStateTracker())

    assert q.traffic_state.vehicle_count == s.traffic_state.vehicle_count
    assert s.traffic_state.occupancy > q.traffic_state.occupancy


def test_full_span_tracks_keep_the_previous_occupancy_exactly():
    """Regression guard for the synthetic scenarios: every hand-authored
    fixture builds tracks spanning the whole window, where mean concurrency
    equals vehicle_count, so their verified numbers must not move."""
    tracks = [_track(str(i), 0.0, 10.0, speed_value=2.0) for i in range(15)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())

    assert result.mean_concurrency == pytest.approx(15.0)
    assert result.traffic_state.occupancy == pytest.approx(15 / 20)  # identical to the old formula


def test_occupancy_still_clamps_at_full_capacity():
    tracks = [_track(str(i), 0.0, 10.0, speed_value=1.0) for i in range(40)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    assert result.traffic_state.occupancy == 1.0


def test_mean_concurrency_counts_only_the_in_window_part_of_a_track():
    """A track straddling the window boundary contributes only its overlap."""
    tracks = [_track("straddler", 5.0, 25.0, speed_value=5.0)]  # 5s of a 10s window
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    assert result.mean_concurrency == pytest.approx(0.5)
