import pytest

from anvesh.perception.motion import (
    ImageSpaceSpeedEstimate,
    RobustSpeedResult,
    RobustSpeedStatus,
    WorldSpaceSpeedEstimate,
    compute_image_space_speed,
    compute_robust_speed,
    compute_world_space_speed,
)


def test_straight_line_constant_speed():
    # 10 px per 1.0s step, three samples -> two 10 px/s steps, zero variance
    trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 2.0)]
    estimate = compute_image_space_speed(trajectory)

    assert estimate.speed_px_per_s == pytest.approx(10.0)
    assert estimate.error_px_per_s == pytest.approx(0.0)
    assert estimate.sample_count == 2
    assert estimate.space == "image"


def test_single_sample_has_no_measurable_motion():
    estimate = compute_image_space_speed([(5.0, 5.0, 0.0)])
    assert estimate.speed_px_per_s == 0.0
    assert estimate.sample_count == 1


def test_empty_trajectory_has_no_measurable_motion():
    estimate = compute_image_space_speed([])
    assert estimate.speed_px_per_s == 0.0
    assert estimate.sample_count == 0


def test_zero_duration_steps_are_skipped_not_divided_by_zero():
    trajectory = [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0), (10.0, 0.0, 2.0)]
    estimate = compute_image_space_speed(trajectory)
    # the (1.0 -> 1.0) step has dt=0 and must be skipped, leaving one 5px/1s step
    assert estimate.speed_px_per_s == pytest.approx(5.0)
    assert estimate.sample_count == 1


def test_diagonal_displacement_uses_euclidean_distance():
    trajectory = [(0.0, 0.0, 0.0), (3.0, 4.0, 1.0)]  # 3-4-5 triangle
    estimate = compute_image_space_speed(trajectory)
    assert estimate.speed_px_per_s == pytest.approx(5.0)


def test_negative_speed_rejected():
    with pytest.raises(ValueError):
        ImageSpaceSpeedEstimate(speed_px_per_s=-1.0, error_px_per_s=0.0, sample_count=1)


def test_wrong_space_tag_rejected():
    with pytest.raises(ValueError):
        ImageSpaceSpeedEstimate(speed_px_per_s=1.0, error_px_per_s=0.0, sample_count=1, space="world")


def test_world_space_straight_line_constant_speed():
    # 5 metres per 1.0s step -> 5 m/s, zero variance
    trajectory_world = [(0.0, 0.0, 0.0), (5.0, 0.0, 1.0), (10.0, 0.0, 2.0)]
    estimate = compute_world_space_speed(trajectory_world)

    assert isinstance(estimate, WorldSpaceSpeedEstimate)
    assert estimate.speed_m_per_s == pytest.approx(5.0)
    assert estimate.error_m_per_s == pytest.approx(0.0)
    assert estimate.sample_count == 2
    assert estimate.space == "world"


def test_world_space_negative_speed_rejected():
    with pytest.raises(ValueError):
        WorldSpaceSpeedEstimate(speed_m_per_s=-1.0, error_m_per_s=0.0, sample_count=1)


def test_world_space_wrong_space_tag_rejected():
    with pytest.raises(ValueError):
        WorldSpaceSpeedEstimate(speed_m_per_s=1.0, error_m_per_s=0.0, sample_count=1, space="image")


def test_image_and_world_estimates_are_never_interchangeable_types():
    """Proves the image/world distinction is enforced by type, not just by
    convention: the same numeric trajectory produces two estimates whose
    types, field names, and unit tags are all distinct -- there is no way
    to accidentally read a pixels/second value as metres/second."""
    trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0)]

    image_estimate = compute_image_space_speed(trajectory)
    world_estimate = compute_world_space_speed(trajectory)

    assert type(image_estimate) is not type(world_estimate)
    assert image_estimate.space == "image"
    assert world_estimate.space == "world"
    assert not hasattr(image_estimate, "speed_m_per_s")
    assert not hasattr(world_estimate, "speed_px_per_s")
    # same numeric trajectory -> same magnitude, but under unit-specific field names only
    assert image_estimate.speed_px_per_s == pytest.approx(world_estimate.speed_m_per_s)


# --- Commit 6: compute_robust_speed ---


def test_robust_straight_line_known_speed():
    # 10 units per 1.0s step, three samples -> total 20 units / 2.0s = 10.0
    trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 2.0)]
    result = compute_robust_speed(trajectory)

    assert result.status == RobustSpeedStatus.OK
    assert result.speed == pytest.approx(10.0)
    assert result.error == pytest.approx(0.0)
    assert result.sample_count == 2
    assert result.elapsed_s == pytest.approx(2.0)


def test_robust_zero_movement_is_ok_not_an_error():
    trajectory = [(5.0, 5.0, 0.0), (5.0, 5.0, 1.0), (5.0, 5.0, 2.0)]
    result = compute_robust_speed(trajectory)

    assert result.status == RobustSpeedStatus.OK
    assert result.speed == pytest.approx(0.0)
    assert result.error == pytest.approx(0.0)


def test_robust_speed_is_timestamp_based_not_detection_count_based():
    """The same 3 samples, but the middle one compressed into a much
    shorter real-time window, must change the computed speed -- proving
    the result depends on actual elapsed time, not merely "how many
    detections happened to arrive" (the exact V0 defect this module
    exists to avoid, see the forensic audit SS2.3)."""
    slow_trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 2.0)]
    fast_trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.1), (20.0, 0.0, 0.2)]

    slow_result = compute_robust_speed(slow_trajectory)
    fast_result = compute_robust_speed(fast_trajectory)

    assert slow_result.speed == pytest.approx(10.0)
    assert fast_result.speed == pytest.approx(100.0)
    assert slow_result.speed != fast_result.speed


def test_robust_irregular_timestamps_weighted_correctly():
    # steps: 5 units/0.3s, 10 units/0.2s, 5 units/0.5s -> 20 units / 1.0s = 20.0
    trajectory = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.3), (15.0, 0.0, 0.5), (20.0, 0.0, 1.0)]
    result = compute_robust_speed(trajectory)

    assert result.status == RobustSpeedStatus.OK
    assert result.speed == pytest.approx(20.0)
    assert result.elapsed_s == pytest.approx(1.0)
    assert result.sample_count == 3


def test_robust_insufficient_observations():
    result = compute_robust_speed([(0.0, 0.0, 0.0)], min_samples=2)

    assert result.status == RobustSpeedStatus.INSUFFICIENT_OBSERVATIONS
    assert result.speed is None
    assert result.error is None
    assert result.sample_count == 0
    assert result.elapsed_s is None


def test_robust_insufficient_elapsed_time():
    trajectory = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.05)]
    result = compute_robust_speed(trajectory, min_elapsed_s=0.1)

    assert result.status == RobustSpeedStatus.INSUFFICIENT_ELAPSED_TIME
    assert result.speed is None


def test_robust_large_temporal_gap_with_no_remaining_steps():
    trajectory = [(0.0, 0.0, 0.0), (1000.0, 0.0, 100.0)]
    result = compute_robust_speed(trajectory, max_gap_s=2.0)

    assert result.status == RobustSpeedStatus.LARGE_TEMPORAL_GAP
    assert result.speed is None


def test_robust_large_gap_excluded_but_enough_steps_remain():
    """A large gap is excluded from the totals, but if other, normal
    steps remain, the result is still OK -- using only the remaining
    continuous segment, not the whole span."""
    trajectory = [(0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (1000.0, 0.0, 100.0)]
    result = compute_robust_speed(trajectory, max_gap_s=2.0)

    assert result.status == RobustSpeedStatus.OK
    assert result.speed == pytest.approx(10.0)
    assert result.sample_count == 1


def test_robust_invalid_coordinates_nan():
    trajectory = [(0.0, 0.0, 0.0), (float("nan"), 0.0, 1.0)]
    result = compute_robust_speed(trajectory)

    assert result.status == RobustSpeedStatus.INVALID_TRAJECTORY
    assert result.speed is None


def test_robust_invalid_coordinates_infinite():
    trajectory = [(0.0, 0.0, 0.0), (float("inf"), 0.0, 1.0)]
    result = compute_robust_speed(trajectory)

    assert result.status == RobustSpeedStatus.INVALID_TRAJECTORY


def test_robust_noisy_outlier_step_does_not_dominate_speed():
    """One near-duplicate-timestamp jitter step (0.1 units in 0.0001s ->
    a 1000 units/s instantaneous speed) among otherwise-consistent
    10 units/s motion must NOT drag the robust estimate anywhere near an
    unweighted mean (which would be skewed to ~340) -- the time-weighted
    total_distance/total_time stays close to the true 10 units/s signal."""
    trajectory = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 1.0),
        (10.1, 0.0, 1.0001),  # jitter: 0.1 units in 0.0001s -> 1000 units/s instantaneous
        (20.1, 0.0, 2.0001),
    ]
    result = compute_robust_speed(trajectory)

    assert result.status == RobustSpeedStatus.OK
    # naive unweighted mean of [10, 1000, 10] would be ~340 -- robust must stay near 10
    assert result.speed == pytest.approx(10.05, abs=0.5)
    assert result.speed < 20.0


def test_robust_result_ok_requires_all_numeric_fields():
    with pytest.raises(ValueError):
        RobustSpeedResult(status=RobustSpeedStatus.OK, speed=None, error=0.0, sample_count=1, elapsed_s=1.0)


def test_robust_result_non_ok_must_not_carry_numeric_fields():
    with pytest.raises(ValueError):
        RobustSpeedResult(
            status=RobustSpeedStatus.INSUFFICIENT_OBSERVATIONS,
            speed=5.0,
            error=0.0,
            sample_count=0,
            elapsed_s=None,
        )


def test_robust_default_min_samples_matches_existing_two_point_convention():
    """Default min_samples=2 matches compute_image_space_speed/
    compute_world_space_speed's own minimum for "any measurable motion"."""
    result = compute_robust_speed([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)])
    assert result.status == RobustSpeedStatus.OK
