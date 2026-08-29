import pytest

from anvesh.perception.motion import (
    ImageSpaceSpeedEstimate,
    WorldSpaceSpeedEstimate,
    compute_image_space_speed,
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
