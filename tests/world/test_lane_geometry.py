import pytest

from anvesh.world.lane_geometry import (
    LaneAssignmentStatus,
    LaneRegion,
    assign_lane,
)


def _lane_1():
    return LaneRegion(lane_id="lane_1", vertices=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))


def _lane_2():
    return LaneRegion(lane_id="lane_2", vertices=((10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)))


def test_point_clearly_inside_a_lane_is_assigned():
    result = assign_lane((5.0, 5.0), (_lane_1(), _lane_2()))
    assert result.status == LaneAssignmentStatus.ASSIGNED
    assert result.lane_id == "lane_1"
    assert result.candidate_lane_ids == ("lane_1",)


def test_point_clearly_outside_all_lanes_is_unknown_outside_regions():
    result = assign_lane((100.0, 100.0), (_lane_1(), _lane_2()))
    assert result.status == LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS
    assert result.lane_id is None
    assert result.candidate_lane_ids == ()


def test_point_on_a_shared_lane_boundary_is_ambiguous():
    # (10.0, 5.0) sits exactly on the shared edge between lane_1 and lane_2.
    result = assign_lane((10.0, 5.0), (_lane_1(), _lane_2()))
    assert result.status == LaneAssignmentStatus.AMBIGUOUS
    assert result.lane_id is None
    assert result.candidate_lane_ids == ("lane_1", "lane_2")


def test_overlapping_lane_regions_are_ambiguous():
    overlapping = LaneRegion(lane_id="lane_1_overlap", vertices=((5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0)))
    result = assign_lane((7.0, 5.0), (_lane_1(), overlapping))
    assert result.status == LaneAssignmentStatus.AMBIGUOUS
    assert result.candidate_lane_ids == ("lane_1", "lane_1_overlap")


def test_invalid_polygon_too_few_vertices_raises_at_construction():
    with pytest.raises(ValueError):
        LaneRegion(lane_id="bad", vertices=((0.0, 0.0), (1.0, 1.0)))


def test_invalid_polygon_degenerate_collinear_raises_at_construction():
    with pytest.raises(ValueError):
        LaneRegion(lane_id="bad", vertices=((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)))


def test_duplicate_lane_ids_are_invalid_geometry_at_query_time():
    dup = LaneRegion(lane_id="lane_1", vertices=((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)))
    result = assign_lane((5.0, 5.0), (_lane_1(), dup))
    assert result.status == LaneAssignmentStatus.INVALID_GEOMETRY
    assert result.lane_id is None
    assert result.candidate_lane_ids == ("lane_1",)


def test_no_regions_at_all_is_unknown_no_geometry():
    result = assign_lane((5.0, 5.0), ())
    assert result.status == LaneAssignmentStatus.UNKNOWN_NO_GEOMETRY
    assert result.candidate_lane_ids == ()


def test_multiple_lanes_have_deterministic_stable_ids():
    lane_3 = LaneRegion(lane_id="lane_3", vertices=((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)))
    regions = (_lane_1(), _lane_2(), lane_3)
    assert assign_lane((5.0, 5.0), regions).lane_id == "lane_1"
    assert assign_lane((15.0, 5.0), regions).lane_id == "lane_2"
    assert assign_lane((25.0, 5.0), regions).lane_id == "lane_3"


def test_fractional_coordinates_are_handled():
    result = assign_lane((5.5, 5.25), (_lane_1(), _lane_2()))
    assert result.status == LaneAssignmentStatus.ASSIGNED
    assert result.lane_id == "lane_1"


def test_repeated_evaluation_gives_identical_result():
    regions = (_lane_1(), _lane_2())
    point = (5.0, 5.0)
    first = assign_lane(point, regions)
    for _ in range(10):
        assert assign_lane(point, regions) == first


def test_boundary_ambiguity_is_independent_of_region_order():
    """A point exactly on a shared lane boundary must not randomly change
    lanes due to implementation ordering (req 6): the same regions, tested
    in reversed order, must produce an identical result object, not just an
    identical status."""
    forward = assign_lane((10.0, 5.0), (_lane_1(), _lane_2()))
    reversed_order = assign_lane((10.0, 5.0), (_lane_2(), _lane_1()))
    assert forward == reversed_order
    assert forward.candidate_lane_ids == ("lane_1", "lane_2")


def test_assigned_result_order_independent_too():
    forward = assign_lane((5.0, 5.0), (_lane_1(), _lane_2()))
    reversed_order = assign_lane((5.0, 5.0), (_lane_2(), _lane_1()))
    assert forward == reversed_order
