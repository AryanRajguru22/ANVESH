from anvesh.perception.lane_assignment import (
    LaneAssignmentSample,
    TrackLaneHistory,
    assign_detection_lane,
    build_track_lane_history,
)
from anvesh.perception.schema_conversion import build_vehicle_track
from anvesh.perception.track_manager import TrackManager, TrackState
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import OcclusionState, VehicleClass, VehicleTrack
from anvesh.world.lane_config import default_lane_config_path, load_lane_config
from anvesh.world.lane_geometry import LaneAssignmentStatus, LaneRegion, assign_lane


def _detection(bbox_xyxy, frame_index=0, timestamp=0.0):
    return RawDetection(
        frame_index=frame_index,
        timestamp=timestamp,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.9,
        bbox_xyxy=bbox_xyxy,
    )


def _detection_centered_on(x, y, half_size=5.0, frame_index=0, timestamp=0.0):
    return _detection(
        (x - half_size, y - half_size, x + half_size, y + half_size),
        frame_index=frame_index,
        timestamp=timestamp,
    )


# --- requirement 13: vehicle clearly inside c006 L1 / c009 L1 -> ASSIGNED L1 ---


def test_vehicle_clearly_inside_c006_l1_is_assigned():
    config = load_lane_config(default_lane_config_path("s02_c006"))
    detection = _detection_centered_on(705.0, 300.0)  # inside c006 L1 by construction (Commit 9A.1 geometry)

    result = assign_detection_lane(detection, config.regions)

    assert result.status == LaneAssignmentStatus.ASSIGNED
    assert result.lane_id == "L1"


def test_vehicle_clearly_inside_c009_l1_is_assigned():
    config = load_lane_config(default_lane_config_path("s02_c009"))
    detection = _detection_centered_on(700.0, 350.0)  # inside c009 L1 by construction

    result = assign_detection_lane(detection, config.regions)

    assert result.status == LaneAssignmentStatus.ASSIGNED
    assert result.lane_id == "L1"


def test_vehicle_outside_lane_is_unknown_outside_regions():
    config = load_lane_config(default_lane_config_path("s02_c006"))
    detection = _detection_centered_on(10.0, 10.0)  # nowhere near L1's polygon

    result = assign_detection_lane(detection, config.regions)

    assert result.status == LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS
    assert result.lane_id is None


def test_ambiguous_boundary_point_is_ambiguous():
    lane_a = LaneRegion(lane_id="L1", vertices=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    lane_b = LaneRegion(lane_id="L2", vertices=((10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)))
    # ground_point = ((x1+x2)/2, y2) -> this box's ground_point is
    # (10.0, 5.0), exactly on the shared edge between L1 and L2.
    detection = RawDetection(
        frame_index=0,
        timestamp=0.0,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.9,
        bbox_xyxy=(5.0, 0.0, 15.0, 5.0),
    )
    assert detection.ground_point == (10.0, 5.0)

    result = assign_detection_lane(detection, (lane_a, lane_b))

    assert result.status == LaneAssignmentStatus.AMBIGUOUS
    assert result.lane_id is None
    assert result.candidate_lane_ids == ("L1", "L2")


# --- requirement: ground point, not centroid, determines assignment ---


def test_ground_point_not_centroid_determines_assignment():
    region = LaneRegion(lane_id="L1", vertices=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    # Tall box: centroid sits above the region (y < 0), but the ground
    # point (bottom-center) sits inside it.
    detection = RawDetection(
        frame_index=0,
        timestamp=0.0,
        class_name="bus",
        vehicle_class=VehicleClass.BUS,
        confidence=0.9,
        bbox_xyxy=(40.0, -100.0, 60.0, 50.0),
    )
    assert detection.centroid == (50.0, -25.0)
    assert detection.ground_point == (50.0, 50.0)

    ground_point_result = assign_detection_lane(detection, (region,))
    centroid_result = assign_lane(detection.centroid, (region,))

    assert ground_point_result.status == LaneAssignmentStatus.ASSIGNED
    assert ground_point_result.lane_id == "L1"
    # If this had used centroid instead, it would NOT be assigned -- this
    # is the failure mode ground_point exists to avoid (audit SS1.3/SS3.2).
    assert centroid_result.status == LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS


# --- requirement: lane assignment survives the perception/schema conversion path ---


def test_lane_assignment_survives_schema_conversion_path():
    """`build_vehicle_track` (schema_conversion.py) and
    `build_track_lane_history` (lane_assignment.py) both consume the same
    TrackState -- proving lane data is derivable alongside schema
    conversion without being folded into the frozen VehicleTrack schema."""
    state = TrackState(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=0.1,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=[(705.0, 300.0, 0.0), (705.0, 300.0, 0.1)],
        ground_point_trajectory=[(705.0, 300.0, 0.0), (705.0, 300.0, 0.1)],
    )
    config = load_lane_config(default_lane_config_path("s02_c006"))

    track = build_vehicle_track("s02_c006", state)
    lane_history = build_track_lane_history("s02_c006", state, config.regions)

    assert isinstance(track, VehicleTrack)
    assert track.track_id == "s02_c006:1"
    # VehicleTrack itself carries no lane field -- unchanged frozen schema.
    assert not hasattr(track, "lane_id")
    assert not hasattr(track, "lane_assignment")

    assert isinstance(lane_history, TrackLaneHistory)
    assert lane_history.track_id == "s02_c006:1"
    assert len(lane_history.samples) == 2
    for sample in lane_history.samples:
        assert isinstance(sample, LaneAssignmentSample)
        assert sample.result.status == LaneAssignmentStatus.ASSIGNED
        assert sample.result.lane_id == "L1"


def test_existing_schema_conversion_behavior_is_unchanged_by_ground_point_trajectory():
    """A TrackState with ground_point_trajectory populated must produce an
    identical VehicleTrack to one without it -- build_vehicle_track never
    reads the new field."""
    common_kwargs = dict(
        track_id="1",
        vehicle_class=VehicleClass.CAR,
        first_seen=0.0,
        last_seen=1.0,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=[(0.0, 0.0, 0.0), (10.0, 0.0, 1.0)],
    )
    state_without = TrackState(**common_kwargs)
    state_with = TrackState(ground_point_trajectory=[(0.0, 5.0, 0.0), (10.0, 5.0, 1.0)], **common_kwargs)

    track_without = build_vehicle_track("cam-01", state_without)
    track_with = build_vehicle_track("cam-01", state_with)

    assert track_without.position_history == track_with.position_history
    assert track_without.speed_estimate == track_with.speed_estimate
    assert track_without.motion_space == track_with.motion_space


# --- requirement 14: integration test, detection -> ground_point -> lane assignment -> representation ---


def test_integration_detection_through_track_manager_to_lane_history():
    """Full path: a RawDetection flows through TrackedDetection ->
    TrackManager.update() (accumulating ground_point_trajectory,
    unrelated to the unchanged centroid trajectory) -> build_track_lane_history
    (lane_assignment.py) -> a TrackLaneHistory naming the correct lane,
    using the real c006 config from Commit 3."""
    config = load_lane_config(default_lane_config_path("s02_c006"))

    # bbox chosen so ground_point (bottom-center) = (705.0, 300.0), well
    # inside c006 L1 (Commit 9A.1 geometry), while the box is tall enough
    # that centroid differs.
    detection = RawDetection(
        frame_index=0,
        timestamp=0.0,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.9,
        bbox_xyxy=(685.0, 250.0, 725.0, 300.0),
    )
    assert detection.ground_point == (705.0, 300.0)
    assert detection.centroid != detection.ground_point

    manager = TrackManager(max_missed_frames=3, partial_after_missed_frames=1)
    manager.update(0.0, [TrackedDetection(detection=detection, track_id="7")])
    manager.finalize()

    state = manager.finished_tracks[0]
    assert state.ground_point_trajectory == [(705.0, 300.0, 0.0)]

    lane_history = build_track_lane_history("s02_c006", state, config.regions)

    assert lane_history.track_id == "s02_c006:7"
    assert len(lane_history.samples) == 1
    assert lane_history.samples[0].result.status == LaneAssignmentStatus.ASSIGNED
    assert lane_history.samples[0].result.lane_id == "L1"
