import pytest

from anvesh.evidence.evidence_builder import (
    EvidenceDirection,
    EvidenceItem,
    EvidenceThresholds,
    build_evidence_for_camera,
)
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.schemas import (
    CameraRole,
    Evidence,
    EvidenceCompleteness,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
)

WINDOW = (0.0, 10.0)


def _track(track_id, first_seen, last_seen, speed_value, motion_space=MotionSpace.WORLD):
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-A",
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=0.0),
        motion_space=motion_space,
    )


def _build(tracks, motion_space=MotionSpace.WORLD, camera_role=CameraRole.DOWNSTREAM, reliability=0.9):
    aggregation = aggregate_traffic_state("cam-A", *WINDOW, tracks, motion_space, CongestionStateTracker())
    return build_evidence_for_camera("cam-A", WINDOW, camera_role, aggregation, tracks, reliability)


def _by_id(items):
    return {item.evidence.hypothesis_id: item for item in items}


def test_produces_exactly_six_items_for_h1_through_h6():
    items = _build([])
    assert len(items) == 6
    assert {item.evidence.hypothesis_id for item in items} == {"H1", "H2", "H3", "H4", "H5", "H6"}


def test_every_item_is_a_valid_evidence_schema_instance():
    items = _build([_track("1", 0.0, 10.0, speed_value=10.0)])
    for item in items:
        assert isinstance(item, EvidenceItem)
        assert isinstance(item.evidence, Evidence)
        assert item.evidence.window == WINDOW
        assert item.evidence.camera_id == "cam-A"


def test_h4_and_h6_are_always_insufficient():
    items = _by_id(_build([_track("1", 0.0, 10.0, speed_value=1.0)]))
    assert items["H4"].evidence.evidence_completeness == EvidenceCompleteness.NONE
    assert items["H4"].direction == EvidenceDirection.INSUFFICIENT
    assert items["H6"].evidence.evidence_completeness == EvidenceCompleteness.NONE
    assert items["H6"].direction == EvidenceDirection.INSUFFICIENT
    # missing evidence must not be scored as contradiction
    assert items["H4"].evidence.contradicting is False
    assert items["H6"].evidence.contradicting is False


def test_single_stopped_vehicle_supports_h2():
    stopped = _track("1", 0.0, 8.0, speed_value=0.5)  # slow, sustained >= min_dwell
    items = _by_id(_build([stopped]))
    assert items["H2"].direction == EvidenceDirection.SUPPORTS
    assert items["H2"].evidence.supporting_track_refs == ["1"]
    assert items["H3"].direction != EvidenceDirection.SUPPORTS


def test_multiple_stopped_vehicles_support_h3_and_contradict_h2():
    stopped = [_track(str(i), 0.0, 8.0, speed_value=0.5) for i in range(3)]
    items = _by_id(_build(stopped))
    assert items["H3"].direction == EvidenceDirection.SUPPORTS
    assert items["H2"].direction == EvidenceDirection.CONTRADICTS


def test_congestion_without_obstruction_supports_h1():
    # 15 slow-moving-but-not-stopped vehicles -> CONGESTED, no stopped vehicle
    tracks = [_track(str(i), 0.0, 10.0, speed_value=2.0) for i in range(15)]
    items = _by_id(_build(tracks))
    assert items["H1"].direction == EvidenceDirection.SUPPORTS


def test_stopped_vehicle_contradicts_h1():
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)] + [
        _track(str(i), 0.0, 10.0, speed_value=2.0) for i in range(1, 15)
    ]
    items = _by_id(_build(tracks))
    assert items["H1"].direction == EvidenceDirection.CONTRADICTS


def test_elevated_count_with_no_obstruction_supports_h5():
    tracks = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(10)]
    items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM))
    assert items["H5"].direction == EvidenceDirection.SUPPORTS


def test_h5_is_camera_role_sensitive():
    """Camera-specific evidence: identical traffic, different camera role
    (blueprint Part 7: upstream is 'most informative' for demand surge)."""
    tracks = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(10)]
    upstream_items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM))
    downstream_items = _by_id(_build(tracks, camera_role=CameraRole.DOWNSTREAM))
    assert upstream_items["H5"].evidence.signature_match_score > downstream_items["H5"].evidence.signature_match_score


def test_stopped_vehicle_contradicts_h5():
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)]
    items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM))
    assert items["H5"].direction == EvidenceDirection.CONTRADICTS


def test_image_space_camera_cannot_detect_stopped_vehicles():
    """H2/H3 need a metric speed threshold; IMAGE space cannot provide one."""
    tracks = [_track("1", 0.0, 8.0, speed_value=0.1, motion_space=MotionSpace.IMAGE)]
    items = _by_id(_build(tracks, motion_space=MotionSpace.IMAGE))
    assert items["H2"].evidence.evidence_completeness == EvidenceCompleteness.NONE
    assert items["H3"].evidence.evidence_completeness == EvidenceCompleteness.NONE
    assert items["H2"].direction == EvidenceDirection.INSUFFICIENT


def test_image_space_h1_is_only_partial_completeness():
    tracks = [_track(str(i), 0.0, 10.0, speed_value=50.0, motion_space=MotionSpace.IMAGE) for i in range(15)]
    items = _by_id(_build(tracks, motion_space=MotionSpace.IMAGE))
    assert items["H1"].evidence.evidence_completeness == EvidenceCompleteness.PARTIAL


def test_evidence_carries_provenance():
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)]
    items = _build(tracks, reliability=0.42)
    for item in items:
        assert item.reliability == 0.42
        assert item.source
        assert isinstance(item.metadata, dict)


def test_no_traffic_at_all_yields_neutral_not_manufactured_evidence():
    """Empty road: no vehicles at all -- must not manufacture support for
    any hypothesis (design rule #8)."""
    items = _by_id(_build([]))
    for hid in ("H1", "H2", "H3", "H5"):
        assert items[hid].direction in (EvidenceDirection.NEUTRAL, EvidenceDirection.CONTRADICTS)
        assert items[hid].evidence.signature_match_score <= 0.15


def test_deterministic_repeated_calls():
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)]
    first = _build(tracks)
    second = _build(tracks)
    assert [i.evidence.signature_match_score for i in first] == [i.evidence.signature_match_score for i in second]
    assert [i.direction for i in first] == [i.direction for i in second]


def test_custom_thresholds_are_respected():
    strict = EvidenceThresholds(stopped_speed_m_per_s=0.1, min_dwell_seconds=100.0, elevated_vehicle_count=8)
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)]  # would normally count as "stopped"
    aggregation = aggregate_traffic_state("cam-A", *WINDOW, tracks, MotionSpace.WORLD, CongestionStateTracker())
    items = _by_id(
        build_evidence_for_camera("cam-A", WINDOW, CameraRole.DOWNSTREAM, aggregation, tracks, 0.9, thresholds=strict)
    )
    # min_dwell_seconds=100 means an 8-second track never counts as "stopped"
    assert items["H2"].direction != EvidenceDirection.SUPPORTS
