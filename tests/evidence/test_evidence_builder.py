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

# H5's graded branch needs a caller-supplied site capacity (see
# evidence_builder's module docstring). 1.6 veh/s over this 10s window is the
# rate-unit restatement of the 16-vehicles-per-window ramp these H5 tests were
# originally written against, so their expectations are unchanged.
H5_CAPACITY = EvidenceThresholds(capacity_flow_veh_per_s=1.6)


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


def _build(tracks, motion_space=MotionSpace.WORLD, camera_role=CameraRole.DOWNSTREAM, reliability=0.9,
           thresholds=None):
    aggregation = aggregate_traffic_state("cam-A", *WINDOW, tracks, motion_space, CongestionStateTracker())
    return build_evidence_for_camera("cam-A", WINDOW, camera_role, aggregation, tracks, reliability,
                                     thresholds=thresholds)


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


def test_elevated_flow_with_no_obstruction_supports_h5():
    tracks = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(10)]
    items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM, thresholds=H5_CAPACITY))
    assert items["H5"].direction == EvidenceDirection.SUPPORTS


def test_h5_is_camera_role_sensitive():
    """Camera-specific evidence: identical traffic, different camera role
    (blueprint Part 7: upstream is 'most informative' for demand surge)."""
    tracks = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(10)]
    upstream_items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM, thresholds=H5_CAPACITY))
    downstream_items = _by_id(_build(tracks, camera_role=CameraRole.DOWNSTREAM, thresholds=H5_CAPACITY))
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
    items = _by_id(_build([], thresholds=H5_CAPACITY))
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
    strict = EvidenceThresholds(stopped_speed_m_per_s=0.1, min_dwell_seconds=100.0, capacity_flow_veh_per_s=1.6)
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)]  # would normally count as "stopped"
    aggregation = aggregate_traffic_state("cam-A", *WINDOW, tracks, MotionSpace.WORLD, CongestionStateTracker())
    items = _by_id(
        build_evidence_for_camera("cam-A", WINDOW, CameraRole.DOWNSTREAM, aggregation, tracks, 0.9, thresholds=strict)
    )
    # min_dwell_seconds=100 means an 8-second track never counts as "stopped"
    assert items["H2"].direction != EvidenceDirection.SUPPORTS


# ---------------------------------------------------------------------------
# H5 needs a caller-supplied capacity reference: "elevated" is comparative.
# ---------------------------------------------------------------------------


def test_h5_is_unavailable_without_a_capacity_reference():
    """No site capacity supplied -> H5's graded branch reports NONE rather
    than scoring against an unfounded absolute cutoff."""
    tracks = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(30)]
    items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM))  # default thresholds: capacity is None
    assert items["H5"].evidence.evidence_completeness == EvidenceCompleteness.NONE
    assert items["H5"].direction == EvidenceDirection.INSUFFICIENT
    assert items["H5"].evidence.signature_match_score == 0.0
    assert items["H5"].evidence.contradicting is False  # missing != contradicted
    assert "capacity_flow_veh_per_s" in items["H5"].metadata["reason"]


def test_h5_gate_is_a_rate_so_it_does_not_change_meaning_with_window_length():
    """The same steady flow judged over a 10s and a 20s window must reach the
    same verdict. A per-window count threshold would flip here."""
    thresholds = EvidenceThresholds(capacity_flow_veh_per_s=1.6)
    tracks = [_track(str(i), float(i) * 0.5, float(i) * 0.5 + 0.4, speed_value=9.0) for i in range(40)]

    def h5_for(window):
        agg = aggregate_traffic_state("cam-A", *window, tracks, MotionSpace.WORLD, CongestionStateTracker())
        items = build_evidence_for_camera("cam-A", window, CameraRole.UPSTREAM, agg, tracks, 0.9,
                                          thresholds=thresholds)
        return _by_id(items)["H5"]

    short, long = h5_for((0.0, 10.0)), h5_for((0.0, 20.0))
    assert short.direction == long.direction
    assert short.evidence.signature_match_score == pytest.approx(long.evidence.signature_match_score, abs=0.05)


def test_h5_score_scales_with_flow_relative_to_capacity():
    thresholds = EvidenceThresholds(capacity_flow_veh_per_s=1.6)
    light = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(9)]
    heavy = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(15)]
    light_h5 = _by_id(_build(light, camera_role=CameraRole.UPSTREAM, thresholds=thresholds))["H5"]
    heavy_h5 = _by_id(_build(heavy, camera_role=CameraRole.UPSTREAM, thresholds=thresholds))["H5"]
    assert light_h5.direction == EvidenceDirection.SUPPORTS
    assert heavy_h5.evidence.signature_match_score > light_h5.evidence.signature_match_score


def test_h5_below_the_elevated_fraction_is_neutral_not_supporting():
    thresholds = EvidenceThresholds(capacity_flow_veh_per_s=1.6)  # elevated at >= 0.8 veh/s == 8 per 10s
    tracks = [_track(str(i), 0.0, 10.0, speed_value=9.0) for i in range(5)]
    items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM, thresholds=thresholds))
    assert items["H5"].direction == EvidenceDirection.NEUTRAL


def test_h5_stopped_vehicle_contradiction_needs_no_capacity_reference():
    """The obstruction branch is decidable without a baseline, so it must
    still be evaluated when capacity is unknown."""
    tracks = [_track("1", 0.0, 8.0, speed_value=0.5)]
    items = _by_id(_build(tracks, camera_role=CameraRole.UPSTREAM))  # no capacity supplied
    assert items["H5"].direction == EvidenceDirection.CONTRADICTS
    assert items["H5"].evidence.evidence_completeness == EvidenceCompleteness.FULL


def test_capacity_flow_must_be_positive_when_supplied():
    with pytest.raises(ValueError):
        EvidenceThresholds(capacity_flow_veh_per_s=0.0)
