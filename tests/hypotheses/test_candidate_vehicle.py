import pytest

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem, build_evidence_for_camera
from anvesh.fusion.ds_fusion import single_camera_result
from anvesh.hypotheses.candidate_vehicle import (
    CandidateVehicleResolution,
    CandidateVehicleStatus,
    resolve_candidate_vehicle,
)
from anvesh.hypotheses.ranking import build_ranking
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.schemas import GeometryValidity
from anvesh.storage.schemas import (
    CameraRole,
    CandidateCauseHypothesis,
    CandidateOutcome,
    ConfidenceTier,
    Evidence,
    EvidenceCompleteness,
    Measurement,
    MotionSpace,
    OcclusionState,
    RankedHypothesisEntry,
    VehicleClass,
    VehicleTrack,
)

WINDOW = (0.0, 10.0)


# ---------------------------------------------------------------------------
# Fixture helpers -- hand-built ranking/evidence, same style as
# tests/hypotheses/test_ranking.py and tests/evidence/test_evidence_builder.py
# ---------------------------------------------------------------------------


def _entry(hypothesis_id, belief, plausibility, supporting=()):
    return RankedHypothesisEntry(
        hypothesis_id=hypothesis_id,
        belief=belief,
        plausibility=plausibility,
        confidence_tier=ConfidenceTier.MEDIUM,
        supporting_evidence_refs=list(supporting),
        contradicting_evidence_refs=[],
    )


def _ranking(outcome, entries):
    return CandidateCauseHypothesis(
        ranking_id="rank-1",
        corridor_state_id="cs-1",
        outcome=outcome,
        ranked_list=list(entries),
        engine_model_id="test",
        engine_model_version="v1",
    )


def _evidence_item(evidence_id, hypothesis_id, camera_id, track_refs, direction=EvidenceDirection.SUPPORTS):
    evidence = Evidence(
        evidence_id=evidence_id,
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hypothesis_id,
        signature_match_score=0.8,
        supporting_track_refs=list(track_refs),
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(
        evidence=evidence, evidence_type="test", measured_value=None, direction=direction,
        reliability=1.0, source="test", metadata={},
    )


# ---------------------------------------------------------------------------
# a/b: RESOLVED (single supporting vehicle) for H2 and H3
# ---------------------------------------------------------------------------


def test_ranked_h2_with_one_supporting_vehicle_is_resolved():
    ev = _evidence_item("cam-A:H2:0-10", "H2", "cam-A", ["t1"])
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H2", 0.8, 0.9, supporting=["cam-A:H2:0-10"])])

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev,)})

    assert isinstance(result, CandidateVehicleResolution)
    assert result.status == CandidateVehicleStatus.RESOLVED
    assert result.track_id == "t1"
    assert result.candidate_track_ids == ("t1",)
    assert result.hypothesis_id == "H2"


def test_ranked_h3_with_one_supporting_vehicle_is_resolved():
    ev = _evidence_item("cam-A:H3:0-10", "H3", "cam-A", ["t7"])
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H3", 0.75, 0.85, supporting=["cam-A:H3:0-10"])])

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev,)})

    assert result.status == CandidateVehicleStatus.RESOLVED
    assert result.track_id == "t7"
    assert result.hypothesis_id == "H3"


# ---------------------------------------------------------------------------
# c: AMBIGUOUS -- two equally supported vehicles
# ---------------------------------------------------------------------------


def test_two_equally_supported_vehicles_is_ambiguous():
    ev_a = _evidence_item("cam-A:H3:0-10", "H3", "cam-A", ["t1", "t2"])
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H3", 0.75, 0.85, supporting=["cam-A:H3:0-10"])])

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev_a,)})

    assert result.status == CandidateVehicleStatus.AMBIGUOUS
    assert result.track_id is None
    assert result.candidate_track_ids == ("t1", "t2")


def test_two_vehicles_from_separate_evidence_items_is_also_ambiguous():
    ev_a = _evidence_item("cam-A:H2:0-10", "H2", "cam-A", ["t1"])
    ev_b = _evidence_item("cam-B:H2:0-10", "H2", "cam-B", ["t2"])
    ranking = _ranking(
        CandidateOutcome.RANKED,
        [_entry("H2", 0.8, 0.9, supporting=["cam-A:H2:0-10", "cam-B:H2:0-10"])],
    )

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev_a,), "cam-B": (ev_b,)})

    assert result.status == CandidateVehicleStatus.AMBIGUOUS
    assert result.candidate_track_ids == ("t1", "t2")


# ---------------------------------------------------------------------------
# d/e/f: NO_VEHICLE_FOR_HYPOTHESIS -- H1/H5/H6 winning with no vehicle ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hypothesis_id", ["H1", "H5", "H6"])
def test_winning_hypothesis_with_no_vehicle_reference_has_no_vehicle(hypothesis_id):
    ev = _evidence_item(f"cam-A:{hypothesis_id}:0-10", hypothesis_id, "cam-A", [])
    ranking = _ranking(
        CandidateOutcome.RANKED,
        [_entry(hypothesis_id, 0.8, 0.9, supporting=[f"cam-A:{hypothesis_id}:0-10"])],
    )

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev,)})

    assert result.status == CandidateVehicleStatus.NO_VEHICLE_FOR_HYPOTHESIS
    assert result.track_id is None
    assert result.candidate_track_ids == ()
    assert result.hypothesis_id == hypothesis_id


def test_winning_hypothesis_with_no_supporting_evidence_at_all_has_no_vehicle():
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H1", 0.6, 0.7, supporting=[])])

    result = resolve_candidate_vehicle(ranking, {})

    assert result.status == CandidateVehicleStatus.NO_VEHICLE_FOR_HYPOTHESIS


# ---------------------------------------------------------------------------
# g: NOT_RANKED -- ANVESH abstained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", [CandidateOutcome.HIGH_CONFLICT, CandidateOutcome.INSUFFICIENT_EVIDENCE])
def test_abstained_ranking_is_not_ranked(outcome):
    ev = _evidence_item("cam-A:H2:0-10", "H2", "cam-A", ["t1"])
    ranking = _ranking(outcome, [_entry("H2", 0.1, 0.9, supporting=["cam-A:H2:0-10"])])

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev,)})

    assert result.status == CandidateVehicleStatus.NOT_RANKED
    assert result.hypothesis_id is None
    assert result.belief is None
    assert result.plausibility is None
    assert result.track_id is None
    assert result.candidate_track_ids == ()


def test_empty_ranked_list_is_not_ranked():
    ranking = _ranking(CandidateOutcome.RANKED, [])

    result = resolve_candidate_vehicle(ranking, {})

    assert result.status == CandidateVehicleStatus.NOT_RANKED


# ---------------------------------------------------------------------------
# h: lower-ranked hypothesis has a vehicle ref, winner does not
# ---------------------------------------------------------------------------


def test_only_top_ranked_hypothesis_is_ever_consulted():
    ev_h1 = _evidence_item("cam-A:H1:0-10", "H1", "cam-A", [])
    ev_h2 = _evidence_item("cam-A:H2:0-10", "H2", "cam-A", ["t1"])
    ranking = _ranking(
        CandidateOutcome.RANKED,
        [
            _entry("H1", 0.9, 0.95, supporting=["cam-A:H1:0-10"]),  # winner: no vehicle
            _entry("H2", 0.2, 0.3, supporting=["cam-A:H2:0-10"]),  # loser: has a vehicle ref
        ],
    )

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev_h1, ev_h2)})

    assert result.status == CandidateVehicleStatus.NO_VEHICLE_FOR_HYPOTHESIS
    assert result.hypothesis_id == "H1"
    assert result.track_id is None


# ---------------------------------------------------------------------------
# i: track_id is taken from evidence, never generated/guessed
# ---------------------------------------------------------------------------


def test_resolved_track_id_is_exactly_the_one_named_in_evidence():
    ev = _evidence_item("cam-A:H2:0-10", "H2", "cam-A", ["track-xyz-999"])
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H2", 0.8, 0.9, supporting=["cam-A:H2:0-10"])])

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev,)})

    assert result.track_id == "track-xyz-999"
    assert result.track_id in ev.evidence.supporting_track_refs
    # never any track_id absent from the input evidence
    assert set(result.candidate_track_ids) <= set(ev.evidence.supporting_track_refs)


def test_missing_evidence_item_is_never_fabricated_into_a_track():
    # supporting_evidence_refs names an evidence_id that isn't in the
    # supplied evidence_by_camera at all -- must not invent a track for it.
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H2", 0.8, 0.9, supporting=["cam-A:H2:0-10"])])

    result = resolve_candidate_vehicle(ranking, {})

    assert result.status == CandidateVehicleStatus.NO_VEHICLE_FOR_HYPOTHESIS
    assert result.track_id is None


# ---------------------------------------------------------------------------
# j: belief/plausibility preserved as evidential quantities, unchanged
# ---------------------------------------------------------------------------


def test_belief_and_plausibility_are_preserved_unchanged():
    ev = _evidence_item("cam-A:H2:0-10", "H2", "cam-A", ["t1"])
    ranking = _ranking(CandidateOutcome.RANKED, [_entry("H2", belief=0.73, plausibility=0.88, supporting=["cam-A:H2:0-10"])])

    result = resolve_candidate_vehicle(ranking, {"cam-A": (ev,)})

    assert result.belief == 0.73
    assert result.plausibility == 0.88
    # never renormalized/converted into some other "responsibility" quantity
    assert 0.0 <= result.belief <= 1.0
    assert 0.0 <= result.plausibility <= 1.0


# ---------------------------------------------------------------------------
# Integration-style test using the real evidence builder + ranking pipeline
# ---------------------------------------------------------------------------


def _stopped_track(track_id):
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-A",
        first_seen=0.0,
        last_seen=8.0,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=0.5, error=0.0),  # slow, sustained -> "stopped"
        motion_space=MotionSpace.WORLD, geometry_validity=GeometryValidity.WORLD_VALIDATED,
    )


def test_integration_single_stopped_vehicle_resolves_through_real_pipeline():
    tracks = [_stopped_track("track-42")]
    aggregation = aggregate_traffic_state("cam-A", *WINDOW, tracks, MotionSpace.WORLD, CongestionStateTracker())
    evidence = build_evidence_for_camera("cam-A", WINDOW, CameraRole.DOWNSTREAM, aggregation, tracks, reliability=0.9)
    evidence_by_camera = {"cam-A": evidence}

    fusion = single_camera_result("cam-A", evidence, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, evidence_by_camera)

    result = resolve_candidate_vehicle(ranking, evidence_by_camera)

    assert ranking.outcome == CandidateOutcome.RANKED
    assert ranking.ranked_list[0].hypothesis_id == "H2"
    assert result.status == CandidateVehicleStatus.RESOLVED
    assert result.track_id == "track-42"
    assert result.hypothesis_id == "H2"
    assert result.belief == ranking.ranked_list[0].belief
    assert result.plausibility == ranking.ranked_list[0].plausibility


def test_integration_no_evidence_at_all_abstains_and_is_not_ranked():
    ev = tuple(
        _evidence_item(f"cam-A:{h}:0-10", h, "cam-A", [], direction=EvidenceDirection.INSUFFICIENT)
        for h in ["H1", "H2", "H3", "H4", "H5", "H6"]
    )
    fusion = single_camera_result("cam-A", ev, WINDOW)
    ranking = build_ranking("rank-2", "cs-1", fusion, {"cam-A": ev})

    result = resolve_candidate_vehicle(ranking, {"cam-A": ev})

    assert ranking.outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE
    assert result.status == CandidateVehicleStatus.NOT_RANKED
