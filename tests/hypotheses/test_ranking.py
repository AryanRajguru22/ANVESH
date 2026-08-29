import pytest

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.fusion.ds_fusion import fuse_two_cameras, single_camera_result
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.hypotheses.ranking import attach_ranking_to_corridor_state, build_ranking, determine_outcome
from anvesh.storage.schemas import (
    CandidateCauseHypothesis,
    CandidateOutcome,
    CongestionLevel,
    CorridorState,
    Evidence,
    EvidenceCompleteness,
    Measurement,
    MotionSpace,
    TrafficState,
)
from anvesh.world.corridor_state import UNRANKED_PLACEHOLDER

WINDOW = (0.0, 10.0)


def _evidence_item(camera_id, hypothesis_id, score, direction):
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hypothesis_id}:0-10",
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hypothesis_id,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(
        evidence=evidence, evidence_type="test", measured_value=None, direction=direction,
        reliability=1.0, source="test", metadata={},
    )


def _neutral_except(camera_id, hid, score, direction):
    return tuple(
        _evidence_item(camera_id, h, score if h == hid else 0.0, direction if h == hid else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )


def _traffic_state(camera_id):
    return TrafficState(
        camera_id=camera_id, window_start=0.0, window_end=10.0, occupancy=0.3,
        mean_speed=Measurement(value=8.0, error=0.5), vehicle_count=3, flow_rate=0.3, density=3.0,
        congestion_level=CongestionLevel.FREE_FLOW, motion_space=MotionSpace.WORLD,
    )


def _corridor_state():
    return CorridorState(
        corridor_state_id="cs-1", window=WINDOW, segment_states=[_traffic_state("cam-A")],
        fused_congestion_level=CongestionLevel.FREE_FLOW, active_ranking_id=UNRANKED_PLACEHOLDER,
    )


# ---------------------------------------------------------------------------
# Ordering / determinism / ties
# ---------------------------------------------------------------------------


def test_ranked_list_is_ordered_by_belief_descending():
    ev_a = _neutral_except("cam-A", "H1", 0.9, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H1", 0.8, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)

    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})

    beliefs = [e.belief for e in ranking.ranked_list]
    assert beliefs == sorted(beliefs, reverse=True)
    assert ranking.ranked_list[0].hypothesis_id == "H1"


def test_all_six_hypotheses_always_present_in_ranked_list():
    ev_a = _neutral_except("cam-A", "H1", 0.5, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H1", 0.5, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    assert {e.hypothesis_id for e in ranking.ranked_list} == set(HYPOTHESIS_IDS)


def test_tie_break_is_by_hypothesis_id():
    # no evidence at all -> every hypothesis has belief=0.0, plausibility=1.0 (pure ignorance)
    ev_a = tuple(_evidence_item("cam-A", h, 0.0, EvidenceDirection.NEUTRAL) for h in HYPOTHESIS_IDS)
    ev_b = tuple(_evidence_item("cam-B", h, 0.0, EvidenceDirection.NEUTRAL) for h in HYPOTHESIS_IDS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)

    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})

    assert [e.hypothesis_id for e in ranking.ranked_list] == sorted(HYPOTHESIS_IDS)


def test_deterministic_repeated_execution():
    ev_a = _neutral_except("cam-A", "H2", 0.7, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H2", 0.6, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 0.8, "cam-B", ev_b, 0.9, WINDOW)

    r1 = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    r2 = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})

    assert [(e.hypothesis_id, e.belief, e.plausibility) for e in r1.ranked_list] == [
        (e.hypothesis_id, e.belief, e.plausibility) for e in r2.ranked_list
    ]
    assert r1.outcome == r2.outcome


# ---------------------------------------------------------------------------
# Outcomes: ranked / insufficient_evidence / high_conflict
# ---------------------------------------------------------------------------


def test_outcome_ranked_when_evidence_is_clear():
    ev_a = _neutral_except("cam-A", "H1", 0.8, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H1", 0.7, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    assert ranking.outcome == CandidateOutcome.RANKED


def test_outcome_insufficient_evidence_when_no_signal():
    ev_a = tuple(_evidence_item("cam-A", h, 0.0, EvidenceDirection.INSUFFICIENT) for h in HYPOTHESIS_IDS)
    ev_b = tuple(_evidence_item("cam-B", h, 0.0, EvidenceDirection.INSUFFICIENT) for h in HYPOTHESIS_IDS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    assert ranking.outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE


def test_outcome_high_conflict_when_cameras_disagree_totally():
    ev_a = _neutral_except("cam-A", "H1", 1.0, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H2", 1.0, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    assert ranking.outcome == CandidateOutcome.HIGH_CONFLICT


def test_determine_outcome_thresholds_directly():
    ev_a = tuple(_evidence_item("cam-A", h, 0.0, EvidenceDirection.INSUFFICIENT) for h in HYPOTHESIS_IDS)
    fusion = single_camera_result("cam-A", ev_a, WINDOW)
    assert determine_outcome(fusion) == CandidateOutcome.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Provenance / traceability
# ---------------------------------------------------------------------------


def test_ranking_is_traceable_to_supporting_evidence():
    ev_a = _neutral_except("cam-A", "H1", 0.8, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H1", 0.7, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})

    h1_entry = next(e for e in ranking.ranked_list if e.hypothesis_id == "H1")
    assert "cam-A:H1:0-10" in h1_entry.supporting_evidence_refs
    assert "cam-B:H1:0-10" in h1_entry.supporting_evidence_refs


def test_contradicting_evidence_is_referenced_too():
    ev_a = _neutral_except("cam-A", "H2", 0.9, EvidenceDirection.CONTRADICTS)
    ev_b = _neutral_except("cam-B", "H2", 0.0, EvidenceDirection.NEUTRAL)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})

    h2_entry = next(e for e in ranking.ranked_list if e.hypothesis_id == "H2")
    assert "cam-A:H2:0-10" in h2_entry.contradicting_evidence_refs


def test_ranking_is_valid_schema_instance():
    ev_a = _neutral_except("cam-A", "H1", 0.8, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H1", 0.7, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    assert isinstance(ranking, CandidateCauseHypothesis)
    assert ranking.corridor_state_id == "cs-1"


# ---------------------------------------------------------------------------
# UNRANKED_PLACEHOLDER replacement
# ---------------------------------------------------------------------------


def test_attach_ranking_replaces_unranked_placeholder():
    corridor_state = _corridor_state()
    assert corridor_state.active_ranking_id == UNRANKED_PLACEHOLDER

    ev_a = _neutral_except("cam-A", "H1", 0.8, EvidenceDirection.SUPPORTS)
    ev_b = _neutral_except("cam-B", "H1", 0.7, EvidenceDirection.SUPPORTS)
    fusion = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    ranking = build_ranking("rank-real-1", "cs-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})

    updated = attach_ranking_to_corridor_state(corridor_state, ranking)

    assert updated.active_ranking_id == "rank-real-1"
    assert updated.active_ranking_id != UNRANKED_PLACEHOLDER
    # everything else must be preserved unchanged
    assert updated.corridor_state_id == corridor_state.corridor_state_id
    assert updated.segment_states == corridor_state.segment_states
    assert updated.fused_congestion_level == corridor_state.fused_congestion_level


def test_attach_ranking_rejects_mismatched_corridor_state_id():
    corridor_state = _corridor_state()
    ev_a = _neutral_except("cam-A", "H1", 0.5, EvidenceDirection.SUPPORTS)
    fusion = single_camera_result("cam-A", ev_a, WINDOW)
    ranking = build_ranking("rank-1", "cs-DIFFERENT", fusion, {"cam-A": ev_a})

    with pytest.raises(ValueError):
        attach_ranking_to_corridor_state(corridor_state, ranking)
