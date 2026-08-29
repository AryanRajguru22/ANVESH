import pytest

from anvesh.evaluation.baselines import AnveshComparisonResult, compare_anvesh_vs_baseline_c
from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import (
    DataCompleteness,
    Evidence,
    EvidenceCompleteness,
    HypothesisUpdateOutcome,
    Measurement,
    PropagationObservation,
    PropagationPrediction,
)

WINDOW = (0.0, 10.0)


def _evidence_item(camera_id, hid, score, direction):
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hid}:0-10",
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hid,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(evidence=evidence, evidence_type="test", measured_value=None, direction=direction, reliability=1.0, source="test", metadata={})


def _neutral_except(camera_id, hid, score):
    return tuple(
        _evidence_item(camera_id, h, score if h == hid else 0.0, EvidenceDirection.SUPPORTS if h == hid else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )


def _evidence_by_camera():
    return {"cam-A": _neutral_except("cam-A", "H1", 0.8), "cam-B": _neutral_except("cam-B", "H1", 0.7)}


def _prediction():
    return PropagationPrediction(
        prediction_id="pred-1", based_on_ranking_id="rank-c",
        predicted_shockwave_speed=Measurement(value=2.0, error=0.0),
        predicted_arrival_camera="cam-B", predicted_arrival_time=100.0, predicted_queue_growth_rate=5.0,
        method="rankine_hugoniot_kinematic_wave_v1",
    )


def _observation(camera="cam-B", completeness=DataCompleteness.FULL, arrival_time=100.0, growth_rate=5.0):
    return PropagationObservation(
        observation_id="obs-1", prediction_id="pred-1", actual_arrival_camera=camera,
        actual_arrival_time=arrival_time, actual_queue_growth_rate=growth_rate, data_completeness=completeness,
    )


def test_baseline_c_and_anvesh_start_from_identical_evidence():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.9, "cam-B": 0.9}
    result = compare_anvesh_vs_baseline_c(
        "rank-c", "cs-1", WINDOW, evidence, reliability, _prediction(), _observation(), "upd-1", "rank-anvesh"
    )
    assert isinstance(result, AnveshComparisonResult)
    # ANVESH's feedback step starts from the exact same object as baseline_c_ranking
    assert result.anvesh_feedback_result.prior_ranking is result.baseline_c_ranking


def test_confirmed_feedback_strengthens_anvesh_vs_baseline_c():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.9, "cam-B": 0.9}
    result = compare_anvesh_vs_baseline_c(
        "rank-c", "cs-1", WINDOW, evidence, reliability, _prediction(), _observation(), "upd-1", "rank-anvesh"
    )
    c_top = result.baseline_c_ranking.ranked_list[0]
    anvesh_top = result.anvesh_ranking.ranked_list[0]
    assert anvesh_top.hypothesis_id == c_top.hypothesis_id
    assert anvesh_top.belief > c_top.belief  # feedback confirmed and boosted it
    assert result.anvesh_feedback_result.update.outcome == HypothesisUpdateOutcome.CONFIRMED


def test_evidence_missing_leaves_anvesh_ranking_identical_to_baseline_c():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.9, "cam-B": 0.9}
    missing_observation = _observation(completeness=DataCompleteness.MISSING)

    result = compare_anvesh_vs_baseline_c(
        "rank-c", "cs-1", WINDOW, evidence, reliability, _prediction(), missing_observation, "upd-1", "rank-anvesh"
    )

    assert result.anvesh_feedback_result.update.outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING
    assert result.anvesh_ranking is result.baseline_c_ranking  # literally unchanged, not force-revised
    c_beliefs = [(e.hypothesis_id, e.belief) for e in result.baseline_c_ranking.ranked_list]
    anvesh_beliefs = [(e.hypothesis_id, e.belief) for e in result.anvesh_ranking.ranked_list]
    assert c_beliefs == anvesh_beliefs


def test_contradiction_is_visible_and_changes_anvesh_ranking():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.9, "cam-B": 0.9}
    contradicting_observation = _observation(camera="cam-A", completeness=DataCompleteness.FULL)  # wrong camera

    result = compare_anvesh_vs_baseline_c(
        "rank-c", "cs-1", WINDOW, evidence, reliability, _prediction(), contradicting_observation, "upd-1", "rank-anvesh"
    )

    assert result.anvesh_feedback_result.update.outcome == HypothesisUpdateOutcome.CONTRADICTED
    c_top_belief = result.baseline_c_ranking.ranked_list[0].belief
    anvesh_top_hid = result.anvesh_ranking.ranked_list[0].hypothesis_id
    original_top_hid = result.baseline_c_ranking.ranked_list[0].hypothesis_id
    anvesh_entry_for_original_top = next(e for e in result.anvesh_ranking.ranked_list if e.hypothesis_id == original_top_hid)
    assert anvesh_entry_for_original_top.belief < c_top_belief  # visibly downgraded


def test_complete_m4_to_m5_integration_is_deterministic():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.9, "cam-B": 0.9}
    r1 = compare_anvesh_vs_baseline_c(
        "rank-c", "cs-1", WINDOW, evidence, reliability, _prediction(), _observation(), "upd-1", "rank-anvesh"
    )
    r2 = compare_anvesh_vs_baseline_c(
        "rank-c", "cs-1", WINDOW, evidence, reliability, _prediction(), _observation(), "upd-1", "rank-anvesh"
    )
    assert [(e.hypothesis_id, e.belief) for e in r1.anvesh_ranking.ranked_list] == [
        (e.hypothesis_id, e.belief) for e in r2.anvesh_ranking.ranked_list
    ]
