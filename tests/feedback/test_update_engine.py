import pytest

from anvesh.feedback.update_engine import (
    ComparisonResult,
    FeedbackAdjustmentConfig,
    FeedbackThresholds,
    apply_feedback,
    compare_propagation,
    revise_ranking,
)
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import (
    CandidateCauseHypothesis,
    CandidateOutcome,
    ConfidenceTier,
    DataCompleteness,
    Discrepancy,
    HypothesisUpdate,
    HypothesisUpdateOutcome,
    Measurement,
    PropagationObservation,
    PropagationPrediction,
    RankedHypothesisEntry,
)


def _prediction(arrival_camera="cam-B", arrival_time=100.0, growth_rate=5.0):
    return PropagationPrediction(
        prediction_id="pred-1",
        based_on_ranking_id="rank-orig",
        predicted_shockwave_speed=Measurement(value=2.0, error=0.0),
        predicted_arrival_camera=arrival_camera,
        predicted_arrival_time=arrival_time,
        predicted_queue_growth_rate=growth_rate,
        method="rankine_hugoniot_kinematic_wave_v1",
    )


def _observation(camera="cam-B", arrival_time=100.0, growth_rate=5.0, completeness=DataCompleteness.FULL):
    return PropagationObservation(
        observation_id="obs-1",
        prediction_id="pred-1",
        actual_arrival_camera=camera,
        actual_arrival_time=arrival_time,
        actual_queue_growth_rate=growth_rate,
        data_completeness=completeness,
    )


def _ranking(ranking_id="rank-orig"):
    return CandidateCauseHypothesis(
        ranking_id=ranking_id,
        corridor_state_id="cs-1",
        outcome=CandidateOutcome.RANKED,
        ranked_list=[
            RankedHypothesisEntry("H1", 0.6, 0.7, ConfidenceTier.HIGH, ["ev-a1"], []),
            RankedHypothesisEntry("H2", 0.2, 0.4, ConfidenceTier.LOW, ["ev-a2"], []),
            RankedHypothesisEntry("H3", 0.0, 0.2, ConfidenceTier.LOW, [], []),
            RankedHypothesisEntry("H4", 0.0, 0.2, ConfidenceTier.LOW, [], []),
            RankedHypothesisEntry("H5", 0.0, 0.2, ConfidenceTier.LOW, [], []),
            RankedHypothesisEntry("H6", 0.0, 0.2, ConfidenceTier.LOW, [], []),
        ],
        engine_model_id="ds_fusion",
        engine_model_version="v1",
    )


# ---------------------------------------------------------------------------
# compare_propagation: the four branches
# ---------------------------------------------------------------------------


def test_confirmed_when_camera_and_timing_match():
    result = compare_propagation(_prediction(), _observation())
    assert result.outcome == HypothesisUpdateOutcome.CONFIRMED
    assert result.location_match is True
    assert result.discrepancy.time_error == 0.0


def test_partial_when_camera_matches_but_timing_diverges():
    prediction = _prediction(arrival_time=100.0, growth_rate=5.0)
    observation = _observation(arrival_time=300.0, growth_rate=5.0)  # far outside time tolerance
    result = compare_propagation(prediction, observation)
    assert result.outcome == HypothesisUpdateOutcome.PARTIAL
    assert result.location_match is True
    assert result.discrepancy.time_error == pytest.approx(200.0)


def test_contradicted_when_full_observation_disagrees_on_camera():
    prediction = _prediction(arrival_camera="cam-B")
    observation = _observation(camera="cam-A", completeness=DataCompleteness.FULL)
    result = compare_propagation(prediction, observation)
    assert result.outcome == HypothesisUpdateOutcome.CONTRADICTED
    assert result.location_match is False


def test_evidence_missing_when_observation_unavailable():
    prediction = _prediction()
    observation = _observation(completeness=DataCompleteness.MISSING)
    result = compare_propagation(prediction, observation)
    assert result.outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING
    assert result.discrepancy is None
    assert result.location_match is None


def test_partial_completeness_never_produces_contradicted():
    """Unreliable/incomplete observation must not manufacture a false contradiction."""
    prediction = _prediction(arrival_camera="cam-B")
    observation = _observation(camera="cam-A", completeness=DataCompleteness.PARTIAL)  # wrong camera, but PARTIAL data
    result = compare_propagation(prediction, observation)
    assert result.outcome != HypothesisUpdateOutcome.CONTRADICTED
    assert result.outcome == HypothesisUpdateOutcome.PARTIAL


def test_partial_completeness_can_still_confirm_if_it_matches():
    prediction = _prediction()
    observation = _observation(completeness=DataCompleteness.PARTIAL)  # matches exactly, just incomplete
    result = compare_propagation(prediction, observation)
    assert result.outcome == HypothesisUpdateOutcome.CONFIRMED


def test_comparison_is_deterministic():
    prediction, observation = _prediction(), _observation()
    r1 = compare_propagation(prediction, observation)
    r2 = compare_propagation(prediction, observation)
    assert r1.outcome == r2.outcome
    assert r1.discrepancy == r2.discrepancy


def test_custom_thresholds_change_confirmed_vs_partial():
    prediction = _prediction(arrival_time=100.0)
    observation = _observation(arrival_time=110.0)  # 10s off
    strict = FeedbackThresholds(time_tolerance_s=5.0, rate_tolerance_veh_per_min=2.0)
    loose = FeedbackThresholds(time_tolerance_s=20.0, rate_tolerance_veh_per_min=2.0)
    assert compare_propagation(prediction, observation, strict).outcome == HypothesisUpdateOutcome.PARTIAL
    assert compare_propagation(prediction, observation, loose).outcome == HypothesisUpdateOutcome.CONFIRMED


# ---------------------------------------------------------------------------
# revise_ranking
# ---------------------------------------------------------------------------


def test_revise_ranking_boosts_top_hypothesis_on_confirmed():
    original = _ranking()
    revised = revise_ranking(original, HypothesisUpdateOutcome.CONFIRMED, "rank-revised-1")
    revised_h1 = next(e for e in revised.ranked_list if e.hypothesis_id == "H1")
    original_h1 = next(e for e in original.ranked_list if e.hypothesis_id == "H1")
    assert revised_h1.belief > original_h1.belief


def test_revise_ranking_handles_a_partial_ranked_list():
    """Regression: nothing in the frozen schema guarantees a prior ranking
    covers all six hypotheses -- a ranking with only one entry must still
    revise cleanly (missing hypotheses default to empty evidence refs),
    not raise a KeyError."""
    partial = CandidateCauseHypothesis(
        ranking_id="rank-partial",
        corridor_state_id="cs-1",
        outcome=CandidateOutcome.RANKED,
        ranked_list=[
            RankedHypothesisEntry("H1", 0.6, 0.8, ConfidenceTier.MEDIUM, ["ev-1"], []),
        ],
        engine_model_id="ds_fusion",
        engine_model_version="v1",
    )
    revised = revise_ranking(partial, HypothesisUpdateOutcome.CONFIRMED, "rank-revised-partial")
    assert {e.hypothesis_id for e in revised.ranked_list} == set(HYPOTHESIS_IDS)
    h2_entry = next(e for e in revised.ranked_list if e.hypothesis_id == "H2")
    assert h2_entry.supporting_evidence_refs == []
    assert h2_entry.contradicting_evidence_refs == []


def test_revise_ranking_penalizes_top_hypothesis_on_contradicted():
    original = _ranking()
    revised = revise_ranking(original, HypothesisUpdateOutcome.CONTRADICTED, "rank-revised-2")
    revised_h1 = next(e for e in revised.ranked_list if e.hypothesis_id == "H1")
    original_h1 = next(e for e in original.ranked_list if e.hypothesis_id == "H1")
    assert revised_h1.belief < original_h1.belief


def test_contradiction_lets_an_alternative_rise():
    original = _ranking()  # H1=0.6 top, H2=0.2 second
    revised = revise_ranking(original, HypothesisUpdateOutcome.CONTRADICTED, "rank-revised-3")
    assert revised.ranked_list[0].hypothesis_id == "H2"  # H1 penalized enough that H2 overtakes it


def test_revise_ranking_rejects_evidence_missing():
    with pytest.raises(ValueError):
        revise_ranking(_ranking(), HypothesisUpdateOutcome.EVIDENCE_MISSING, "rank-x")


def test_revise_ranking_preserves_evidence_refs():
    original = _ranking()
    revised = revise_ranking(original, HypothesisUpdateOutcome.CONFIRMED, "rank-revised-4")
    revised_h1 = next(e for e in revised.ranked_list if e.hypothesis_id == "H1")
    assert revised_h1.supporting_evidence_refs == ["ev-a1"]


def test_revise_ranking_does_not_mutate_original():
    original = _ranking()
    original_beliefs = [e.belief for e in original.ranked_list]
    revise_ranking(original, HypothesisUpdateOutcome.CONFIRMED, "rank-revised-5")
    assert [e.belief for e in original.ranked_list] == original_beliefs


def test_custom_adjustment_config_is_respected():
    original = _ranking()
    mild = FeedbackAdjustmentConfig(confirmed_boost_factor=1.01, partial_boost_factor=1.0, contradicted_penalty_factor=0.9)
    strong = FeedbackAdjustmentConfig(confirmed_boost_factor=2.0, partial_boost_factor=1.0, contradicted_penalty_factor=0.9)
    mild_revised = revise_ranking(original, HypothesisUpdateOutcome.CONFIRMED, "r1", mild)
    strong_revised = revise_ranking(original, HypothesisUpdateOutcome.CONFIRMED, "r2", strong)
    mild_h1 = next(e for e in mild_revised.ranked_list if e.hypothesis_id == "H1").belief
    strong_h1 = next(e for e in strong_revised.ranked_list if e.hypothesis_id == "H1").belief
    assert strong_h1 > mild_h1


# ---------------------------------------------------------------------------
# apply_feedback: full orchestration, provenance, history preservation
# ---------------------------------------------------------------------------


def test_apply_feedback_confirmed_preserves_history_and_strengthens():
    prior = _ranking()
    result = apply_feedback("upd-1", "rank-revised-6", prior, _prediction(), _observation())

    assert isinstance(result.update, HypothesisUpdate)
    assert result.update.outcome == HypothesisUpdateOutcome.CONFIRMED
    assert result.prior_ranking is prior  # original object, untouched
    assert result.prior_ranking.ranking_id == "rank-orig"
    assert result.revised_ranking.ranking_id == "rank-revised-6"
    assert result.revised_ranking.ranking_id != result.prior_ranking.ranking_id
    revised_h1 = next(e for e in result.revised_ranking.ranked_list if e.hypothesis_id == "H1")
    original_h1 = next(e for e in prior.ranked_list if e.hypothesis_id == "H1")
    assert revised_h1.belief >= original_h1.belief
    assert result.reason  # a human-readable explanation is present


def test_apply_feedback_evidence_missing_does_not_change_ranking():
    prior = _ranking()
    observation = _observation(completeness=DataCompleteness.MISSING)
    result = apply_feedback("upd-2", "rank-would-be-revised", prior, _prediction(), observation)

    assert result.update.outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING
    assert result.revised_ranking is prior  # literally the same object -- no fabricated revision
    assert result.update.revised_ranking_id == prior.ranking_id  # points back at the original, not a new id
    assert result.update.discrepancy is None


def test_apply_feedback_contradicted_downgrades_and_promotes_alternative():
    prior = _ranking()
    prediction = _prediction(arrival_camera="cam-B")
    observation = _observation(camera="cam-A", completeness=DataCompleteness.FULL)

    result = apply_feedback("upd-3", "rank-revised-7", prior, prediction, observation)

    assert result.update.outcome == HypothesisUpdateOutcome.CONTRADICTED
    assert result.revised_ranking.ranked_list[0].hypothesis_id == "H2"
    assert result.prior_ranking.ranked_list[0].hypothesis_id == "H1"  # history untouched


def test_apply_feedback_records_correct_update_linkage():
    prior = _ranking()
    result = apply_feedback("upd-4", "rank-revised-8", prior, _prediction(), _observation())
    assert result.update.prior_ranking_id == prior.ranking_id
    assert result.update.propagation_observation_id == "obs-1"
    assert result.update.revised_ranking_id == result.revised_ranking.ranking_id


def test_apply_feedback_deterministic_repeated_execution():
    prior = _ranking()
    r1 = apply_feedback("upd-5", "rank-revised-9", prior, _prediction(), _observation())
    r2 = apply_feedback("upd-5", "rank-revised-9", prior, _prediction(), _observation())
    assert [(e.hypothesis_id, e.belief) for e in r1.revised_ranking.ranked_list] == [
        (e.hypothesis_id, e.belief) for e in r2.revised_ranking.ranked_list
    ]
    assert r1.update.outcome == r2.update.outcome
