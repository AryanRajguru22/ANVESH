import pytest

from anvesh.evaluation.metrics import (
    ALL_BASELINES,
    compute_abstention_metrics,
    compute_determinism_summary,
    compute_false_confidence,
    compute_feedback_effect,
    compute_propagation_errors,
    compute_ranking_metrics,
    compute_reliability_effect,
    compute_runtime_summary,
    evaluate,
    run_all_scenarios,
    run_scenario,
)
from anvesh.evaluation.scenarios import load_scenarios
from anvesh.storage.schemas import BaselineType, CandidateOutcome, HypothesisUpdateOutcome

SCENARIOS = load_scenarios()


@pytest.fixture(scope="module")
def results():
    return run_all_scenarios(SCENARIOS)


def test_run_scenario_produces_all_four_baselines():
    result = run_scenario(SCENARIOS[0])
    assert set(result.runs.keys()) == set(ALL_BASELINES)


def test_baseline_a_b_c_receive_the_same_evidence_object(results):
    """A/B/C must all be run against the identical evidence_by_camera --
    verified here by checking each baseline's ranking was in fact built
    from the same underlying Evidence records (via supporting refs), not
    merely by trusting the call site."""
    scenario = SCENARIOS[0]  # s1: both cameras support H1 only
    result = results[0]
    for baseline in (BaselineType.A, BaselineType.B, BaselineType.C):
        ranking = result.runs[baseline].ranking
        h1_entry = next(e for e in ranking.ranked_list if e.hypothesis_id == "H1")
        refs = set(h1_entry.supporting_evidence_refs)
        # every ref must trace back to an evidence_id actually present in the scenario's evidence
        all_scenario_refs = {
            item.evidence.evidence_id for items in scenario.evidence_by_camera.values() for item in items
        }
        assert refs <= all_scenario_refs


def test_anvesh_starts_from_baseline_c_ranking_when_confirmed(results):
    # s1 is CONFIRMED -- ANVESH's belief must be >= Baseline C's (strengthened, never contradicted)
    result = next(r for r in results if r.scenario.scenario_id == "s1_clear_cross_camera_agreement")
    c_top = result.runs[BaselineType.C].ranking.ranked_list[0]
    anvesh_top = result.runs[BaselineType.ANVESH].ranking.ranked_list[0]
    assert anvesh_top.hypothesis_id == c_top.hypothesis_id
    assert anvesh_top.belief >= c_top.belief


def _build_evidence_item(camera_id, hypothesis_id, score, direction):
    from anvesh.storage.schemas import Evidence, EvidenceCompleteness
    from anvesh.evidence.evidence_builder import EvidenceItem

    evidence = Evidence(
        evidence_id=f"{camera_id}:{hypothesis_id}:0-10",
        camera_id=camera_id,
        window=(0.0, 10.0),
        hypothesis_id=hypothesis_id,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=False,
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(evidence=evidence, evidence_type="test", measured_value=None, direction=direction, reliability=1.0, source="test", metadata={})


def _build_camera_evidence(camera_id, hid_with_score, score, direction):
    from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
    from anvesh.evidence.evidence_builder import EvidenceDirection

    return tuple(
        _build_evidence_item(camera_id, h, score if h == hid_with_score else 0.0, direction if h == hid_with_score else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )


def test_anvesh_equals_baseline_c_when_no_propagation_data():
    """Scenarios with propagation=None: ANVESH has nothing to revise
    Baseline C's ranking with, so it must equal it exactly."""
    from anvesh.evaluation.scenarios import Scenario, DesignLabel, ScenarioDesignLabelKind
    from anvesh.evidence.evidence_builder import EvidenceDirection

    scenario = Scenario(
        scenario_id="no-propagation-test",
        description="test",
        design_label=DesignLabel(ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1", "test scenario"),
        evidence_by_camera={
            "cam-A": _build_camera_evidence("cam-A", "H1", 0.7, EvidenceDirection.SUPPORTS),
            "cam-B": _build_camera_evidence("cam-B", "H1", 0.6, EvidenceDirection.SUPPORTS),
        },
        reliability_by_camera={"cam-A": 0.9, "cam-B": 0.9},
        single_camera_id="cam-A",
        propagation=None,
    )
    result = run_scenario(scenario)
    c_beliefs = [(e.hypothesis_id, e.belief) for e in result.runs[BaselineType.C].ranking.ranked_list]
    anvesh_beliefs = [(e.hypothesis_id, e.belief) for e in result.runs[BaselineType.ANVESH].ranking.ranked_list]
    assert c_beliefs == anvesh_beliefs
    assert result.anvesh_feedback_result is None


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def test_ranking_metrics_top1_accuracy_correct_for_clean_scenario(results):
    metrics = compute_ranking_metrics(results)
    # A, B, C should all get s1/s3/s4/s6/s8 right at minimum (H1 is the
    # dominant, uncontested signal); their accuracy must be high.
    assert metrics.top1_accuracy[BaselineType.A] >= 0.8
    assert metrics.applicable_scenario_count == 6  # 8 scenarios minus s2 (unresolvable) minus s5 (no cause)


def test_ranking_metrics_excludes_non_applicable_scenarios(results):
    metrics = compute_ranking_metrics(results)
    total_specific = sum(
        1 for r in results if r.scenario.design_label.kind.value == "specific_hypothesis"
    )
    assert metrics.applicable_scenario_count == total_specific


def test_outcome_distribution_sums_to_total_scenarios(results):
    metrics = compute_ranking_metrics(results)
    for baseline in ALL_BASELINES:
        total = sum(metrics.outcome_distribution[baseline].values())
        assert total == len(results)


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_abstention_metrics_reflect_the_no_cause_scenario(results):
    metrics = compute_abstention_metrics(results)
    for baseline in ALL_BASELINES:
        assert metrics.insufficient_evidence_count[baseline] >= 1  # s5 must trigger this for every baseline
    assert metrics.total_scenarios == len(SCENARIOS)


def test_abstention_rate_is_a_fraction(results):
    metrics = compute_abstention_metrics(results)
    for baseline in ALL_BASELINES:
        assert 0.0 <= metrics.insufficient_evidence_rate[baseline] <= 1.0
        assert 0.0 <= metrics.high_conflict_rate[baseline] <= 1.0


# ---------------------------------------------------------------------------
# Reliability effect
# ---------------------------------------------------------------------------


def test_reliability_effect_shows_a_positive_shift_toward_designed_correct(results):
    effect = compute_reliability_effect(results)
    assert effect is not None
    assert effect.designed_correct_hypothesis == "H1"
    # discounting the spurious camera must move belief in the correct
    # direction, not away from it
    assert effect.belief_shift_toward_designed_correct > 0


def test_reliability_effect_returns_none_for_missing_pair():
    effect = compute_reliability_effect((), "does-not-exist-1", "does-not-exist-2")
    assert effect is None


# ---------------------------------------------------------------------------
# Feedback effect
# ---------------------------------------------------------------------------


def test_feedback_effect_covers_all_outcomes(results):
    summary = compute_feedback_effect(results)
    assert set(summary.outcome_counts.keys()) == {"confirmed", "partial", "contradicted", "evidence_missing"}


def test_feedback_effect_detects_top_candidate_change_on_contradicted(results):
    summary = compute_feedback_effect(results)
    entry = next(e for e in summary.entries if e.scenario_id == "s7_propagation_contradicted_feedback")
    assert entry.feedback_outcome == "contradicted"
    assert entry.top_candidate_changed is True


def test_feedback_effect_no_change_on_evidence_missing(results):
    summary = compute_feedback_effect(results)
    entry = next(e for e in summary.entries if e.scenario_id == "s8_propagation_evidence_missing_feedback")
    assert entry.top_candidate_changed is False
    assert entry.baseline_c_top_belief == entry.anvesh_top_belief


def test_feedback_effect_confirmed_strengthens_belief(results):
    summary = compute_feedback_effect(results)
    entry = next(e for e in summary.entries if e.scenario_id == "s1_clear_cross_camera_agreement")
    assert entry.feedback_outcome == "confirmed"
    assert entry.anvesh_top_belief >= entry.baseline_c_top_belief


# ---------------------------------------------------------------------------
# Propagation errors
# ---------------------------------------------------------------------------


def test_propagation_errors_none_for_evidence_missing(results):
    errors = compute_propagation_errors(results)
    entry = next(e for e in errors if e.scenario_id == "s8_propagation_evidence_missing_feedback")
    assert entry.location_error is None
    assert entry.time_error is None


def test_propagation_errors_present_for_contradicted(results):
    errors = compute_propagation_errors(results)
    entry = next(e for e in errors if e.scenario_id == "s7_propagation_contradicted_feedback")
    assert entry.location_error is not None and entry.location_error > 0
    assert entry.time_error == 0.0


def test_propagation_errors_no_fake_zero_for_missing_case():
    """The N/A case must be None, never a manufactured 0.0."""
    errors = compute_propagation_errors(run_all_scenarios(load_scenarios()))
    missing_entry = next(e for e in errors if e.scenario_id == "s8_propagation_evidence_missing_feedback")
    assert missing_entry.location_error is None  # not 0.0
    assert missing_entry.time_error is None  # not 0.0


# ---------------------------------------------------------------------------
# False confidence
# ---------------------------------------------------------------------------


def test_false_confidence_reports_na_when_no_high_tier_predictions():
    # Verify the N/A contract directly via the function's documented
    # behavior on an empty results tuple (zero scenarios -> zero HIGH-tier count).
    result = compute_false_confidence(())
    for baseline in ALL_BASELINES:
        assert result.high_tier_count[baseline] == 0
        assert result.rate[baseline] is None  # N/A, not 0.0


def test_false_confidence_definition_is_documented(results):
    result = compute_false_confidence(results)
    assert "HIGH" in result.definition
    assert len(result.definition) > 20


def test_false_confidence_rate_is_valid_fraction_or_none(results):
    result = compute_false_confidence(results)
    for baseline in ALL_BASELINES:
        rate = result.rate[baseline]
        assert rate is None or 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_summary_all_deterministic(results):
    summary = compute_determinism_summary(results)
    assert summary.all_deterministic is True
    assert summary.non_deterministic_scenario_ids == ()


def test_each_scenario_result_carries_its_own_determinism_flag(results):
    for r in results:
        assert r.determinism_ok is True


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def test_runtime_summary_reports_wall_clock_note(results):
    summary = compute_runtime_summary(results)
    assert "wall-clock" in summary.note.lower()
    assert "scalability" not in summary.note.lower() or "not a scalability" in summary.note.lower()
    for baseline in ALL_BASELINES:
        assert summary.total_seconds[baseline] >= 0.0


# ---------------------------------------------------------------------------
# Full evaluate() bundle
# ---------------------------------------------------------------------------


def test_evaluate_returns_a_complete_report():
    report = evaluate(load_scenarios())
    assert len(report.scenario_results) == 8
    assert report.determinism.all_deterministic is True
    assert report.reliability_effect is not None
    assert len(report.feedback_effect.entries) == 8  # every scenario in this set has propagation data
