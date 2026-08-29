"""Purpose-built evaluation metrics for the fixed scenario set
(`evaluation.scenarios`), computed by actually running
`evaluation.baselines.run_baseline`/`compare_anvesh_vs_baseline_c` --
no fusion/ranking/feedback logic is duplicated here.

Deliberately NOT built on the frozen `storage.schemas.EvaluationMetrics`:
that type has 11 mandatory fields including `congestion_p/r/f1`, which
validate M3's congestion-*detection* mechanism, not the M4/M5 cause-
*ranking* mechanism this evaluation targets -- forcing a number into them
would be exactly the "fake zero for an inapplicable metric" this phase
was told not to do. Every result type below is scoped to what this
evaluation actually measures, and every metric that cannot be honestly
computed for a given scenario/baseline is `None` (reported as "N/A" by
`scripts/run_evaluation.py`), never a fabricated 0.0.

Every "correctness" check compares against a scenario's
`design_label` -- a SYNTHETIC SCENARIO DESIGN LABEL (see
`evaluation.scenarios`'s module docstring), never real-world ground
truth. Scenarios labeled `GENUINELY_UNRESOLVABLE` or `NO_CAUSE_PRESENT`
have no correct answer to check against and are excluded from top-1/
top-k/false-confidence accounting (not scored as wrong).

Belief/plausibility values handled here are Dempster-Shafer quantities
(blueprint Part 6), never calibrated probabilities -- nothing in this
module computes or reports Brier score/ECE (see the module docstring of
`scripts/run_evaluation.py` for why they were omitted entirely, per this
phase's explicit instruction not to make them a headline metric).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from anvesh.evaluation.baselines import compare_anvesh_vs_baseline_c, run_baseline
from anvesh.evaluation.scenarios import Scenario, ScenarioDesignLabelKind
from anvesh.storage.schemas import BaselineType, CandidateOutcome, ConfidenceTier

ALL_BASELINES = (BaselineType.A, BaselineType.B, BaselineType.C, BaselineType.ANVESH)


# ---------------------------------------------------------------------------
# Per-scenario execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineRunResult:
    baseline: BaselineType
    ranking: object  # schemas.CandidateCauseHypothesis
    runtime_seconds: float


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    runs: dict  # BaselineType -> BaselineRunResult (A, B, C, ANVESH)
    anvesh_feedback_result: object  # feedback.update_engine.FeedbackResult
    determinism_ok: bool


def _timed(fn) -> tuple:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _run_all_baselines(scenario: Scenario) -> tuple:
    """Runs A, B, C on `scenario`'s evidence, then ANVESH via
    `compare_anvesh_vs_baseline_c` (which itself calls Baseline C again
    internally -- that second C call is intentionally not reused here so
    Baseline C's OWN reported runtime reflects a single, standalone call,
    matching what a user actually invoking Baseline C would experience).
    Returns (runs: dict[BaselineType, BaselineRunResult], feedback_result).
    """
    runs = {}

    for baseline in (BaselineType.A, BaselineType.B, BaselineType.C):
        kwargs = {}
        if baseline == BaselineType.A:
            kwargs["single_camera_id"] = scenario.single_camera_id
        else:
            kwargs["reliability_by_camera"] = scenario.reliability_by_camera
        ranking, elapsed = _timed(
            lambda b=baseline, k=kwargs: run_baseline(
                b, f"eval-{scenario.scenario_id}-{b.value}", "cs-eval", (0.0, 10.0), scenario.evidence_by_camera, **k
            )
        )
        runs[baseline] = BaselineRunResult(baseline=baseline, ranking=ranking, runtime_seconds=elapsed)

    feedback_result = None
    if scenario.propagation is not None:
        comparison, elapsed = _timed(
            lambda: compare_anvesh_vs_baseline_c(
                f"eval-{scenario.scenario_id}-anvesh-c",
                "cs-eval",
                (0.0, 10.0),
                scenario.evidence_by_camera,
                scenario.reliability_by_camera,
                scenario.propagation.prediction,
                scenario.propagation.observation,
                f"eval-{scenario.scenario_id}-update",
                f"eval-{scenario.scenario_id}-anvesh-revised",
                corridor_topology=scenario.propagation.corridor_topology,
            )
        )
        runs[BaselineType.ANVESH] = BaselineRunResult(
            baseline=BaselineType.ANVESH, ranking=comparison.anvesh_ranking, runtime_seconds=elapsed
        )
        feedback_result = comparison.anvesh_feedback_result
    else:
        # No propagation data for this scenario -- ANVESH has nothing to
        # revise Baseline C's ranking with, so it IS Baseline C's ranking
        # here. This is reported explicitly, not silently assumed.
        runs[BaselineType.ANVESH] = BaselineRunResult(
            baseline=BaselineType.ANVESH, ranking=runs[BaselineType.C].ranking, runtime_seconds=0.0
        )

    return runs, feedback_result


def _rankings_equal(a, b) -> bool:
    a_entries = [(e.hypothesis_id, e.belief, e.plausibility) for e in a.ranked_list]
    b_entries = [(e.hypothesis_id, e.belief, e.plausibility) for e in b.ranked_list]
    return a_entries == b_entries and a.outcome == b.outcome


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Run one scenario through A/B/C/ANVESH once, plus a second full
    rerun purely to check determinism (design constraint: no tuning, no
    randomness -- reruns must be bit-for-bit identical)."""
    runs, feedback_result = _run_all_baselines(scenario)
    rerun, _rerun_feedback = _run_all_baselines(scenario)

    determinism_ok = all(_rankings_equal(runs[b].ranking, rerun[b].ranking) for b in ALL_BASELINES)

    return ScenarioResult(scenario=scenario, runs=runs, anvesh_feedback_result=feedback_result, determinism_ok=determinism_ok)


def run_all_scenarios(scenarios: tuple) -> tuple:
    return tuple(run_scenario(s) for s in scenarios)


# ---------------------------------------------------------------------------
# Ranking metrics: top-1 / top-2 correctness, outcome distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingMetrics:
    applicable_scenario_count: int  # scenarios with a SPECIFIC_HYPOTHESIS design label
    top1_correct: dict  # BaselineType -> int
    top1_accuracy: dict  # BaselineType -> float or None (N/A if applicable_scenario_count == 0)
    top2_correct: dict
    top2_accuracy: dict
    outcome_distribution: dict  # BaselineType -> {CandidateOutcome: count}


def compute_ranking_metrics(results: tuple) -> RankingMetrics:
    applicable = [r for r in results if r.scenario.design_label.kind == ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS]
    n = len(applicable)

    top1_correct = {b: 0 for b in ALL_BASELINES}
    top2_correct = {b: 0 for b in ALL_BASELINES}
    outcome_distribution = {b: {} for b in ALL_BASELINES}

    for result in results:
        for baseline in ALL_BASELINES:
            ranking = result.runs[baseline].ranking
            outcome_distribution[baseline][ranking.outcome] = outcome_distribution[baseline].get(ranking.outcome, 0) + 1

    for result in applicable:
        expected = result.scenario.design_label.hypothesis_id
        for baseline in ALL_BASELINES:
            ranked_list = result.runs[baseline].ranking.ranked_list
            top1_ids = [e.hypothesis_id for e in ranked_list[:1]]
            top2_ids = [e.hypothesis_id for e in ranked_list[:2]]
            if expected in top1_ids:
                top1_correct[baseline] += 1
            if expected in top2_ids:
                top2_correct[baseline] += 1

    top1_accuracy = {b: (top1_correct[b] / n if n > 0 else None) for b in ALL_BASELINES}
    top2_accuracy = {b: (top2_correct[b] / n if n > 0 else None) for b in ALL_BASELINES}

    return RankingMetrics(
        applicable_scenario_count=n,
        top1_correct=top1_correct,
        top1_accuracy=top1_accuracy,
        top2_correct=top2_correct,
        top2_accuracy=top2_accuracy,
        outcome_distribution=outcome_distribution,
    )


# ---------------------------------------------------------------------------
# Abstention / conflict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionMetrics:
    total_scenarios: int
    insufficient_evidence_count: dict  # BaselineType -> int
    insufficient_evidence_rate: dict  # BaselineType -> float
    high_conflict_count: dict
    high_conflict_rate: dict


def compute_abstention_metrics(results: tuple) -> AbstentionMetrics:
    n = len(results)
    insufficient = {b: 0 for b in ALL_BASELINES}
    high_conflict = {b: 0 for b in ALL_BASELINES}

    for result in results:
        for baseline in ALL_BASELINES:
            outcome = result.runs[baseline].ranking.outcome
            if outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE:
                insufficient[baseline] += 1
            elif outcome == CandidateOutcome.HIGH_CONFLICT:
                high_conflict[baseline] += 1

    return AbstentionMetrics(
        total_scenarios=n,
        insufficient_evidence_count=insufficient,
        insufficient_evidence_rate={b: insufficient[b] / n for b in ALL_BASELINES},
        high_conflict_count=high_conflict,
        high_conflict_rate={b: high_conflict[b] / n for b in ALL_BASELINES},
    )


# ---------------------------------------------------------------------------
# Reliability effect: paired equal-vs-degraded comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityEffectResult:
    equal_scenario_id: str
    degraded_scenario_id: str
    equal_top_hypothesis: str
    equal_top_belief: float
    degraded_top_hypothesis: str
    degraded_top_belief: float
    designed_correct_hypothesis: str
    belief_shift_toward_designed_correct: float  # degraded_belief_of_designed - equal_belief_of_designed


def compute_reliability_effect(
    results: tuple, equal_scenario_id: str = "s3_reliability_equal", degraded_scenario_id: str = "s4_reliability_degraded"
) -> object:
    equal_result = next((r for r in results if r.scenario.scenario_id == equal_scenario_id), None)
    degraded_result = next((r for r in results if r.scenario.scenario_id == degraded_scenario_id), None)
    if equal_result is None or degraded_result is None:
        return None  # the paired scenarios aren't both present -- N/A, not fabricated

    designed = equal_result.scenario.design_label.hypothesis_id
    equal_c = equal_result.runs[BaselineType.C].ranking
    degraded_c = degraded_result.runs[BaselineType.C].ranking

    equal_belief_of_designed = next(e.belief for e in equal_c.ranked_list if e.hypothesis_id == designed)
    degraded_belief_of_designed = next(e.belief for e in degraded_c.ranked_list if e.hypothesis_id == designed)

    return ReliabilityEffectResult(
        equal_scenario_id=equal_scenario_id,
        degraded_scenario_id=degraded_scenario_id,
        equal_top_hypothesis=equal_c.ranked_list[0].hypothesis_id,
        equal_top_belief=equal_c.ranked_list[0].belief,
        degraded_top_hypothesis=degraded_c.ranked_list[0].hypothesis_id,
        degraded_top_belief=degraded_c.ranked_list[0].belief,
        designed_correct_hypothesis=designed,
        belief_shift_toward_designed_correct=degraded_belief_of_designed - equal_belief_of_designed,
    )


# ---------------------------------------------------------------------------
# Feedback effect: Baseline C -> ANVESH, per outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedbackEffectEntry:
    scenario_id: str
    feedback_outcome: str
    baseline_c_top: str
    baseline_c_top_belief: float
    anvesh_top: str
    anvesh_top_belief: float
    top_candidate_changed: bool


@dataclass(frozen=True)
class FeedbackEffectSummary:
    entries: tuple
    outcome_counts: dict  # str (outcome value) -> int


def compute_feedback_effect(results: tuple) -> FeedbackEffectSummary:
    entries = []
    outcome_counts = {}

    for result in results:
        if result.anvesh_feedback_result is None:
            continue
        update = result.anvesh_feedback_result.update
        c_ranking = result.runs[BaselineType.C].ranking
        anvesh_ranking = result.runs[BaselineType.ANVESH].ranking

        entries.append(
            FeedbackEffectEntry(
                scenario_id=result.scenario.scenario_id,
                feedback_outcome=update.outcome.value,
                baseline_c_top=c_ranking.ranked_list[0].hypothesis_id,
                baseline_c_top_belief=c_ranking.ranked_list[0].belief,
                anvesh_top=anvesh_ranking.ranked_list[0].hypothesis_id,
                anvesh_top_belief=anvesh_ranking.ranked_list[0].belief,
                top_candidate_changed=(c_ranking.ranked_list[0].hypothesis_id != anvesh_ranking.ranked_list[0].hypothesis_id),
            )
        )
        outcome_counts[update.outcome.value] = outcome_counts.get(update.outcome.value, 0) + 1

    return FeedbackEffectSummary(entries=tuple(entries), outcome_counts=outcome_counts)


# ---------------------------------------------------------------------------
# Propagation location/time error (only where a FULL comparison happened)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationErrorEntry:
    scenario_id: str
    location_error: object  # float, or None if no discrepancy was computed (e.g. EVIDENCE_MISSING)
    time_error: object


def compute_propagation_errors(results: tuple) -> tuple:
    entries = []
    for result in results:
        if result.anvesh_feedback_result is None:
            continue
        discrepancy = result.anvesh_feedback_result.update.discrepancy
        if discrepancy is None:
            entries.append(PropagationErrorEntry(result.scenario.scenario_id, None, None))
        else:
            entries.append(PropagationErrorEntry(result.scenario.scenario_id, discrepancy.location_error, discrepancy.time_error))
    return tuple(entries)


# ---------------------------------------------------------------------------
# False-confidence rate -- explicit definition (see module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FalseConfidenceResult:
    definition: str
    high_tier_count: dict  # BaselineType -> int
    high_tier_wrong_count: dict  # BaselineType -> int
    rate: dict  # BaselineType -> float or None (N/A if high_tier_count == 0)


FALSE_CONFIDENCE_DEFINITION = (
    "Among this baseline's HIGH-confidence-tier top-1 selections (on scenarios with a "
    "SPECIFIC_HYPOTHESIS design label only), the fraction whose top-1 hypothesis_id does "
    "NOT match the scenario's design label. Reported as N/A, not 0.0, when a baseline "
    "produced zero HIGH-tier top-1 selections."
)


def compute_false_confidence(results: tuple) -> FalseConfidenceResult:
    applicable = [r for r in results if r.scenario.design_label.kind == ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS]

    high_tier_count = {b: 0 for b in ALL_BASELINES}
    high_tier_wrong_count = {b: 0 for b in ALL_BASELINES}

    for result in applicable:
        expected = result.scenario.design_label.hypothesis_id
        for baseline in ALL_BASELINES:
            top_entry = result.runs[baseline].ranking.ranked_list[0]
            if top_entry.confidence_tier == ConfidenceTier.HIGH:
                high_tier_count[baseline] += 1
                if top_entry.hypothesis_id != expected:
                    high_tier_wrong_count[baseline] += 1

    rate = {
        b: (high_tier_wrong_count[b] / high_tier_count[b] if high_tier_count[b] > 0 else None) for b in ALL_BASELINES
    }

    return FalseConfidenceResult(
        definition=FALSE_CONFIDENCE_DEFINITION,
        high_tier_count=high_tier_count,
        high_tier_wrong_count=high_tier_wrong_count,
        rate=rate,
    )


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeSummary:
    note: str
    total_seconds: dict  # BaselineType -> float
    mean_seconds_per_scenario: dict  # BaselineType -> float


def compute_runtime_summary(results: tuple) -> RuntimeSummary:
    n = len(results)
    total = {b: 0.0 for b in ALL_BASELINES}
    for result in results:
        for baseline in ALL_BASELINES:
            total[baseline] += result.runs[baseline].runtime_seconds
    return RuntimeSummary(
        note="Wall-clock runtime on this machine for this run -- not a scalability benchmark.",
        total_seconds=total,
        mean_seconds_per_scenario={b: (total[b] / n if n > 0 else 0.0) for b in ALL_BASELINES},
    )


# ---------------------------------------------------------------------------
# Determinism summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterminismSummary:
    all_deterministic: bool
    non_deterministic_scenario_ids: tuple


def compute_determinism_summary(results: tuple) -> DeterminismSummary:
    failing = tuple(r.scenario.scenario_id for r in results if not r.determinism_ok)
    return DeterminismSummary(all_deterministic=(len(failing) == 0), non_deterministic_scenario_ids=failing)


# ---------------------------------------------------------------------------
# Top-level report bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationReport:
    scenario_results: tuple
    ranking_metrics: RankingMetrics
    abstention_metrics: AbstentionMetrics
    reliability_effect: object
    feedback_effect: FeedbackEffectSummary
    propagation_errors: tuple
    false_confidence: FalseConfidenceResult
    runtime: RuntimeSummary
    determinism: DeterminismSummary


def evaluate(scenarios: tuple) -> EvaluationReport:
    results = run_all_scenarios(scenarios)
    return EvaluationReport(
        scenario_results=results,
        ranking_metrics=compute_ranking_metrics(results),
        abstention_metrics=compute_abstention_metrics(results),
        reliability_effect=compute_reliability_effect(results),
        feedback_effect=compute_feedback_effect(results),
        propagation_errors=compute_propagation_errors(results),
        false_confidence=compute_false_confidence(results),
        runtime=compute_runtime_summary(results),
        determinism=compute_determinism_summary(results),
    )
