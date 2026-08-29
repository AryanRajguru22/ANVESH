"""Evaluation phase entry point: runs the fixed synthetic scenario set
(`anvesh/evaluation/scenarios.py`) through Baseline A/B/C/ANVESH via the
EXISTING `evaluation.baselines.run_baseline`/`compare_anvesh_vs_baseline_c`
machinery, and prints a concise comparison report.

Usage (from the repository root, with the venv active):

    python scripts/run_evaluation.py

*** THIS IS A CONTROLLED SYNTHETIC MECHANISM EVALUATION ***

It answers exactly one question: "given evidence WE authored to imply a
specific candidate cause (or deliberately NOT to), does each baseline's
output match what we designed?" It does NOT and CANNOT establish:

    real-world accuracy          real CCTV performance
    production readiness          causal inference
    calibrated probabilities      statistical significance
    generalization to real Indian traffic
    scalability

Every "correct"/"accuracy" number below is scored against a SYNTHETIC
SCENARIO DESIGN LABEL (see `evaluation/scenarios.py`'s module docstring)
-- never real-world ground truth, expert adjudication, or proof that
ANVESH is right about anything real. Belief/plausibility values are
Dempster-Shafer quantities, not probabilities. "Candidate-cause ranking"
is the only correct term for what this system produces -- never "causal
inference."
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.evaluation.metrics import ALL_BASELINES, evaluate
from anvesh.evaluation.scenarios import ScenarioDesignLabelKind, load_scenarios
from anvesh.storage.schemas import BaselineType

_BASELINE_HEADER = f"{'A':>8} {'B':>8} {'C':>8} {'ANVESH':>8}"


def _fmt_pct(value) -> str:
    return "N/A" if value is None else f"{value * 100:5.1f}%"


_LABEL_ABBREVIATIONS = {
    ScenarioDesignLabelKind.GENUINELY_UNRESOLVABLE: "unresolvable",
    ScenarioDesignLabelKind.NO_CAUSE_PRESENT: "no_cause",
}


def _label_str(design_label) -> str:
    if design_label.kind == ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS:
        return design_label.hypothesis_id
    return _LABEL_ABBREVIATIONS[design_label.kind]


def main() -> int:
    print("=" * 78)
    print("ANVESH EVALUATION PHASE -- controlled synthetic mechanism comparison")
    print("A = single-camera | B = naive averaging | C = reliability-aware fusion | ANVESH = C + feedback")
    print("=" * 78)

    scenarios = load_scenarios()
    report = evaluate(scenarios)

    # -----------------------------------------------------------------
    # Scenario-by-scenario top-1 comparison: A -> B -> C, and C -> ANVESH
    # -----------------------------------------------------------------
    print(f"\n--- Scenario | design label | top-1 candidate per baseline ({len(scenarios)} scenarios) ---")
    print(f"{'scenario_id':<42} {'label':>13} {'A':>6} {'B':>6} {'C':>6} {'ANVESH':>8} {'outcome(ANVESH)':>18}")
    for result in report.scenario_results:
        tops = {b: result.runs[b].ranking.ranked_list[0].hypothesis_id for b in ALL_BASELINES}
        anvesh_outcome = result.runs[BaselineType.ANVESH].ranking.outcome.value
        print(
            f"{result.scenario.scenario_id:<42} {_label_str(result.scenario.design_label):>13} "
            f"{tops[BaselineType.A]:>6} {tops[BaselineType.B]:>6} {tops[BaselineType.C]:>6} "
            f"{tops[BaselineType.ANVESH]:>8} {anvesh_outcome:>18}"
        )

    # -----------------------------------------------------------------
    # Ranking metrics
    # -----------------------------------------------------------------
    rm = report.ranking_metrics
    print(f"\n--- Ranking quality (against SYNTHETIC design labels; N={rm.applicable_scenario_count} applicable scenarios) ---")
    print(f"{'metric':<20}{_BASELINE_HEADER}")
    print(f"{'top-1 accuracy':<20}" + " ".join(f"{_fmt_pct(rm.top1_accuracy[b]):>8}" for b in ALL_BASELINES))
    print(f"{'top-2 accuracy':<20}" + " ".join(f"{_fmt_pct(rm.top2_accuracy[b]):>8}" for b in ALL_BASELINES))
    print("NOTE: s2 (genuinely_unresolvable) and s5 (no_cause_present) are excluded -- they have no correct answer to score against.")

    # -----------------------------------------------------------------
    # Abstention / conflict
    # -----------------------------------------------------------------
    am = report.abstention_metrics
    print(f"\n--- Abstention / conflict behavior (N={am.total_scenarios} scenarios) ---")
    print(f"{'metric':<28}{_BASELINE_HEADER}")
    print(f"{'insufficient_evidence rate':<28}" + " ".join(f"{_fmt_pct(am.insufficient_evidence_rate[b]):>8}" for b in ALL_BASELINES))
    print(f"{'high_conflict rate':<28}" + " ".join(f"{_fmt_pct(am.high_conflict_rate[b]):>8}" for b in ALL_BASELINES))

    # -----------------------------------------------------------------
    # Reliability effect
    # -----------------------------------------------------------------
    re = report.reliability_effect
    print("\n--- Reliability effect (s3 equal-reliability vs s4 degraded-reliability, IDENTICAL evidence) ---")
    if re is None:
        print("  N/A -- paired scenarios not found.")
    else:
        print(f"  designed-correct hypothesis: {re.designed_correct_hypothesis}")
        print(f"  s3 (equal reliability)   -> Baseline C top={re.equal_top_hypothesis}, belief={re.equal_top_belief:.3f}")
        print(f"  s4 (degraded reliability)-> Baseline C top={re.degraded_top_hypothesis}, belief={re.degraded_top_belief:.3f}")
        direction = "TOWARD" if re.belief_shift_toward_designed_correct > 0 else "AWAY FROM"
        print(f"  belief shift {direction} the designed-correct hypothesis: {re.belief_shift_toward_designed_correct:+.3f}")

    # -----------------------------------------------------------------
    # Feedback effect (C -> ANVESH)
    # -----------------------------------------------------------------
    fe = report.feedback_effect
    print(f"\n--- Feedback effect: Baseline C -> ANVESH, same starting ranking (outcome counts: {fe.outcome_counts}) ---")
    print(f"{'scenario_id':<42} {'outcome':>17} {'C top/belief':>16} {'ANVESH top/belief':>18} {'top changed?':>13}")
    for entry in fe.entries:
        c_cell = f"{entry.baseline_c_top}/{entry.baseline_c_top_belief:.3f}"
        anvesh_cell = f"{entry.anvesh_top}/{entry.anvesh_top_belief:.3f}"
        print(
            f"{entry.scenario_id:<42} {entry.feedback_outcome:>17} "
            f"{c_cell:>16} {anvesh_cell:>18} {str(entry.top_candidate_changed):>13}"
        )

    # -----------------------------------------------------------------
    # Propagation location/time error (only where a real comparison happened)
    # -----------------------------------------------------------------
    print("\n--- Propagation prediction error (location_error m, time_error s) -- N/A where no FULL observation existed ---")
    for entry in report.propagation_errors:
        loc = "N/A" if entry.location_error is None else f"{entry.location_error:.1f} m"
        t = "N/A" if entry.time_error is None else f"{entry.time_error:.1f} s"
        print(f"  {entry.scenario_id:<42} location_error={loc:>10}  time_error={t:>8}")

    # -----------------------------------------------------------------
    # False confidence
    # -----------------------------------------------------------------
    fc = report.false_confidence
    print(f"\n--- False-confidence rate ---\n  Definition: {fc.definition}")
    print(f"{'':<20}{_BASELINE_HEADER}")
    print(f"{'HIGH-tier count':<20}" + " ".join(f"{fc.high_tier_count[b]:>8}" for b in ALL_BASELINES))
    print(f"{'false-confidence':<20}" + " ".join(f"{_fmt_pct(fc.rate[b]):>8}" for b in ALL_BASELINES))

    # -----------------------------------------------------------------
    # Determinism + runtime
    # -----------------------------------------------------------------
    print(f"\n--- Determinism (each scenario run twice, compared bit-for-bit) ---")
    print(f"  all scenarios deterministic: {report.determinism.all_deterministic}")
    if not report.determinism.all_deterministic:
        print(f"  NON-deterministic scenarios: {report.determinism.non_deterministic_scenario_ids}")

    rt = report.runtime
    print(f"\n--- Runtime ({rt.note}) ---")
    print(f"{'mean seconds/scenario':<24}" + " ".join(f"{rt.mean_seconds_per_scenario[b]:>8.5f}" for b in ALL_BASELINES))

    # -----------------------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("This is a CONTROLLED SYNTHETIC MECHANISM EVALUATION on hand-authored")
    print("scenarios. It does NOT establish real-world accuracy, real CCTV")
    print("performance, production readiness, causal inference, calibrated")
    print("probabilities, statistical significance, generalization to real Indian")
    print("traffic, or scalability. 'Design labels' are OUR authoring intent, not")
    print("ground truth. All belief/plausibility values are Dempster-Shafer")
    print("quantities, never probabilities. ANVESH's output is a candidate-cause")
    print("ranking, never a causal claim.")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
