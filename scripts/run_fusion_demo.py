"""M4 demo: reliability-discounted Dempster-Shafer cross-camera evidence
fusion, four deterministic scenarios.

Usage (from the repository root, with the venv active):

    python scripts/run_fusion_demo.py

Uses hand-constructed, clearly-synthetic Evidence (not real video) so each
scenario's expected behavior is exact and auditable. This demonstrates the
FUSION LAYER specifically (evidence -> belief -> ranking) -- it is not a
perception/M1-M3 demo (see scripts/run_pipeline.py and
scripts/run_corridor_state_demo.py for those).

Terminology note (per M4 scope): this is candidate-cause reasoning under
uncertainty, not causal inference, not accident detection, not
propagation prediction. Nothing printed here is a calibrated probability.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.fusion.ds_fusion import fuse_two_cameras
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.hypotheses.ranking import build_ranking
from anvesh.storage.schemas import Evidence, EvidenceCompleteness

WINDOW = (0.0, 10.0)


def _item(camera_id, hid, score, direction, completeness=EvidenceCompleteness.FULL):
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hid}:{WINDOW[0]}-{WINDOW[1]}",
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hid,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=completeness,
    )
    return EvidenceItem(
        evidence=evidence, evidence_type="demo", measured_value=None, direction=direction,
        reliability=1.0, source="demo_script", metadata={},
    )


def _evidence_set(strong_hid=None, strong_score=0.0, strong_direction=EvidenceDirection.SUPPORTS, camera_id="cam"):
    """All six hypotheses NEUTRAL/insufficient except (optionally) one."""
    items = []
    for hid in HYPOTHESIS_IDS:
        if hid == strong_hid:
            items.append(_item(camera_id, hid, strong_score, strong_direction))
        else:
            items.append(_item(camera_id, hid, 0.0, EvidenceDirection.NEUTRAL))
    return tuple(items)


def _all_insufficient(camera_id):
    return tuple(
        _item(camera_id, hid, 0.0, EvidenceDirection.INSUFFICIENT, EvidenceCompleteness.NONE)
        for hid in HYPOTHESIS_IDS
    )


def _print_camera_evidence(label, items):
    print(f"  {label}:")
    for item in items:
        print(
            f"    {item.evidence.hypothesis_id}: score={item.evidence.signature_match_score:.2f} "
            f"direction={item.direction.value:<11} completeness={item.evidence.evidence_completeness.value}"
        )


def _print_fusion_result(fusion_result):
    print(f"  conflict_k={fusion_result.conflict_k:.4f}  total_conflict={fusion_result.total_conflict}  "
          f"high_conflict={fusion_result.high_conflict}  lambda_shared={fusion_result.lambda_shared_used}")
    print(f"  final mass assignment: {{{', '.join(f'{h}={m:.3f}' for h, m in fusion_result.final_bpa.masses.items() if m > 0.001)}}}"
          f"  theta(ignorance)={fusion_result.final_bpa.theta_mass:.3f}")
    bel, pl = fusion_result.belief, fusion_result.plausibility
    print("  belief / plausibility (NOT probabilities):")
    for hid in HYPOTHESIS_IDS:
        if bel[hid] > 0.001 or pl[hid] < 0.999:
            print(f"    {hid}: Bel={bel[hid]:.3f}  Pl={pl[hid]:.3f}")


def _print_ranking(ranking):
    print(f"  outcome={ranking.outcome.value}")
    for entry in ranking.ranked_list[:3]:
        print(
            f"    {entry.hypothesis_id}: belief={entry.belief:.3f} plausibility={entry.plausibility:.3f} "
            f"tier={entry.confidence_tier.value} supporting_refs={entry.supporting_evidence_refs} "
            f"contradicting_refs={entry.contradicting_evidence_refs}"
        )


def scenario_1_compatible_evidence_strengthens():
    print("\n=== Scenario 1: Camera A strong for H1, Camera B compatible -> fusion should strengthen H1 ===")
    ev_a = _evidence_set("H1", 0.8, camera_id="cam-A")
    ev_b = _evidence_set("H1", 0.7, camera_id="cam-B")
    _print_camera_evidence("Camera A evidence", ev_a)
    _print_camera_evidence("Camera B evidence", ev_b)
    print("  reliability: cam-A=0.9, cam-B=0.9")

    fusion = fuse_two_cameras("cam-A", ev_a, 0.9, "cam-B", ev_b, 0.9, WINDOW)
    _print_fusion_result(fusion)
    ranking = build_ranking("rank-scenario-1", "cs-demo-1", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    _print_ranking(ranking)
    print(f"  CHECK: fused H1 belief ({fusion.belief['H1']:.3f}) > either camera's own score (0.8 / 0.7): "
          f"{fusion.belief['H1'] > 0.8}")


def scenario_2_disagreement_shows_conflict():
    print("\n=== Scenario 2: Camera A supports H1, Camera B supports H2 -> conflict must be visible ===")
    ev_a = _evidence_set("H1", 0.9, camera_id="cam-A")
    ev_b = _evidence_set("H2", 0.9, camera_id="cam-B")
    _print_camera_evidence("Camera A evidence", ev_a)
    _print_camera_evidence("Camera B evidence", ev_b)
    print("  reliability: cam-A=0.9, cam-B=0.9")

    fusion = fuse_two_cameras("cam-A", ev_a, 0.9, "cam-B", ev_b, 0.9, WINDOW)
    _print_fusion_result(fusion)
    ranking = build_ranking("rank-scenario-2", "cs-demo-2", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    _print_ranking(ranking)
    print(f"  CHECK: conflict_k is high and visible: {fusion.conflict_k:.3f}")


def scenario_3_low_reliability_reduces_influence():
    print("\n=== Scenario 3: Camera B has poor reliability -> its influence must be reduced ===")
    ev_a = _evidence_set("H1", 0.9, camera_id="cam-A")
    ev_b = _evidence_set("H2", 0.9, camera_id="cam-B")
    _print_camera_evidence("Camera A evidence", ev_a)
    _print_camera_evidence("Camera B evidence", ev_b)
    print("  reliability: cam-A=0.9 (good), cam-B=0.1 (poor -- e.g. degraded/occluded camera)")

    fusion = fuse_two_cameras("cam-A", ev_a, 0.9, "cam-B", ev_b, 0.1, WINDOW)
    _print_fusion_result(fusion)
    ranking = build_ranking("rank-scenario-3", "cs-demo-3", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    _print_ranking(ranking)
    print(f"  CHECK: H1 (cam-A's claim) dominates over H2 (cam-B's claim) now that B is discounted: "
          f"Bel(H1)={fusion.belief['H1']:.3f} > Bel(H2)={fusion.belief['H2']:.3f}")


def scenario_4_insufficient_evidence():
    print("\n=== Scenario 4: Both cameras lack sufficient evidence -> must abstain, not invent a cause ===")
    ev_a = _all_insufficient("cam-A")
    ev_b = _all_insufficient("cam-B")
    _print_camera_evidence("Camera A evidence", ev_a)
    _print_camera_evidence("Camera B evidence", ev_b)
    print("  reliability: cam-A=0.9, cam-B=0.9")

    fusion = fuse_two_cameras("cam-A", ev_a, 0.9, "cam-B", ev_b, 0.9, WINDOW)
    _print_fusion_result(fusion)
    ranking = build_ranking("rank-scenario-4", "cs-demo-4", fusion, {"cam-A": ev_a, "cam-B": ev_b})
    _print_ranking(ranking)
    print(f"  CHECK: outcome is insufficient_evidence, not a fabricated ranking: {ranking.outcome.value}")


def main() -> int:
    print("ANVESH M4 demo -- reliability-discounted Dempster-Shafer fusion (SYNTHETIC evidence)")
    scenario_1_compatible_evidence_strengthens()
    scenario_2_disagreement_shows_conflict()
    scenario_3_low_reliability_reduces_influence()
    scenario_4_insufficient_evidence()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
