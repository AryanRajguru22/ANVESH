"""Baseline ladder (blueprint Part 10): A (single-camera) / B (naive
averaging) / C (reliability-aware fusion) / ANVESH (same fusion foundation
as C -- M5/M6 will extend it with propagation feedback).

M4 scope, per the blueprint's own note: "the important thing is that
A/B/C can be executed independently on the SAME evidence" -- this module
is a thin, uniform dispatch over `fusion.ds_fusion`, not the full
evaluation/metrics framework (blueprint Part 11's ablation harness,
`evaluation/metrics.py`, is still unimplemented -- milestone M8).

All four baselines take the SAME `evidence_by_camera`/`reliability_by_camera`
inputs and produce a `CandidateCauseHypothesis` via the SAME
`hypotheses.ranking.build_ranking`, so their outputs are directly
comparable.
"""

from __future__ import annotations

from anvesh.fusion.ds_fusion import FusionResult, evidence_items_to_bpa, fuse_two_cameras, normalize_masses, single_camera_result
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.hypotheses.ranking import build_ranking
from anvesh.storage.schemas import BaselineType, CandidateCauseHypothesis


def _naive_average(camera_a_id: str, camera_a_evidence: tuple, camera_b_id: str, camera_b_evidence: tuple, window: tuple) -> FusionResult:
    """Baseline B (blueprint Part 10): both cameras' raw BPAs (no
    reliability discount), combined by simple arithmetic mean per
    hypothesis, renormalized. No Dempster combination rule, no conflict
    handling, no shared-uncertainty discount -- isolates whether "more
    data, naively combined" helps at all."""
    bpa_a = evidence_items_to_bpa(camera_a_evidence)
    bpa_b = evidence_items_to_bpa(camera_b_evidence)
    averaged_raw = {hid: (bpa_a.masses[hid] + bpa_b.masses[hid]) / 2.0 for hid in HYPOTHESIS_IDS}
    final_bpa = normalize_masses(averaged_raw)

    return FusionResult(
        camera_ids=(camera_a_id, camera_b_id),
        window=window,
        final_bpa=final_bpa,
        conflict_k=0.0,  # naive averaging never computes a Dempster conflict mass
        total_conflict=False,
        high_conflict=False,
        lambda_shared_used=1.0,
        shared_condition=False,
        per_camera_discounted_bpa={camera_a_id: bpa_a, camera_b_id: bpa_b},
        method="naive_averaging_v1",
    )


def _as_reliability_value(reliability) -> float:
    return reliability.score if hasattr(reliability, "score") else float(reliability)


def run_baseline(
    baseline: BaselineType,
    ranking_id: str,
    corridor_state_id: str,
    window: tuple,
    evidence_by_camera: dict,
    reliability_by_camera: dict = None,
    single_camera_id: str = None,
    shared_condition: bool = False,
) -> CandidateCauseHypothesis:
    """Run one baseline on the SAME evidence and return its ranking.

    `evidence_by_camera` maps camera_id -> tuple[EvidenceItem, ...];
    `reliability_by_camera` maps camera_id -> a `ReliabilityScore` or a
    bare float in [0, 1] (both accepted). Baseline A requires
    `single_camera_id` explicitly -- it does not guess which camera to
    use, matching the blueprint's own Baseline A definition.
    """
    reliability_by_camera = reliability_by_camera or {}

    if baseline == BaselineType.A:
        if single_camera_id is None or single_camera_id not in evidence_by_camera:
            raise ValueError("Baseline A requires a valid single_camera_id present in evidence_by_camera")
        fusion_result = single_camera_result(single_camera_id, evidence_by_camera[single_camera_id], window)
        used_evidence = {single_camera_id: evidence_by_camera[single_camera_id]}

    elif baseline == BaselineType.B:
        camera_ids = sorted(evidence_by_camera)
        if len(camera_ids) != 2:
            raise ValueError("Baseline B requires exactly two cameras' evidence")
        a_id, b_id = camera_ids
        fusion_result = _naive_average(a_id, evidence_by_camera[a_id], b_id, evidence_by_camera[b_id], window)
        used_evidence = evidence_by_camera

    elif baseline in (BaselineType.C, BaselineType.ANVESH):
        camera_ids = sorted(evidence_by_camera)
        if len(camera_ids) != 2:
            raise ValueError(f"Baseline {baseline.value} requires exactly two cameras' evidence")
        a_id, b_id = camera_ids
        reliability_a = _as_reliability_value(reliability_by_camera.get(a_id, 1.0))
        reliability_b = _as_reliability_value(reliability_by_camera.get(b_id, 1.0))
        fusion_result = fuse_two_cameras(
            a_id,
            evidence_by_camera[a_id],
            reliability_a,
            b_id,
            evidence_by_camera[b_id],
            reliability_b,
            window,
            shared_condition=shared_condition,
        )
        used_evidence = evidence_by_camera

    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    return build_ranking(
        ranking_id=ranking_id,
        corridor_state_id=corridor_state_id,
        fusion_result=fusion_result,
        evidence_by_camera=used_evidence,
        engine_model_id=f"baseline_{baseline.value.lower()}",
        engine_model_version="v1",
    )
