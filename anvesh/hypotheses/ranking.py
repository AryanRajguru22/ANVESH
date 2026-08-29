"""Ranked candidate-cause result (blueprint Part 3, module 12).

Turns one `fusion.ds_fusion.FusionResult` plus the `Evidence` that produced
it into a frozen `anvesh.storage.schemas.CandidateCauseHypothesis` -- the
"ranked candidate-cause result" at the end of the fusion pipeline:

    Camera A evidence --\\
                          > reliability-aware fusion --> fused belief --> ranking
    Camera B evidence --/

Never a probability (M4 design rule): `belief`/`plausibility` here are
Dempster-Shafer quantities carried straight from `ds_fusion.py`, mapped to
a categorical `ConfidenceTier` for display -- nothing in this module
prints "confidence = NN%".

Outcome (`schemas.CandidateOutcome`, M4 addition of `HIGH_CONFLICT` --
see `storage/schemas.py`'s module docstring for why):
  - `HIGH_CONFLICT`: the fusion hit total or near-total conflict, or its
    conflict mass K is at/above `high_conflict_threshold`. Design rule #5
    ("conflict must remain visible") -- kept distinct from
    `insufficient_evidence`, never silently folded into it.
  - `INSUFFICIENT_EVIDENCE`: almost all fused mass is ignorance (no
    hypothesis distinguishes itself). Design rule #7: "a ranking must be
    allowed to abstain."
  - `RANKED`: otherwise.

Determinism: `ranked_list` is sorted by (belief desc, plausibility desc,
hypothesis_id asc) -- the hypothesis_id tiebreak makes ordering fully
deterministic even when two hypotheses land on identical belief and
plausibility.
"""

from __future__ import annotations

import dataclasses

from anvesh.evidence.evidence_builder import EvidenceDirection
from anvesh.fusion.ds_fusion import (
    DEFAULT_HIGH_CONFLICT_THRESHOLD,
    FusionResult,
    belief,
    confidence_tier_for,
    plausibility,
)
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import CandidateCauseHypothesis, CandidateOutcome, CorridorState, RankedHypothesisEntry

DEFAULT_INSUFFICIENT_EVIDENCE_THETA_THRESHOLD = 0.9


def _evidence_refs(evidence_by_hypothesis: dict, hypothesis_id: str, direction: EvidenceDirection) -> list:
    items = evidence_by_hypothesis.get(hypothesis_id, ())
    return [item.evidence.evidence_id for item in items if item.direction == direction]


def determine_outcome(
    fusion_result: FusionResult,
    insufficient_evidence_theta_threshold: float = DEFAULT_INSUFFICIENT_EVIDENCE_THETA_THRESHOLD,
    high_conflict_threshold: float = DEFAULT_HIGH_CONFLICT_THRESHOLD,
) -> CandidateOutcome:
    if fusion_result.total_conflict or fusion_result.conflict_k >= high_conflict_threshold:
        return CandidateOutcome.HIGH_CONFLICT
    if fusion_result.final_bpa.theta_mass >= insufficient_evidence_theta_threshold:
        return CandidateOutcome.INSUFFICIENT_EVIDENCE
    return CandidateOutcome.RANKED


def build_ranking(
    ranking_id: str,
    corridor_state_id: str,
    fusion_result: FusionResult,
    evidence_by_camera: dict,
    engine_model_id: str = "ds_fusion",
    engine_model_version: str = "v1",
    insufficient_evidence_theta_threshold: float = DEFAULT_INSUFFICIENT_EVIDENCE_THETA_THRESHOLD,
    high_conflict_threshold: float = DEFAULT_HIGH_CONFLICT_THRESHOLD,
) -> CandidateCauseHypothesis:
    """Build the ranked result.

    `evidence_by_camera` maps camera_id -> the tuple of `EvidenceItem`s
    that camera contributed (from `evidence_builder.build_evidence_for_
    camera`) -- used only to populate `supporting_evidence_refs`/
    `contradicting_evidence_refs` (every ranking is traceable to the
    evidence that produced it, design rule #14); the belief/plausibility
    numbers themselves come from `fusion_result`, already computed.
    """
    bel = belief(fusion_result.final_bpa)
    pl = plausibility(fusion_result.final_bpa)

    evidence_by_hypothesis = {hid: [] for hid in HYPOTHESIS_IDS}
    for camera_evidence in evidence_by_camera.values():
        for item in camera_evidence:
            evidence_by_hypothesis[item.evidence.hypothesis_id].append(item)

    entries = [
        RankedHypothesisEntry(
            hypothesis_id=hid,
            belief=bel[hid],
            plausibility=pl[hid],
            confidence_tier=confidence_tier_for(bel[hid], pl[hid]),
            supporting_evidence_refs=_evidence_refs(evidence_by_hypothesis, hid, EvidenceDirection.SUPPORTS),
            contradicting_evidence_refs=_evidence_refs(evidence_by_hypothesis, hid, EvidenceDirection.CONTRADICTS),
        )
        for hid in HYPOTHESIS_IDS
    ]
    entries.sort(key=lambda e: (-e.belief, -e.plausibility, e.hypothesis_id))

    outcome = determine_outcome(fusion_result, insufficient_evidence_theta_threshold, high_conflict_threshold)

    return CandidateCauseHypothesis(
        ranking_id=ranking_id,
        corridor_state_id=corridor_state_id,
        outcome=outcome,
        ranked_list=entries,
        engine_model_id=engine_model_id,
        engine_model_version=engine_model_version,
    )


def attach_ranking_to_corridor_state(corridor_state: CorridorState, ranking: CandidateCauseHypothesis) -> CorridorState:
    """Replace M3's `UNRANKED_PLACEHOLDER` with the real `ranking_id` now
    that a `CandidateCauseHypothesis` exists for this corridor state
    (blueprint Part 5's `active_ranking_id` FK). `CorridorState` is
    frozen, so this returns a NEW instance (`dataclasses.replace`) rather
    than mutating -- this is the exact point in the pipeline where the M3
    placeholder is replaced; see `anvesh/world/corridor_state.py`'s
    `UNRANKED_PLACEHOLDER` for the flagged M3 gap this closes.
    """
    if ranking.corridor_state_id != corridor_state.corridor_state_id:
        raise ValueError(
            f"ranking.corridor_state_id ({ranking.corridor_state_id!r}) does not match "
            f"corridor_state.corridor_state_id ({corridor_state.corridor_state_id!r})"
        )
    return dataclasses.replace(corridor_state, active_ranking_id=ranking.ranking_id)
