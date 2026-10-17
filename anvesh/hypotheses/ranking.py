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
    hypothesis distinguishes itself), OR fewer than
    `MIN_EVALUABLE_HYPOTHESES` hypotheses could be evaluated at all (M-A:
    see below). Design rule #7: "a ranking must be allowed to abstain."
  - `RANKED`: otherwise.

WHAT A SINGLE-CAMERA RANKING DOES AND DOES NOT ESTABLISH (M-A.1).
The production single-camera path is `fusion.ds_fusion.single_camera_discounted_result`:
one camera's evidence scores -> a BPA -> Shafer-discounted by THAT camera's
reliability score. There is no second observation to combine with or to corroborate it.
  - A `RANKED` outcome means: among the hypotheses this window could actually TEST
    (`evaluable_hypothesis_ids`), this one is best supported by the measured data, to the
    degree its camera was reliable. H1 in particular means "congestion was observed and
    no stopped vehicle was found among the vehicles whose speed was measured".
  - It does NOT establish a cause (no causal inference, no temporal precedence, no
    queue-head test), a calibrated probability (the scores are heuristics: H1's is
    `0.5 + 0.5 * occupancy`, occupancy being a capacity proxy), or that anything absent from
    the evidence was ruled out. A hypothesis that could not be tested is excluded from the
    comparison, never counted as contradicted.
  - `INSUFFICIENT_EVIDENCE` is the expected, honest result for any image-space or
    unvalidated-geometry camera (nothing there can be tested) and for any validated camera
    whose speeds were mostly not measured.

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
from anvesh.evidence.evidence_builder import EvidenceDirection
from anvesh.hypotheses.cause_model import COMPETITOR_PREREQUISITES, HYPOTHESIS_IDS
from anvesh.storage.schemas import (
    CandidateCauseHypothesis,
    CandidateOutcome,
    ConfidenceTier,
    CorridorState,
    EvidenceCompleteness,
    RankedHypothesisEntry,
)

DEFAULT_INSUFFICIENT_EVIDENCE_THETA_THRESHOLD = 0.9

# M-A (ANVESH_PRODUCT_GAP_AUDIT.md finding 3 / SS20 P0): a "ranking" is a
# comparison between competing causes. If only ONE hypothesis could be tested at
# all (every other one is NONE-completeness, i.e. "cannot be checked here" -- zero
# mass, neither for nor against), there is nothing to compare, and the result read
# as "H1, high" -- a property of what could be measured, not a finding about the
# road. In image-space (uncalibrated or unvalidated-scale) cameras H2-H6 are always
# NONE (they need a metric speed threshold or lane/signal context), so every ranked
# window of such a camera was "H1" by construction (13/13 on the demo video).
#
# CORRECTION (M-A.1): the original comment blamed Dempster NORMALISATION for the high
# belief. That is wrong for the single-camera path, which never normalises or
# combines: the belief was simply the raw evidence score passed through undiscounted
# (see `ds_fusion.single_camera_discounted_result`, which now applies the camera's
# reliability). Abstaining on a lone candidate is still right, for the reason above.
#
# M-A.1 also derives eligibility from the evidence itself (`evaluable_hypothesis_ids`),
# not from a completeness flag alone. This changes NO evidence score or threshold.
MIN_EVALUABLE_HYPOTHESES = 2


def evaluable_hypothesis_ids(evidence_by_camera: dict) -> set:
    """Hypotheses that are ELIGIBLE COMPETITORS in this window, derived from the
    evidence actually present (M-A.1) rather than from a single self-declared flag:

      - the evidence must carry some completeness (not NONE) AND must not itself
        report `INSUFFICIENT` (a record that says "cannot evaluate" is never a
        competitor, whatever completeness it was stamped with);
      - a hypothesis with prerequisites (`COMPETITOR_PREREQUISITES`, e.g. H1 needs
        an obstruction test to have been possible) is eligible on a camera only if
        at least one prerequisite was eligible on that same camera.

    NOTE what eligibility is NOT: it does not require the evidence to SUPPORT the
    hypothesis. "We looked and found nothing for H2" is a real, evaluated result;
    only "we could not look" is excluded."""
    ids = set()
    for camera_evidence in evidence_by_camera.values():
        testable = {
            item.evidence.hypothesis_id
            for item in camera_evidence
            if item.evidence.evidence_completeness != EvidenceCompleteness.NONE
            and item.direction != EvidenceDirection.INSUFFICIENT
        }
        for hid in testable:
            prereqs = COMPETITOR_PREREQUISITES.get(hid)
            if prereqs and not any(p in testable for p in prereqs):
                continue
            ids.add(hid)
    return ids


def _evidence_refs(evidence_by_hypothesis: dict, hypothesis_id: str, direction: EvidenceDirection) -> list:
    items = evidence_by_hypothesis.get(hypothesis_id, ())
    return [item.evidence.evidence_id for item in items if item.direction == direction]


def determine_outcome(
    fusion_result: FusionResult,
    insufficient_evidence_theta_threshold: float = DEFAULT_INSUFFICIENT_EVIDENCE_THETA_THRESHOLD,
    high_conflict_threshold: float = DEFAULT_HIGH_CONFLICT_THRESHOLD,
    evaluable_hypotheses: int = None,
) -> CandidateOutcome:
    if fusion_result.total_conflict or fusion_result.conflict_k >= high_conflict_threshold:
        return CandidateOutcome.HIGH_CONFLICT
    if evaluable_hypotheses is not None and evaluable_hypotheses < MIN_EVALUABLE_HYPOTHESES:
        return CandidateOutcome.INSUFFICIENT_EVIDENCE
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

    outcome = determine_outcome(
        fusion_result, insufficient_evidence_theta_threshold, high_conflict_threshold,
        evaluable_hypotheses=len(evaluable_hypothesis_ids(evidence_by_camera)),
    )
    if outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE:
        # The abstention must not be undercut by a per-entry "high" label: the
        # raw belief/plausibility stay (auditable), the displayed tier does not
        # claim confidence the ranking has just declined to assert.
        entries = [dataclasses.replace(e, confidence_tier=ConfidenceTier.LOW) for e in entries]

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
