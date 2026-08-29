"""Feedback engine (blueprint Part 3, module 14; Part 9's 4-branch loop).

Compares a `PropagationPrediction` (`propagation/shockwave.py`) against
the `PropagationObservation` actually measured later, and revises the
prior `CandidateCauseHypothesis` ranking accordingly. Implements the four
explicit branches exactly:

  - CONFIRMED: the observation matches the prediction within tolerance.
  - PARTIAL: some match, but not a strong/complete one -- a modest
    adjustment, not a sharp re-ranking off one ambiguous window (Part 9:
    "deliberately does NOT trigger a sharp re-ranking off one ambiguous
    window, to avoid overreacting to noise"). Also the ceiling outcome
    for any observation whose own `data_completeness` is only PARTIAL --
    an unreliable/incomplete observation must never manufacture a
    confident CONTRADICTED (see `compare_propagation`).
  - CONTRADICTED: a FULLY-complete observation materially disagrees
    (wrong arrival camera entirely).
  - EVIDENCE_MISSING: `observation.data_completeness == MISSING` -- no
    confidence change in either direction (Part 9: "missing evidence !=
    low confidence"), and the ORIGINAL ranking is returned completely
    unchanged (same object, not a fabricated no-op revision).

Never a probability: the belief adjustment applied on CONFIRMED/PARTIAL/
CONTRADICTED is a small, explicit, documented, configurable multiplicative
factor (`FeedbackAdjustmentConfig`), reusing `ds_fusion.normalize_masses`
for the actual Dempster-Shafer-consistent renormalization -- it is never
presented as a learned or calibrated probability.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.fusion.ds_fusion import confidence_tier_for, normalize_masses
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import (
    CandidateCauseHypothesis,
    CandidateOutcome,
    DataCompleteness,
    Discrepancy,
    HypothesisUpdate,
    HypothesisUpdateOutcome,
    PropagationObservation,
    PropagationPrediction,
    RankedHypothesisEntry,
)


@dataclass(frozen=True)
class FeedbackThresholds:
    """Explicit, documented tolerances -- no hidden constants."""

    time_tolerance_s: float = 60.0
    rate_tolerance_veh_per_min: float = 2.0


@dataclass(frozen=True)
class FeedbackAdjustmentConfig:
    """Explicit, documented, deterministic belief-adjustment factors.

    These are NOT probabilities and NOT learned -- plain multipliers
    applied to a Dempster-Shafer mass before renormalization, in the same
    spirit as blueprint Part 9's "boost"/"slight_boost_or_unchanged"/
    "penalize" pseudocode.
    """

    confirmed_boost_factor: float = 1.3
    partial_boost_factor: float = 1.05
    contradicted_penalty_factor: float = 0.3

    def __post_init__(self) -> None:
        for name, value in (
            ("confirmed_boost_factor", self.confirmed_boost_factor),
            ("partial_boost_factor", self.partial_boost_factor),
            ("contradicted_penalty_factor", self.contradicted_penalty_factor),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class ComparisonResult:
    """The outcome of comparing one prediction against one observation,
    before any ranking revision is attempted."""

    outcome: HypothesisUpdateOutcome
    discrepancy: object  # Discrepancy, or None for EVIDENCE_MISSING
    reason: str
    location_match: object  # bool, or None for EVIDENCE_MISSING


def compare_propagation(
    prediction: PropagationPrediction,
    observation: PropagationObservation,
    thresholds: FeedbackThresholds = None,
    corridor_topology=None,
) -> ComparisonResult:
    """Compare predicted vs. observed propagation (blueprint Part 9).

    `data_completeness == MISSING` short-circuits to EVIDENCE_MISSING
    with `discrepancy=None` -- the numeric fields on `observation` are a
    documented placeholder in that case (see
    `PropagationObservation`-constructing callers) and must never be read.
    `data_completeness == PARTIAL` can produce CONFIRMED (if it still
    happens to match) or PARTIAL, but never CONTRADICTED -- an unreliable/
    incomplete observation must not manufacture a confident contradiction.
    Only a FULLY-complete observation with the wrong arrival camera
    produces CONTRADICTED.
    """
    thresholds = thresholds or FeedbackThresholds()

    if observation.data_completeness == DataCompleteness.MISSING:
        return ComparisonResult(
            outcome=HypothesisUpdateOutcome.EVIDENCE_MISSING,
            discrepancy=None,
            reason="no propagation observation was available for this window; confidence is left unchanged",
            location_match=None,
        )

    location_match = observation.actual_arrival_camera == prediction.predicted_arrival_camera
    time_error = abs(prediction.predicted_arrival_time - observation.actual_arrival_time)
    rate_error = abs(prediction.predicted_queue_growth_rate - observation.actual_queue_growth_rate)
    location_error = 0.0
    if not location_match:
        location_error = (
            corridor_topology.distance_between_m(prediction.predicted_arrival_camera, observation.actual_arrival_camera)
            if corridor_topology is not None
            else 1.0
        )
    discrepancy = Discrepancy(location_error=location_error, time_error=time_error, rate_error=rate_error)

    within_tolerance = time_error <= thresholds.time_tolerance_s and rate_error <= thresholds.rate_tolerance_veh_per_min

    if observation.data_completeness == DataCompleteness.PARTIAL:
        if location_match and within_tolerance:
            outcome = HypothesisUpdateOutcome.CONFIRMED
            reason = "partial observation still matched the prediction within tolerance"
        else:
            outcome = HypothesisUpdateOutcome.PARTIAL
            reason = "observation is only partially complete -- cannot confidently confirm or contradict"
    elif not location_match:
        outcome = HypothesisUpdateOutcome.CONTRADICTED
        reason = (
            f"reliable (FULL) observation shows the effect at '{observation.actual_arrival_camera}', "
            f"materially disagreeing with the predicted '{prediction.predicted_arrival_camera}'"
        )
    elif within_tolerance:
        outcome = HypothesisUpdateOutcome.CONFIRMED
        reason = (
            f"observed arrival matched the predicted camera within tolerance "
            f"(time_error={time_error:.1f}s, rate_error={rate_error:.2f} veh/min)"
        )
    else:
        outcome = HypothesisUpdateOutcome.PARTIAL
        reason = (
            f"observed arrival camera matched, but timing/rate diverged beyond tolerance "
            f"(time_error={time_error:.1f}s, rate_error={rate_error:.2f} veh/min)"
        )

    return ComparisonResult(outcome=outcome, discrepancy=discrepancy, reason=reason, location_match=location_match)


def revise_ranking(
    original_ranking: CandidateCauseHypothesis,
    outcome: HypothesisUpdateOutcome,
    revised_ranking_id: str,
    adjustment_config: FeedbackAdjustmentConfig = None,
) -> CandidateCauseHypothesis:
    """Apply the outcome's documented adjustment factor to the top-ranked
    hypothesis's mass, renormalize (`ds_fusion.normalize_masses`), and
    re-rank -- the same mass bookkeeping M4 already uses, so a revised
    ranking is exactly as valid a `CandidateCauseHypothesis` as an
    original one.

    Must not be called with `outcome == EVIDENCE_MISSING` -- the caller
    (`apply_feedback`) never revises in that case; it returns the
    original ranking object unchanged instead.
    """
    if outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING:
        raise ValueError("revise_ranking must not be called for EVIDENCE_MISSING -- the ranking must stay unchanged")
    if not original_ranking.ranked_list:
        return original_ranking

    adjustment_config = adjustment_config or FeedbackAdjustmentConfig()
    top_hypothesis_id = original_ranking.ranked_list[0].hypothesis_id
    raw_masses = {entry.hypothesis_id: entry.belief for entry in original_ranking.ranked_list}

    if outcome == HypothesisUpdateOutcome.CONFIRMED:
        raw_masses[top_hypothesis_id] *= adjustment_config.confirmed_boost_factor
    elif outcome == HypothesisUpdateOutcome.PARTIAL:
        raw_masses[top_hypothesis_id] *= adjustment_config.partial_boost_factor
    elif outcome == HypothesisUpdateOutcome.CONTRADICTED:
        raw_masses[top_hypothesis_id] *= adjustment_config.contradicted_penalty_factor

    revised_bpa = normalize_masses(raw_masses)

    # `original_ranking.ranked_list` normally covers all of HYPOTHESIS_IDS
    # (every M4 `build_ranking()` output does), but nothing in the frozen
    # schema guarantees that -- default a hypothesis absent from the prior
    # ranking to empty evidence refs rather than assuming its presence.
    refs_by_hypothesis = {entry.hypothesis_id: entry for entry in original_ranking.ranked_list}
    revised_entries = [
        RankedHypothesisEntry(
            hypothesis_id=hid,
            belief=revised_bpa.masses[hid],
            plausibility=revised_bpa.masses[hid] + revised_bpa.theta_mass,
            confidence_tier=confidence_tier_for(revised_bpa.masses[hid], revised_bpa.masses[hid] + revised_bpa.theta_mass),
            supporting_evidence_refs=refs_by_hypothesis[hid].supporting_evidence_refs if hid in refs_by_hypothesis else [],
            contradicting_evidence_refs=refs_by_hypothesis[hid].contradicting_evidence_refs if hid in refs_by_hypothesis else [],
        )
        for hid in HYPOTHESIS_IDS
    ]
    revised_entries.sort(key=lambda e: (-e.belief, -e.plausibility, e.hypothesis_id))

    revised_outcome = (
        CandidateOutcome.INSUFFICIENT_EVIDENCE if revised_bpa.theta_mass >= 0.9 else CandidateOutcome.RANKED
    )

    return CandidateCauseHypothesis(
        ranking_id=revised_ranking_id,
        corridor_state_id=original_ranking.corridor_state_id,
        outcome=revised_outcome,
        ranked_list=revised_entries,
        engine_model_id=original_ranking.engine_model_id,
        engine_model_version=original_ranking.engine_model_version,
    )


@dataclass(frozen=True)
class FeedbackResult:
    """The full audit trail (design rule: provenance must never be
    destroyed) -- answers exactly: what did ANVESH believe before
    feedback? what did it predict? what happened? what feedback outcome?
    what changed, and why?"""

    prior_ranking: CandidateCauseHypothesis
    prediction: PropagationPrediction
    observation: PropagationObservation
    update: HypothesisUpdate
    revised_ranking: CandidateCauseHypothesis
    reason: str


def apply_feedback(
    update_id: str,
    revised_ranking_id: str,
    prior_ranking: CandidateCauseHypothesis,
    prediction: PropagationPrediction,
    observation: PropagationObservation,
    thresholds: FeedbackThresholds = None,
    adjustment_config: FeedbackAdjustmentConfig = None,
    corridor_topology=None,
) -> FeedbackResult:
    """Orchestrate compare -> (maybe) revise -> record, preserving the
    prior ranking untouched either way (`prior_ranking` is never mutated
    -- `CandidateCauseHypothesis` is frozen)."""
    comparison = compare_propagation(prediction, observation, thresholds, corridor_topology)

    if comparison.outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING:
        revised_ranking = prior_ranking
        effective_revised_ranking_id = prior_ranking.ranking_id  # no new ranking minted; points back at the original
    else:
        revised_ranking = revise_ranking(prior_ranking, comparison.outcome, revised_ranking_id, adjustment_config)
        effective_revised_ranking_id = revised_ranking.ranking_id

    update = HypothesisUpdate(
        update_id=update_id,
        prior_ranking_id=prior_ranking.ranking_id,
        propagation_observation_id=observation.observation_id,
        outcome=comparison.outcome,
        revised_ranking_id=effective_revised_ranking_id,
        discrepancy=comparison.discrepancy,
    )

    return FeedbackResult(
        prior_ranking=prior_ranking,
        prediction=prediction,
        observation=observation,
        update=update,
        revised_ranking=revised_ranking,
        reason=comparison.reason,
    )
