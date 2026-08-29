"""Reliability-discounted Dempster-Shafer fusion (blueprint Part 6).

Dempster-Shafer combination and Shafer discounting are textbook, decades-
old mathematics -- nothing in this module claims the mathematics itself
is novel. What is specific to ANVESH is the instantiation described in
the blueprint: applying it to fixed-camera candidate-cause evidence with
an explicit, logged shared-uncertainty discount.

Frame of discernment: Theta = {H1..H6} (blueprint `hypotheses.cause_model.
HYPOTHESIS_IDS`). H7 ("insufficient evidence") is the ignorance mass on
Theta itself, never a seventh singleton -- see `cause_model.py`.

Pipeline (Part 6, Steps 1-3):
  1. `evidence_items_to_bpa` -- per-camera basic probability assignment
     (BPA) from that camera's `Evidence` items.
  2. `discount_by_reliability` -- classical Shafer discounting by the
     camera's `ReliabilityScore`.
  3. `dempster_combine` + `shared_uncertainty_blend` -- combine two
     discounted BPAs, blending the raw Dempster combination with a
     conservative element-wise-max fallback when the two cameras share a
     logged, correlated degradation condition (fog, glare, ...), so two
     cameras "agreeing" under a *shared* failure cannot manufacture more
     confidence than either camera alone already had.

Never a probability: `score`/`mass`/`belief`/`plausibility` values here
are Dempster-Shafer quantities, not calibrated probabilities -- nothing in
this module or its callers should print "confidence = NN%".

Never NaN/inf: near-total conflict (K -> 1, blueprint's own documented
critique of vanilla Dempster's rule) is detected and handled explicitly
(see `_TOTAL_CONFLICT_EPSILON` and `dempster_combine`) rather than dividing
by a near-zero denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import ConfidenceTier

_TOTAL_CONFLICT_EPSILON = 1e-9
_MASS_SUM_EPSILON = 1e-6


def _require_valid_masses(masses: dict, theta_mass: float, label: str) -> None:
    for hid, m in masses.items():
        if m < -_MASS_SUM_EPSILON:
            raise ValueError(f"{label}: mass for {hid} must be >= 0, got {m}")
    if theta_mass < -_MASS_SUM_EPSILON:
        raise ValueError(f"{label}: theta_mass must be >= 0, got {theta_mass}")
    total = sum(masses.values()) + theta_mass
    if abs(total - 1.0) > _MASS_SUM_EPSILON:
        raise ValueError(f"{label}: masses + theta_mass must sum to 1.0, got {total}")


@dataclass(frozen=True)
class BeliefMassAssignment:
    """A basic probability assignment (BPA) over Theta = {H1..H6} plus the
    ignorance mass on Theta itself. `masses` must cover exactly
    `HYPOTHESIS_IDS`, all values >= 0, and `sum(masses) + theta_mass == 1`."""

    masses: dict
    theta_mass: float

    def __post_init__(self) -> None:
        if set(self.masses.keys()) != set(HYPOTHESIS_IDS):
            raise ValueError(f"masses must cover exactly {HYPOTHESIS_IDS}, got {sorted(self.masses.keys())}")
        _require_valid_masses(self.masses, self.theta_mass, "BeliefMassAssignment")


def normalize_masses(raw_masses: dict) -> BeliefMassAssignment:
    """Build a valid BPA from non-negative, possibly-unnormalized raw
    per-hypothesis masses. If their sum exceeds 1.0, all are scaled down
    proportionally (documented, not silently clipped) so the remainder can
    become ignorance; if their sum is 0 (no supporting evidence anywhere),
    the result is pure ignorance (`theta_mass = 1.0`) -- this is exactly
    how "insufficient evidence" (H7) falls out of the mass model rather
    than needing special-case code.
    """
    masses = {hid: max(0.0, raw_masses.get(hid, 0.0)) for hid in HYPOTHESIS_IDS}
    total = sum(masses.values())
    if total > 1.0:
        masses = {hid: v / total for hid, v in masses.items()}
        total = 1.0
    theta_mass = max(0.0, 1.0 - total)
    return BeliefMassAssignment(masses=masses, theta_mass=theta_mass)


def evidence_items_to_bpa(evidence_items: tuple) -> BeliefMassAssignment:
    """Convert one camera's six `EvidenceItem`s (H1-H6, from
    `evidence_builder.build_evidence_for_camera`) into a BPA.

    Only `SUPPORTS`-direction evidence contributes raw singleton mass;
    `CONTRADICTS`/`NEUTRAL`/`INSUFFICIENT` evidence contributes zero raw
    mass for that hypothesis (a singleton-only frame has no subset to
    redirect "against" mass to -- the contradiction is still fully
    preserved and auditable via `EvidenceItem.direction` and the
    resulting `Evidence.contradicting` flag, just not as negative mass,
    which Dempster-Shafer masses cannot represent). Evidence with
    `evidence_completeness=NONE` never contributes support, so missing
    evidence widens ignorance -- it never manufactures belief.
    """
    raw = {}
    for item in evidence_items:
        hid = item.evidence.hypothesis_id
        raw[hid] = item.evidence.signature_match_score if item.direction == EvidenceDirection.SUPPORTS else 0.0
    return normalize_masses(raw)


def discount_by_reliability(bpa: BeliefMassAssignment, reliability: float) -> BeliefMassAssignment:
    """Classical Shafer discounting (blueprint Part 6, Step 2):
    m'(Hk) = r * m(Hk); m'(Theta) = 1 - r*(1 - m(Theta))."""
    if not (0.0 <= reliability <= 1.0):
        raise ValueError(f"reliability must be within [0, 1], got {reliability}")
    discounted = {hid: reliability * m for hid, m in bpa.masses.items()}
    theta = 1.0 - reliability * (1.0 - bpa.theta_mass)
    return BeliefMassAssignment(masses=discounted, theta_mass=theta)


@dataclass(frozen=True)
class CombinationResult:
    """Output of one Dempster combination: the combined BPA (meaningful
    only if not `total_conflict`), the raw conflict mass K, and whether K
    was so close to 1 that combination was unsafe to perform."""

    combined: object  # BeliefMassAssignment, or None if total_conflict
    conflict_k: float
    total_conflict: bool


def dempster_combine(bpa_i: BeliefMassAssignment, bpa_j: BeliefMassAssignment) -> CombinationResult:
    """Dempster's combination rule for two BPAs over the same singleton
    frame (blueprint Part 6): K = sum over distinct hypothesis pairs of
    m_i(p)*m_j(q); m(Hk) = [m_i(Hk)m_j(Hk) + m_i(Hk)m_j(Theta) +
    m_i(Theta)m_j(Hk)] / (1-K); m(Theta) = [m_i(Theta)m_j(Theta)] / (1-K).

    Near-total conflict (1-K < `_TOTAL_CONFLICT_EPSILON`) is the
    documented, decades-old failure mode of vanilla Dempster combination
    (blueprint Part 6's own "known limitation" note): dividing by a
    near-zero denominator would either raise or silently produce NaN/inf,
    so this function instead reports `total_conflict=True` with
    `combined=None` and lets the caller (`fuse_two_cameras`) decide the
    safe fallback (all mass to ignorance) -- conflict stays visible via
    `conflict_k` either way, it is never hidden.
    """
    k = sum(
        bpa_i.masses[p] * bpa_j.masses[q]
        for p in HYPOTHESIS_IDS
        for q in HYPOTHESIS_IDS
        if p != q
    )
    denom = 1.0 - k
    if denom < _TOTAL_CONFLICT_EPSILON:
        return CombinationResult(combined=None, conflict_k=k, total_conflict=True)

    combined_masses = {
        hid: (
            bpa_i.masses[hid] * bpa_j.masses[hid]
            + bpa_i.masses[hid] * bpa_j.theta_mass
            + bpa_i.theta_mass * bpa_j.masses[hid]
        )
        / denom
        for hid in HYPOTHESIS_IDS
    }
    combined_theta = (bpa_i.theta_mass * bpa_j.theta_mass) / denom
    combined = BeliefMassAssignment(masses=combined_masses, theta_mass=combined_theta)
    return CombinationResult(combined=combined, conflict_k=k, total_conflict=False)


def shared_uncertainty_blend(
    m_ds: BeliefMassAssignment,
    bpa_i_discounted: BeliefMassAssignment,
    bpa_j_discounted: BeliefMassAssignment,
    lambda_shared: float,
) -> BeliefMassAssignment:
    """Blueprint Part 6, Step 3: blend the raw Dempster combination with a
    conservative element-wise-max fallback, so two cameras "agreeing"
    under a *shared*, correlated degradation (fog, glare -- logged via the
    caller's `shared_condition` flag) cannot manufacture more confidence
    than the single more-confident reliable camera already had.

    `lambda_shared` is the documented, versioned config constant from
    Part 6 (1.0 when no shared-condition flag is active; ~0.3-0.5 when it
    is), not learned -- pending the blueprint's own SH1 ablation.
    """
    if not (0.0 <= lambda_shared <= 1.0):
        raise ValueError(f"lambda_shared must be within [0, 1], got {lambda_shared}")

    m_conserv = normalize_masses(
        {hid: max(bpa_i_discounted.masses[hid], bpa_j_discounted.masses[hid]) for hid in HYPOTHESIS_IDS}
    )

    blended_raw = {
        hid: lambda_shared * m_ds.masses[hid] + (1.0 - lambda_shared) * m_conserv.masses[hid]
        for hid in HYPOTHESIS_IDS
    }
    # blended_raw + blended theta already sum to 1 by construction (convex
    # combination of two valid BPAs); normalize_masses is used anyway so
    # any floating-point drift is corrected the same documented way as
    # everywhere else, rather than trusting exact arithmetic.
    return normalize_masses(blended_raw)


def belief(bpa: BeliefMassAssignment) -> dict:
    """Bel(Hk) = m(Hk) on a singleton-only frame (no subsets besides Theta)."""
    return dict(bpa.masses)


def plausibility(bpa: BeliefMassAssignment) -> dict:
    """Pl(Hk) = m(Hk) + m(Theta) on a singleton-only frame."""
    return {hid: m + bpa.theta_mass for hid, m in bpa.masses.items()}


def confidence_tier_for(bel: float, pl: float) -> ConfidenceTier:
    """Documented, explicit Bel/Pl -> categorical tier mapping (blueprint
    Part 6: the displayed value stays categorical; Bel/Pl themselves are
    tracked internally for metrics, never shown as a bare percentage)."""
    interval_width = pl - bel
    if bel >= 0.5 and interval_width < 0.3:
        return ConfidenceTier.HIGH
    if bel >= 0.25:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


@dataclass(frozen=True)
class FusionResult:
    """Everything needed to audit one fusion decision."""

    camera_ids: tuple
    window: tuple
    final_bpa: object  # BeliefMassAssignment; masses are all 0 and theta=1 if total_conflict
    conflict_k: float
    total_conflict: bool
    high_conflict: bool
    lambda_shared_used: float
    shared_condition: bool
    per_camera_discounted_bpa: dict = field(default_factory=dict)
    method: str = "reliability_discounted_dempster_shafer_v1"

    @property
    def belief(self) -> dict:
        return belief(self.final_bpa)

    @property
    def plausibility(self) -> dict:
        return plausibility(self.final_bpa)


DEFAULT_LAMBDA_SHARED_WHEN_SHARED = 0.4  # documented, versioned constant (blueprint Part 6: "0.3-0.5")
DEFAULT_HIGH_CONFLICT_THRESHOLD = 0.85


def fuse_two_cameras(
    camera_a_id: str,
    camera_a_evidence: tuple,
    camera_a_reliability: float,
    camera_b_id: str,
    camera_b_evidence: tuple,
    camera_b_reliability: float,
    window: tuple,
    shared_condition: bool = False,
    lambda_shared_when_shared: float = DEFAULT_LAMBDA_SHARED_WHEN_SHARED,
    high_conflict_threshold: float = DEFAULT_HIGH_CONFLICT_THRESHOLD,
) -> FusionResult:
    """The full Part 6 pipeline for two cameras: BPA -> reliability
    discount -> Dempster combine -> shared-uncertainty blend."""
    bpa_a = discount_by_reliability(evidence_items_to_bpa(camera_a_evidence), camera_a_reliability)
    bpa_b = discount_by_reliability(evidence_items_to_bpa(camera_b_evidence), camera_b_reliability)

    combination = dempster_combine(bpa_a, bpa_b)
    lambda_shared = 1.0 if not shared_condition else lambda_shared_when_shared

    if combination.total_conflict:
        # Safe fallback: total mass to ignorance -- never divide by ~0,
        # never fabricate a ranking out of an undefined combination.
        final_bpa = normalize_masses({hid: 0.0 for hid in HYPOTHESIS_IDS})
        conflict_k = combination.conflict_k
    else:
        final_bpa = shared_uncertainty_blend(combination.combined, bpa_a, bpa_b, lambda_shared)
        conflict_k = combination.conflict_k

    return FusionResult(
        camera_ids=(camera_a_id, camera_b_id),
        window=window,
        final_bpa=final_bpa,
        conflict_k=conflict_k,
        total_conflict=combination.total_conflict,
        high_conflict=(combination.total_conflict or conflict_k >= high_conflict_threshold),
        lambda_shared_used=lambda_shared,
        shared_condition=shared_condition,
        per_camera_discounted_bpa={camera_a_id: bpa_a, camera_b_id: bpa_b},
    )


def single_camera_result(camera_id: str, evidence: tuple, window: tuple) -> FusionResult:
    """Baseline A (blueprint Part 10): one camera's own BPA, undiscounted,
    passed straight through as the "fused" result -- no combination, no
    reliability weighting, matching the blueprint's own Baseline A
    definition exactly."""
    bpa = evidence_items_to_bpa(evidence)
    return FusionResult(
        camera_ids=(camera_id,),
        window=window,
        final_bpa=bpa,
        conflict_k=0.0,
        total_conflict=False,
        high_conflict=False,
        lambda_shared_used=1.0,
        shared_condition=False,
        per_camera_discounted_bpa={camera_id: bpa},
        method="single_camera_v1",
    )
