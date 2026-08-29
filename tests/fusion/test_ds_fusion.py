import math

import pytest

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.fusion.ds_fusion import (
    BeliefMassAssignment,
    DEFAULT_LAMBDA_SHARED_WHEN_SHARED,
    confidence_tier_for,
    dempster_combine,
    discount_by_reliability,
    evidence_items_to_bpa,
    fuse_two_cameras,
    normalize_masses,
    plausibility,
    shared_uncertainty_blend,
    single_camera_result,
)
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import ConfidenceTier, Evidence, EvidenceCompleteness

WINDOW = (0.0, 10.0)


def _evidence_item(hypothesis_id, score, direction, completeness=EvidenceCompleteness.FULL):
    evidence = Evidence(
        evidence_id=f"cam:{hypothesis_id}:0-10",
        camera_id="cam",
        window=WINDOW,
        hypothesis_id=hypothesis_id,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=completeness,
    )
    return EvidenceItem(
        evidence=evidence,
        evidence_type="test",
        measured_value=None,
        direction=direction,
        reliability=1.0,
        source="test",
        metadata={},
    )


def _all_hypotheses_neutral_except(hid, score, direction):
    items = []
    for h in HYPOTHESIS_IDS:
        if h == hid:
            items.append(_evidence_item(h, score, direction))
        else:
            items.append(_evidence_item(h, 0.0, EvidenceDirection.NEUTRAL))
    return tuple(items)


def _bpa(masses: dict, theta: float) -> BeliefMassAssignment:
    full = {h: masses.get(h, 0.0) for h in HYPOTHESIS_IDS}
    return BeliefMassAssignment(masses=full, theta_mass=theta)


# ---------------------------------------------------------------------------
# BeliefMassAssignment / normalize_masses
# ---------------------------------------------------------------------------


def test_bpa_rejects_wrong_key_set():
    with pytest.raises(ValueError):
        BeliefMassAssignment(masses={"H1": 1.0}, theta_mass=0.0)


def test_bpa_rejects_masses_that_do_not_sum_to_one():
    masses = {h: 0.0 for h in HYPOTHESIS_IDS}
    masses["H1"] = 0.5
    with pytest.raises(ValueError):
        BeliefMassAssignment(masses=masses, theta_mass=0.9)  # sums to 1.4


def test_bpa_rejects_negative_mass():
    masses = {h: 0.0 for h in HYPOTHESIS_IDS}
    masses["H1"] = -0.1
    with pytest.raises(ValueError):
        BeliefMassAssignment(masses=masses, theta_mass=1.1)


def test_normalize_masses_leaves_room_for_ignorance():
    result = normalize_masses({"H1": 0.3, "H2": 0.2})
    assert result.masses["H1"] == pytest.approx(0.3)
    assert result.masses["H2"] == pytest.approx(0.2)
    assert result.theta_mass == pytest.approx(0.5)


def test_normalize_masses_scales_down_when_sum_exceeds_one():
    result = normalize_masses({"H1": 0.7, "H2": 0.5})  # sums to 1.2
    assert result.masses["H1"] == pytest.approx(0.7 / 1.2)
    assert result.masses["H2"] == pytest.approx(0.5 / 1.2)
    assert result.theta_mass == pytest.approx(0.0)


def test_normalize_masses_all_zero_is_pure_ignorance():
    """This is how H7 (insufficient evidence) falls out of the mass model."""
    result = normalize_masses({h: 0.0 for h in HYPOTHESIS_IDS})
    assert all(v == 0.0 for v in result.masses.values())
    assert result.theta_mass == 1.0


# ---------------------------------------------------------------------------
# Evidence -> BPA
# ---------------------------------------------------------------------------


def test_evidence_items_to_bpa_only_supports_contribute_mass():
    items = (
        _evidence_item("H1", 0.8, EvidenceDirection.SUPPORTS),
        _evidence_item("H2", 0.9, EvidenceDirection.CONTRADICTS),
        _evidence_item("H3", 0.5, EvidenceDirection.NEUTRAL),
        _evidence_item("H4", 0.0, EvidenceDirection.INSUFFICIENT, EvidenceCompleteness.NONE),
        _evidence_item("H5", 0.1, EvidenceDirection.SUPPORTS),
        _evidence_item("H6", 0.0, EvidenceDirection.INSUFFICIENT, EvidenceCompleteness.NONE),
    )
    bpa = evidence_items_to_bpa(items)
    assert bpa.masses["H1"] == pytest.approx(0.8)
    assert bpa.masses["H2"] == 0.0
    assert bpa.masses["H3"] == 0.0
    assert bpa.masses["H4"] == 0.0
    assert bpa.masses["H5"] == pytest.approx(0.1)
    assert bpa.masses["H6"] == 0.0
    assert bpa.theta_mass == pytest.approx(1.0 - 0.8 - 0.1)


def test_all_insufficient_evidence_yields_pure_ignorance():
    items = tuple(_evidence_item(h, 0.0, EvidenceDirection.INSUFFICIENT, EvidenceCompleteness.NONE) for h in HYPOTHESIS_IDS)
    bpa = evidence_items_to_bpa(items)
    assert bpa.theta_mass == 1.0


# ---------------------------------------------------------------------------
# Reliability discounting
# ---------------------------------------------------------------------------


def test_discount_by_reliability_matches_hand_computation():
    bpa = _bpa({"H1": 0.6}, theta=0.4)
    discounted = discount_by_reliability(bpa, reliability=0.5)
    assert discounted.masses["H1"] == pytest.approx(0.3)  # r * m(H1)
    assert discounted.theta_mass == pytest.approx(1 - 0.5 * (1 - 0.4))  # 1 - r*(1-theta) = 0.7


def test_discount_by_full_reliability_is_identity():
    bpa = _bpa({"H1": 0.6, "H2": 0.1}, theta=0.3)
    discounted = discount_by_reliability(bpa, reliability=1.0)
    assert discounted.masses == bpa.masses
    assert discounted.theta_mass == pytest.approx(bpa.theta_mass)


def test_discount_by_zero_reliability_yields_pure_ignorance():
    bpa = _bpa({"H1": 0.9}, theta=0.1)
    discounted = discount_by_reliability(bpa, reliability=0.0)
    assert all(v == 0.0 for v in discounted.masses.values())
    assert discounted.theta_mass == pytest.approx(1.0)


def test_reliability_out_of_range_rejected():
    bpa = _bpa({"H1": 0.5}, theta=0.5)
    with pytest.raises(ValueError):
        discount_by_reliability(bpa, reliability=1.5)
    with pytest.raises(ValueError):
        discount_by_reliability(bpa, reliability=-0.1)


# ---------------------------------------------------------------------------
# Dempster combination -- hand-computed reference values
# ---------------------------------------------------------------------------


def test_dempster_combine_matches_hand_computed_example():
    bpa_i = _bpa({"H1": 0.6}, theta=0.4)
    bpa_j = _bpa({"H2": 0.5}, theta=0.5)

    result = dempster_combine(bpa_i, bpa_j)

    assert result.total_conflict is False
    assert result.conflict_k == pytest.approx(0.30)  # 0.6 * 0.5
    assert result.combined.masses["H1"] == pytest.approx(3 / 7)
    assert result.combined.masses["H2"] == pytest.approx(2 / 7)
    assert result.combined.theta_mass == pytest.approx(2 / 7)
    total = sum(result.combined.masses.values()) + result.combined.theta_mass
    assert total == pytest.approx(1.0)


def test_dempster_combine_conflict_is_visible_when_cameras_disagree():
    bpa_i = _bpa({"H1": 1.0}, theta=0.0)
    bpa_j = _bpa({"H1": 0.2, "H2": 0.8}, theta=0.0)
    result = dempster_combine(bpa_i, bpa_j)
    assert result.conflict_k == pytest.approx(0.8)  # only the H2 mass conflicts with i's pure H1


def test_dempster_combine_no_conflict_when_cameras_fully_agree():
    bpa_i = _bpa({"H1": 0.7}, theta=0.3)
    bpa_j = _bpa({"H1": 0.6}, theta=0.4)
    result = dempster_combine(bpa_i, bpa_j)
    assert result.conflict_k == pytest.approx(0.0)


def test_exact_total_conflict_is_handled_safely_no_exception_no_nan():
    bpa_i = _bpa({"H1": 1.0}, theta=0.0)
    bpa_j = _bpa({"H2": 1.0}, theta=0.0)

    result = dempster_combine(bpa_i, bpa_j)

    assert result.total_conflict is True
    assert result.combined is None
    assert result.conflict_k == pytest.approx(1.0)
    assert not math.isnan(result.conflict_k)
    assert not math.isinf(result.conflict_k)


def test_near_total_conflict_is_also_caught():
    bpa_i = _bpa({"H1": 1 - 1e-12}, theta=1e-12)
    bpa_j = _bpa({"H2": 1 - 1e-12}, theta=1e-12)

    result = dempster_combine(bpa_i, bpa_j)

    assert result.total_conflict is True
    assert result.conflict_k == pytest.approx(1.0, abs=1e-6)
    assert not math.isnan(result.conflict_k)


# ---------------------------------------------------------------------------
# Shared-uncertainty blend (SH1: two cameras agreeing under a shared
# condition must not manufacture more confidence than either camera alone)
# ---------------------------------------------------------------------------


def test_shared_condition_reduces_confidence_vs_naive_combination():
    bpa_i_discounted = _bpa({"H1": 0.9}, theta=0.1)
    bpa_j_discounted = _bpa({"H1": 0.9}, theta=0.1)
    m_ds = dempster_combine(bpa_i_discounted, bpa_j_discounted).combined
    assert m_ds.masses["H1"] == pytest.approx(0.99)  # naive DS inflates past either camera's own 0.9

    unshared = shared_uncertainty_blend(m_ds, bpa_i_discounted, bpa_j_discounted, lambda_shared=1.0)
    shared = shared_uncertainty_blend(m_ds, bpa_i_discounted, bpa_j_discounted, lambda_shared=DEFAULT_LAMBDA_SHARED_WHEN_SHARED)

    assert unshared.masses["H1"] == pytest.approx(0.99)
    assert shared.masses["H1"] == pytest.approx(
        DEFAULT_LAMBDA_SHARED_WHEN_SHARED * 0.99 + (1 - DEFAULT_LAMBDA_SHARED_WHEN_SHARED) * 0.9
    )
    assert shared.masses["H1"] < unshared.masses["H1"]


def test_conservative_fallback_never_exceeds_the_more_confident_camera():
    bpa_i_discounted = _bpa({"H1": 0.3}, theta=0.7)
    bpa_j_discounted = _bpa({"H1": 0.7}, theta=0.3)
    m_ds = dempster_combine(bpa_i_discounted, bpa_j_discounted).combined

    fully_shared = shared_uncertainty_blend(m_ds, bpa_i_discounted, bpa_j_discounted, lambda_shared=0.0)
    assert fully_shared.masses["H1"] == pytest.approx(0.7)  # pure element-wise max


def test_lambda_shared_out_of_range_rejected():
    m_ds = _bpa({"H1": 0.5}, theta=0.5)
    with pytest.raises(ValueError):
        shared_uncertainty_blend(m_ds, m_ds, m_ds, lambda_shared=1.5)


# ---------------------------------------------------------------------------
# Belief / plausibility / confidence tier
# ---------------------------------------------------------------------------


def test_plausibility_adds_ignorance_to_belief():
    bpa = _bpa({"H1": 0.4}, theta=0.6)
    pl = plausibility(bpa)
    assert pl["H1"] == pytest.approx(1.0)


def test_confidence_tier_thresholds():
    assert confidence_tier_for(bel=0.6, pl=0.65) == ConfidenceTier.HIGH
    assert confidence_tier_for(bel=0.3, pl=0.5) == ConfidenceTier.MEDIUM
    assert confidence_tier_for(bel=0.1, pl=0.9) == ConfidenceTier.LOW
    assert confidence_tier_for(bel=0.6, pl=0.95) == ConfidenceTier.MEDIUM  # high bel but wide interval


# ---------------------------------------------------------------------------
# End-to-end fuse_two_cameras / single_camera_result (Baseline A)
# ---------------------------------------------------------------------------


def test_fuse_two_cameras_end_to_end_deterministic():
    ev_a = _all_hypotheses_neutral_except("H1", 0.8, EvidenceDirection.SUPPORTS)
    ev_b = _all_hypotheses_neutral_except("H1", 0.7, EvidenceDirection.SUPPORTS)

    result1 = fuse_two_cameras("cam-A", ev_a, 0.9, "cam-B", ev_b, 0.9, WINDOW)
    result2 = fuse_two_cameras("cam-A", ev_a, 0.9, "cam-B", ev_b, 0.9, WINDOW)

    assert result1.final_bpa.masses == result2.final_bpa.masses
    assert result1.conflict_k == pytest.approx(result2.conflict_k)
    assert result1.final_bpa.masses["H1"] > 0.7  # agreement should not be *lower* than either camera alone


def test_fuse_two_cameras_low_reliability_camera_has_less_influence():
    ev_a = _all_hypotheses_neutral_except("H1", 0.9, EvidenceDirection.SUPPORTS)
    ev_b = _all_hypotheses_neutral_except("H2", 0.9, EvidenceDirection.SUPPORTS)

    full_reliability = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)
    low_reliability_b = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 0.1, WINDOW)

    assert low_reliability_b.final_bpa.masses["H2"] < full_reliability.final_bpa.masses["H2"]


def test_fuse_two_cameras_total_conflict_returns_pure_ignorance_safely():
    ev_a = _all_hypotheses_neutral_except("H1", 1.0, EvidenceDirection.SUPPORTS)
    ev_b = _all_hypotheses_neutral_except("H2", 1.0, EvidenceDirection.SUPPORTS)

    result = fuse_two_cameras("cam-A", ev_a, 1.0, "cam-B", ev_b, 1.0, WINDOW)

    assert result.total_conflict is True
    assert result.high_conflict is True
    assert result.final_bpa.theta_mass == pytest.approx(1.0)
    for v in result.final_bpa.masses.values():
        assert not math.isnan(v)
        assert not math.isinf(v)


def test_single_camera_result_is_undiscounted_raw_bpa():
    ev = _all_hypotheses_neutral_except("H1", 0.6, EvidenceDirection.SUPPORTS)
    result = single_camera_result("cam-A", ev, WINDOW)
    assert result.camera_ids == ("cam-A",)
    assert result.final_bpa.masses["H1"] == pytest.approx(0.6)
    assert result.method == "single_camera_v1"
