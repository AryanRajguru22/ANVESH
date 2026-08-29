import pytest

from anvesh.evaluation.baselines import run_baseline
from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.evidence.reliability import ReliabilityScore
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import BaselineType, CandidateCauseHypothesis, CandidateOutcome, Evidence, EvidenceCompleteness

WINDOW = (0.0, 10.0)


def _evidence_item(camera_id, hypothesis_id, score, direction):
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hypothesis_id}:0-10",
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hypothesis_id,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(
        evidence=evidence, evidence_type="test", measured_value=None, direction=direction,
        reliability=1.0, source="test", metadata={},
    )


def _neutral_except(camera_id, hid, score, direction=EvidenceDirection.SUPPORTS):
    return tuple(
        _evidence_item(camera_id, h, score if h == hid else 0.0, direction if h == hid else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )


def _evidence_by_camera():
    return {
        "cam-A": _neutral_except("cam-A", "H1", 0.8),
        "cam-B": _neutral_except("cam-B", "H1", 0.7),
    }


def test_baseline_a_requires_single_camera_id():
    with pytest.raises(ValueError):
        run_baseline(BaselineType.A, "rank-1", "cs-1", WINDOW, _evidence_by_camera())


def test_baseline_a_uses_only_the_named_camera():
    evidence = _evidence_by_camera()
    ranking = run_baseline(BaselineType.A, "rank-a", "cs-1", WINDOW, evidence, single_camera_id="cam-A")
    assert isinstance(ranking, CandidateCauseHypothesis)
    h1 = next(e for e in ranking.ranked_list if e.hypothesis_id == "H1")
    assert h1.belief == pytest.approx(0.8)  # cam-A's own raw evidence, undiscounted
    assert "cam-B:H1:0-10" not in h1.supporting_evidence_refs


def test_baseline_b_requires_exactly_two_cameras():
    with pytest.raises(ValueError):
        run_baseline(BaselineType.B, "rank-1", "cs-1", WINDOW, {"cam-A": _neutral_except("cam-A", "H1", 0.8)})


def test_baseline_b_is_simple_average_no_conflict_math():
    evidence = _evidence_by_camera()
    ranking = run_baseline(BaselineType.B, "rank-b", "cs-1", WINDOW, evidence)
    h1 = next(e for e in ranking.ranked_list if e.hypothesis_id == "H1")
    assert h1.belief == pytest.approx((0.8 + 0.7) / 2.0)


def test_baseline_c_uses_reliability_and_ds_combination():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": ReliabilityScore("cam-A", 0.0, 10.0, 0.9, {}), "cam-B": ReliabilityScore("cam-B", 0.0, 10.0, 0.9, {})}
    ranking = run_baseline(BaselineType.C, "rank-c", "cs-1", WINDOW, evidence, reliability_by_camera=reliability)
    h1 = next(e for e in ranking.ranked_list if e.hypothesis_id == "H1")
    # DS combination of two agreeing sources must differ from the plain naive average
    naive = run_baseline(BaselineType.B, "rank-b2", "cs-1", WINDOW, evidence)
    h1_naive = next(e for e in naive.ranked_list if e.hypothesis_id == "H1")
    assert h1.belief != pytest.approx(h1_naive.belief)


def test_baseline_c_accepts_bare_float_reliability():
    evidence = _evidence_by_camera()
    ranking = run_baseline(
        BaselineType.C, "rank-c2", "cs-1", WINDOW, evidence, reliability_by_camera={"cam-A": 0.5, "cam-B": 0.9}
    )
    assert isinstance(ranking, CandidateCauseHypothesis)


def test_anvesh_baseline_matches_c_in_m4():
    """ANVESH == C until M5/M6 adds propagation feedback (documented)."""
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.8, "cam-B": 0.8}
    c_ranking = run_baseline(BaselineType.C, "rank-c3", "cs-1", WINDOW, evidence, reliability_by_camera=reliability)
    anvesh_ranking = run_baseline(BaselineType.ANVESH, "rank-anvesh", "cs-1", WINDOW, evidence, reliability_by_camera=reliability)

    c_beliefs = {e.hypothesis_id: e.belief for e in c_ranking.ranked_list}
    anvesh_beliefs = {e.hypothesis_id: e.belief for e in anvesh_ranking.ranked_list}
    assert c_beliefs == anvesh_beliefs


def test_all_four_baselines_run_on_the_same_evidence_deterministically():
    evidence = _evidence_by_camera()
    reliability = {"cam-A": 0.9, "cam-B": 0.8}

    results = {
        BaselineType.A: run_baseline(BaselineType.A, "r-a", "cs-1", WINDOW, evidence, single_camera_id="cam-A"),
        BaselineType.B: run_baseline(BaselineType.B, "r-b", "cs-1", WINDOW, evidence),
        BaselineType.C: run_baseline(BaselineType.C, "r-c", "cs-1", WINDOW, evidence, reliability_by_camera=reliability),
        BaselineType.ANVESH: run_baseline(
            BaselineType.ANVESH, "r-anvesh", "cs-1", WINDOW, evidence, reliability_by_camera=reliability
        ),
    }
    for baseline, ranking in results.items():
        assert isinstance(ranking, CandidateCauseHypothesis)
        assert ranking.outcome in (CandidateOutcome.RANKED, CandidateOutcome.INSUFFICIENT_EVIDENCE, CandidateOutcome.HIGH_CONFLICT)

    # rerun for determinism
    rerun = run_baseline(BaselineType.C, "r-c", "cs-1", WINDOW, evidence, reliability_by_camera=reliability)
    original_beliefs = [(e.hypothesis_id, e.belief) for e in results[BaselineType.C].ranked_list]
    rerun_beliefs = [(e.hypothesis_id, e.belief) for e in rerun.ranked_list]
    assert original_beliefs == rerun_beliefs
