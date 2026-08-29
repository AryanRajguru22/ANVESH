import pytest

from anvesh.evaluation.scenarios import (
    DesignLabel,
    Scenario,
    ScenarioDesignLabelKind,
    load_scenarios,
)
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.storage.schemas import HypothesisUpdateOutcome


def test_scenario_count_within_target_range():
    scenarios = load_scenarios()
    assert 5 <= len(scenarios) <= 8


def test_scenario_ids_are_unique():
    scenarios = load_scenarios()
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))


def test_every_scenario_has_a_non_empty_design_label_note():
    for s in load_scenarios():
        assert isinstance(s.design_label, DesignLabel)
        assert s.design_label.note
        assert len(s.design_label.note) > 10  # a real explanation, not a stub


def test_every_scenario_has_description_and_evidence():
    for s in load_scenarios():
        assert s.description
        assert set(s.evidence_by_camera.keys()) == {"cam-A", "cam-B"}
        for camera_id, items in s.evidence_by_camera.items():
            assert len(items) == len(HYPOTHESIS_IDS)
            assert {i.evidence.hypothesis_id for i in items} == set(HYPOTHESIS_IDS)


def test_required_scenario_kinds_all_present():
    scenarios = load_scenarios()
    kinds = {s.design_label.kind for s in scenarios}
    assert ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS in kinds
    assert ScenarioDesignLabelKind.GENUINELY_UNRESOLVABLE in kinds
    assert ScenarioDesignLabelKind.NO_CAUSE_PRESENT in kinds


def test_all_four_feedback_outcomes_are_covered():
    scenarios = load_scenarios()
    outcomes = {s.propagation.expected_feedback_outcome for s in scenarios if s.propagation is not None}
    assert outcomes == {
        HypothesisUpdateOutcome.CONFIRMED,
        HypothesisUpdateOutcome.PARTIAL,
        HypothesisUpdateOutcome.CONTRADICTED,
        HypothesisUpdateOutcome.EVIDENCE_MISSING,
    }


def test_reliability_pair_scenarios_share_identical_evidence():
    """s3/s4 must differ ONLY in reliability_by_camera -- otherwise the
    reliability-effect comparison would not be isolated."""
    scenarios = {s.scenario_id: s for s in load_scenarios()}
    s3, s4 = scenarios["s3_reliability_equal"], scenarios["s4_reliability_degraded"]

    for camera_id in s3.evidence_by_camera:
        s3_scores = [(i.evidence.hypothesis_id, i.evidence.signature_match_score, i.direction) for i in s3.evidence_by_camera[camera_id]]
        s4_scores = [(i.evidence.hypothesis_id, i.evidence.signature_match_score, i.direction) for i in s4.evidence_by_camera[camera_id]]
        assert s3_scores == s4_scores

    assert s3.reliability_by_camera != s4.reliability_by_camera


def test_design_label_rejects_empty_note():
    with pytest.raises(ValueError):
        DesignLabel(ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1", note="")


def test_design_label_specific_hypothesis_requires_hypothesis_id():
    with pytest.raises(ValueError):
        DesignLabel(ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, None, note="something")


def test_design_label_non_specific_kinds_must_not_name_a_hypothesis():
    with pytest.raises(ValueError):
        DesignLabel(ScenarioDesignLabelKind.GENUINELY_UNRESOLVABLE, "H1", note="something")
    with pytest.raises(ValueError):
        DesignLabel(ScenarioDesignLabelKind.NO_CAUSE_PRESENT, "H1", note="something")


def test_scenario_requires_matching_camera_sets():
    with pytest.raises(ValueError):
        Scenario(
            scenario_id="bad",
            description="x",
            design_label=DesignLabel(ScenarioDesignLabelKind.NO_CAUSE_PRESENT, None, "test"),
            evidence_by_camera={"cam-A": ()},
            reliability_by_camera={"cam-A": 0.9, "cam-B": 0.9},
            single_camera_id="cam-A",
            propagation=None,
        )


def test_scenario_single_camera_id_must_be_in_evidence():
    with pytest.raises(ValueError):
        Scenario(
            scenario_id="bad",
            description="x",
            design_label=DesignLabel(ScenarioDesignLabelKind.NO_CAUSE_PRESENT, None, "test"),
            evidence_by_camera={"cam-A": (), "cam-B": ()},
            reliability_by_camera={"cam-A": 0.9, "cam-B": 0.9},
            single_camera_id="cam-Z",
            propagation=None,
        )


def test_scenarios_are_deterministic_across_calls():
    first = load_scenarios()
    second = load_scenarios()
    for s1, s2 in zip(first, second):
        assert s1.scenario_id == s2.scenario_id
        for camera_id in s1.evidence_by_camera:
            scores1 = [i.evidence.signature_match_score for i in s1.evidence_by_camera[camera_id]]
            scores2 = [i.evidence.signature_match_score for i in s2.evidence_by_camera[camera_id]]
            assert scores1 == scores2
