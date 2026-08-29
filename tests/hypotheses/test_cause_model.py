import pytest

from anvesh.hypotheses.cause_model import (
    ALL_HYPOTHESES,
    HYPOTHESES_BY_ID,
    HYPOTHESIS_IDS,
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
    H7,
    HypothesisSpec,
    to_schema_record,
)
from anvesh.storage.schemas import CauseHypothesis


def test_all_seven_hypotheses_exist():
    assert len(ALL_HYPOTHESES) == 7
    ids = {spec.hypothesis_id for spec in ALL_HYPOTHESES}
    assert ids == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}


def test_hypothesis_ids_excludes_h7():
    """H7 maps to DS ignorance mass, not a seventh singleton (blueprint Part 6)."""
    assert HYPOTHESIS_IDS == ("H1", "H2", "H3", "H4", "H5", "H6")
    assert "H7" not in HYPOTHESIS_IDS


def test_hypotheses_by_id_lookup():
    for spec in ALL_HYPOTHESES:
        assert HYPOTHESES_BY_ID[spec.hypothesis_id] is spec


@pytest.mark.parametrize("spec", ALL_HYPOTHESES)
def test_each_hypothesis_has_required_fields(spec: HypothesisSpec):
    assert spec.hypothesis_id
    assert spec.name
    assert spec.description
    assert isinstance(spec.expected_evidence, tuple)
    assert isinstance(spec.supporting_signatures, tuple)
    assert isinstance(spec.contradiction_signatures, tuple)
    assert isinstance(spec.required_evidence, tuple)
    assert isinstance(spec.evidence_that_increases_belief, tuple)
    assert isinstance(spec.evidence_that_decreases_belief, tuple)
    assert spec.camera_observations


def test_h1_through_h6_have_non_empty_supporting_signatures():
    for spec in (H1, H2, H3, H4, H5, H6):
        assert len(spec.supporting_signatures) > 0


def test_h7_is_the_honest_abstention_hypothesis():
    assert H7.hypothesis_id == "H7"
    assert H7.expected_evidence == ()
    assert H7.supporting_signatures == ()


def test_no_duplicate_hypothesis_ids():
    ids = [spec.hypothesis_id for spec in ALL_HYPOTHESES]
    assert len(ids) == len(set(ids))


def test_hypothesis_spec_rejects_empty_id():
    with pytest.raises(ValueError):
        HypothesisSpec(
            hypothesis_id="",
            name="x",
            description="d",
            expected_evidence=(),
            supporting_signatures=(),
            contradiction_signatures=(),
            required_evidence=(),
            evidence_that_increases_belief=(),
            evidence_that_decreases_belief=(),
            camera_observations="n/a",
        )


@pytest.mark.parametrize("spec", ALL_HYPOTHESES)
def test_to_schema_record_produces_valid_cause_hypothesis(spec: HypothesisSpec):
    record = to_schema_record(spec)
    assert isinstance(record, CauseHypothesis)
    assert record.hypothesis_id == spec.hypothesis_id
    assert record.name == spec.name
    assert "description" in record.expected_signature
    assert "contradiction_signatures" in record.contradiction_rules


def test_model_is_deterministic_across_imports():
    from anvesh.hypotheses import cause_model as reimported

    assert reimported.ALL_HYPOTHESES == ALL_HYPOTHESES
