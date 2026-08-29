"""Fixed, deterministic evaluation scenario set (evaluation phase, post-M5).

Each `Scenario` bundles the SAME `evidence_by_camera`/`reliability_by_camera`
inputs that `evaluation.baselines.run_baseline`/`compare_anvesh_vs_baseline_c`
already consume -- this module only *constructs* scenarios, using the exact
`EvidenceItem`/`Evidence`/`PropagationPrediction`/`PropagationObservation`
style already established in `scripts/run_fusion_demo.py` and
`scripts/run_feedback_demo.py`. It duplicates no fusion/ranking/feedback
logic; `evaluation.metrics` and `scripts/run_evaluation.py` do the actual
running and measuring.

*** SYNTHETIC SCENARIO DESIGN LABEL -- READ BEFORE USING ***

Every `Scenario.design_label` is a label WE chose when we authored the
scenario's evidence -- it records what we *intended* the evidence to
imply. It is:

    NOT real-world ground truth
    NOT expert/adjudicated interpretation
    NOT evidence that ANVESH (or any baseline) is correct about anything
    real

It only supports one honest claim: "given evidence WE constructed to
imply X, did each baseline's output match X?" That is a controlled
mechanism check, not a validation of real-world accuracy. See
`ScenarioDesignLabelKind` for the three label kinds this module uses, and
never call any of this "ground truth" in code, tests, or printed output.

No randomness is used anywhere in this module -- every scenario is a
fixed, hand-authored set of numbers, so "fixed seed" is moot; determinism
is structural, not probabilistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.propagation.shockwave import predict_propagation
from anvesh.storage.schemas import (
    CameraRole,
    CongestionLevel,
    DataCompleteness,
    Evidence,
    EvidenceCompleteness,
    HypothesisUpdateOutcome,
    Measurement,
    MotionSpace,
    PropagationObservation,
    TrafficState,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology

WINDOW = (0.0, 10.0)
SEGMENT_LENGTH_M = 100.0
CAMERA_A = "cam-A"
CAMERA_B = "cam-B"


class ScenarioDesignLabelKind(str, Enum):
    """What kind of intent a scenario's design label expresses -- see the
    module docstring's SYNTHETIC SCENARIO DESIGN LABEL warning."""

    SPECIFIC_HYPOTHESIS = "specific_hypothesis"  # a definite intended top cause, e.g. "H1"
    GENUINELY_UNRESOLVABLE = "genuinely_unresolvable"  # deliberately symmetric/conflicting by construction
    NO_CAUSE_PRESENT = "no_cause_present"  # deliberately no discriminating evidence at all


@dataclass(frozen=True)
class DesignLabel:
    kind: ScenarioDesignLabelKind
    hypothesis_id: object  # str, or None for GENUINELY_UNRESOLVABLE / NO_CAUSE_PRESENT
    note: str  # why this label was chosen -- required, non-empty

    def __post_init__(self) -> None:
        if not self.note:
            raise ValueError("DesignLabel.note must explain why this label was chosen -- it must not be empty")
        if self.kind == ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS and not self.hypothesis_id:
            raise ValueError("SPECIFIC_HYPOTHESIS labels must name a hypothesis_id")
        if self.kind != ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS and self.hypothesis_id is not None:
            raise ValueError(f"{self.kind.value} labels must not name a hypothesis_id")


@dataclass(frozen=True)
class PropagationScenario:
    """The M5 half of a scenario: a prediction, the observation that will
    be compared against it, and the DESIGNED expected feedback outcome
    (documentation of intent, not something the runner blindly asserts as
    if it were guaranteed by the mechanism)."""

    prediction: object  # schemas.PropagationPrediction
    observation: PropagationObservation
    expected_feedback_outcome: HypothesisUpdateOutcome
    corridor_topology: CorridorTopology


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    design_label: DesignLabel
    evidence_by_camera: dict
    reliability_by_camera: dict
    single_camera_id: str  # which camera Baseline A uses -- fixed, not guessed
    propagation: object  # PropagationScenario, or None if this scenario doesn't exercise feedback

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if not self.description:
            raise ValueError("description must be a non-empty string")
        if sorted(self.evidence_by_camera) != sorted(self.reliability_by_camera):
            raise ValueError("evidence_by_camera and reliability_by_camera must cover the same cameras")
        if self.single_camera_id not in self.evidence_by_camera:
            raise ValueError("single_camera_id must be one of the cameras in evidence_by_camera")


def _evidence_item(camera_id: str, hypothesis_id: str, score: float, direction: EvidenceDirection) -> EvidenceItem:
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hypothesis_id}:{WINDOW[0]}-{WINDOW[1]}",
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hypothesis_id,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(
        evidence=evidence,
        evidence_type="evaluation_scenario",
        measured_value=None,
        direction=direction,
        reliability=1.0,
        source="evaluation.scenarios",
        metadata={},
    )


def _camera_evidence(camera_id: str, supports: dict) -> tuple:
    """`supports` maps hypothesis_id -> (score, direction) for hypotheses
    with a designed signal; every other hypothesis in HYPOTHESIS_IDS gets
    a neutral, zero-score item (matching `evidence_builder`'s convention
    of always producing all six)."""
    items = []
    for hid in HYPOTHESIS_IDS:
        score, direction = supports.get(hid, (0.0, EvidenceDirection.NEUTRAL))
        items.append(_evidence_item(camera_id, hid, score, direction))
    return tuple(items)


def _topology() -> CorridorTopology:
    return CorridorTopology(
        corridor_id="corridor-eval",
        placements=(
            CameraPlacement(camera_id=CAMERA_A, role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id=CAMERA_B, role=CameraRole.DOWNSTREAM, distance_from_upstream_m=SEGMENT_LENGTH_M),
        ),
    )


def _shared_traffic_states():
    """One fixed, congested-upstream / clearing-downstream physical
    setup, reused by every scenario that needs a propagation prediction
    -- the corridor's physical state is not what these scenarios vary;
    the per-hypothesis evidence is."""
    upstream = TrafficState(
        camera_id=CAMERA_A, window_start=0.0, window_end=10.0, occupancy=1.0,
        mean_speed=Measurement(value=1.0, error=0.0), vehicle_count=20, flow_rate=2.0,
        density=20.0, congestion_level=CongestionLevel.CONGESTED, motion_space=MotionSpace.WORLD,
    )
    downstream = TrafficState(
        camera_id=CAMERA_B, window_start=0.0, window_end=10.0, occupancy=0.25,
        mean_speed=Measurement(value=9.0, error=0.0), vehicle_count=5, flow_rate=1.0,
        density=5.0, congestion_level=CongestionLevel.FREE_FLOW, motion_space=MotionSpace.WORLD,
    )
    return upstream, downstream


def _shared_prediction(scenario_id: str, ranking_id: str):
    topology = _topology()
    upstream, downstream = _shared_traffic_states()
    return predict_propagation(
        f"pred-{scenario_id}", ranking_id, topology, upstream, downstream, SEGMENT_LENGTH_M, reference_timestamp=10.0
    ), topology


def _confirmed_propagation(scenario_id: str, ranking_id: str) -> PropagationScenario:
    prediction, topology = _shared_prediction(scenario_id, ranking_id)
    observation = PropagationObservation(
        observation_id=f"obs-{scenario_id}", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )
    return PropagationScenario(prediction, observation, HypothesisUpdateOutcome.CONFIRMED, topology)


def _partial_propagation(scenario_id: str, ranking_id: str) -> PropagationScenario:
    prediction, topology = _shared_prediction(scenario_id, ranking_id)
    observation = PropagationObservation(
        observation_id=f"obs-{scenario_id}", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,  # right camera
        actual_arrival_time=prediction.predicted_arrival_time + 5.0,  # but timing diverges
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate * 2.5,  # and rate diverges
        data_completeness=DataCompleteness.PARTIAL,
    )
    return PropagationScenario(prediction, observation, HypothesisUpdateOutcome.PARTIAL, topology)


def _contradicted_propagation(scenario_id: str, ranking_id: str) -> PropagationScenario:
    prediction, topology = _shared_prediction(scenario_id, ranking_id)
    wrong_camera = CAMERA_A if prediction.predicted_arrival_camera == CAMERA_B else CAMERA_B
    observation = PropagationObservation(
        observation_id=f"obs-{scenario_id}", prediction_id=prediction.prediction_id,
        actual_arrival_camera=wrong_camera,  # effect showed up at the wrong camera entirely
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )
    return PropagationScenario(prediction, observation, HypothesisUpdateOutcome.CONTRADICTED, topology)


def _missing_propagation(scenario_id: str, ranking_id: str) -> PropagationScenario:
    prediction, topology = _shared_prediction(scenario_id, ranking_id)
    # Documented placeholder values -- data_completeness=MISSING means these
    # must never be read as real measurements (see feedback.update_engine).
    observation = PropagationObservation(
        observation_id=f"obs-{scenario_id}", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.MISSING,
    )
    return PropagationScenario(prediction, observation, HypothesisUpdateOutcome.EVIDENCE_MISSING, topology)


def _scenario_1_clear_agreement() -> Scenario:
    scenario_id = "s1_clear_cross_camera_agreement"
    evidence = {
        CAMERA_A: _camera_evidence(CAMERA_A, {"H1": (0.8, EvidenceDirection.SUPPORTS)}),
        CAMERA_B: _camera_evidence(CAMERA_B, {"H1": (0.7, EvidenceDirection.SUPPORTS)}),
    }
    return Scenario(
        scenario_id=scenario_id,
        description="Both cameras compatibly support H1 -- the clean-agreement case fusion should strengthen.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1",
            "Evidence was authored so both cameras point at H1 with no contradiction; H1 is the intended top candidate by construction.",
        ),
        evidence_by_camera=evidence,
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_confirmed_propagation(scenario_id, "rank-s1"),
    )


def _scenario_2_disagreement_conflict() -> Scenario:
    scenario_id = "s2_camera_disagreement_conflict"
    evidence = {
        CAMERA_A: _camera_evidence(CAMERA_A, {"H1": (0.9, EvidenceDirection.SUPPORTS)}),
        CAMERA_B: _camera_evidence(CAMERA_B, {"H2": (0.9, EvidenceDirection.SUPPORTS)}),
    }
    return Scenario(
        scenario_id=scenario_id,
        description="Camera A strongly supports H1, camera B strongly supports H2 -- deliberate, symmetric conflict.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.GENUINELY_UNRESOLVABLE, None,
            "Evidence was authored to be equally strong and mutually exclusive on purpose -- there is no single 'correct' "
            "top candidate to check against; the point of this scenario is to observe conflict-handling, not accuracy.",
        ),
        evidence_by_camera=evidence,
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_confirmed_propagation(scenario_id, "rank-s2"),
    )


def _reliability_pair_evidence() -> dict:
    # Camera B's H2 claim is the one DESIGNED to be spurious -- this pair
    # of scenarios tests whether discounting it (unequal reliability)
    # changes the outcome relative to trusting it equally.
    return {
        CAMERA_A: _camera_evidence(CAMERA_A, {"H1": (0.8, EvidenceDirection.SUPPORTS)}),
        CAMERA_B: _camera_evidence(CAMERA_B, {"H2": (0.75, EvidenceDirection.SUPPORTS)}),
    }


def _scenario_3_reliability_equal() -> Scenario:
    scenario_id = "s3_reliability_equal"
    return Scenario(
        scenario_id=scenario_id,
        description="Same conflicting evidence as s4, but BOTH cameras trusted equally (reliability=0.9/0.9).",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1",
            "Camera A's H1 claim is designed to be correct; camera B's H2 claim is designed to be a spurious reading "
            "from an unreliable sensor. This 'equal reliability' variant is the control for s4.",
        ),
        evidence_by_camera=_reliability_pair_evidence(),
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_confirmed_propagation(scenario_id, "rank-s3"),
    )


def _scenario_4_reliability_degraded() -> Scenario:
    scenario_id = "s4_reliability_degraded"
    return Scenario(
        scenario_id=scenario_id,
        description="IDENTICAL evidence to s3, but camera B's reliability is degraded to 0.2 -- isolates the effect of reliability discounting.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1",
            "Same intent as s3: camera A's H1 claim is designed to be correct, camera B's H2 claim spurious. "
            "Only reliability_by_camera differs from s3 -- everything else is byte-identical.",
        ),
        evidence_by_camera=_reliability_pair_evidence(),
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.2},
        single_camera_id=CAMERA_A,
        propagation=_confirmed_propagation(scenario_id, "rank-s4"),
    )


def _scenario_5_insufficient_evidence() -> Scenario:
    scenario_id = "s5_insufficient_evidence_abstention"
    evidence = {
        CAMERA_A: _camera_evidence(CAMERA_A, {}),
        CAMERA_B: _camera_evidence(CAMERA_B, {}),
    }
    return Scenario(
        scenario_id=scenario_id,
        description="Both cameras report only neutral evidence for every hypothesis -- no discriminating signal at all.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.NO_CAUSE_PRESENT, None,
            "Evidence was authored to contain no support for any hypothesis on purpose -- the system is expected to "
            "abstain (insufficient_evidence), not invent a cause.",
        ),
        evidence_by_camera=evidence,
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_confirmed_propagation(scenario_id, "rank-s5"),
    )


def _tight_gap_evidence() -> dict:
    # A closer H1/H2 gap than s1 -- deliberately chosen so a CONTRADICTED
    # penalty on H1 is large enough to let H2 overtake it (see s7).
    return {
        CAMERA_A: _camera_evidence(CAMERA_A, {"H1": (0.7, EvidenceDirection.SUPPORTS)}),
        CAMERA_B: _camera_evidence(
            CAMERA_B, {"H1": (0.4, EvidenceDirection.SUPPORTS), "H2": (0.5, EvidenceDirection.SUPPORTS)}
        ),
    }


def _scenario_6_partial_feedback() -> Scenario:
    scenario_id = "s6_propagation_partial_feedback"
    return Scenario(
        scenario_id=scenario_id,
        description="Clear H1 > H2 agreement (as s1), but the propagation observation is only partially complete and diverges on timing/rate.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1",
            "Same intent as s1 (H1 is the designed top candidate); this scenario varies only the propagation observation to exercise the PARTIAL feedback branch.",
        ),
        evidence_by_camera={
            CAMERA_A: _camera_evidence(CAMERA_A, {"H1": (0.8, EvidenceDirection.SUPPORTS)}),
            CAMERA_B: _camera_evidence(CAMERA_B, {"H1": (0.7, EvidenceDirection.SUPPORTS)}),
        },
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_partial_propagation(scenario_id, "rank-s6"),
    )


def _scenario_7_contradicted_feedback() -> Scenario:
    scenario_id = "s7_propagation_contradicted_feedback"
    return Scenario(
        scenario_id=scenario_id,
        description="H1 > H2 but with a tight gap; a FULL, reliable propagation observation shows the effect at the wrong camera entirely.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1",
            "H1 is the designed initial top candidate (by a tight margin over H2); this scenario is designed so that a "
            "materially contradicting propagation observation should be able to demote H1 below H2 -- testing whether "
            "feedback can actually change the top candidate, not just nudge a number.",
        ),
        evidence_by_camera=_tight_gap_evidence(),
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_contradicted_propagation(scenario_id, "rank-s7"),
    )


def _scenario_8_evidence_missing_feedback() -> Scenario:
    scenario_id = "s8_propagation_evidence_missing_feedback"
    return Scenario(
        scenario_id=scenario_id,
        description="Clear H1 > H2 agreement (as s1), but no subsequent propagation observation is available at all.",
        design_label=DesignLabel(
            ScenarioDesignLabelKind.SPECIFIC_HYPOTHESIS, "H1",
            "Same intent as s1; this scenario varies only the propagation observation (MISSING) to verify feedback "
            "never forces a revision when there is nothing to compare against.",
        ),
        evidence_by_camera={
            CAMERA_A: _camera_evidence(CAMERA_A, {"H1": (0.8, EvidenceDirection.SUPPORTS)}),
            CAMERA_B: _camera_evidence(CAMERA_B, {"H1": (0.7, EvidenceDirection.SUPPORTS)}),
        },
        reliability_by_camera={CAMERA_A: 0.9, CAMERA_B: 0.9},
        single_camera_id=CAMERA_A,
        propagation=_missing_propagation(scenario_id, "rank-s8"),
    )


def load_scenarios() -> tuple:
    """The fixed, deterministic evaluation scenario set (8 scenarios,
    each exercising a distinct mechanism -- see the module docstring)."""
    return (
        _scenario_1_clear_agreement(),
        _scenario_2_disagreement_conflict(),
        _scenario_3_reliability_equal(),
        _scenario_4_reliability_degraded(),
        _scenario_5_insufficient_evidence(),
        _scenario_6_partial_feedback(),
        _scenario_7_contradicted_feedback(),
        _scenario_8_evidence_missing_feedback(),
    )
