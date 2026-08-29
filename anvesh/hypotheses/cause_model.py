"""H1-H7 candidate-cause model (blueprint Part 7).

Static, machine-readable definitions only -- this module holds no
per-incident state and performs no evidence scoring itself (that is
`anvesh/evidence/evidence_builder.py`). `HypothesisSpec` is this module's
own richer in-code representation (supporting/contradiction *signature
names* that the evidence builder can check for); `to_schema_record()`
projects each spec into the frozen `anvesh.storage.schemas.CauseHypothesis`
Part 5 entity for persistence/interchange.

Frame of discernment (blueprint Part 6): Θ = {H1..H6}, the six substantive
causes. H7 ("insufficient evidence") is NOT a seventh singleton -- it maps
to the Dempster-Shafer ignorance mass on Θ itself. `HYPOTHESIS_IDS` below
is exactly {H1..H6}; H7's spec exists for documentation/UI purposes only
and is deliberately excluded from `HYPOTHESIS_IDS`.

Do not invent additional hypotheses -- this is the complete, closed H1-H7
set from the blueprint; no eighth hypothesis, no auto-generated new causes
(blueprint Part 2, explicit LATER item).
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.storage.schemas import CauseHypothesis


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    name: str
    description: str
    expected_evidence: tuple
    supporting_signatures: tuple
    contradiction_signatures: tuple
    required_evidence: tuple
    evidence_that_increases_belief: tuple
    evidence_that_decreases_belief: tuple
    camera_observations: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be a non-empty string")
        if not self.name:
            raise ValueError("name must be a non-empty string")


H1 = HypothesisSpec(
    hypothesis_id="H1",
    name="downstream_bottleneck",
    description=(
        "Congestion caused by a fixed, capacity-limited point downstream "
        "(no stopped/blocking object) -- queue backs up from that point."
    ),
    expected_evidence=(
        "sustained high occupancy at a fixed downstream point",
        "queue tail growing upstream",
        "no stationary/blocking object within the queue",
    ),
    supporting_signatures=("sustained_congestion_no_obstruction",),
    contradiction_signatures=("stopped_vehicle_present", "moving_origin_point"),
    required_evidence=("congestion_level", "stopped_vehicle_presence"),
    evidence_that_increases_belief=("repeated windows of congestion at an unchanging capacity-limited point",),
    evidence_that_decreases_belief=("a stationary blocking object found in the queue",),
    camera_observations="Downstream camera: consistent capacity-limited flow at a fixed point. Upstream camera: backward-growing queue.",
)

H2 = HypothesisSpec(
    hypothesis_id="H2",
    name="stopped_disabled_vehicle",
    description=(
        "A single tracked vehicle at near-zero speed, sustained, not at a "
        "marked stop point -- the queue originates at that vehicle."
    ),
    expected_evidence=(
        "a specific tracked vehicle at near-zero speed, sustained",
        "not at a marked/legitimate stop point",
    ),
    supporting_signatures=("single_stopped_vehicle",),
    contradiction_signatures=("no_stopped_vehicle", "multiple_stopped_vehicles"),
    required_evidence=("stopped_vehicle_presence", "stopped_vehicle_count"),
    evidence_that_increases_belief=("the stopped vehicle confirmed at the queue head in a later window",),
    evidence_that_decreases_belief=(
        "the 'stopped' vehicle is actually moving slowly (misclassification)",
        "the vehicle is at a legitimate signal stop",
    ),
    camera_observations="Origin camera: a stationary tracked object. Downstream of it: free flow.",
)

H3 = HypothesisSpec(
    hypothesis_id="H3",
    name="collision_like_obstruction",
    description=(
        "Multiple vehicles abnormally clustered/stopped -- a higher-severity, "
        "longer-persistence version of H2's signature."
    ),
    expected_evidence=(
        "multiple vehicles abnormally clustered/stopped",
        "abrupt onset",
        "possible full (not partial) lane blockage",
    ),
    supporting_signatures=("multiple_stopped_vehicles",),
    contradiction_signatures=("single_stopped_vehicle", "no_stopped_vehicle"),
    required_evidence=("stopped_vehicle_count",),
    evidence_that_increases_belief=("visual clustering cues with abrupt onset relative to the demand trend",),
    evidence_that_decreases_belief=("an orderly single-vehicle stop", "flow continuing around the location"),
    camera_observations="Origin camera: abnormal multi-vehicle cluster. Up/downstream: stronger version of H2's signature.",
)

H4 = HypothesisSpec(
    hypothesis_id="H4",
    name="lane_blockage_non_incident",
    description=(
        "A fixed non-vehicle obstruction (or a vehicle disabled far longer than "
        "plausible) causing partial-capacity reduction at one point."
    ),
    expected_evidence=(
        "a fixed-position obstruction persisting far longer than a plausible disabled-vehicle clearance time",
        "repeated lane-change/swerve behavior at that point",
    ),
    supporting_signatures=("lane_level_swerve_clustering",),
    contradiction_signatures=("object_cleared_quickly", "all_lanes_equally_affected"),
    required_evidence=("lane_level_swerve_clustering",),
    evidence_that_increases_belief=("repeated swerve/lane-change detections clustered at one fixed point",),
    evidence_that_decreases_belief=(
        "the object moves/disappears quickly (favors H2/H3)",
        "all lanes are equally affected (favors H1)",
    ),
    camera_observations="Camera at the obstruction: consistent lane-change clustering at a fixed point.",
)

H5 = HypothesisSpec(
    hypothesis_id="H5",
    name="upstream_demand_surge",
    description=(
        "No obstruction anywhere in the corridor -- congestion is explained by "
        "elevated inflow from upstream/a side entry."
    ),
    expected_evidence=(
        "no stopped/obstructing object anywhere in the corridor",
        "elevated inflow measured upstream, preceding downstream congestion",
    ),
    supporting_signatures=("elevated_inflow_no_obstruction",),
    contradiction_signatures=("stopped_vehicle_present",),
    required_evidence=("vehicle_count", "stopped_vehicle_presence"),
    evidence_that_increases_belief=("a measurable count/inflow increase preceding congestion onset",),
    evidence_that_decreases_belief=(
        "any stopped vehicle or lane blockage found in the corridor",
        "no consistent slow-down under normal demand at the downstream point",
    ),
    camera_observations="Upstream camera is most informative: elevated counts/density before downstream congestion appears.",
)

H6 = HypothesisSpec(
    hypothesis_id="H6",
    name="signal_intersection_restriction",
    description=(
        "Congestion originates at/upstream of a known signalized point, with "
        "queue length oscillating abnormally relative to the historical baseline."
    ),
    expected_evidence=(
        "congestion originates at/upstream of a known signalized point",
        "queue length oscillates with signal phase, deviating from the historical baseline",
    ),
    supporting_signatures=("abnormal_signal_phase_oscillation",),
    contradiction_signatures=("oscillation_matches_known_baseline", "persists_through_green_phase"),
    required_evidence=("multi_window_cyclical_history",),
    evidence_that_increases_belief=("oscillation amplitude/period deviating from the historical baseline",),
    evidence_that_decreases_belief=(
        "oscillation matches the known-normal signal timing",
        "queue persists through green phases (favors a downstream cause)",
    ),
    camera_observations=(
        "Camera covering the intersection approach; V1 infers phase purely from cyclical queue "
        "behavior, no direct signal-controller feed (documented limitation)."
    ),
)

H7 = HypothesisSpec(
    hypothesis_id="H7",
    name="insufficient_evidence",
    description=(
        "None of H1-H6's signatures are sufficiently supported, or the corridor's "
        "cameras do not cover the plausibly-relevant area. Maps to Dempster-Shafer "
        "ignorance mass on Θ={H1..H6}, not a seventh singleton (blueprint Part 6)."
    ),
    expected_evidence=(),
    supporting_signatures=(),
    contradiction_signatures=(),
    required_evidence=(),
    evidence_that_increases_belief=(),
    evidence_that_decreases_belief=("any of H1-H6 gaining strong support",),
    camera_observations="N/A -- this is the honest 'we don't/can't know' state.",
)

# Θ per blueprint Part 6 -- the six substantive, fusible hypotheses. H7 is
# intentionally excluded: it is represented structurally (ignorance mass),
# never scored as a singleton.
HYPOTHESIS_IDS = ("H1", "H2", "H3", "H4", "H5", "H6")

ALL_HYPOTHESES = (H1, H2, H3, H4, H5, H6, H7)
HYPOTHESES_BY_ID = {spec.hypothesis_id: spec for spec in ALL_HYPOTHESES}


def to_schema_record(spec: HypothesisSpec) -> CauseHypothesis:
    """Project a HypothesisSpec into the frozen Part 5 `CauseHypothesis` record."""
    return CauseHypothesis(
        hypothesis_id=spec.hypothesis_id,
        name=spec.name,
        expected_signature={
            "description": spec.description,
            "expected_evidence": list(spec.expected_evidence),
            "supporting_signatures": list(spec.supporting_signatures),
            "camera_observations": spec.camera_observations,
        },
        contradiction_rules={
            "contradiction_signatures": list(spec.contradiction_signatures),
            "evidence_that_decreases_belief": list(spec.evidence_that_decreases_belief),
        },
    )
