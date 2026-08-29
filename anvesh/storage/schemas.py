"""Frozen data contracts for ANVESH V1 (blueprint Part 5).

These are plain, immutable `dataclasses` -- the simplest standard-library
approach, matching the pattern already used in `anvesh/config.py` and the
blueprint's own tech-stack choice (Python 3.11+, no extra modeling
dependency introduced for this scale).

Scope: data contracts only. No SQLite, repositories, persistence, database
connections, API, or business logic live here -- see blueprint Part 14,
milestones M3+ for where each schema starts being populated/persisted.

Every entity carries `schema_version`, per the Part 5 preamble note that
this field is required on every record even though it is omitted from the
tables themselves for brevity.

Note on naming: Part 5's `VehicleTrack.class` field is renamed here to
`vehicle_class` because `class` is a reserved word in Python.

M2 addition -- `VehicleTrack.motion_space` (`MotionSpace` enum):
Part 5's text assumes `position_history`/`speed_estimate` are always
calibrated world values, but M1 shipped without camera calibration
(`world/calibration.py` did not exist yet) and so populated both fields
with image-space pixels, documented only in `anvesh/perception/
schema_conversion.py`'s module docstring -- a real record's caller had no
schema-level way to tell pixels from metres. M2 adds calibration, so a
given VehicleTrack may now legitimately be in either space depending on
whether its camera has a valid CalibrationProfile. `motion_space` makes
that machine-checkable instead of documentation-only: `IMAGE` means
`position_history` is `(x, y, t)` pixel tuples and `speed_estimate.value`
is pixels/second; `WORLD` means `position_history` is `WorldPosition`
instances (metres) and `speed_estimate.value` is metres/second. This is a
required field (no default), so every existing call site that constructs
a VehicleTrack had to be updated to pass it explicitly -- see
`anvesh/perception/schema_conversion.py` and `tests/test_schemas.py`.
No other Part 5 entity was touched; `Measurement` itself still carries no
unit field, since its other users (`TrafficState.mean_speed`,
`PropagationPrediction.predicted_shockwave_speed`) are out of M2's scope
and were left alone rather than changed pre-emptively.

M3 addition -- `TrafficState.motion_space` (reuses the same `MotionSpace`
enum): M3 aggregates per-camera `TrafficState` directly from M2's
calibrated-or-not `VehicleTrack`s, so the exact same pixels-vs-metres
ambiguity flagged above for `VehicleTrack` applies to `TrafficState.
mean_speed` too -- and, less obviously, to `density` (vehicles per unit
*length*, which is equally meaningless without knowing whether that
length is in pixels or metres). `flow_rate` (vehicles per unit *time*)
and `occupancy` (a dimensionless ratio) do NOT need this tag: time is
always real wall-clock seconds regardless of calibration, and occupancy
is a ratio of same-unit extents, so it cancels units either way. Same
rule as `VehicleTrack`: `WORLD` means `mean_speed` is m/s and `density`
is vehicles/metre; `IMAGE` means `mean_speed` is px/s and `density` is a
non-metric proxy (vehicles per pixel-span) -- see
`anvesh/perception/traffic_state.py`'s module docstring for how `density`
itself is computed (a documented proxy in both cases; V1 has no surveyed
per-camera segment length to compute a textbook vehicles/metre density
from, only an optional manually-supplied `segment_length_m`). Required
field, no default -- `anvesh/perception/traffic_state.py` and
`tests/test_schemas.py` were updated for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Shared value types (not their own Part 5 tables; used to type fields that
# Part 5 specifies inline, e.g. "float ± error" and "(start, end)" windows).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """A `float ± error` value, as used throughout Part 5 (e.g. speed_estimate)."""

    value: float
    error: float

    def __post_init__(self) -> None:
        if self.error < 0:
            raise ValueError("error must be >= 0")


@dataclass(frozen=True)
class WorldPosition:
    """One (world_x, world_y, timestamp) sample from a VehicleTrack.position_history."""

    world_x: float
    world_y: float
    timestamp: float


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_range(value: float, low: float, high: float, field_name: str) -> None:
    if not (low <= value <= high):
        raise ValueError(f"{field_name} must be within [{low}, {high}], got {value}")


def _validate_window(window: tuple[float, float], field_name: str = "window") -> None:
    if len(window) != 2:
        raise ValueError(f"{field_name} must be a (start, end) pair")
    start, end = window
    if end < start:
        raise ValueError(f"{field_name} end must be >= start")


# ---------------------------------------------------------------------------
# Enumerations (one per Part 5 `enum{...}` field)
# ---------------------------------------------------------------------------


class CameraRole(str, Enum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"


class VehicleClass(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"


class OcclusionState(str, Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    LOST = "lost"


class MotionSpace(str, Enum):
    """How to interpret VehicleTrack.position_history/speed_estimate.

    Added in M2 (see the module docstring's "M2 addition" note) -- this
    was not part of the original Part 5 table text, which assumed
    position_history/speed_estimate would always be calibrated world
    values. `IMAGE` is the fallback for a camera with no valid
    CalibrationProfile.
    """

    IMAGE = "image"
    WORLD = "world"


class CongestionLevel(str, Enum):
    FREE_FLOW = "free_flow"
    BUILDING = "building"
    CONGESTED = "congested"
    DISSIPATING = "dissipating"


class EvidenceCompleteness(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class ConfidenceTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CandidateOutcome(str, Enum):
    RANKED = "ranked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DataCompleteness(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    MISSING = "missing"


class HypothesisUpdateOutcome(str, Enum):
    CONFIRMED = "confirmed"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    EVIDENCE_MISSING = "evidence_missing"


class BaselineType(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    ANVESH = "ANVESH"


# ---------------------------------------------------------------------------
# Part 5 entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    camera_id: str
    position_description: str
    role: CameraRole
    calibration_profile_id: str
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.camera_id, "camera_id")
        _require_non_empty(self.position_description, "position_description")
        _require_non_empty(self.calibration_profile_id, "calibration_profile_id")
        object.__setattr__(self, "role", CameraRole(self.role))


@dataclass(frozen=True)
class CameraObservation:
    observation_id: str
    camera_id: str
    frame_timestamp: float
    detections: list
    model_id: str
    model_version: str
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.observation_id, "observation_id")
        _require_non_empty(self.camera_id, "camera_id")
        _require_non_empty(self.model_id, "model_id")
        _require_non_empty(self.model_version, "model_version")
        if not isinstance(self.detections, list):
            raise TypeError("detections must be a list")


@dataclass(frozen=True)
class VehicleTrack:
    track_id: str
    camera_id: str
    first_seen: float
    last_seen: float
    vehicle_class: VehicleClass
    occlusion_state: OcclusionState
    position_history: list
    speed_estimate: Measurement
    motion_space: MotionSpace
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.track_id, "track_id")
        _require_non_empty(self.camera_id, "camera_id")
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must be >= first_seen")
        if not isinstance(self.speed_estimate, Measurement):
            raise TypeError("speed_estimate must be a Measurement instance")
        object.__setattr__(self, "vehicle_class", VehicleClass(self.vehicle_class))
        object.__setattr__(self, "occlusion_state", OcclusionState(self.occlusion_state))
        object.__setattr__(self, "motion_space", MotionSpace(self.motion_space))


@dataclass(frozen=True)
class TrafficState:
    camera_id: str
    window_start: float
    window_end: float
    occupancy: float
    mean_speed: Measurement
    vehicle_count: int
    flow_rate: float
    density: float
    congestion_level: CongestionLevel
    motion_space: MotionSpace
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.camera_id, "camera_id")
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        _require_range(self.occupancy, 0.0, 1.0, "occupancy")
        if self.vehicle_count < 0:
            raise ValueError("vehicle_count must be >= 0")
        if not isinstance(self.mean_speed, Measurement):
            raise TypeError("mean_speed must be a Measurement instance")
        object.__setattr__(self, "congestion_level", CongestionLevel(self.congestion_level))
        object.__setattr__(self, "motion_space", MotionSpace(self.motion_space))


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    camera_id: str
    window: tuple
    hypothesis_id: str
    signature_match_score: float
    supporting_track_refs: list
    contradicting: bool
    evidence_completeness: EvidenceCompleteness
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.evidence_id, "evidence_id")
        _require_non_empty(self.camera_id, "camera_id")
        _require_non_empty(self.hypothesis_id, "hypothesis_id")
        _validate_window(self.window)
        _require_range(self.signature_match_score, 0.0, 1.0, "signature_match_score")
        object.__setattr__(self, "evidence_completeness", EvidenceCompleteness(self.evidence_completeness))


@dataclass(frozen=True)
class CauseHypothesis:
    """Static H1-H7 definition record (Part 7) -- not a per-incident output."""

    hypothesis_id: str
    name: str
    expected_signature: dict
    contradiction_rules: dict
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.hypothesis_id, "hypothesis_id")
        _require_non_empty(self.name, "name")


@dataclass(frozen=True)
class RankedHypothesisEntry:
    """One entry of CandidateCauseHypothesis.ranked_list."""

    hypothesis_id: str
    belief: float
    plausibility: float
    confidence_tier: ConfidenceTier
    supporting_evidence_refs: list
    contradicting_evidence_refs: list

    def __post_init__(self) -> None:
        _require_non_empty(self.hypothesis_id, "hypothesis_id")
        _require_range(self.belief, 0.0, 1.0, "belief")
        _require_range(self.plausibility, 0.0, 1.0, "plausibility")
        if self.plausibility < self.belief:
            raise ValueError("plausibility must be >= belief")
        object.__setattr__(self, "confidence_tier", ConfidenceTier(self.confidence_tier))


@dataclass(frozen=True)
class CandidateCauseHypothesis:
    ranking_id: str
    corridor_state_id: str
    outcome: CandidateOutcome
    ranked_list: list
    engine_model_id: str
    engine_model_version: str
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.ranking_id, "ranking_id")
        _require_non_empty(self.corridor_state_id, "corridor_state_id")
        _require_non_empty(self.engine_model_id, "engine_model_id")
        _require_non_empty(self.engine_model_version, "engine_model_version")
        if not all(isinstance(entry, RankedHypothesisEntry) for entry in self.ranked_list):
            raise TypeError("ranked_list must contain only RankedHypothesisEntry instances")
        object.__setattr__(self, "outcome", CandidateOutcome(self.outcome))


@dataclass(frozen=True)
class PropagationPrediction:
    prediction_id: str
    based_on_ranking_id: str
    predicted_shockwave_speed: Measurement
    predicted_arrival_camera: str
    predicted_arrival_time: float
    predicted_queue_growth_rate: float
    method: str
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.prediction_id, "prediction_id")
        _require_non_empty(self.based_on_ranking_id, "based_on_ranking_id")
        _require_non_empty(self.predicted_arrival_camera, "predicted_arrival_camera")
        _require_non_empty(self.method, "method")
        if not isinstance(self.predicted_shockwave_speed, Measurement):
            raise TypeError("predicted_shockwave_speed must be a Measurement instance")


@dataclass(frozen=True)
class PropagationObservation:
    observation_id: str
    prediction_id: str
    actual_arrival_camera: str
    actual_arrival_time: float
    actual_queue_growth_rate: float
    data_completeness: DataCompleteness
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.observation_id, "observation_id")
        _require_non_empty(self.prediction_id, "prediction_id")
        _require_non_empty(self.actual_arrival_camera, "actual_arrival_camera")
        object.__setattr__(self, "data_completeness", DataCompleteness(self.data_completeness))


@dataclass(frozen=True)
class Discrepancy:
    location_error: float
    time_error: float
    rate_error: float


@dataclass(frozen=True)
class HypothesisUpdate:
    update_id: str
    prior_ranking_id: str
    propagation_observation_id: str
    outcome: HypothesisUpdateOutcome
    revised_ranking_id: str
    discrepancy: object
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.update_id, "update_id")
        _require_non_empty(self.prior_ranking_id, "prior_ranking_id")
        _require_non_empty(self.propagation_observation_id, "propagation_observation_id")
        _require_non_empty(self.revised_ranking_id, "revised_ranking_id")
        object.__setattr__(self, "outcome", HypothesisUpdateOutcome(self.outcome))
        if self.discrepancy is not None and not isinstance(self.discrepancy, Discrepancy):
            raise TypeError("discrepancy must be a Discrepancy instance or None")


@dataclass(frozen=True)
class CorridorState:
    corridor_state_id: str
    window: tuple
    segment_states: list
    fused_congestion_level: CongestionLevel
    active_ranking_id: str
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.corridor_state_id, "corridor_state_id")
        _validate_window(self.window)
        _require_non_empty(self.active_ranking_id, "active_ranking_id")
        if not all(isinstance(state, TrafficState) for state in self.segment_states):
            raise TypeError("segment_states must contain only TrafficState instances")
        object.__setattr__(self, "fused_congestion_level", CongestionLevel(self.fused_congestion_level))


@dataclass(frozen=True)
class EvaluationMetrics:
    top1_accuracy: float
    topk_accuracy: float
    brier_score: float
    ece: float
    false_confidence_rate: float
    propagation_location_error: float
    propagation_time_error: float
    hypothesis_revision_accuracy: float
    congestion_p: float
    congestion_r: float
    congestion_f1: float


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    experiment_run_id: str
    baseline: BaselineType
    scenario_id: str
    metrics: EvaluationMetrics
    ground_truth_ref: str
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.record_id, "record_id")
        _require_non_empty(self.experiment_run_id, "experiment_run_id")
        _require_non_empty(self.scenario_id, "scenario_id")
        _require_non_empty(self.ground_truth_ref, "ground_truth_ref")
        if not isinstance(self.metrics, EvaluationMetrics):
            raise TypeError("metrics must be an EvaluationMetrics instance")
        object.__setattr__(self, "baseline", BaselineType(self.baseline))
