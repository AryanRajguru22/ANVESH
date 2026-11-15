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

M4 addition -- `CandidateOutcome.HIGH_CONFLICT`: M4's Dempster-Shafer
fusion (`fusion/ds_fusion.py`) can detect that two cameras' evidence is
in near-total conflict (the classical, decades-old critique of vanilla
Dempster combination -- blueprint Part 6's own documented limitation).
That is a materially different situation from ordinary
`insufficient_evidence` (no camera has much to say) and design rule #5
("conflict between cameras must remain visible") requires it stay
distinguishable, not silently folded into `insufficient_evidence`. This
is a single additive enum value -- `RANKED` and `INSUFFICIENT_EVIDENCE`
are unchanged, no other field on any entity was touched, and no existing
code path treated "high_conflict" as an invalid value to construct
(verified against `tests/test_schemas.py` before making the change).

M7 addition -- `SafetyEvent`: M7 (blueprint-external -- Part 5 has no
safety-event concept at all) adds the first capability outside
candidate-cause reasoning: persistent stopped-vehicle detection
(`anvesh/safety/stopped_vehicle.py`). Its per-track diagnostic output
(`StoppedVehicleClassification`) deliberately does NOT live here, in the
same spirit as M3's `TrafficStateAggregation`/M4's `ReliabilityScore`
(heuristic, non-probability diagnostics stay next to the code that
produces them, not in the frozen contract). `SafetyEvent` is the one
piece that IS schema-worthy: a small, generic, reusable record of "an
event of some kind was detected," deliberately not named or shaped only
for stopped vehicles -- `event_type` is a free-text field precisely so a
future detector (still out of scope; none is implemented here) can reuse
this same entity for a different kind of event without a further schema
change. `motion_space` and `confidence_tier` reuse the existing enums
above rather than introducing new ones. No existing entity or field was
touched to add this.
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


class GeometryValidity(str, Enum):
    """What kind of quantity a camera's geometry can honestly produce (M-A
    speed-safety invariant, ANVESH_PRODUCT_GAP_AUDIT.md finding 2).

      IMAGE             -- no usable transform; only pixel quantities exist.
      WORLD_UNVALIDATED -- a transform exists but its output scale/units are
                           not independently confirmed (e.g. CityFlow's
                           supplied homography). Non-metric: never m/s, km/h,
                           or an input to an m/s-thresholded consumer.
      WORLD_VALIDATED   -- transform exists AND metric scale validated; the
                           only state that may produce m/s or km/h.

    Relationship to `MotionSpace`: `MotionSpace.WORLD` means EXACTLY
    "WORLD_VALIDATED" -- position_history is metres, speed_estimate is m/s.
    `MotionSpace.IMAGE` covers both IMAGE and WORLD_UNVALIDATED: a track on a
    camera whose transform is unvalidated carries image-space (px, px/s)
    kinematics, and records `geometry_validity=WORLD_UNVALIDATED` so a
    consumer can tell "no transform exists" from "a transform exists but is
    not trusted". Every existing `motion_space == WORLD` consumer therefore
    fails closed for an unvalidated calibration without further edits.
    """

    IMAGE = "image"
    WORLD_UNVALIDATED = "world_unvalidated"
    WORLD_VALIDATED = "world_validated"


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
    HIGH_CONFLICT = "high_conflict"


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
    schema_version: str = "1.1.0"
    # M-A (schema 1.1.0): see `GeometryValidity`. M-A.1: a WORLD (metres) track
    # MUST declare `geometry_validity=WORLD_VALIDATED` explicitly -- omitting it
    # raises, because silently deriving "validated" from "world" is exactly how an
    # unvalidated transform used to be promoted to metres. Only the conservative
    # IMAGE case may be omitted (it can only ever be derived as IMAGE).
    geometry_validity: GeometryValidity = None
    # False when `speed_estimate` is NOT a real measurement (too few samples /
    # implausible): `speed_estimate.value` is then a placeholder that no
    # consumer may threshold or average. Never a fabricated "0 m/s".
    speed_available: bool = True

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
        if self.geometry_validity is None:
            if self.motion_space == MotionSpace.WORLD:
                raise ValueError(
                    "a motion_space=WORLD (metres) VehicleTrack must declare geometry_validity explicitly "
                    "(GeometryValidity.WORLD_VALIDATED); it is never derived from motion_space"
                )
            object.__setattr__(self, "geometry_validity", GeometryValidity.IMAGE)
        else:
            object.__setattr__(self, "geometry_validity", GeometryValidity(self.geometry_validity))
        # Structural speed-safety invariant: metres only from validated geometry.
        if self.motion_space == MotionSpace.WORLD and self.geometry_validity != GeometryValidity.WORLD_VALIDATED:
            raise ValueError("motion_space=WORLD (metres) requires geometry_validity=WORLD_VALIDATED")
        if self.motion_space == MotionSpace.IMAGE and self.geometry_validity == GeometryValidity.WORLD_VALIDATED:
            raise ValueError("geometry_validity=WORLD_VALIDATED requires motion_space=WORLD")
        # Unit consistency: a metric track's positions are `WorldPosition` (metres);
        # an image track's are plain (x_px, y_px, t) tuples. A mix, or the wrong kind,
        # means the declared space does not describe the data.
        is_world_pos = [isinstance(p, WorldPosition) for p in self.position_history]
        if self.motion_space == MotionSpace.WORLD and not all(is_world_pos):
            raise ValueError("motion_space=WORLD requires every position_history entry to be a WorldPosition (metres)")
        if self.motion_space == MotionSpace.IMAGE and any(is_world_pos):
            raise ValueError("motion_space=IMAGE (pixels) must not carry WorldPosition (metre) entries")


TRAFFIC_STATE_SCHEMA_VERSION = "1.2.0"


def _version_tuple(version: str) -> tuple:
    try:
        return tuple(int(part) for part in str(version).split("."))
    except ValueError:
        return (0,)  # unparseable == oldest: treated as legacy (fail closed)


@dataclass(frozen=True)
class TrafficState:
    # Hardening A2 (schema 1.2.0): like `VehicleTrack`, a new TrafficState MUST declare
    # `geometry_validity`, and `motion_space=WORLD` is structurally tied to WORLD_VALIDATED.
    # LEGACY states (schema_version < 1.2.0, e.g. rehydrated from old storage) may have
    # `geometry_validity=None`: that means UNVERIFIED -- it is never treated as metric-validated
    # (`supports_metric` is False, `measured_mean_speed()` withholds a WORLD speed).
    # `mean_speed_available=False` means `mean_speed` is a 0.0 placeholder, not a measurement.
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
    schema_version: str = TRAFFIC_STATE_SCHEMA_VERSION
    # M-A.1 (schema 1.1.0): False when NO vehicle in the window carried a real
    # speed measurement. `mean_speed.value` is then a 0.0 PLACEHOLDER -- not
    # "traffic at 0 m/s" -- and no consumer or display may read it as a speed.
    mean_speed_available: bool = True
    # Hardening A2 (schema 1.2.0): what kind of quantity this camera's geometry can honestly
    # produce. Required for schema >= 1.2.0; None only for legacy states (= unverified).
    geometry_validity: GeometryValidity = None

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
        if self.geometry_validity is not None:
            object.__setattr__(self, "geometry_validity", GeometryValidity(self.geometry_validity))
        legacy = _version_tuple(self.schema_version) < _version_tuple(TRAFFIC_STATE_SCHEMA_VERSION)
        if self.geometry_validity is None and not legacy:
            raise ValueError(
                f"a TrafficState (schema {self.schema_version}) must declare geometry_validity explicitly; "
                "it is never derived from motion_space"
            )
        if self.geometry_validity is not None:
            if self.motion_space == MotionSpace.WORLD and self.geometry_validity != GeometryValidity.WORLD_VALIDATED:
                raise ValueError("motion_space=WORLD (metres) requires geometry_validity=WORLD_VALIDATED")
            if self.motion_space == MotionSpace.IMAGE and self.geometry_validity == GeometryValidity.WORLD_VALIDATED:
                raise ValueError("geometry_validity=WORLD_VALIDATED requires motion_space=WORLD")

    @property
    def supports_metric(self) -> bool:
        """True only for a WORLD state whose geometry is declared WORLD_VALIDATED. A legacy state
        with no recorded validity is NOT metric, whatever its `motion_space` says."""
        return self.motion_space == MotionSpace.WORLD and self.geometry_validity == GeometryValidity.WORLD_VALIDATED

    def measured_mean_speed(self):
        """The window's mean speed if it was actually measured, else `None`.

        `mean_speed` is a `Measurement` and cannot be absent, so when
        `mean_speed_available` is False it holds a 0.0 placeholder. Display and
        threshold code should read this accessor rather than `mean_speed.value`,
        so a placeholder can never be mistaken for "traffic at 0 speed". A legacy WORLD
        state with no recorded validity also yields None: its metres cannot be vouched for."""
        if not self.mean_speed_available:
            return None
        if self.motion_space == MotionSpace.WORLD and not self.supports_metric:
            return None
        return self.mean_speed


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


@dataclass(frozen=True)
class SafetyEvent:
    """A generic, reusable "an event of some kind was detected" record
    (M7 addition -- see the module docstring's "M7 addition" note).
    `event_type` is free-text on purpose so a future safety-event detector
    can reuse this same entity without another schema change; M7 itself
    only ever constructs one with `event_type="stopped_vehicle"`
    (`anvesh/safety/stopped_vehicle.py:build_safety_event`)."""

    event_id: str
    event_type: str
    camera_id: str
    track_id: str
    window: tuple
    motion_space: MotionSpace
    confidence_tier: ConfidenceTier
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        _require_non_empty(self.event_id, "event_id")
        _require_non_empty(self.event_type, "event_type")
        _require_non_empty(self.camera_id, "camera_id")
        _require_non_empty(self.track_id, "track_id")
        _validate_window(self.window)
        object.__setattr__(self, "motion_space", MotionSpace(self.motion_space))
        object.__setattr__(self, "confidence_tier", ConfidenceTier(self.confidence_tier))
