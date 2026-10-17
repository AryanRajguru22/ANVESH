"""Per-camera evidence generation (blueprint Part 3, module 8): turn one
camera's per-window `TrafficState` (+ contributing `VehicleTrack`s) into
explicit `Evidence` records against each of H1-H6 (blueprint Part 7).

Evidence != interpretation (M4 design rule #1): this module only checks
observable signals against each hypothesis's documented signature
(`cause_model.py`) and records what it found -- it does not rank, combine,
or discount anything (that is `fusion/ds_fusion.py` and `hypotheses/
ranking.py`).

Missing evidence != contradictory evidence (M4 design rule #2): several
signals genuinely cannot be computed from V1's M1-M3 primitives -- see the
per-hypothesis notes below. Those `Evidence` records use
`evidence_completeness=NONE` and `signature_match_score=0.0`,
`contradicting=False`; the *reason* is always "we cannot check this here,"
never "this hypothesis is false." `ds_fusion.py` treats NONE-completeness
evidence as contributing zero raw mass -- neither for nor against -- which
is what correctly keeps missing evidence from being read as contradiction.

Documented V1 limitations (by design, not oversight):
  - H4 (lane blockage) needs per-lane swerve-clustering detection; V1's
    perception stack has no lane geometry at all. Always NONE.
  - H6 (signal/intersection restriction) needs a multi-window cyclical
    history compared against a historical baseline; this module only ever
    sees ONE window. Always NONE.
  - H2/H3's stopped-vehicle detection needs a metric speed threshold, so it
    is NONE for any camera with `motion_space=IMAGE` (uncalibrated) --
    consistent with the whole project's "never treat pixels/sec as a
    metric speed" rule.
  - M-A.1 (OBSTRUCTION-DETECTION AVAILABILITY). Validated geometry does NOT
    imply that speeds were measured: a track whose speed could not be
    estimated carries `speed_available=False` and a placeholder value. The
    ability to look for stopped vehicles is therefore derived ONCE, from the
    tracks actually present (`_obstruction_detection`): the camera's space
    must be WORLD *and* enough of the window's vehicles must carry a real
    measurement (`EvidenceThresholds.min_speed_coverage`). H1 ("congestion with
    no obstruction"), H2, H3 and the stopped branch of H5 all depend on that
    same ability, so they become unavailable (NONE) together. H1 in particular
    is NOT evidence of anything when nobody could look for the obstruction.
    Where coverage is real but incomplete the evidence is PARTIAL: stopped
    vehicles that WERE measured still count, but a failure to find one is only
    a statement about the measured vehicles.
  - H1/H5's "propagation-trend" nuance (a *growing* vs a *stable* queue, an
    inflow *increase* vs merely a *high* count) is explicitly blueprint
    Part 8/9 territory (milestone M5's propagation model) and is
    deliberately not attempted here from a single window. H5 does use
    `camera_role`, though: the blueprint's own Part 7 signature says the
    upstream camera is "most informative" for demand-surge evidence, and
    that much *is* checkable without trend data.
  - H5's graded branch is NONE unless the caller supplies
    `EvidenceThresholds.capacity_flow_veh_per_s`. "Elevated" is a
    comparative claim and V1 has no historical volume baseline for an
    arbitrary camera to compare against -- the same reason H6 is always
    NONE. This is the project's own M6 finding (see configs/experiment/
    m6_real_camera_thresholds.toml, which records that no defensible
    "elevated" cutoff could be derived for either real camera); scoring
    against an absolute per-window count instead would both smuggle in
    that unfounded cutoff and silently change meaning with the window
    length. The stopped-vehicle CONTRADICTS branch needs no baseline and
    is still evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.perception.traffic_state import TrafficStateAggregation
from anvesh.storage.schemas import CameraRole, CongestionLevel, Evidence, EvidenceCompleteness, MotionSpace


class EvidenceDirection(str, Enum):
    """Richer than `schemas.Evidence.contradicting: bool` -- used for
    diagnostics/audit only; `Evidence.contradicting` is derived from this."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class EvidenceItem:
    """The frozen `Evidence` record plus provenance/diagnostics that don't
    belong in the Part 5 schema (evidence_type, measured_value, direction,
    reliability, source, metadata)."""

    evidence: Evidence
    evidence_type: str
    measured_value: object
    direction: EvidenceDirection
    reliability: float
    source: str
    metadata: dict


@dataclass(frozen=True)
class EvidenceThresholds:
    """Explicit, documented thresholds -- no hidden constants."""

    stopped_speed_m_per_s: float = 1.0  # at/below this (WORLD only), a track counts as "stopped"
    min_dwell_seconds: float = 3.0  # minimum duration to count as sustained-stopped, not a momentary slow-down
    # M-A.1: minimum fraction of the window's vehicles that must carry a REAL speed
    # measurement (`speed_available`) before "no stopped vehicle found" means anything.
    # A policy parameter, not tuned to any dataset; 0.5 = "at least half were measured".
    min_speed_coverage: float = 0.5

    # H5 ("elevated inflow") is a COMPARATIVE claim -- "elevated" is
    # meaningless without a per-site reference for what normal inflow is.
    # `capacity_flow_veh_per_s` is that reference, and it is caller-supplied
    # (real or explicitly-synthetic), never defaulted: the same discipline
    # `CalibrationProfile`/`segment_length_m` already follow. When it is
    # None, H5's graded branch reports NONE completeness rather than
    # scoring against an unfounded cutoff -- see `_evidence_h5`.
    capacity_flow_veh_per_s: float = None  # site capacity flow, vehicles/second
    elevated_flow_fraction_of_capacity: float = 0.5  # flow at/above this fraction of capacity is "elevated"

    def __post_init__(self) -> None:
        if self.capacity_flow_veh_per_s is not None and self.capacity_flow_veh_per_s <= 0:
            raise ValueError(f"capacity_flow_veh_per_s must be > 0 when supplied, got {self.capacity_flow_veh_per_s}")
        if not (0.0 < self.elevated_flow_fraction_of_capacity <= 1.0):
            raise ValueError(
                f"elevated_flow_fraction_of_capacity must be within (0, 1], got {self.elevated_flow_fraction_of_capacity}"
            )


@dataclass(frozen=True)
class ObstructionDetection:
    """Whether (and how completely) this window's tracks allow a stopped-vehicle
    / obstruction search. Derived from the data, never asserted by a caller."""

    usable: bool
    completeness: EvidenceCompleteness  # FULL / PARTIAL when usable, NONE otherwise
    measured: int  # tracks carrying a real metric speed
    total: int  # tracks in the window
    reason: str  # "" when usable

    @property
    def coverage(self) -> float:
        return self.measured / self.total if self.total else 1.0  # nothing to measure -> nothing missed


def _obstruction_detection(tracks: list, motion_space: MotionSpace, thresholds: EvidenceThresholds) -> ObstructionDetection:
    total = len(tracks)
    if motion_space != MotionSpace.WORLD:
        return ObstructionDetection(
            False, EvidenceCompleteness.NONE, 0, total,
            "stopped-vehicle detection requires validated metric geometry (WORLD); this camera is image-space or unvalidated",
        )
    measured = sum(1 for t in tracks if t.motion_space == MotionSpace.WORLD and t.speed_available)
    if total and measured / total < thresholds.min_speed_coverage:
        return ObstructionDetection(
            False, EvidenceCompleteness.NONE, measured, total,
            f"speed was measured for only {measured}/{total} vehicles (minimum coverage "
            f"{thresholds.min_speed_coverage:.2f}); a missing speed is not evidence of anything",
        )
    completeness = EvidenceCompleteness.FULL if measured == total else EvidenceCompleteness.PARTIAL
    return ObstructionDetection(True, completeness, measured, total, "")


def _stopped_tracks(tracks: list, motion_space: MotionSpace, thresholds: EvidenceThresholds) -> list:
    if motion_space != MotionSpace.WORLD:
        return []  # cannot apply a metric speed threshold to image-space (px/s) tracks
    # Defence in depth for the speed-safety invariant (M-A): the m/s
    # threshold is applied ONLY to tracks that are themselves validated-metric
    # (MotionSpace.WORLD) AND carry a real speed measurement -- the window's
    # declared space is not trusted on its own, and a placeholder value
    # (`speed_available=False`) is "missing", not "0 m/s, i.e. stopped".
    return [
        t
        for t in tracks
        if t.motion_space == MotionSpace.WORLD
        and t.speed_available
        and t.speed_estimate.value <= thresholds.stopped_speed_m_per_s
        and (t.last_seen - t.first_seen) >= thresholds.min_dwell_seconds
    ]


def _make_evidence(
    evidence_id: str,
    camera_id: str,
    window: tuple,
    hypothesis_id: str,
    score: float,
    direction: EvidenceDirection,
    completeness: EvidenceCompleteness,
    supporting_track_refs: list,
    evidence_type: str,
    measured_value,
    reliability: float,
    source: str,
    metadata: dict,
) -> EvidenceItem:
    evidence = Evidence(
        evidence_id=evidence_id,
        camera_id=camera_id,
        window=window,
        hypothesis_id=hypothesis_id,
        signature_match_score=score,
        supporting_track_refs=list(supporting_track_refs),
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=completeness,
    )
    return EvidenceItem(
        evidence=evidence,
        evidence_type=evidence_type,
        measured_value=measured_value,
        direction=direction,
        reliability=reliability,
        source=source,
        metadata=dict(metadata),
    )


def build_evidence_for_camera(
    camera_id: str,
    window: tuple,
    camera_role: CameraRole,
    aggregation: TrafficStateAggregation,
    tracks: list,
    reliability: float,
    thresholds: EvidenceThresholds = None,
) -> tuple:
    """Build one Evidence item per H1-H6 (always all six, some NONE) for
    one camera's window. `tracks` should be the same overlapping-track
    list used to produce `aggregation` (see `traffic_state.
    aggregate_traffic_state`)."""
    thresholds = thresholds or EvidenceThresholds()
    ts = aggregation.traffic_state
    motion_space = ts.motion_space
    congested = ts.congestion_level in (CongestionLevel.BUILDING, CongestionLevel.CONGESTED)

    obstruction = _obstruction_detection(tracks, motion_space, thresholds)
    stopped = _stopped_tracks(tracks, motion_space, thresholds) if obstruction.usable else []
    stopped_refs = [t.track_id for t in stopped]
    speed_detection_available = obstruction.usable

    items = {
        "H1": _evidence_h1(camera_id, window, congested, ts, stopped, obstruction, reliability),
        "H2": _evidence_h2(camera_id, window, congested, stopped, stopped_refs, obstruction, reliability),
        "H3": _evidence_h3(camera_id, window, congested, stopped, stopped_refs, obstruction, reliability),
        "H4": _evidence_h4(camera_id, window, reliability),
        "H5": _evidence_h5(camera_id, window, camera_role, ts, stopped, speed_detection_available, thresholds, reliability),
        "H6": _evidence_h6(camera_id, window, reliability),
    }
    return tuple(items[hid] for hid in HYPOTHESIS_IDS)


def _evidence_h1(camera_id, window, congested, ts, stopped, obstruction, reliability):
    if not obstruction.usable:
        # "Congestion with NO obstruction" is a statement about what a search for
        # obstructions did not find. If no search was possible (image-space /
        # unvalidated geometry, or too few measured speeds) it asserts nothing --
        # it must not be SUPPORTS, and it must not count as an evaluable competitor.
        return _make_evidence(
            evidence_id=f"{camera_id}:H1:{window[0]}-{window[1]}",
            camera_id=camera_id, window=window, hypothesis_id="H1",
            score=0.0, direction=EvidenceDirection.INSUFFICIENT,
            completeness=EvidenceCompleteness.NONE, supporting_track_refs=[],
            evidence_type="sustained_congestion_no_obstruction", measured_value=None,
            reliability=reliability, source="unavailable",
            metadata={
                "reason": obstruction.reason,
                "congestion_level": ts.congestion_level.value,
                "speed_measured_tracks": obstruction.measured, "tracks": obstruction.total,
            },
        )
    completeness = obstruction.completeness
    if congested and not stopped:
        # H1 is an ABSENCE claim ("no obstruction found"); it is only as strong as the search
        # behind it. When only a fraction of the window's vehicles had a measured speed the
        # score is scaled by that fraction (a no-op at full coverage), so a half-searched window
        # cannot reach the confidence of a fully searched one.
        score = min(1.0, 0.5 + 0.5 * ts.occupancy) * obstruction.coverage
        direction = EvidenceDirection.SUPPORTS
    elif congested and stopped:
        score = 0.15
        direction = EvidenceDirection.CONTRADICTS  # a stopped vehicle better explains this (H2/H3)
    else:
        score = 0.1
        direction = EvidenceDirection.NEUTRAL
    return _make_evidence(
        evidence_id=f"{camera_id}:H1:{window[0]}-{window[1]}",
        camera_id=camera_id,
        window=window,
        hypothesis_id="H1",
        score=score,
        direction=direction,
        completeness=completeness,
        supporting_track_refs=[],
        evidence_type="sustained_congestion_no_obstruction",
        measured_value=ts.occupancy,
        reliability=reliability,
        source="traffic_state_aggregation",
        metadata={
            "congestion_level": ts.congestion_level.value, "occupancy": ts.occupancy,
            "speed_measured_tracks": obstruction.measured, "tracks": obstruction.total,
        },
    )


def _evidence_h2(camera_id, window, congested, stopped, stopped_refs, obstruction, reliability):
    if not obstruction.usable:
        return _make_evidence(
            evidence_id=f"{camera_id}:H2:{window[0]}-{window[1]}",
            camera_id=camera_id,
            window=window,
            hypothesis_id="H2",
            score=0.0,
            direction=EvidenceDirection.INSUFFICIENT,
            completeness=EvidenceCompleteness.NONE,
            supporting_track_refs=[],
            evidence_type="single_stopped_vehicle",
            measured_value=None,
            reliability=reliability,
            source="traffic_state_aggregation",
            metadata={"reason": obstruction.reason},
        )
    if len(stopped) == 1:
        score, direction = 0.8, EvidenceDirection.SUPPORTS
    elif len(stopped) >= 2:
        score, direction = 0.2, EvidenceDirection.CONTRADICTS  # favors H3 instead
    elif congested:
        score, direction = 0.15, EvidenceDirection.CONTRADICTS  # congested but genuinely no stopped vehicle found
    else:
        score, direction = 0.05, EvidenceDirection.NEUTRAL
    return _make_evidence(
        evidence_id=f"{camera_id}:H2:{window[0]}-{window[1]}",
        camera_id=camera_id,
        window=window,
        hypothesis_id="H2",
        score=score,
        direction=direction,
        completeness=obstruction.completeness,
        supporting_track_refs=stopped_refs if len(stopped) == 1 else [],
        evidence_type="single_stopped_vehicle",
        measured_value=len(stopped),
        reliability=reliability,
        source="track_stopped_detection",
        metadata={
            "stopped_track_count": len(stopped),
            "speed_measured_tracks": obstruction.measured, "tracks": obstruction.total,
        },
    )


def _evidence_h3(camera_id, window, congested, stopped, stopped_refs, obstruction, reliability):
    if not obstruction.usable:
        return _make_evidence(
            evidence_id=f"{camera_id}:H3:{window[0]}-{window[1]}",
            camera_id=camera_id,
            window=window,
            hypothesis_id="H3",
            score=0.0,
            direction=EvidenceDirection.INSUFFICIENT,
            completeness=EvidenceCompleteness.NONE,
            supporting_track_refs=[],
            evidence_type="multiple_stopped_vehicles",
            measured_value=None,
            reliability=reliability,
            source="traffic_state_aggregation",
            metadata={"reason": obstruction.reason},
        )
    if len(stopped) >= 2:
        score, direction = 0.75, EvidenceDirection.SUPPORTS
    elif len(stopped) == 1:
        score, direction = 0.2, EvidenceDirection.CONTRADICTS  # favors H2 more specifically
    else:
        score, direction = 0.05, EvidenceDirection.NEUTRAL
    return _make_evidence(
        evidence_id=f"{camera_id}:H3:{window[0]}-{window[1]}",
        camera_id=camera_id,
        window=window,
        hypothesis_id="H3",
        score=score,
        direction=direction,
        completeness=obstruction.completeness,
        supporting_track_refs=stopped_refs if len(stopped) >= 2 else [],
        evidence_type="multiple_stopped_vehicles",
        measured_value=len(stopped),
        reliability=reliability,
        source="track_stopped_detection",
        metadata={"stopped_track_count": len(stopped), "congested": congested},
    )


def _evidence_h4(camera_id, window, reliability):
    return _make_evidence(
        evidence_id=f"{camera_id}:H4:{window[0]}-{window[1]}",
        camera_id=camera_id,
        window=window,
        hypothesis_id="H4",
        score=0.0,
        direction=EvidenceDirection.INSUFFICIENT,
        completeness=EvidenceCompleteness.NONE,
        supporting_track_refs=[],
        evidence_type="lane_level_swerve_clustering",
        measured_value=None,
        reliability=reliability,
        source="unavailable",
        metadata={"reason": "V1 perception has no per-lane geometry / swerve-clustering detection"},
    )


def _evidence_h5(camera_id, window, camera_role, ts, stopped, speed_available, thresholds, reliability):
    has_stopped = speed_available and len(stopped) > 0
    capacity = thresholds.capacity_flow_veh_per_s
    # Part 7: "Upstream camera is most informative" for demand-surge evidence --
    # a downstream camera only sees the *result* of upstream demand, so its
    # inflow is a weaker signal for this specific hypothesis.
    role_weight = 1.0 if camera_role == CameraRole.UPSTREAM else 0.5

    # A stopped vehicle contradicts "free-flowing demand surge" regardless of
    # whether a capacity reference exists -- that branch needs no baseline, so
    # it is still decided here rather than being reported unavailable.
    if has_stopped:
        return _make_evidence(
            evidence_id=f"{camera_id}:H5:{window[0]}-{window[1]}",
            camera_id=camera_id, window=window, hypothesis_id="H5",
            score=0.15, direction=EvidenceDirection.CONTRADICTS,
            completeness=EvidenceCompleteness.FULL, supporting_track_refs=[],
            evidence_type="elevated_inflow_no_obstruction", measured_value=ts.flow_rate,
            reliability=reliability, source="track_stopped_detection",
            metadata={
                "flow_rate_veh_per_s": ts.flow_rate,
                "stopped_track_count": len(stopped),
                "camera_role": camera_role.value,
            },
        )

    if capacity is None:
        return _make_evidence(
            evidence_id=f"{camera_id}:H5:{window[0]}-{window[1]}",
            camera_id=camera_id, window=window, hypothesis_id="H5",
            score=0.0, direction=EvidenceDirection.INSUFFICIENT,
            completeness=EvidenceCompleteness.NONE, supporting_track_refs=[],
            evidence_type="elevated_inflow_no_obstruction", measured_value=ts.flow_rate,
            reliability=reliability, source="unavailable",
            metadata={
                "flow_rate_veh_per_s": ts.flow_rate,
                "camera_role": camera_role.value,
                "reason": (
                    "'elevated' inflow is a comparative claim; grading it requires a caller-supplied "
                    "capacity_flow_veh_per_s reference for this site, and none was provided"
                ),
            },
        )

    elevated_threshold = thresholds.elevated_flow_fraction_of_capacity * capacity
    if ts.flow_rate >= elevated_threshold:
        base = 0.6 + 0.4 * min(1.0, ts.flow_rate / capacity)
        score = min(1.0, base * role_weight)
        direction = EvidenceDirection.SUPPORTS
    else:
        score, direction = 0.1, EvidenceDirection.NEUTRAL
    return _make_evidence(
        evidence_id=f"{camera_id}:H5:{window[0]}-{window[1]}",
        camera_id=camera_id,
        window=window,
        hypothesis_id="H5",
        score=score,
        direction=direction,
        completeness=EvidenceCompleteness.FULL,
        supporting_track_refs=[],
        evidence_type="elevated_inflow_no_obstruction",
        measured_value=ts.flow_rate,
        reliability=reliability,
        source="traffic_state_aggregation",
        metadata={
            "flow_rate_veh_per_s": ts.flow_rate,
            "capacity_flow_veh_per_s": capacity,
            "elevated_threshold_veh_per_s": elevated_threshold,
            "camera_role": camera_role.value,
        },
    )


def _evidence_h6(camera_id, window, reliability):
    return _make_evidence(
        evidence_id=f"{camera_id}:H6:{window[0]}-{window[1]}",
        camera_id=camera_id,
        window=window,
        hypothesis_id="H6",
        score=0.0,
        direction=EvidenceDirection.INSUFFICIENT,
        completeness=EvidenceCompleteness.NONE,
        supporting_track_refs=[],
        evidence_type="abnormal_signal_phase_oscillation",
        measured_value=None,
        reliability=reliability,
        source="unavailable",
        metadata={
            "reason": "requires multi-window cyclical history vs. a historical baseline; not available from one window"
        },
    )
