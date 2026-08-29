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
  - H1/H5's "propagation-trend" nuance (a *growing* vs a *stable* queue, an
    inflow *increase* vs merely a *high* count) is explicitly blueprint
    Part 8/9 territory (milestone M5's propagation model) and is
    deliberately not attempted here from a single window. H5 does use
    `camera_role`, though: the blueprint's own Part 7 signature says the
    upstream camera is "most informative" for demand-surge evidence, and
    that much *is* checkable without trend data.
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
    elevated_vehicle_count: int = 8  # single-window proxy for "elevated inflow" (see module docstring)


def _stopped_tracks(tracks: list, motion_space: MotionSpace, thresholds: EvidenceThresholds) -> list:
    if motion_space != MotionSpace.WORLD:
        return []  # cannot apply a metric speed threshold to image-space (px/s) tracks
    return [
        t
        for t in tracks
        if t.speed_estimate.value <= thresholds.stopped_speed_m_per_s
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

    stopped = _stopped_tracks(tracks, motion_space, thresholds)
    stopped_refs = [t.track_id for t in stopped]
    speed_detection_available = motion_space == MotionSpace.WORLD

    items = {
        "H1": _evidence_h1(camera_id, window, congested, ts, stopped, speed_detection_available, reliability),
        "H2": _evidence_h2(camera_id, window, congested, stopped, stopped_refs, speed_detection_available, reliability),
        "H3": _evidence_h3(camera_id, window, congested, stopped, stopped_refs, speed_detection_available, reliability),
        "H4": _evidence_h4(camera_id, window, reliability),
        "H5": _evidence_h5(camera_id, window, camera_role, ts, stopped, speed_detection_available, thresholds, reliability),
        "H6": _evidence_h6(camera_id, window, reliability),
    }
    return tuple(items[hid] for hid in HYPOTHESIS_IDS)


def _evidence_h1(camera_id, window, congested, ts, stopped, speed_available, reliability):
    completeness = EvidenceCompleteness.FULL if speed_available else EvidenceCompleteness.PARTIAL
    if congested and not stopped:
        score = min(1.0, 0.5 + 0.5 * ts.occupancy)
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
        metadata={"congestion_level": ts.congestion_level.value, "occupancy": ts.occupancy},
    )


def _evidence_h2(camera_id, window, congested, stopped, stopped_refs, speed_available, reliability):
    if not speed_available:
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
            metadata={"reason": "stopped-vehicle detection requires a calibrated (WORLD) speed threshold"},
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
        completeness=EvidenceCompleteness.FULL,
        supporting_track_refs=stopped_refs if len(stopped) == 1 else [],
        evidence_type="single_stopped_vehicle",
        measured_value=len(stopped),
        reliability=reliability,
        source="track_stopped_detection",
        metadata={"stopped_track_count": len(stopped)},
    )


def _evidence_h3(camera_id, window, congested, stopped, stopped_refs, speed_available, reliability):
    if not speed_available:
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
            metadata={"reason": "stopped-vehicle detection requires a calibrated (WORLD) speed threshold"},
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
        completeness=EvidenceCompleteness.FULL,
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
    elevated = ts.vehicle_count >= thresholds.elevated_vehicle_count
    has_stopped = speed_available and len(stopped) > 0
    # Part 7: "Upstream camera is most informative" for demand-surge evidence --
    # a downstream camera only sees the *result* of upstream demand, so its
    # inflow count is a weaker signal for this specific hypothesis.
    role_weight = 1.0 if camera_role == CameraRole.UPSTREAM else 0.5

    if has_stopped:
        score, direction = 0.15, EvidenceDirection.CONTRADICTS
    elif elevated:
        base = 0.6 + 0.4 * min(1.0, ts.vehicle_count / (2 * thresholds.elevated_vehicle_count))
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
        measured_value=ts.vehicle_count,
        reliability=reliability,
        source="traffic_state_aggregation",
        metadata={
            "vehicle_count": ts.vehicle_count,
            "elevated_threshold": thresholds.elevated_vehicle_count,
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
