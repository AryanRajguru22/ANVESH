"""Persistent stopped-vehicle detection (M7, blueprint-external).

Scope: this module detects that ONE camera's `VehicleTrack` shows a
sustained, low-motion dwell segment. It is deliberately NOT:
  - accident/collision detection
  - wrong-way detection
  - speeding enforcement
  - lane violation detection
  - N-camera fusion (each track is classified independently, one camera
    at a time -- exactly the scope M1-M3 already establish per track)

It is a reusable primitive: `classify_stopped_vehicle` produces a
diagnostic (`StoppedVehicleClassification`), and `build_safety_event`
turns a STOPPED classification into the frozen, schema-level
`SafetyEvent` record so a future detector (not implemented here) could
produce the same kind of record for a different `event_type`.

WHY VehicleTrack.speed_estimate IS NOT USED:
`speed_estimate` (see `anvesh/perception/motion.py` /
`anvesh/perception/schema_conversion.py`) is a single mean +/- error over
the ENTIRE track lifetime. A vehicle that drove for 20s then stopped for
10s and one that drove slowly the whole 30s can produce the same lifetime
average -- the average cannot distinguish them. This module recomputes
per-step speed directly from `position_history` instead, exactly the
same step-speed math as `motion.py`'s internal `_step_speeds` (not
imported from there, to avoid depending on that module's private helper;
duplicated here as ~10 lines rather than exposing new public surface on
M1's motion module for a single caller).

WHY VehicleTrack.occlusion_state IS NOT A STOPPED SIGNAL:
A track ending in `OcclusionState.LOST` means the tracker stopped seeing
it -- it does NOT mean the vehicle stopped moving. M6.5's own real-video
diagnostics found real tracks ending in LOST purely from a detector
confidence dip near the frame edge, with the vehicle still visibly
moving. `occlusion_state` is therefore never used to decide STOPPED here;
it is only carried into `StoppedVehicleClassification.factors` as
context, with an explicit note when a STOPPED classification's track
happened to end in LOST (see `_build_factors`).

IMAGE vs. WORLD:
`VehicleTrack.motion_space` selects which threshold applies
(`image_speed_threshold_px_per_s` heuristic, or
`world_speed_threshold_m_per_s` metric) and which coordinate fields are
read from `position_history` (plain `(x, y, t)` tuples for `IMAGE`,
`WorldPosition.world_x/world_y/timestamp` for `WORLD`, mirroring
`schema_conversion.py`'s own branch). The result's `motion_space` field
always records which one was actually used; pixels/second are never
converted to, or presented as, metres/second or km/h.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum

from anvesh.storage.schemas import ConfidenceTier, MotionSpace, OcclusionState, SafetyEvent, VehicleTrack


class StoppedVehicleStatus(str, Enum):
    STOPPED = "stopped"
    MOVING = "moving"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class StoppedVehicleThresholds:
    """Explicit, documented thresholds -- no hidden constants. These
    defaults are illustrative/configurable choices, not derived from any
    traffic-engineering standard or tuned against any specific video --
    see M6.5's threshold-derivation discipline for why that distinction
    matters. Callers needing a specific camera's own justified values
    should construct their own instance, the same way M6.5 did for
    `EvidenceThresholds`/`CongestionThresholds`."""

    min_observations: int = 5  # fewer samples than this cannot support any dwell/coverage judgment
    min_temporal_coverage: float = 0.7  # fraction of the track's own span that must be free of large gaps
    dwell_seconds: float = 3.0  # minimum sustained low-speed duration to call it STOPPED, not a momentary slowdown
    max_gap_seconds: float = 1.0  # an inter-sample gap larger than this breaks dwell continuity and hurts coverage
    image_speed_threshold_px_per_s: float = 5.0  # IMAGE-space heuristic only -- never a physical speed
    world_speed_threshold_m_per_s: float = 0.5  # WORLD-space metric threshold (calibrated cameras only)

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("min_observations must be >= 2 (at least one step is needed)")
        if not (0.0 <= self.min_temporal_coverage <= 1.0):
            raise ValueError(f"min_temporal_coverage must be within [0, 1], got {self.min_temporal_coverage}")
        if self.dwell_seconds <= 0:
            raise ValueError("dwell_seconds must be > 0")
        if self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be > 0")
        if self.image_speed_threshold_px_per_s < 0:
            raise ValueError("image_speed_threshold_px_per_s must be >= 0")
        if self.world_speed_threshold_m_per_s < 0:
            raise ValueError("world_speed_threshold_m_per_s must be >= 0")


@dataclass(frozen=True)
class StoppedVehicleClassification:
    """Per-track diagnostic output. NOT part of the frozen schema (same
    treatment as `traffic_state.TrafficStateAggregation` /
    `evidence.reliability.ReliabilityScore`): a heuristic explanation
    object, not a Part 5 contract.

    `heuristic_score` is explicitly NOT a probability -- it is a bounded
    [0, 1] heuristic derived only from `factors` below (temporal coverage
    and how far the observed dwell/motion is from the configured
    thresholds), with no calibration against any labeled outcome. It is
    0.0 for INSUFFICIENT_EVIDENCE by convention (no classification was
    actually made -- this is NOT evidence that the vehicle is "definitely
    not stopped"), matching this project's existing convention for
    insufficient-evidence outcomes (see M4 ranking belief=0.0)."""

    track_id: str
    camera_id: str
    status: StoppedVehicleStatus
    motion_space: MotionSpace
    heuristic_score: float
    confidence_tier: ConfidenceTier
    factors: dict
    dwell_window: tuple = None  # (start, end) of the qualifying low-speed run; only set when status == STOPPED

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", StoppedVehicleStatus(self.status))
        object.__setattr__(self, "motion_space", MotionSpace(self.motion_space))
        object.__setattr__(self, "confidence_tier", ConfidenceTier(self.confidence_tier))
        if not (0.0 <= self.heuristic_score <= 1.0):
            raise ValueError(f"heuristic_score must be within [0, 1], got {self.heuristic_score}")
        if self.status == StoppedVehicleStatus.STOPPED and self.dwell_window is None:
            raise ValueError("a STOPPED classification must carry a dwell_window")
        if self.dwell_window is not None:
            start, end = self.dwell_window
            if end < start:
                raise ValueError("dwell_window end must be >= start")


def _extract_positions(track: VehicleTrack) -> list:
    """(x, y, t) triples in whichever space `track.motion_space` says --
    branches exactly like `schema_conversion.py:build_vehicle_track`."""
    if track.motion_space == MotionSpace.WORLD:
        return [(p.world_x, p.world_y, p.timestamp) for p in track.position_history]
    return [(x, y, t) for (x, y, t) in track.position_history]


def _steps(positions: list) -> list:
    """One (t0, t1, dt, speed) tuple per consecutive sample pair with
    dt > 0. Same math as `motion.py`'s private `_step_speeds`, duplicated
    here rather than imported (see module docstring)."""
    steps = []
    for (x0, y0, t0), (x1, y1, t1) in zip(positions, positions[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        distance = math.hypot(x1 - x0, y1 - y0)
        steps.append((t0, t1, dt, distance / dt))
    return steps


def _longest_low_speed_run(steps: list, speed_threshold: float, max_gap_seconds: float) -> tuple:
    """The longest contiguous run of steps that are BOTH within
    `max_gap_seconds` (no bridging a large gap) AND at/below
    `speed_threshold`. Returns (start, end, duration); (None, None, 0.0)
    if no step qualifies at all."""
    best_start = best_end = None
    best_duration = 0.0
    run_start = run_end = None
    run_duration = 0.0

    for t0, t1, dt, speed in steps:
        qualifies = dt <= max_gap_seconds and speed <= speed_threshold
        if qualifies:
            if run_start is None:
                run_start = t0
            run_end = t1
            run_duration += dt
        else:
            if run_duration > best_duration:
                best_duration, best_start, best_end = run_duration, run_start, run_end
            run_start = run_end = None
            run_duration = 0.0

    if run_duration > best_duration:
        best_duration, best_start, best_end = run_duration, run_start, run_end

    return best_start, best_end, best_duration


def _confidence_tier_for(score: float) -> ConfidenceTier:
    if score >= 0.67:
        return ConfidenceTier.HIGH
    if score >= 0.34:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


def _build_factors(
    track: VehicleTrack,
    observation_count: int,
    observed_duration: float,
    temporal_coverage: float,
    gap_count: int,
    gap_time_beyond_threshold: float,
    step_speeds: list,
    speed_threshold: float,
    dwell_duration: float,
    thresholds: StoppedVehicleThresholds,
    status: StoppedVehicleStatus,
    reasons: list,
) -> dict:
    reasons = list(reasons)
    if status == StoppedVehicleStatus.STOPPED and track.occlusion_state == OcclusionState.LOST:
        reasons.append(
            "track ended with occlusion_state=LOST -- tracking was lost after the last observed "
            "position; this is not evidence the vehicle is still there or still stopped now"
        )
    return {
        "observed_duration": observed_duration,
        "observation_count": observation_count,
        "temporal_coverage": temporal_coverage,
        "min_temporal_coverage_required": thresholds.min_temporal_coverage,
        "dwell_seconds_observed": dwell_duration,
        "dwell_seconds_required": thresholds.dwell_seconds,
        "mean_step_speed": statistics.fmean(step_speeds) if step_speeds else None,
        "min_step_speed": min(step_speeds) if step_speeds else None,
        "motion_threshold_used": speed_threshold,
        "motion_space": track.motion_space.value,
        "occlusion_state": track.occlusion_state.value,
        "gap_count_beyond_max_gap_seconds": gap_count,
        "gap_time_beyond_max_gap_seconds": gap_time_beyond_threshold,
        "reasons": reasons,
    }


def classify_stopped_vehicle(
    track: VehicleTrack,
    thresholds: StoppedVehicleThresholds = None,
) -> StoppedVehicleClassification:
    """Classify one VehicleTrack as STOPPED / MOVING / INSUFFICIENT_EVIDENCE.

    Never uses `track.speed_estimate` (lifetime average -- see module
    docstring) and never uses `track.occlusion_state` to decide the
    status (a vanished track is not evidence of stopping -- see module
    docstring). `thresholds` defaults to `StoppedVehicleThresholds()`.
    """
    thresholds = thresholds or StoppedVehicleThresholds()
    speed_threshold = (
        thresholds.world_speed_threshold_m_per_s
        if track.motion_space == MotionSpace.WORLD
        else thresholds.image_speed_threshold_px_per_s
    )

    observation_count = len(track.position_history)
    observed_duration = track.last_seen - track.first_seen

    if observation_count < thresholds.min_observations:
        factors = _build_factors(
            track, observation_count, observed_duration, 0.0, 0, 0.0, [], speed_threshold, 0.0, thresholds,
            StoppedVehicleStatus.INSUFFICIENT_EVIDENCE, ["insufficient_observations"],
        )
        return StoppedVehicleClassification(
            track_id=track.track_id, camera_id=track.camera_id,
            status=StoppedVehicleStatus.INSUFFICIENT_EVIDENCE, motion_space=track.motion_space,
            heuristic_score=0.0, confidence_tier=ConfidenceTier.LOW, factors=factors,
        )

    positions = _extract_positions(track)
    steps = _steps(positions)
    step_speeds = [speed for (_, _, _, speed) in steps]

    gaps = [dt for (_, _, dt, _) in steps if dt > thresholds.max_gap_seconds]
    gap_time_beyond_threshold = sum(gaps)
    if observed_duration > 0:
        temporal_coverage = max(0.0, 1.0 - (gap_time_beyond_threshold / observed_duration))
    else:
        temporal_coverage = 0.0

    if observed_duration <= 0 or temporal_coverage < thresholds.min_temporal_coverage:
        factors = _build_factors(
            track, observation_count, observed_duration, temporal_coverage, len(gaps), gap_time_beyond_threshold,
            step_speeds, speed_threshold, 0.0, thresholds,
            StoppedVehicleStatus.INSUFFICIENT_EVIDENCE, ["insufficient_temporal_coverage"],
        )
        return StoppedVehicleClassification(
            track_id=track.track_id, camera_id=track.camera_id,
            status=StoppedVehicleStatus.INSUFFICIENT_EVIDENCE, motion_space=track.motion_space,
            heuristic_score=0.0, confidence_tier=ConfidenceTier.LOW, factors=factors,
        )

    dwell_start, dwell_end, dwell_duration = _longest_low_speed_run(steps, speed_threshold, thresholds.max_gap_seconds)

    if dwell_duration >= thresholds.dwell_seconds:
        status = StoppedVehicleStatus.STOPPED
        dwell_window = (dwell_start, dwell_end)
        dwell_ratio = min(1.0, dwell_duration / (thresholds.dwell_seconds * 2.0))
        heuristic_score = round(temporal_coverage * dwell_ratio, 6)
        reasons = ["sustained_low_speed_dwell_observed"]
    else:
        status = StoppedVehicleStatus.MOVING
        dwell_window = None
        shortfall_ratio = min(1.0, dwell_duration / thresholds.dwell_seconds)
        heuristic_score = round(temporal_coverage * (1.0 - shortfall_ratio), 6)
        reasons = ["no_qualifying_dwell_segment"] if dwell_duration == 0.0 else ["dwell_segment_below_threshold"]

    factors = _build_factors(
        track, observation_count, observed_duration, temporal_coverage, len(gaps), gap_time_beyond_threshold,
        step_speeds, speed_threshold, dwell_duration, thresholds, status, reasons,
    )

    return StoppedVehicleClassification(
        track_id=track.track_id, camera_id=track.camera_id, status=status, motion_space=track.motion_space,
        heuristic_score=heuristic_score, confidence_tier=_confidence_tier_for(heuristic_score), factors=factors,
        dwell_window=dwell_window,
    )


def build_safety_event(classification: StoppedVehicleClassification) -> SafetyEvent:
    """STOPPED -> a SafetyEvent scoped to the actual qualifying dwell
    interval (never the track's full lifetime). MOVING/INSUFFICIENT_EVIDENCE
    -> None (no event exists to record)."""
    if classification.status != StoppedVehicleStatus.STOPPED:
        return None
    return SafetyEvent(
        event_id=f"{classification.track_id}:stopped_vehicle:{classification.dwell_window[0]}-{classification.dwell_window[1]}",
        event_type="stopped_vehicle",
        camera_id=classification.camera_id,
        track_id=classification.track_id,
        window=classification.dwell_window,
        motion_space=classification.motion_space,
        confidence_tier=classification.confidence_tier,
    )
