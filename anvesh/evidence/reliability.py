"""Evidence reliability scoring (blueprint Part 3, module 9).

Produces a per-camera, per-window reliability SCORE: a heuristic, documented
weighted combination of observation count, tracking continuity, calibration
status, and temporal coverage. This is NOT a probability and NOT
cross-camera fusion -- the Dempster-Shafer reliability discounting in
blueprint Part 6 (`fusion/ds_fusion.py`, milestone M4) is a different,
still-unimplemented thing that will eventually *consume* a score like this
one as one input. Nothing here should be read as a calibrated confidence
level; it exists only to give M4 (and this milestone's own diagnostics) a
documented way to say "trust this camera's evidence more/less this window."
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.perception.traffic_state import TrafficStateAggregation
from anvesh.storage.schemas import MotionSpace, OcclusionState


@dataclass(frozen=True)
class ReliabilityWeights:
    """Explicit, documented weights -- no hidden constants. Must sum to 1.0."""

    observation_count: float = 0.25
    tracking_continuity: float = 0.25
    calibration: float = 0.25
    temporal_coverage: float = 0.25

    def __post_init__(self) -> None:
        total = self.observation_count + self.tracking_continuity + self.calibration + self.temporal_coverage
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ReliabilityWeights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ReliabilityScore:
    """A heuristic [0, 1] quality score for one camera's evidence in one
    window. NOT a probability -- see module docstring."""

    camera_id: str
    window_start: float
    window_end: float
    score: float
    factors: dict
    method: str = "weighted_heuristic_v1"

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be within [0, 1], got {self.score}")
        for name, value in self.factors.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"factor '{name}' must be within [0, 1], got {value}")


def _observation_count_factor(observation_count: int, saturating_count: int) -> float:
    if saturating_count <= 0:
        raise ValueError("saturating_count must be > 0")
    return min(1.0, observation_count / saturating_count)


def _tracking_continuity_factor(tracks: list) -> float:
    if not tracks:
        return 0.0
    visible = sum(1 for t in tracks if t.occlusion_state == OcclusionState.VISIBLE)
    return visible / len(tracks)


def _calibration_factor(motion_space: MotionSpace) -> float:
    # Image-space (uncalibrated) evidence cannot be cross-camera compared
    # against known geometry, so it is trusted less -- not zero, since it
    # is still real per-camera evidence, just less useful downstream.
    return 1.0 if motion_space == MotionSpace.WORLD else 0.5


def _temporal_coverage_factor(tracks: list, window_start: float, window_end: float) -> float:
    window_duration = window_end - window_start
    if window_duration <= 0 or not tracks:
        return 0.0
    covered = 0.0
    for t in tracks:
        overlap_start = max(t.first_seen, window_start)
        overlap_end = min(t.last_seen, window_end)
        covered += max(0.0, overlap_end - overlap_start)
    return min(1.0, covered / window_duration)


def compute_reliability(
    camera_id: str,
    window_start: float,
    window_end: float,
    tracks: list,
    aggregation: TrafficStateAggregation,
    weights: ReliabilityWeights = None,
    saturating_observation_count: int = 3,
) -> ReliabilityScore:
    """Score one camera's evidence quality for one window.

    `tracks` should be the same overlapping-track list that produced
    `aggregation` (see `traffic_state.aggregate_traffic_state`) -- this
    function does not re-filter by window overlap itself for continuity/
    coverage, it trusts the caller's `tracks` argument as already scoped
    to the window (matching `contributing_track_ids` on `aggregation`).
    """
    weights = weights or ReliabilityWeights()
    motion_space = aggregation.traffic_state.motion_space

    factors = {
        "observation_count": _observation_count_factor(len(tracks), saturating_observation_count),
        "tracking_continuity": _tracking_continuity_factor(tracks),
        "calibration": _calibration_factor(motion_space),
        "temporal_coverage": _temporal_coverage_factor(tracks, window_start, window_end),
    }

    score = (
        weights.observation_count * factors["observation_count"]
        + weights.tracking_continuity * factors["tracking_continuity"]
        + weights.calibration * factors["calibration"]
        + weights.temporal_coverage * factors["temporal_coverage"]
    )

    return ReliabilityScore(
        camera_id=camera_id,
        window_start=window_start,
        window_end=window_end,
        score=score,
        factors=factors,
    )
