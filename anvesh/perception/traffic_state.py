"""Traffic-state extraction (blueprint Part 3, module 5): per-camera
occupancy/speed/count over a wall-clock time window, assembled from
`anvesh.storage.schemas.VehicleTrack`s.

Congestion classification (`TrafficState.congestion_level`) is
hysteresis-damped per Part 5's own note ("derived, hysteresis-damped") --
see `CongestionStateTracker` below. This is a stateful, per-camera object
the caller must keep and reuse across consecutive windows; a fresh
tracker per window would defeat the point of hysteresis.

DENSITY -- what "vehicles per unit length" means here:
V1 has no surveyed, fixed per-camera segment length, so a textbook
vehicles/metre density cannot be measured, only estimated. If the caller
supplies `segment_length_m` (a manually-configured constant, same spirit
as M2's manually-supplied calibration reference points) AND the camera is
calibrated (`motion_space=WORLD`), `density = vehicle_count /
segment_length_m` is a real vehicles/metre figure. Otherwise `density` is
a documented, explicitly non-metric PROXY (`vehicle_count` itself, i.e.
"vehicles per observed view", vehicles per pixel-span is not attempted
since no reliable observed-pixel-extent is available either) -- flagged
as `DataQuality.ESTIMATED`, never presented as a real density. Occupancy
is a proxy too (`vehicle_count / max_capacity_vehicles`, a documented
config constant) since V1 has no per-camera occupancy-length geometry;
`flow_rate` (vehicles / real elapsed seconds) is the one quantity here
that is always exactly measurable regardless of calibration.

SPEED -- per `motion_space`: `mean_speed.value` is m/s when a track's
camera is calibrated, px/s otherwise -- callers must not mix tracks from
differently-calibrated cameras into one `aggregate_traffic_state()` call
(the function assumes a single, uniform `motion_space` for its inputs and
does not check for mixed input; see `tests/perception/test_traffic_state.py`
for what happens if a caller violates this).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from anvesh.storage.schemas import CongestionLevel, Measurement, MotionSpace, TrafficState


class DataQuality(str, Enum):
    """Per-quantity provenance tag -- NOT part of the frozen Part 5 schema
    (adding a quality field to every TrafficState quantity would be schema
    bloat for a concept M3 only needs for diagnostics/tests); lives only
    on the internal `TrafficStateAggregation` wrapper below."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TrafficStateAggregation:
    """The frozen `TrafficState` plus diagnostics that don't belong in the
    Part 5 schema: per-quantity data quality, class counts, and which
    tracks actually contributed (provenance)."""

    traffic_state: TrafficState
    class_counts: dict
    contributing_track_ids: tuple
    speed_quality: DataQuality
    density_quality: DataQuality
    occupancy_quality: DataQuality


@dataclass(frozen=True)
class CongestionThresholds:
    """Explicit, documented congestion thresholds -- no hidden constants.

    Speed thresholds only apply to WORLD (calibrated) cameras: a fixed
    pixels/second threshold is not a defensible universal congestion
    signal (pixel scale is camera-specific), so an IMAGE-space camera
    classifies on the occupancy proxy alone. This is a known, documented
    V1 limitation (see the M3 report), not a claim that occupancy alone
    is as informative as speed.
    """

    free_speed_m_per_s: float = 8.0  # ~29 km/h: at/above this, traffic flows freely
    congested_speed_m_per_s: float = 3.0  # ~11 km/h: at/below this, traffic is congested
    high_occupancy: float = 0.6  # occupancy proxy at/above this reinforces "congested"
    low_occupancy: float = 0.3  # occupancy proxy at/below this reinforces "free" (image-space only)


def _raw_pressure(mean_speed_value: float, occupancy: float, motion_space: MotionSpace, thresholds: CongestionThresholds) -> str:
    """One window's instantaneous congestion signal, before hysteresis.

    Returns "low", "medium", or "high". Not exported: only
    CongestionStateTracker should turn this into a CongestionLevel.
    """
    if motion_space == MotionSpace.WORLD:
        if mean_speed_value >= thresholds.free_speed_m_per_s and occupancy < thresholds.high_occupancy:
            return "low"
        if mean_speed_value <= thresholds.congested_speed_m_per_s or occupancy >= thresholds.high_occupancy:
            return "high"
        return "medium"
    # IMAGE space: no defensible universal speed threshold -- occupancy proxy only.
    if occupancy >= thresholds.high_occupancy:
        return "high"
    if occupancy <= thresholds.low_occupancy:
        return "low"
    return "medium"


class CongestionStateTracker:
    """Per-camera hysteresis state machine over `CongestionLevel`.

    Uses the blueprint's own 4-state vocabulary as the hysteresis
    mechanism itself: BUILDING and DISSIPATING act as one-window
    confirmation buffers, so a single noisy window cannot flip
    FREE_FLOW <-> CONGESTED directly -- it must pass through (and be
    reconfirmed by) the transitional state first. This directly targets
    the "one noisy frame shouldn't flip state" regression.
    """

    def __init__(self, thresholds: CongestionThresholds = None) -> None:
        self._thresholds = thresholds or CongestionThresholds()
        self._state = CongestionLevel.FREE_FLOW

    @property
    def state(self) -> CongestionLevel:
        return self._state

    def update(self, mean_speed_value: float, occupancy: float, motion_space: MotionSpace) -> CongestionLevel:
        raw = _raw_pressure(mean_speed_value, occupancy, motion_space, self._thresholds)

        if self._state == CongestionLevel.FREE_FLOW:
            self._state = CongestionLevel.BUILDING if raw == "high" else CongestionLevel.FREE_FLOW
        elif self._state == CongestionLevel.BUILDING:
            if raw == "high":
                self._state = CongestionLevel.CONGESTED
            elif raw == "low":
                self._state = CongestionLevel.FREE_FLOW
            # raw == "medium": stay BUILDING, still ambiguous
        elif self._state == CongestionLevel.CONGESTED:
            if raw != "high":
                self._state = CongestionLevel.DISSIPATING
            # raw == "high": stay CONGESTED
        elif self._state == CongestionLevel.DISSIPATING:
            if raw == "low":
                self._state = CongestionLevel.FREE_FLOW
            elif raw == "high":
                self._state = CongestionLevel.CONGESTED
            # raw == "medium": stay DISSIPATING, still ambiguous

        return self._state


def _tracks_overlapping_window(tracks: list, window_start: float, window_end: float) -> list:
    return [t for t in tracks if t.first_seen < window_end and t.last_seen > window_start]


def aggregate_traffic_state(
    camera_id: str,
    window_start: float,
    window_end: float,
    tracks: list,
    motion_space: MotionSpace,
    congestion_tracker: CongestionStateTracker,
    segment_length_m: float = None,
    max_capacity_vehicles: int = 20,
) -> TrafficStateAggregation:
    """Aggregate one camera's VehicleTracks overlapping [window_start,
    window_end] into a TrafficStateAggregation (frozen TrafficState +
    diagnostics).

    `tracks` may be pre-filtered to the window or not -- overlap is
    (re-)checked here via `first_seen < window_end and last_seen >
    window_start`. `congestion_tracker` is mutated (advanced by exactly
    one window) as a side effect -- own one instance per camera and reuse
    it across consecutive calls, or hysteresis has no effect.
    """
    if window_end < window_start:
        raise ValueError("window_end must be >= window_start")

    overlapping = _tracks_overlapping_window(tracks, window_start, window_end)
    window_duration = window_end - window_start
    vehicle_count = len(overlapping)
    class_counts = dict(Counter(t.vehicle_class for t in overlapping))
    contributing_track_ids = tuple(t.track_id for t in overlapping)

    flow_rate = vehicle_count / window_duration if window_duration > 0 else 0.0

    if vehicle_count == 0:
        mean_speed = Measurement(value=0.0, error=0.0)
        speed_quality = DataQuality.UNAVAILABLE
        occupancy = 0.0
        occupancy_quality = DataQuality.OBSERVED  # zero vehicles genuinely means zero occupied
        density = 0.0
        density_quality = DataQuality.OBSERVED
    else:
        speeds = [t.speed_estimate.value for t in overlapping]
        mean_speed_value = statistics.fmean(speeds)
        speed_error = statistics.pstdev(speeds) if len(speeds) > 1 else overlapping[0].speed_estimate.error
        mean_speed = Measurement(value=mean_speed_value, error=speed_error)
        speed_quality = DataQuality.OBSERVED

        occupancy = min(1.0, vehicle_count / max_capacity_vehicles)
        occupancy_quality = DataQuality.ESTIMATED  # a capacity-based proxy, not a measured occupied-length fraction

        if segment_length_m is not None and motion_space == MotionSpace.WORLD:
            density = vehicle_count / segment_length_m
            density_quality = DataQuality.OBSERVED
        else:
            density = float(vehicle_count)  # non-metric proxy: "vehicles observed", not vehicles/length
            density_quality = DataQuality.ESTIMATED

    congestion_level = congestion_tracker.update(mean_speed.value, occupancy, motion_space)

    traffic_state = TrafficState(
        camera_id=camera_id,
        window_start=window_start,
        window_end=window_end,
        occupancy=occupancy,
        mean_speed=mean_speed,
        vehicle_count=vehicle_count,
        flow_rate=flow_rate,
        density=density,
        congestion_level=congestion_level,
        motion_space=motion_space,
    )

    return TrafficStateAggregation(
        traffic_state=traffic_state,
        class_counts=class_counts,
        contributing_track_ids=contributing_track_ids,
        speed_quality=speed_quality,
        density_quality=density_quality,
        occupancy_quality=occupancy_quality,
    )
