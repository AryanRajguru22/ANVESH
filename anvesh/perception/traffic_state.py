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
as `DataQuality.ESTIMATED`, never presented as a real density.
`flow_rate` (vehicles / real elapsed seconds) is the one quantity here
that is always exactly measurable regardless of calibration.

OCCUPANCY -- why the numerator is `mean_concurrency`, not `vehicle_count`:
Occupancy asks "how much of the view is occupied", so its numerator must
be a count of vehicles PRESENT, not a count of vehicles that passed
through. `vehicle_count` is the latter: it counts every distinct track
overlapping the window, so on through-traffic it grows with the window
length and with vehicle speed -- faster free-flowing traffic pushes more
distinct vehicles through the window and would RAISE an occupancy built
on it, inverting the quantity's meaning. `mean_concurrency` (the
time-average of how many tracks are simultaneously in view, i.e. total
in-window track dwell / window duration) is the presence-count occupancy
actually needs: it is invariant to window length for steady traffic and
falls, correctly, when the same demand moves through faster. Occupancy
remains a capacity-based PROXY (`mean_concurrency / max_capacity_vehicles`,
a documented config constant), since V1 still has no per-camera
occupancy-length geometry -- but it is now dimensionally the right kind
of quantity to compare against that capacity. `vehicle_count`,
`flow_rate`, and `density` are unchanged: they are honest throughput
measures and other milestones consume them as such.

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

from anvesh.storage.schemas import CongestionLevel, GeometryValidity, Measurement, MotionSpace, TrafficState


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
    mean_concurrency: float = 0.0  # the occupancy numerator, exposed for audit (see module docstring)


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

    def reset(self) -> None:
        """Forget all hysteresis memory (back to the start-up state). Used when the
        camera was UNAVAILABLE for a window: the previous level says nothing about
        what the road looked like after the outage, so the next observed window is
        classified as if it were the first one."""
        self._state = CongestionLevel.FREE_FLOW

    def update(self, mean_speed_value: float, occupancy: float, motion_space: MotionSpace) -> CongestionLevel:
        raw = _raw_pressure(mean_speed_value, occupancy, motion_space, self._thresholds)
        return self._advance(raw)

    def update_empty(self) -> CongestionLevel:
        """Advance one window with zero vehicles observed.

        Not the same as calling `update(0.0, 0.0, ...)`: a genuine 0 m/s
        mean speed over *present* vehicles is a strong congestion signal,
        but a 0 m/s placeholder over *zero* vehicles means "empty road,"
        which is a free-flow signal, not a congestion one -- an empty
        window must never be misread as gridlock. See
        `aggregate_traffic_state`'s vehicle_count == 0 branch, which is
        the only caller of this method.
        """
        return self._advance("low")

    def _advance(self, raw: str) -> CongestionLevel:
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


def _mean_concurrency(tracks: list, window_start: float, window_end: float) -> float:
    """Time-averaged number of tracks simultaneously present in the window.

    Sum of each track's dwell INSIDE the window, divided by the window's
    duration -- the continuous-time equivalent of averaging a per-frame
    "how many vehicles are on screen right now" count, computed from
    `first_seen`/`last_seen` alone so it needs no frame-level data. A
    track spanning the whole window contributes 1.0; one present for half
    of it contributes 0.5. Not clamped: it is a vehicle count, and
    clamping belongs to the capacity ratio in `aggregate_traffic_state`,
    not here.
    """
    window_duration = window_end - window_start
    if window_duration <= 0:
        return 0.0
    dwell = sum(
        max(0.0, min(t.last_seen, window_end) - max(t.first_seen, window_start))
        for t in tracks
    )
    return dwell / window_duration


def _resolve_geometry_validity(camera_id: str, tracks: list, motion_space: MotionSpace, declared) -> GeometryValidity:
    """The window's geometry validity (hardening A2). Explicit wins; otherwise it comes from the
    tracks themselves (each already carries a structurally enforced validity) or, with no tracks,
    from an IMAGE camera. It is NEVER invented for an empty WORLD window: validated geometry is a
    property of the camera's calibration, which only the caller knows."""
    if declared is not None:
        declared = GeometryValidity(declared)
        if motion_space == MotionSpace.WORLD and declared != GeometryValidity.WORLD_VALIDATED:
            raise ValueError(f"aggregate_traffic_state: motion_space=WORLD requires geometry_validity=WORLD_VALIDATED, got {declared.value}")
        if motion_space == MotionSpace.IMAGE and declared == GeometryValidity.WORLD_VALIDATED:
            raise ValueError("aggregate_traffic_state: geometry_validity=WORLD_VALIDATED requires motion_space=WORLD")
        return declared
    if tracks:
        kinds = {t.geometry_validity for t in tracks}
        if motion_space == MotionSpace.WORLD:
            return GeometryValidity.WORLD_VALIDATED  # every WORLD track is WORLD_VALIDATED (VehicleTrack invariant)
        # IMAGE-space tracks may be IMAGE or WORLD_UNVALIDATED; a mixed window takes the weaker claim.
        return GeometryValidity.IMAGE if GeometryValidity.IMAGE in kinds else GeometryValidity.WORLD_UNVALIDATED
    if motion_space == MotionSpace.IMAGE:
        return GeometryValidity.IMAGE
    raise ValueError(
        f"aggregate_traffic_state({camera_id!r}): a window with no tracks on a WORLD camera needs an explicit "
        "geometry_validity (e.g. geometry_validity_for(calibration_profile)); it is never assumed"
    )


def aggregate_traffic_state(
    camera_id: str,
    window_start: float,
    window_end: float,
    tracks: list,
    motion_space: MotionSpace,
    congestion_tracker: CongestionStateTracker,
    segment_length_m: float = None,
    max_capacity_vehicles: int = 20,
    geometry_validity: GeometryValidity = None,
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

    # Speed-safety (M-A): the caller-declared `motion_space` must be the one
    # the tracks actually carry. A WORLD (m/s) label cannot be put on a state
    # computed from image-space or unvalidated-geometry tracks, and image
    # tracks cannot be silently averaged with metric ones.
    mismatched = sorted({t.motion_space.value for t in tracks if t.motion_space != motion_space})
    if mismatched:
        raise ValueError(
            f"aggregate_traffic_state(motion_space={motion_space.value!r}) received tracks in "
            f"{mismatched}: a window must be built from tracks of exactly one motion space"
        )

    geometry_validity = _resolve_geometry_validity(camera_id, tracks, motion_space, geometry_validity)

    overlapping = _tracks_overlapping_window(tracks, window_start, window_end)
    window_duration = window_end - window_start
    vehicle_count = len(overlapping)
    mean_concurrency = _mean_concurrency(overlapping, window_start, window_end)
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
        # `speed_available=False` tracks (a validated-geometry track with too
        # few samples / an implausible speed) have only a placeholder value:
        # they never enter a mean, and a window with none has NO speed.
        measured = [t for t in overlapping if t.speed_available]
        if measured:
            speeds = [t.speed_estimate.value for t in measured]
            mean_speed_value = statistics.fmean(speeds)
            speed_error = statistics.pstdev(speeds) if len(speeds) > 1 else measured[0].speed_estimate.error
            mean_speed = Measurement(value=mean_speed_value, error=speed_error)
            speed_quality = DataQuality.OBSERVED
        else:
            mean_speed = Measurement(value=0.0, error=0.0)
            speed_quality = DataQuality.UNAVAILABLE

        occupancy = min(1.0, mean_concurrency / max_capacity_vehicles)
        occupancy_quality = DataQuality.ESTIMATED  # a capacity-based proxy, not a measured occupied-length fraction

        if segment_length_m is not None and motion_space == MotionSpace.WORLD:
            density = vehicle_count / segment_length_m
            density_quality = DataQuality.OBSERVED
        else:
            density = float(vehicle_count)  # non-metric proxy: "vehicles observed", not vehicles/length
            density_quality = DataQuality.ESTIMATED

    if vehicle_count == 0:
        congestion_level = congestion_tracker.update_empty()
    else:
        # With no usable speed measurement the classification falls back to
        # the occupancy-only rule (the same one image-space cameras use)
        # instead of thresholding a placeholder 0.
        pressure_space = motion_space if speed_quality == DataQuality.OBSERVED else MotionSpace.IMAGE
        congestion_level = congestion_tracker.update(mean_speed.value, occupancy, pressure_space)

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
        # A 0.0 `mean_speed` with no measured vehicle is a placeholder, not a speed.
        mean_speed_available=(speed_quality == DataQuality.OBSERVED),
        geometry_validity=geometry_validity,
    )

    return TrafficStateAggregation(
        traffic_state=traffic_state,
        class_counts=class_counts,
        contributing_track_ids=contributing_track_ids,
        speed_quality=speed_quality,
        density_quality=density_quality,
        occupancy_quality=occupancy_quality,
        mean_concurrency=mean_concurrency,
    )
