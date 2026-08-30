"""M1 -> M3 windowing glue (M6 real-world validation).

`perception.pipeline.run_m1_pipeline()` returns one flat `vehicle_tracks`
list spanning the whole video; `perception.traffic_state.
aggregate_traffic_state()` expects to be called once per fixed time
window. Nothing previously connected the two -- this module is exactly
that connection, and nothing more.

This module does NOT reimplement M3's aggregation math (speed, occupancy,
congestion classification): `generate_windows` only computes window
boundaries, `tracks_in_window` mirrors (does not replace) the same
trivial overlap predicate `aggregate_traffic_state` already applies
internally -- exposed here only so a caller can report raw per-window
track counts/identities *before* aggregation, for sanity-checking real
data. `build_traffic_state_timeline` is pure orchestration: it calls
`aggregate_traffic_state` once per window with one persistent
`CongestionStateTracker` (required for the hysteresis state machine to
mean anything across windows -- a fresh tracker per window would defeat
it, exactly as `traffic_state.py` itself documents); it computes nothing
that module doesn't already compute.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.perception.traffic_state import aggregate_traffic_state


@dataclass(frozen=True)
class Window:
    index: int
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("window end must be >= start")


def generate_windows(total_duration: float, window_seconds: float) -> tuple:
    """Fixed-duration, non-overlapping windows covering [0, total_duration).

    The final window is clipped to `total_duration` (it may be shorter
    than `window_seconds`) rather than silently dropped or overrun.
    `total_duration == 0` yields an empty tuple, not a zero-length window.
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
    if total_duration < 0:
        raise ValueError(f"total_duration must be >= 0, got {total_duration}")

    windows = []
    start = 0.0
    index = 0
    while start < total_duration:
        end = min(start + window_seconds, total_duration)
        windows.append(Window(index=index, start=start, end=end))
        start = end
        index += 1
    return tuple(windows)


def tracks_in_window(tracks: list, window: Window) -> list:
    """Tracks overlapping `window`, preserving original order and identity.

    Mirrors -- does not replace -- the exact overlap predicate
    `aggregate_traffic_state` applies internally
    (`first_seen < window_end and last_seen > window_start`); kept here
    only so a caller can inspect/report which tracks fall in a window
    before/independent of running the actual aggregation.
    """
    return [t for t in tracks if t.first_seen < window.end and t.last_seen > window.start]


def video_duration_from_tracks(tracks: list) -> float:
    """The latest `last_seen` across all tracks -- an honest proxy for
    'how much timeline is there to window over' when only track data
    (not the original frame count/fps) is available. Returns 0.0 for an
    empty track list rather than raising."""
    if not tracks:
        return 0.0
    return max(t.last_seen for t in tracks)


def build_traffic_state_timeline(
    camera_id: str,
    tracks: list,
    motion_space,
    window_seconds: float,
    congestion_tracker,
    total_duration: float = None,
    segment_length_m: float = None,
    max_capacity_vehicles: int = 20,
) -> tuple:
    """Run `aggregate_traffic_state` once per fixed window across the
    track list's timeline, in chronological order, sharing ONE
    `congestion_tracker` across all windows.

    Pure orchestration -- every `TrafficStateAggregation` in the returned
    tuple is produced entirely by `aggregate_traffic_state`; nothing here
    recomputes speed, occupancy, density, or congestion level.
    """
    duration = total_duration if total_duration is not None else video_duration_from_tracks(tracks)
    windows = generate_windows(duration, window_seconds)

    return tuple(
        aggregate_traffic_state(
            camera_id,
            w.start,
            w.end,
            tracks,
            motion_space,
            congestion_tracker,
            segment_length_m=segment_length_m,
            max_capacity_vehicles=max_capacity_vehicles,
        )
        for w in windows
    )
