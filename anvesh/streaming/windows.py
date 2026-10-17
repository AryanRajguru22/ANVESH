"""Sliding windows over the live world model.

A window is `[start, end)` in STREAM TIME (the `Frame.timestamp` clock).
Window `k` has `start = k * stride_s`, `end = start + window_s`; with the
default `stride_s == window_s` they tile the timeline without overlap exactly
like the legacy `perception.windowing.generate_windows`; a smaller stride
gives overlapping windows.

DETERMINISM / NO-EOF: window `k` is CLOSED the first time a frame arrives with
`timestamp >= end + lateness_s` (or at end-of-stream), purely as a function of
stream time -- no wall clock, no dependence on reaching EOF -- so the same
frames in the same order give identical windows whether they came from a file
replayed as fast as possible or a paced/threaded stream. `window_id` is
`"<camera_id>:w<k:06d>"`.

HYSTERESIS CAVEAT: `perception.traffic_state.CongestionStateTracker` is a
per-window state machine written for CONSECUTIVE, NON-OVERLAPPING windows. With
`stride_s < window_s` each overlapping window would advance it once, so the
same traffic would be counted more than once; use the default stride for any
consumer that owns a CongestionStateTracker (the `WindowReasoner` refuses
overlapping windows for that reason).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class WindowAvailability(str, enum.Enum):
    """How much of a window's time the camera actually delivered (M-A.1).

    A window with no vehicles is NOT the same as a window with no camera: the
    first is a healthy empty scene (OBSERVED, zero tracks), the second must never
    be read as "free flow".

      OBSERVED    -- no frame was lost and no silence longer than `max_gap_s`.
      DEGRADED    -- some loss (queue drops or a silence), but at least
                     `min_coverage` of the window's time was observed: counts are
                     UNDER-counts and the window must be read as such.
      UNAVAILABLE -- less than `min_coverage` of the window's time was observed
                     (including zero frames): the camera was effectively absent.
    """

    OBSERVED = "observed"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class WindowSpec:
    window_s: float = 10.0
    stride_s: float = None  # None -> window_s (non-overlapping)
    lateness_s: float = 0.0
    origin_s: float = 0.0
    # Observability policy (M-A.1). A gap between consecutive frames longer than
    # `max_gap_s` is a SILENCE (the same 1 s the stopped-vehicle dwell logic uses
    # as a continuity break); a window with less than `min_coverage` of its time
    # observed is UNAVAILABLE. Policy parameters, not tuned to any dataset.
    max_gap_s: float = 1.0
    min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError("window_s must be > 0")
        if self.stride_s is None:
            object.__setattr__(self, "stride_s", self.window_s)
        if self.stride_s <= 0:
            raise ValueError("stride_s must be > 0")
        if self.lateness_s < 0:
            raise ValueError("lateness_s must be >= 0")
        if self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be > 0")
        if not (0.0 < self.min_coverage <= 1.0):
            raise ValueError("min_coverage must be within (0, 1]")

    @property
    def overlapping(self) -> bool:
        return self.stride_s < self.window_s

    def bounds(self, k: int) -> tuple:
        start = self.origin_s + k * self.stride_s
        return start, start + self.window_s

    def close_time(self, k: int) -> float:
        return self.bounds(k)[1] + self.lateness_s

    def history_needed_s(self) -> float:
        """How far back (stream seconds) trajectories must be retained so the
        oldest still-open window can be built when it closes."""
        return self.window_s + self.lateness_s


@dataclass(frozen=True)
class WindowSnapshot:
    """One closed window's view of the world model: window-clipped
    `VehicleTrack`s plus the observability the downstream reasoner needs to
    know how trustworthy the window is."""

    window_index: int
    window_id: str
    camera_id: str
    start: float
    end: float  # clipped to the stream duration for a final partial window
    tracks: tuple  # tuple of storage.schemas.VehicleTrack
    frames_observed: int
    dropped_frames: int  # frames lost to queue drops OR source gaps while this window was open
    sessions: tuple  # identity sessions that contributed samples
    closed_at_stream_time: float
    final_partial: bool = False
    # Wall-clock capture time of the frame whose arrival closed this window
    # (None for windows flushed at end-of-stream). Observability only: excluded
    # from equality so batch/stream parity compares world-model content, not timing.
    trigger_capture_wall_s: float = field(default=None, compare=False)
    # --- observability (M-A.1). Defaults describe a window built post hoc from a
    # complete recording (no loss information exists), i.e. "observed".
    expected_frames: int = None  # nominal fps * window length; None if unknown
    silent_s: float = 0.0  # seconds inside the window covered only by silences > max_gap_s
    coverage: float = 1.0  # fraction of the window's time that was observed
    availability: WindowAvailability = WindowAvailability.OBSERVED
    # Stop episodes touching this window (`safety.stop_episodes.StopEpisodeView`), computed
    # across window boundaries. None = not computed (e.g. a window cut post hoc from
    # clipped tracks): consumers then fall back to per-window classification, which is
    # only valid for whole-lifetime tracks.
    stop_episodes: tuple = None

    @property
    def degraded(self) -> bool:
        """True if the window is NOT fully observed (frames lost to queue drops,
        a silence, or the camera effectively absent): its counts are under-counts
        and the consumer must not read them as complete."""
        return self.dropped_frames > 0 or self.availability != WindowAvailability.OBSERVED

    @property
    def unavailable(self) -> bool:
        return self.availability == WindowAvailability.UNAVAILABLE

    @property
    def window(self) -> tuple:
        return (self.start, self.end)


class SlidingWindowScheduler:
    """Tracks which windows are still open and which are now due."""

    def __init__(self, spec: WindowSpec) -> None:
        self.spec = spec
        self._next = 0  # index of the oldest window not yet closed

    @property
    def next_window_index(self) -> int:
        return self._next

    def oldest_open_start(self) -> float:
        return self.spec.bounds(self._next)[0]

    def due(self, stream_time: float) -> list:
        """Indices of windows that close at `stream_time` (in order); they
        are consumed -- each index is returned exactly once."""
        out = []
        while self.spec.close_time(self._next) <= stream_time:
            out.append(self._next)
            self._next += 1
        return out

    def remaining_until(self, duration: float) -> list:
        """At end-of-stream: every not-yet-closed window that starts before
        `duration`."""
        out = []
        while self.spec.bounds(self._next)[0] < duration:
            out.append(self._next)
            self._next += 1
        return out
