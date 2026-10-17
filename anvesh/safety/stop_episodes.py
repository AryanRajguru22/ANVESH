"""Stopped-vehicle EPISODES that span window boundaries (M-A.1, Fix 5).

THE DEFECT THIS REPLACES. The per-window classifier (`stopped_vehicle.
classify_stopped_vehicle`) judges a window-CLIPPED track. A vehicle that stops at
7.5 s and moves off at 12.5 s (5 s, well above the 3 s dwell threshold) has only
2.5 s inside window [0, 10) and 2.5 s inside [10, 20): neither window sees a
qualifying stop, so the stop was missed in both (reproduced before the fix). A
stop that DOES qualify in several consecutive windows would, conversely, have been
reported as a new incident in each.

THE MODEL. An EPISODE is one continuous low-speed run of one tracklet, followed
incrementally across the whole stream (independent of how it is cut into windows):

  - the run starts at the first sample of the first qualifying step (the same
    definition `_longest_low_speed_run` uses): a step qualifies if its time gap is
    <= `max_gap_seconds` AND its speed is <= the threshold for the track's space;
  - it is CONFIRMED the first time the run's duration reaches `dwell_seconds`
    (`confirmed_at`); only confirmed runs are episodes;
  - it CLOSES when a step does not qualify (the vehicle moved, or a silence broke
    continuity) or the tracklet is no longer observed.

EVENT IDENTITY / LIFECYCLE. `episode_id` / the SafetyEvent id are keyed on
`(track_id, start)` -- stable however many windows the stop spans. Exactly ONE
window reports a given episode as OPENED (the one containing `confirmed_at`) and
only that window yields a `SafetyEvent`; windows it merely overlaps report it as
CONTINUING, and the window it ends in as CLOSED. Windows must tile (non-overlapping),
as `WindowReasoner` already requires; with overlapping windows the same episode would
be OPENED in more than one.

WHAT "SPEED" MEANS HERE. Image-space episodes (`speed_basis=image_px_per_s_heuristic`)
use a pixel-per-second heuristic that is camera- and scale-specific: they say "the
box barely moved on screen", never "the vehicle was slower than X m/s". Only a track
in validated metric space (`speed_basis=metric_m_per_s`) carries a real-world
threshold. The distinction is on the episode and on the SafetyEvent's motion_space.

NOT ESTABLISHED: that the vehicle is the same physical vehicle after a tracklet
break; that the stop is an incident (queues stop vehicles too).
"""

from __future__ import annotations

import collections
import enum
import math
from dataclasses import dataclass

from anvesh.safety.stopped_vehicle import StoppedVehicleThresholds, _confidence_tier_for
from anvesh.storage.schemas import ConfidenceTier, MotionSpace, SafetyEvent

IMAGE_BASIS = "image_px_per_s_heuristic"
METRIC_BASIS = "metric_m_per_s"
_EPS = 1e-9


class EpisodePhase(str, enum.Enum):
    OPENED = "opened"  # confirmed inside this window: the (only) window that yields the SafetyEvent
    CONTINUING = "continuing"  # confirmed in an earlier window, still going / overlapping this one
    CLOSED = "closed"  # confirmed earlier, ended inside this window


@dataclass(frozen=True)
class StopEpisode:
    """State of one episode AS OF the moment it was read."""

    episode_id: str
    camera_id: str
    track_id: str  # "<camera>:<session>-t<serial>"
    motion_space: MotionSpace
    speed_basis: str  # IMAGE_BASIS | METRIC_BASIS
    speed_threshold: float  # in the units of `speed_basis` (px/s heuristic, or m/s)
    start: float  # first sample of the continuous low-speed run
    confirmed_at: float  # stream time the dwell threshold was first reached
    last_low_speed: float  # latest sample time inside the run
    end: float = None  # None while open
    closed_reason: str = None  # "moved" | "gap" | "track_ended" | "stream_ended"
    dwell_required: float = 3.0

    @property
    def is_open(self) -> bool:
        return self.end is None

    @property
    def dwell_s(self) -> float:
        return self.last_low_speed - self.start

    @property
    def score(self) -> float:
        # Same shape as the per-track classifier's score; continuity is 100% by
        # construction (every step of a run is within max_gap), so only dwell matters.
        return round(min(1.0, self.dwell_s / (self.dwell_required * 2.0)), 6)

    @property
    def confidence_tier(self) -> ConfidenceTier:
        return _confidence_tier_for(self.score)


@dataclass(frozen=True)
class StopEpisodeView:
    """An episode seen from one window."""

    episode: StopEpisode
    phase: EpisodePhase


class _TrackState:
    __slots__ = ("space", "prev", "last_obs", "run_start", "run_last", "confirmed_at")

    def __init__(self, space, first_sample) -> None:
        self.space = space
        self.prev = first_sample  # (t, x, y)
        self.last_obs = first_sample[0]
        self.run_start = None
        self.run_last = None
        self.confirmed_at = None


class StopEpisodeTracker:
    """Incremental episode detection over per-sample positions. Deterministic: the
    result depends only on the samples and their order per track."""

    def __init__(self, camera_id: str, thresholds: StoppedVehicleThresholds = None) -> None:
        self._camera_id = camera_id
        self._th = thresholds or StoppedVehicleThresholds()
        self._tracks: dict = {}  # track_id -> _TrackState (only tracks observed within max_gap)
        self._closed: collections.deque = collections.deque()  # StopEpisode, in closing order

    # -- feeding -----------------------------------------------------------

    def _threshold(self, space: MotionSpace) -> float:
        return self._th.world_speed_threshold_m_per_s if space == MotionSpace.WORLD else self._th.image_speed_threshold_px_per_s

    def observe(self, track_id: str, space: MotionSpace, t: float, x: float, y: float) -> None:
        """One position sample of one tracklet, in `space` units (px for IMAGE, metres for WORLD)."""
        st = self._tracks.get(track_id)
        if st is None:
            self._tracks[track_id] = _TrackState(space, (t, x, y))
            return
        pt, px, py = st.prev
        dt = t - pt
        if dt <= 0:
            return  # duplicate / out-of-order sample: no step
        speed = math.hypot(x - px, y - py) / dt
        if dt <= self._th.max_gap_seconds + _EPS and speed <= self._threshold(st.space):
            if st.run_start is None:
                st.run_start = pt
            st.run_last = t
            if st.confirmed_at is None and st.run_last - st.run_start >= self._th.dwell_seconds - _EPS:
                st.confirmed_at = t
        else:
            self._close_run(track_id, st, "gap" if dt > self._th.max_gap_seconds + _EPS else "moved")
        st.prev = (t, x, y)
        st.last_obs = t

    def advance(self, t: float) -> None:
        """Stream time `t` has been reached: tracklets silent for longer than `max_gap_seconds`
        can no longer extend a run, so their runs are closed and their state dropped (bounded memory)."""
        for track_id in [k for k, s in self._tracks.items() if t - s.last_obs > self._th.max_gap_seconds + _EPS]:
            st = self._tracks.pop(track_id)
            self._close_run(track_id, st, "track_ended")

    def finish(self) -> None:
        """End of stream: close every open run."""
        for track_id in list(self._tracks):
            self._close_run(track_id, self._tracks.pop(track_id), "stream_ended")

    def _make(self, track_id: str, st: _TrackState, end, reason) -> StopEpisode:
        space = st.space
        return StopEpisode(
            episode_id=f"{track_id}:stopped_vehicle:{st.run_start}",
            camera_id=self._camera_id, track_id=track_id, motion_space=space,
            speed_basis=METRIC_BASIS if space == MotionSpace.WORLD else IMAGE_BASIS,
            speed_threshold=self._threshold(space), start=st.run_start, confirmed_at=st.confirmed_at,
            last_low_speed=st.run_last, end=end, closed_reason=reason, dwell_required=self._th.dwell_seconds,
        )

    def _close_run(self, track_id: str, st: _TrackState, reason: str) -> None:
        if st.confirmed_at is not None:
            self._closed.append(self._make(track_id, st, st.run_last, reason))
        st.run_start = st.run_last = st.confirmed_at = None

    # -- reading -----------------------------------------------------------

    def episodes(self) -> list:
        """Every known episode (closed ones still retained + open ones), as of now."""
        out = list(self._closed)
        out.extend(self._make(tid, st, None, None) for tid, st in self._tracks.items() if st.confirmed_at is not None)
        return out

    def views(self, start: float, end: float) -> tuple:
        return episode_views(self.episodes(), start, end)

    def evict_before(self, t: float) -> int:
        """Forget closed episodes that ended before `t` (no open window can need them)."""
        n = 0
        while self._closed and self._closed[0].end < t:
            self._closed.popleft()
            n += 1
        return n

    @property
    def tracked_count(self) -> int:
        return len(self._tracks)


def episode_views(episodes: list, start: float, end: float) -> tuple:
    """The episodes touching window [start, end), each tagged with its phase there,
    ordered by (start, track_id). An episode is in the window if it was confirmed
    before the window ended and had not ended before the window began."""
    out = []
    for ep in episodes:
        if ep.confirmed_at >= end or (ep.end is not None and ep.end < start):
            continue
        if ep.confirmed_at >= start:
            phase = EpisodePhase.OPENED
        elif ep.end is not None and ep.end < end:
            phase = EpisodePhase.CLOSED
        else:
            phase = EpisodePhase.CONTINUING
        out.append(StopEpisodeView(ep, phase))
    out.sort(key=lambda v: (v.episode.start, v.episode.track_id))
    return tuple(out)


def build_stop_event(view: StopEpisodeView) -> SafetyEvent:
    """A SafetyEvent for an episode OPENED in the window (None for any other phase:
    a continuing or closing episode is the same incident, not a new one). Its window is
    the qualifying dwell interval known when it opened; its id is stable for the episode."""
    if view.phase != EpisodePhase.OPENED:
        return None
    ep = view.episode
    return SafetyEvent(
        event_id=ep.episode_id, event_type="stopped_vehicle", camera_id=ep.camera_id, track_id=ep.track_id,
        window=(ep.start, ep.end if ep.end is not None else ep.last_low_speed),
        motion_space=ep.motion_space, confidence_tier=ep.confidence_tier,
    )


def stop_episodes_from_track_states(camera_id: str, states, calibration_profile=None, thresholds: StoppedVehicleThresholds = None) -> list:
    """Post-hoc episodes for a whole recording from its FINAL `TrackState`s, using exactly
    the positions the live engine feeds the tracker (centroid pixels, or ground points
    projected to metres for validated geometry). Track ids match `build_vehicle_track`."""
    from anvesh.perception.schema_conversion import motion_space_for

    space = motion_space_for(calibration_profile)
    samples = {}
    for s in states:
        if space == MotionSpace.WORLD:
            pts = []
            for x, y, t in s.ground_point_trajectory:
                try:
                    wx, wy = calibration_profile.image_to_world((x, y))
                except ValueError:
                    continue
                pts.append((t, wx, wy))
        else:
            pts = [(t, x, y) for x, y, t in s.trajectory]
        samples[f"{camera_id}:{s.track_id}"] = (space, pts)
    return stop_episodes_from_samples(camera_id, samples, thresholds)


def stop_episodes_from_samples(camera_id: str, samples_by_track: dict, thresholds: StoppedVehicleThresholds = None) -> list:
    """Post-hoc episodes from complete per-track samples: `{track_id: (space, [(t, x, y), ...])}`.
    Feeds the SAME tracker the live engine uses, so a recording's episodes equal a stream's."""
    tracker = StopEpisodeTracker(camera_id, thresholds)
    for track_id in sorted(samples_by_track):
        space, samples = samples_by_track[track_id]
        for t, x, y in samples:
            tracker.observe(track_id, space, t, x, y)
    tracker.finish()
    return tracker.episodes()
