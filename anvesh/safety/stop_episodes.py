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

OPT-IN DISPLACEMENT CRITERION (hardening C). The default rule above judges the speed of every
single step. That is fragile twice over: bbox jitter is a few pixels per FRAME, so the implied
px/s scales with the frame rate (a parked car on cached real c006 footage has a MEDIAN per-step
speed of 3.7-7.7 px/s against the 5 px/s threshold), and one gap longer than `max_gap_seconds`
closes the run so the same stop gets a second id. `DisplacementCriterion` (passed as
`displacement=`) replaces both: a vehicle is stopped while each sample stays within a radius of the
mean of its last `anchor_window_s` of samples (frame-rate independent; for a vehicle at constant
speed v that is exactly "v <= 2R / W", so a slow crawl stays ONE continuous episode instead of a
chain of them), a few out-of-radius samples are tolerated as noise, and a short silence over which
the vehicle has not moved is BRIDGED, so the run keeps its original `episode_id`. It is OFF unless asked for; the default behaviour is unchanged. Its
parameters are untuned policy values, camera- and scale-specific in IMAGE space (pixels), exactly
like the heuristic it complements.
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
IMAGE_DISPLACEMENT_BASIS = "image_px_displacement_heuristic"
METRIC_DISPLACEMENT_BASIS = "metric_displacement_m"
_EPS = 1e-9


@dataclass(frozen=True)
class DisplacementCriterion:
    """Opt-in stop criterion: "stays within a radius", instead of "every step is slow".

    The radius is DERIVED from the default rule so the two agree on what "slow" means: a vehicle at
    constant speed v sits v*W/2 from the mean of its last W seconds, so it is stationary iff
    v <= 2R/W. With W = the 3 s dwell, R = 7.5 px <=> 5 px/s (the default image threshold) and
    R = 0.75 m <=> 0.5 m/s (the default metric threshold). The remaining values (exit_seconds,
    bridge_gap_s) are UNTUNED policy defaults, checked only against the cached c006 clip; none of it
    is derived from labelled stops.

      image_radius_px   -- IMAGE space: a sample belongs to the run while within this many pixels
                           of the mean of the run's last `anchor_window_s`. Pixel scale differs per
                           camera/resolution (like the default 5 px/s heuristic it matches).
      world_radius_m    -- validated-metric (WORLD) space: the same, in metres.
      anchor_window_s   -- W above. Keep it equal to the dwell used by the tracker's thresholds.
      exit_seconds      -- how long samples must stay outside the radius before the run is closed
                           (shorter excursions are treated as detection noise and ignored).
      bridge_gap_s      -- a silence longer than `max_gap_seconds` but not longer than this is
                           bridged if the vehicle reappears inside the radius (same run, same id);
                           0 disables bridging. Bridged time is never counted as observed dwell.
    """

    image_radius_px: float = 7.5
    world_radius_m: float = 0.75
    anchor_window_s: float = 3.0
    exit_seconds: float = 1.0
    bridge_gap_s: float = 3.0

    def __post_init__(self) -> None:
        for name in ("image_radius_px", "world_radius_m", "anchor_window_s", "exit_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0 or math.isinf(value):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        if not isinstance(self.bridge_gap_s, (int, float)) or isinstance(self.bridge_gap_s, bool) \
                or not self.bridge_gap_s >= 0 or math.isinf(self.bridge_gap_s):
            raise ValueError(f"bridge_gap_s must be a finite number >= 0, got {self.bridge_gap_s!r}")

    def radius(self, space: MotionSpace) -> float:
        return self.world_radius_m if space == MotionSpace.WORLD else self.image_radius_px


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
    # Displacement criterion only (zero/"step_speed" for the default rule): silences the run was bridged over.
    bridged_s: float = 0.0
    bridged_gaps: int = 0
    criterion: str = "step_speed"

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


class _DispState:
    """Per-track state of the displacement criterion (a different rule, so a different record
    from `_TrackState`; the default rule never touches this)."""

    __slots__ = ("space", "last_obs", "run_start", "win", "sum_x", "sum_y", "run_last", "observed_s",
                 "confirmed_at", "bridged_s", "bridged_gaps", "pending")

    def __init__(self, space) -> None:
        self.space = space
        self.last_obs = None
        self.pending = []  # (t, x, y) samples outside the radius since the last in-radius sample
        self.reset_run()

    def reset_run(self) -> None:
        self.run_start = self.run_last = self.confirmed_at = None
        self.win = collections.deque()  # (t, x, y) of the in-radius samples of the last `anchor_window_s`
        self.sum_x = self.sum_y = 0.0
        self.observed_s = 0.0
        self.bridged_s = 0.0
        self.bridged_gaps = 0
        self.pending = []

    def start_run(self, t: float, x: float, y: float) -> None:
        self.reset_run()
        self.run_start = self.run_last = t
        self.win.append((t, x, y))
        self.sum_x, self.sum_y = x, y

    def push(self, t: float, x: float, y: float, window_s: float) -> None:
        """Add an in-radius sample and drop those older than the anchor window (always keeping one)."""
        self.win.append((t, x, y))
        self.sum_x += x
        self.sum_y += y
        while len(self.win) > 1 and t - self.win[0][0] > window_s + _EPS:
            _, ox, oy = self.win.popleft()
            self.sum_x -= ox
            self.sum_y -= oy

    @property
    def anchor(self) -> tuple:
        """Mean of the window as of the last accepted sample (the candidate-free, last-known position)."""
        n = len(self.win)
        return self.sum_x / n, self.sum_y / n

    def anchor_with(self, t: float, x: float, y: float, window_s: float) -> tuple:
        """Mean of the window as it WOULD be after adding the sample (t, x, y): samples older than
        `window_s` before `t` are excluded and the sample itself is included. Testing a sample against
        this mean (not the pre-sample one) makes the stationary boundary exactly v <= 2R/W whatever
        the frame rate: the newest sample of a constant-speed vehicle is v*W/2 from it."""
        rx = ry = 0.0
        removed = 0
        for wt, wx, wy in self.win:
            if len(self.win) - removed > 1 and t - wt > window_s + _EPS:
                rx += wx
                ry += wy
                removed += 1
            else:
                break
        n = len(self.win) - removed + 1
        return (self.sum_x - rx + x) / n, (self.sum_y - ry + y) / n


class StopEpisodeTracker:
    """Incremental episode detection over per-sample positions. Deterministic: the
    result depends only on the samples and their order per track."""

    def __init__(self, camera_id: str, thresholds: StoppedVehicleThresholds = None,
                 displacement: DisplacementCriterion = None) -> None:
        self._camera_id = camera_id
        self._th = thresholds or StoppedVehicleThresholds()
        self._disp = displacement  # None -> the default per-step speed rule
        self._silence_s = self._th.max_gap_seconds if displacement is None else max(self._th.max_gap_seconds, displacement.bridge_gap_s)
        self._tracks: dict = {}  # track_id -> _TrackState (only tracks observed within max_gap)
        self._closed: collections.deque = collections.deque()  # StopEpisode, in closing order
        self._reported_open: set = set()  # episode ids last handed to a window while still open (see `views`)

    # -- feeding -----------------------------------------------------------

    def _threshold(self, space: MotionSpace) -> float:
        return self._th.world_speed_threshold_m_per_s if space == MotionSpace.WORLD else self._th.image_speed_threshold_px_per_s

    def observe(self, track_id: str, space: MotionSpace, t: float, x: float, y: float) -> None:
        """One position sample of one tracklet, in `space` units (px for IMAGE, metres for WORLD)."""
        if self._disp is not None:
            self._observe_displacement(track_id, space, t, x, y)
            return
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
        for track_id in [k for k, s in self._tracks.items() if t - s.last_obs > self._silence_s + _EPS]:
            st = self._tracks.pop(track_id)
            self._close_run(track_id, st, "track_ended")

    def finish(self) -> None:
        """End of stream: close every open run."""
        for track_id in list(self._tracks):
            self._close_run(track_id, self._tracks.pop(track_id), "stream_ended")

    def _make(self, track_id: str, st, end, reason) -> StopEpisode:
        space = st.space
        if self._disp is not None:
            return StopEpisode(
                episode_id=f"{track_id}:stopped_vehicle:{st.run_start}",
                camera_id=self._camera_id, track_id=track_id, motion_space=space,
                speed_basis=METRIC_DISPLACEMENT_BASIS if space == MotionSpace.WORLD else IMAGE_DISPLACEMENT_BASIS,
                speed_threshold=self._disp.radius(space), start=st.run_start, confirmed_at=st.confirmed_at,
                last_low_speed=st.run_last, end=end, closed_reason=reason, dwell_required=self._th.dwell_seconds,
                bridged_s=st.bridged_s, bridged_gaps=st.bridged_gaps, criterion="displacement",
            )
        return StopEpisode(
            episode_id=f"{track_id}:stopped_vehicle:{st.run_start}",
            camera_id=self._camera_id, track_id=track_id, motion_space=space,
            speed_basis=METRIC_BASIS if space == MotionSpace.WORLD else IMAGE_BASIS,
            speed_threshold=self._threshold(space), start=st.run_start, confirmed_at=st.confirmed_at,
            last_low_speed=st.run_last, end=end, closed_reason=reason, dwell_required=self._th.dwell_seconds,
        )

    def _close_run(self, track_id: str, st, reason: str) -> None:
        if st.confirmed_at is not None:
            self._closed.append(self._make(track_id, st, st.run_last, reason))
        if self._disp is not None:
            st.reset_run()
            return
        st.run_start = st.run_last = st.confirmed_at = None

    # -- displacement criterion (opt-in) -----------------------------------

    def _observe_displacement(self, track_id: str, space: MotionSpace, t: float, x: float, y: float) -> None:
        st = self._tracks.get(track_id)
        if st is None:
            st = self._tracks[track_id] = _DispState(space)
            st.start_run(t, x, y)
            st.last_obs = t
            return
        dt = t - st.last_obs
        if dt <= 0:
            return  # duplicate / out-of-order sample
        radius = self._disp.radius(st.space)
        silent = dt > self._th.max_gap_seconds + _EPS
        if silent:
            # A silence. Bridge it only if the run is still a candidate (nothing pending), the silence is
            # no longer than the bridge limit, and the vehicle reappears where it was LAST KNOWN to be
            # (the pre-silence window mean: the new sample must not define the anchor it is judged by).
            ax, ay = st.anchor
            inside = math.hypot(x - ax, y - ay) <= radius
            if st.run_start is not None and not st.pending and inside and dt <= self._disp.bridge_gap_s + _EPS:
                st.bridged_s += dt
                st.bridged_gaps += 1
                self._accept(st, t, x, y, observed=0.0)  # bridged time is NOT observed dwell
            else:
                self._close_run(track_id, st, "gap")
                st.start_run(t, x, y)
            st.last_obs = t
            return
        ax, ay = st.anchor_with(t, x, y, self._disp.anchor_window_s)
        if math.hypot(x - ax, y - ay) <= radius:
            self._accept(st, t, x, y, observed=t - st.run_last)
        else:
            st.pending.append((t, x, y))
            if t - st.run_last >= self._disp.exit_seconds - _EPS:
                # Out of the radius for long enough: the vehicle moved. Close the run at its last in-radius
                # sample, then re-seed a fresh candidate run from the samples that left (they may be a new stop).
                moved = st.pending
                self._close_run(track_id, st, "moved")
                st.start_run(*moved[0])
                for mt, mx, my in moved[1:]:
                    self._replay(st, mt, mx, my)
        st.last_obs = t

    def _accept(self, st: _DispState, t: float, x: float, y: float, observed: float) -> None:
        st.pending = []
        st.push(t, x, y, self._disp.anchor_window_s)
        st.observed_s += observed
        st.run_last = t
        if st.confirmed_at is None and st.observed_s >= self._th.dwell_seconds - _EPS:
            st.confirmed_at = t

    def _replay(self, st: _DispState, t: float, x: float, y: float) -> None:
        """Feed one already-seen sample into a freshly started run (contiguous by construction)."""
        ax, ay = st.anchor_with(t, x, y, self._disp.anchor_window_s)
        if math.hypot(x - ax, y - ay) <= self._disp.radius(st.space):
            self._accept(st, t, x, y, observed=t - st.run_last)
        else:
            st.pending.append((t, x, y))
            if t - st.run_last >= self._disp.exit_seconds - _EPS:
                moved = st.pending
                st.reset_run()
                st.start_run(*moved[0])
                for mt, mx, my in moved[1:]:
                    self._replay(st, mt, mx, my)

    # -- reading -----------------------------------------------------------

    def episodes(self) -> list:
        """Every known episode (closed ones still retained + open ones), as of now."""
        out = list(self._closed)
        out.extend(self._make(tid, st, None, None) for tid, st in self._tracks.items() if st.confirmed_at is not None)
        return out

    def views(self, start: float, end: float) -> tuple:
        """The episodes touching window [start, end) (`episode_views`), plus LATE CLOSURES.

        A tracklet is only known to be gone once it has been silent for `max_gap_seconds`, and its episode `end` is the LAST low-speed
        sample, i.e. earlier. So an episode can be reported open in window k (its track vanished inside k, the silence limit not yet
        reached) and be closed during window k+1 with `end` < start(k+1). `episode_views` drops such an episode ("ended before the
        window began"), and the consumer that raised the incident for it would never learn that it ended (observed on the demo
        footage: stop incidents stuck at "still open" for the rest of the stream). An episode that was last handed out as open and has
        since closed is therefore reported once, as CLOSED, in the next window."""
        views = episode_views(self.episodes(), start, end)
        shown = {v.episode.episode_id for v in views}
        late = [StopEpisodeView(ep, EpisodePhase.CLOSED) for ep in self._closed if ep.episode_id in self._reported_open and ep.episode_id not in shown]
        if late:
            views = tuple(sorted(list(views) + late, key=lambda v: (v.episode.start, v.episode.track_id)))
        self._reported_open = {v.episode.episode_id for v in views if v.episode.is_open}
        return views

    def evict_before(self, t: float) -> int:
        """Forget closed episodes that ended before `t` (no open window can need them)."""
        n = 0
        keep = collections.deque()
        for ep in self._closed:
            if ep.end < t and ep.episode_id not in self._reported_open:  # a late closure still owes its consumer one CLOSED report
                n += 1
            else:
                keep.append(ep)
        self._closed = keep
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


def stop_episodes_from_track_states(camera_id: str, states, calibration_profile=None, thresholds: StoppedVehicleThresholds = None,
                                    displacement: DisplacementCriterion = None) -> list:
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
    return stop_episodes_from_samples(camera_id, samples, thresholds, displacement)


def stop_episodes_from_samples(camera_id: str, samples_by_track: dict, thresholds: StoppedVehicleThresholds = None,
                               displacement: DisplacementCriterion = None) -> list:
    """Post-hoc episodes from complete per-track samples: `{track_id: (space, [(t, x, y), ...])}`.
    Feeds the SAME tracker the live engine uses, so a recording's episodes equal a stream's."""
    tracker = StopEpisodeTracker(camera_id, thresholds, displacement)
    for track_id in sorted(samples_by_track):
        space, samples = samples_by_track[track_id]
        for t, x, y in samples:
            tracker.observe(track_id, space, t, x, y)
    tracker.finish()
    return tracker.episodes()
