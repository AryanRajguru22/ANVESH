"""The streaming perception engine: ONE implementation of
`frame -> detect+track -> identity -> track store -> sliding windows`, used
for recorded files (batch replay, via `perception.pipeline.run_m1_pipeline`)
and for live/paced streams (via `anvesh.streaming.pipeline`). Because both
call `process_frame`, "batch" and "stream" cannot diverge in logic -- they can
only differ in WHICH frames arrive (drops) and in latency.

ORDER WITHIN `process_frame` (deterministic, and the reason windows never
depend on the trigger frame's own detections):
  1. session check (a new `Frame.epoch` = a reconnect: finalize the old
     identity session, reset the tracker, open a new session);
  2. drop accounting (queue gaps + source-reported gaps);
  3. close every window whose close time <= this frame's stream time, building
     snapshots from the store AS IT WAS BEFORE this frame;
  4. detect + track (the injected `tracker`);
  5. store update (identity assignment) and per-frame callbacks;
  6. bounded-retention eviction.

ARCHIVE MODE (`archive=True`): keeps every per-frame `CameraObservation` and
never trims or evicts -- exactly the legacy batch behaviour, needed because
`M1PipelineResult` returns whole-file observations and tracks. The streaming
and soak runs use `archive=False`, where memory is bounded by construction:
trajectories are trimmed to `history_needed + margin`, finished tracks are
evicted once no open window can reference them, the drop-event and
frame-timestamp deques are trimmed to the oldest open window, and latency uses
a fixed-size reservoir.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import logging
import math
import time
import uuid

from anvesh.perception.schema_conversion import build_camera_observation, motion_space_for
from anvesh.safety.stop_episodes import StopEpisodeTracker
from anvesh.storage.schemas import MotionSpace
from anvesh.perception.track_manager import identified_detections, occlusion_horizon_s
from anvesh.streaming.latency import LatencyRecorder
from anvesh.streaming.store import StreamingTrackStore
from anvesh.streaming.windows import (
    SlidingWindowScheduler,
    WindowAvailability,
    WindowSnapshot,
    WindowSpec,
    frame_completeness,
)

log = logging.getLogger(__name__)

# `run_id` sentinel: derive a run id automatically (the safe default). Pass
# `run_id=None` explicitly to opt OUT and get bare session-scoped ids.
RUN_ID_AUTO = "auto"

DEFAULT_TRACKER_LOST_BUFFER_FRAMES = 30  # Ultralytics bytetrack.yaml `track_buffer`
HISTORY_MARGIN_S = 2.0


class StreamHealth(str, enum.Enum):
    """Liveness of the SOURCE as seen by the engine (M-A.1).

      NO_SIGNAL -- no frame has ever arrived;
      LIVE      -- frames are arriving;
      STALLED   -- no frame for longer than `stalled_after_s`;
      DOWN      -- no frame for longer than `down_after_s`, or the source itself
                   reports it is reconnecting/failed.
    A camera that is DOWN is UNAVAILABLE, which is not the same thing as a road
    that is empty: the windows it spans are marked unavailable, not free-flow."""

    NO_SIGNAL = "no_signal"
    LIVE = "live"
    STALLED = "stalled"
    DOWN = "down"


@dataclasses.dataclass(frozen=True)
class HealthEvent:
    stream_time: float
    previous: StreamHealth
    current: StreamHealth
    reason: str


def _session_of(track_id: str) -> str:
    """'cam:s000-t00012' -> 's000'."""
    local = track_id.split(":", 1)[-1]
    return local.rsplit("-t", 1)[0]


class StreamingPerceptionEngine:
    def __init__(
        self,
        camera_id: str,
        tracker,
        fps: float,
        windows: WindowSpec = None,
        calibration_profile=None,
        recorder: LatencyRecorder = None,
        clock=time.perf_counter,
        session_id: str = None,
        run_id: str = RUN_ID_AUTO,
        min_occlusion_s: float = 0.0,
        archive: bool = False,
        model_id: str = "unknown",
        model_version: str = "unknown",
        on_frame=None,
        on_update=None,
        stalled_after_s: float = 2.0,
        down_after_s: float = 10.0,
        on_health=None,
        stop_thresholds=None,
        stop_displacement=None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if not (0 < stalled_after_s <= down_after_s):
            raise ValueError("need 0 < stalled_after_s <= down_after_s")
        self.camera_id = camera_id
        # IDENTITY SCOPE (M-A.1). Serials restart at 0 in every session and session
        # names restart at s000 in every run, so without a run prefix two runs of
        # the same camera mint identical ids ("cam:s000-t00001") and anything
        # persisted or joined across runs collides. The default is therefore a
        # fresh random run prefix (unique per engine instance); a caller that needs
        # reproducible ids passes `run_id=derive_run_id(<source identity>)`, and a
        # caller that accepts session-only scope passes `run_id=None` explicitly.
        if run_id == RUN_ID_AUTO:
            run_id = uuid.uuid4().hex[:8]
        self._run_id = run_id
        if session_id is None:
            session_id = self._session_name(0)
        self._tracker = tracker
        self._fps = float(fps)
        self._windows = windows
        self._profile = calibration_profile
        self._recorder = recorder if recorder is not None else LatencyRecorder()
        self._clock = clock
        self._archive = archive
        self._model_id = model_id
        self._model_version = model_version
        self._on_frame = on_frame

        buffer_frames = int(getattr(tracker, "lost_buffer_frames", DEFAULT_TRACKER_LOST_BUFFER_FRAMES))
        derived = occlusion_horizon_s(buffer_frames, self._fps)
        if min_occlusion_s and min_occlusion_s < derived:
            log.info(
                "requested occlusion horizon %.3fs is shorter than the tracker's own %d-frame "
                "lost-buffer (%.3fs at %.2f fps); using the tracker's horizon so a track the "
                "tracker can still revive is never finalized early",
                min_occlusion_s, buffer_frames, derived, self._fps,
            )
        self.max_occlusion_s = max(derived, min_occlusion_s or 0.0)

        history_s = None
        if windows is not None and not archive:
            history_s = windows.history_needed_s() + HISTORY_MARGIN_S
        self._store = StreamingTrackStore(
            camera_id,
            session_id=session_id,
            max_occlusion_s=self.max_occlusion_s,
            history_s=history_s,
            on_update=on_update,
        )
        self._scheduler = SlidingWindowScheduler(windows) if windows is not None else None
        # Stop episodes are followed across window boundaries (M-A.1 Fix 5), in the same
        # space the tracks are built in: metres only for validated geometry, else pixels.
        # `stop_displacement` (a `safety.stop_episodes.DisplacementCriterion`) opts in to the jitter- and
        # gap-robust stop rule; None keeps the default per-step speed rule.
        self._stops = StopEpisodeTracker(camera_id, stop_thresholds, stop_displacement)
        self._stop_space = motion_space_for(calibration_profile)

        self._epoch = None
        self._last_index = None
        self._last_ts = None
        self._clock_ts = None  # latest stream time seen, from a frame OR a `tick`
        self._frame_times: collections.deque = collections.deque()  # stream times of frames seen (bounded)
        # (t_from, t_to, n_lost): n_lost frames went missing somewhere between the previous
        # frame (t_from) and this one (t_to) -- attributed to windows by estimated lost-frame
        # time, NOT stamped at t_to (bounded)
        self._drop_events: collections.deque = collections.deque()

        # liveness
        nominal_dt = 1.0 / self._fps
        self._stalled_after_s = max(stalled_after_s, 5.0 * nominal_dt)
        self._down_after_s = max(down_after_s, self._stalled_after_s)
        self._gap_threshold_s = max(self._windows.max_gap_s if self._windows is not None else 1.0, 2.5 * nominal_dt)
        self._on_health = on_health
        self.health = StreamHealth.NO_SIGNAL
        self.health_events: collections.deque = collections.deque(maxlen=256)

        # observability counters
        self.frames_processed = 0
        self.frames_dropped = 0  # queue drops + source gaps
        self.windows_emitted = 0
        self.sessions_started = 1
        self.camera_observations: list = []  # archive mode only

    # -- accessors ---------------------------------------------------------

    @property
    def store(self) -> StreamingTrackStore:
        return self._store

    @property
    def recorder(self) -> LatencyRecorder:
        return self._recorder

    @property
    def last_timestamp(self):
        return self._last_ts

    @property
    def stream_time(self):
        """Latest stream time the engine knows about (frame or tick)."""
        return self._clock_ts

    @property
    def silence_threshold_s(self) -> float:
        """How long the stream may be silent before a tick acts (closes windows, ages tracks). Idle
        polling needs to run at least this often for a silence to be noticed promptly."""
        return self._gap_threshold_s

    # -- main entry --------------------------------------------------------

    def process_frame(self, frame) -> list:
        t_frame = self._clock()

        # 1) session / epoch
        if self._epoch is None:
            self._epoch = frame.epoch
        elif frame.epoch != self._epoch:
            self._start_new_session(frame.epoch)

        # 2) drop accounting: the lost frames belong to the interval BETWEEN the previous
        # frame and this one, so each window gets the share that really fell inside it
        lost = int(frame.gap_before or 0)
        if self._last_index is not None:
            lost += max(0, frame.index - self._last_index - 1)
        if lost:
            self.frames_dropped += lost
            t_prev = self._last_ts if self._last_ts is not None else frame.timestamp
            self._drop_events.append((min(t_prev, frame.timestamp), frame.timestamp, lost))
        self._note_frame_for_health(frame.timestamp)
        self._clock_ts = frame.timestamp if self._clock_ts is None else max(self._clock_ts, frame.timestamp)

        # 3) close due windows from the pre-frame store state
        snapshots = []
        if self._scheduler is not None:
            t0 = self._clock()
            for k in self._scheduler.due(frame.timestamp):
                snapshots.append(self._build_snapshot(k, frame.timestamp, trigger_wall=frame.capture_wall_s))
            if snapshots:
                self._recorder.record("windowing", self._clock() - t0)

        # 4) detect + track
        t0 = self._clock()
        raw_detections, tracked = self._tracker.update(frame.image, frame.index, frame.timestamp)
        self._recorder.record("detect_track", self._clock() - t0)

        # 5) store update + callbacks
        t0 = self._clock()
        assignments, _updates = self._store.update(frame.index, frame.timestamp, tracked)
        identified = identified_detections(assignments)
        self._recorder.record("store_update", self._clock() - t0)
        self._observe_stops(frame.timestamp, assignments)
        if self._on_frame is not None:
            self._on_frame(frame.index, frame.timestamp, identified)
        if self._archive:
            self.camera_observations.append(
                build_camera_observation(
                    camera_id=self.camera_id, frame_index=frame.index, timestamp=frame.timestamp,
                    detections=raw_detections, model_id=self._model_id, model_version=self._model_version,
                )
            )

        # 6) bounded retention
        self._frame_times.append(frame.timestamp)
        self._retain()

        self.frames_processed += 1
        self._last_index = frame.index
        self._last_ts = frame.timestamp
        now = self._clock()
        self._recorder.record("frame_processing", now - t_frame)
        if frame.capture_wall_s is not None:
            self._recorder.record("frame_end_to_end", now - frame.capture_wall_s)
        if frame.decode_s is not None:
            self._recorder.record("decode", frame.decode_s)
        if snapshots and frame.capture_wall_s is not None:
            self._recorder.record("window_emit_after_trigger_frame", now - frame.capture_wall_s)
        self.windows_emitted += len(snapshots)
        return snapshots

    def tick(self, stream_time: float, source_state: str = None) -> list:
        """Advance the engine's clock WITHOUT a frame (M-A.1 liveness).

        A dead or silent camera delivers no frames, so a frame-driven engine would
        never notice: no window would close, no health change would be reported,
        and the world model would sit on its last picture. The owner of the source
        calls this periodically (the threaded pipeline does, on queue timeouts) with
        the current stream time and, if known, the source's own state
        ('reconnecting', 'failed', ...). It (1) updates `health`, (2) closes every
        window whose time has passed -- they are built from the frames that DID
        arrive, so a silent camera yields UNAVAILABLE windows, not empty free-flow
        ones -- and (3) ages out tracks that can no longer be observed. Returns the
        windows closed. Harmless for a non-advancing clock.

        A tick is meaningful ONLY once the source has been silent for longer than a
        silence threshold (or reports itself down). The pipeline polls whenever its queue is
        empty, which in perfectly normal live flow is most of the time (the worker is faster
        than the camera): acting on those ticks would age tracks to PARTIAL between frames
        and close a window before a frame decoded a moment ago has been delivered. Health is
        still evaluated on every tick; windows, tracks and stops are touched only for a
        genuine silence."""
        stream_time = stream_time if self._clock_ts is None else max(stream_time, self._clock_ts)
        self._note_silence_for_health(stream_time, source_state)
        if self._last_ts is None:
            self._clock_ts = stream_time
            return []  # never saw a frame: there is no stream origin to window against
        source_down = source_state in ("reconnecting", "failed")
        if stream_time - self._last_ts <= self._gap_threshold_s and not source_down:
            return []  # ordinary inter-frame idle time, not an outage: change nothing
        self._clock_ts = stream_time
        snapshots = []
        if self._scheduler is not None:
            t0 = self._clock()
            for k in self._scheduler.due(stream_time):
                snapshots.append(self._build_snapshot(k, stream_time))
            if snapshots:
                self._recorder.record("windowing", self._clock() - t0)
        self._store.update(self._last_index if self._last_index is not None else 0, stream_time, [])
        self._stops.advance(stream_time)
        self._retain()
        self.windows_emitted += len(snapshots)
        return snapshots

    def finish(self, duration: float = None) -> list:
        """End of stream: finalize identities and close every remaining window
        that starts before `duration` (default: last frame time + one frame
        interval). The final window is clipped to `duration`."""
        self._store.finish()
        self._stops.finish()
        snapshots = []
        if self._scheduler is not None and self._last_ts is not None:
            if duration is None:
                duration = self._last_ts + 1.0 / self._fps
            else:
                duration = max(duration, self._last_ts + 1.0 / self._fps)
            t0 = self._clock()
            for k in self._scheduler.remaining_until(duration):
                snapshots.append(self._build_snapshot(k, duration, duration=duration))
            if snapshots:
                self._recorder.record("windowing", self._clock() - t0)
            self.windows_emitted += len(snapshots)
        return snapshots

    # -- internals ---------------------------------------------------------

    def _session_name(self, epoch: int) -> str:
        """`s<epoch>` within a run; `<run_id>-s<epoch>` when a run id (see
        `track_manager.derive_run_id`) makes identities unique across runs."""
        base = f"s{epoch:03d}"
        return base if self._run_id is None else f"{self._run_id}-{base}"

    def _unused_session_name(self, epoch: int) -> str:
        """The session name for `epoch`, made unique within this engine: epochs are
        opaque connection numbers (a recreated source may restart at 0, a flapping
        one may repeat), but an identity session is NEVER re-entered, so a repeated
        epoch gets `<name>e<k>` (k = 1, 2, ...). Monotonic epochs keep the plain
        name, so ordinary ids are unchanged."""
        used = self._store.manager.used_sessions
        name = self._session_name(epoch)
        k = 0
        while name in used:
            k += 1
            name = f"{self._session_name(epoch)}e{k}"
        return name

    def _start_new_session(self, new_epoch: int) -> None:
        # Close due windows for the old session happens naturally on the next
        # frame; here we only fence identity: old tracks are finalized and the
        # tracker's own state (raw IDs) is reset.
        session_id = self._unused_session_name(new_epoch)
        self._store.begin_session(session_id)
        reset = getattr(self._tracker, "reset", None)
        if reset is not None:
            reset()
        self._epoch = new_epoch
        self._last_index = None
        self.sessions_started += 1
        log.info("camera %s: new identity session %s (epoch %d)", self.camera_id, session_id, new_epoch)

    # -- retention / health / observability (M-A.1) ---------------------------

    def _retain(self) -> None:
        if self._archive:
            return
        if self._scheduler is None:
            self._frame_times.clear()
            self._drop_events.clear()
            self._stops.evict_before(float("inf"))  # nobody reads episodes without windows
            return
        floor = self._scheduler.oldest_open_start()
        self._store.evict_before(floor)
        self._stops.evict_before(floor)
        # keep ONE frame time before the floor: it is the predecessor that lets the
        # oldest open window measure its leading silence
        while len(self._frame_times) > 1 and self._frame_times[1] < floor:
            self._frame_times.popleft()
        while self._drop_events and self._drop_events[0][1] < floor:
            self._drop_events.popleft()

    def _observe_stops(self, stream_time: float, assignments: list) -> None:
        for a in assignments:
            det = a.tracked.detection
            if self._stop_space == MotionSpace.WORLD:
                try:
                    x, y = self._profile.image_to_world(det.ground_point)
                except ValueError:
                    continue  # no metric position for this sample: it is not a step
            else:
                x, y = det.centroid
            self._stops.observe(f"{self.camera_id}:{a.track_id}", self._stop_space, stream_time, x, y)
        self._stops.advance(stream_time)

    def _set_health(self, state: StreamHealth, stream_time: float, reason: str) -> None:
        if state == self.health:
            return
        event = HealthEvent(stream_time, self.health, state, reason)
        self.health = state
        self.health_events.append(event)
        log.info("camera %s: source health %s -> %s at stream t=%.2fs (%s)",
                 self.camera_id, event.previous.value, state.value, stream_time, reason)
        if self._on_health is not None:
            self._on_health(event)

    def _note_frame_for_health(self, stream_time: float) -> None:
        if self.health == StreamHealth.LIVE:
            return
        if self._last_ts is None:
            self._set_health(StreamHealth.LIVE, stream_time, "first frame")
        else:
            self._set_health(StreamHealth.LIVE, stream_time, f"frame after {stream_time - self._last_ts:.2f}s of silence")

    def _note_silence_for_health(self, stream_time: float, source_state: str = None) -> None:
        if source_state in ("reconnecting", "failed"):
            self._set_health(StreamHealth.DOWN, stream_time, f"source reports '{source_state}'")
            return
        if self._last_ts is None:
            return  # NO_SIGNAL stays NO_SIGNAL until a frame arrives
        silence = stream_time - self._last_ts
        if silence > self._down_after_s:
            self._set_health(StreamHealth.DOWN, stream_time, f"no frame for {silence:.1f}s")
        elif silence > self._stalled_after_s and self.health == StreamHealth.LIVE:
            self._set_health(StreamHealth.STALLED, stream_time, f"no frame for {silence:.1f}s")

    def _lost_frames_in(self, start: float, end: float) -> int:
        """Lost frames whose ESTIMATED capture time falls in [start, end): an event of
        `n` frames lost between `t_from` and `t_to` is spread evenly over that interval."""
        total = 0
        for t_from, t_to, n in self._drop_events:
            if t_to <= t_from:
                total += n if start <= t_to < end else 0
                continue
            step = (t_to - t_from) / (n + 1)
            lo = max(1, math.ceil((start - t_from) / step - 1e-9))
            hi = min(n, math.ceil((end - t_from) / step - 1e-9) - 1)
            total += max(0, hi - lo + 1)
        return total

    def _silent_seconds(self, start: float, end: float) -> float:
        """Seconds of [start, end) lying inside a silence (a gap between consecutive
        frames longer than the gap threshold). A window with no frame at all is
        silent for its whole length."""
        times = list(self._frame_times)
        inside = [t for t in times if start <= t < end]
        if not inside:
            return end - start
        before = [t for t in times if t < start]
        points = [before[-1] if before else start] + inside + [end]
        silent = 0.0
        for a, b in zip(points, points[1:]):
            a, b = max(a, start), min(b, end)
            if b - a > self._gap_threshold_s:
                silent += b - a
        return silent

    def _build_snapshot(self, k: int, stream_time: float, duration: float = None, trigger_wall: float = None) -> WindowSnapshot:
        spec = self._windows
        start, end = spec.bounds(k)
        final_partial = False
        if duration is not None and end > duration:
            end = duration
            final_partial = True
        tracks = tuple(self._store.snapshot(start, end, self._profile))
        frames_observed = sum(1 for t in self._frame_times if start <= t < end)
        dropped = self._lost_frames_in(start, end)
        length = max(end - start, 1e-9)
        silent = min(self._silent_seconds(start, end), length)
        coverage = max(0.0, 1.0 - silent / length)
        if coverage < spec.min_coverage:
            availability = WindowAvailability.UNAVAILABLE
        elif silent > 0.0 or dropped > 0:
            availability = WindowAvailability.DEGRADED
        else:
            availability = WindowAvailability.OBSERVED
        sessions = tuple(sorted({_session_of(t.track_id) for t in tracks}))
        expected_frames = int(round(self._fps * length))
        return WindowSnapshot(
            window_index=k,
            window_id=f"{self.camera_id}:w{k:06d}",
            camera_id=self.camera_id,
            start=start,
            end=end,
            tracks=tracks,
            frames_observed=frames_observed,
            dropped_frames=dropped,
            sessions=sessions,
            closed_at_stream_time=stream_time,
            final_partial=final_partial,
            trigger_capture_wall_s=trigger_wall,
            expected_frames=expected_frames,
            silent_s=silent,
            coverage=coverage,
            frame_completeness=frame_completeness(frames_observed, expected_frames),
            availability=availability,
            stop_episodes=self._stops.views(start, end),
        )

    def stats(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "windows_emitted": self.windows_emitted,
            "sessions_started": self.sessions_started,
            "health": self.health.value,
            "health_transitions": len(self.health_events),
            "run_id": self._run_id,
            "identity_scope": "run" if self._run_id is not None else "session",
            "tracks_created": self._store.manager.total_created,
            "tracks_retained": self._store.retained_track_count(),
            "max_occlusion_s": self.max_occlusion_s,
            "same_frame_raw_id_collisions": self._store.manager.same_frame_raw_id_collisions,
            "stale_raw_id_restarts": self._store.manager.stale_raw_id_restarts,
        }
