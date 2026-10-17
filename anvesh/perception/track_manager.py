"""Track management (blueprint Part 3, module 4 -- lifecycle half): per-track
bookkeeping, deliberately independent of *how* the tracker assigns its IDs.

Given a stream of `TrackedDetection`s (from any `Tracker` implementation),
this module owns:
  - VEHICLE IDENTITY (M-A, ANVESH_PRODUCT_GAP_AUDIT.md finding 4);
  - trajectory accumulation (image-space centroid and ground-point samples);
  - track age (first_seen / last_seen);
  - the visible/partial/lost occlusion state, driven by elapsed TIME since a
    track was last observed.

IDENTITY MODEL
--------------
A raw tracker ID (ByteTrack's `track_id`) is NOT a vehicle identity. It is a
per-run small integer from a class-global counter (`STrack._count`), it is
re-issued from 1 after any tracker reset, and a lost ID can be *revived* by
the tracker after this module has already given up on it. Joining downstream
data on it is therefore unsafe.

Instead every `TrackState` carries a manager-assigned, deterministic identity
key `(camera_id, session_id, serial)`:
  - `session_id` names one uninterrupted tracker run on one camera (a file
    replay is a single session; a stream reconnect or a tracker `reset()`
    starts a new one -- the owner constructs a new manager or calls
    `begin_session`). THIS CLASS never derives it from a clock or RNG: it is
    supplied by the owner, so identical input + identical session ids reproduce
    identical identities. (M-A.1: the owner that matters, `StreamingPerceptionEngine`,
    DOES by default prefix its session ids with a random per-instance run id so that
    two runs can never collide -- so live/streaming ids are NOT reproducible across
    runs unless the caller passes `run_id=derive_run_id(<source identity>)`;
    `run_m1_pipeline` defaults to such a deterministic per-recording run id.)
  - `serial` is a monotonically increasing per-session counter assigned at the
    moment a new `TrackState` is created. It is never reused.
  - `TrackState.track_id` is the printable local form `"<session_id>-t<serial:05d>"`
    and `identity_key` the full tuple form. `schema_conversion` prefixes the
    camera, giving the existing one-colon `"<camera_id>:<session_id>-t<serial>"`
    shape the dashboard already parses.
  - The raw tracker ID is kept on the state as `raw_tracker_id` for audit only.

IDENTITY SCOPE -- WHAT AN IDENTITY IS AND IS NOT (M-A.1)
---------------------------------------------------------
An identity is a TRACKLET: one uninterrupted run of the tracker following one
box-stream, valid only inside its `(camera_id, session_id)` and, when the owner
supplies a `run_id` prefix, unique across runs. Two different identities may be
the same physical vehicle (the tracker lost it and found it again, the camera
reconnected, the vehicle drove into another camera); this module does NOT
re-identify, merge or stitch tracklets, and nothing downstream may read
"same identity" as "same vehicle across sessions or cameras" nor "different
identity" as "different vehicle". A session id is NEVER reused by one manager
(`begin_session` refuses it), so `(camera, session, serial)` cannot collide
inside a manager lifetime; cross-RUN uniqueness is the owner's `run_id` prefix
(the streaming engine applies one by default -- see `StreamingPerceptionEngine`).

INVARIANT (enforced structurally, regression-tested): within one manager there
are never two distinct `TrackState` objects (live or finalized) with the same
identity key. A raw ID is mapped to its *currently live* state only; once that
state is finalized the mapping is deleted, so a returning raw ID can only ever
open a NEW state with a NEW serial -- it cannot resurrect or collide with the
old one. The same applies if the raw ID returns after an excessive time gap
while its state was never finalized (frames dropped / no updates in between):
the stale state is finalized first.

LIFECYCLE IS TIME-BASED
-----------------------
`max_occlusion_s` / `partial_after_s` are seconds, not frame counts: a frame
count means 0.2 s at 25 fps but 0.5 s at 10 fps. ByteTrack itself keeps a lost
track alive for `track_buffer` FRAMES (Ultralytics' `bytetrack.yaml`, default
30, not rescaled by fps) and may re-emit its ID inside that window. To avoid
this manager finalizing a track the tracker would still revive,
`occlusion_horizon_s()` converts the tracker's buffer to seconds at the real
source fps; the pipeline passes `max(requested, horizon)`.

BOUNDED MEMORY (streaming)
--------------------------
`history_s` (default None = keep everything, the batch/archive behaviour)
trims each track's trajectory lists to the most recent `history_s` seconds,
while `sample_count` / `first_seen` keep the whole-lifetime facts.
`evict_finished_before(t)` drops finalized tracks that ended before `t`.
Both are used by `anvesh.streaming`; with defaults the behaviour is unchanged
from M1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from anvesh.perception.types import TrackedDetection
from anvesh.storage.schemas import OcclusionState, VehicleClass

DEFAULT_MAX_OCCLUSION_S = 3.0
DEFAULT_PARTIAL_AFTER_S = 0.0  # PARTIAL as soon as a track is not observed in an update
_TIME_EPS = 1e-9


def derive_run_id(*parts) -> str:
    """Deterministic 8-hex-char run id from caller-supplied identity parts
    (e.g. source path/URL, file size, camera id, run start time for a live
    stream). Prefixing session ids with it (`StreamingPerceptionEngine(run_id=)`)
    makes identity keys unique across RUNS of the same camera, not just within
    one run -- without it every run starts again at `s000-t00000`. It contains
    only [0-9a-f], so it can never look like a '-t<serial>' suffix or a ':'."""
    import hashlib

    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8]


def occlusion_horizon_s(tracker_lost_buffer_frames: int, fps: float, margin_frames: float = 1.5) -> float:
    """Seconds a track must be kept alive so this manager never finalizes a
    track that a tracker with a `tracker_lost_buffer_frames` buffer could
    still revive (`+ margin_frames` for the off-by-one between "frames since
    last seen" and timestamp differences)."""
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return (tracker_lost_buffer_frames + margin_frames) / fps


@dataclass
class TrackState:
    track_id: str  # session-scoped identity ("<session_id>-t<serial:05d>"); NOT the raw tracker ID
    vehicle_class: VehicleClass
    first_seen: float
    last_seen: float
    occlusion_state: OcclusionState
    trajectory: list = field(default_factory=list)
    ground_point_trajectory: list = field(default_factory=list)
    consecutive_missed_frames: int = 0
    is_finished: bool = False
    raw_tracker_id: str = None  # the tracker's own ID, retained for audit only
    camera_id: str = ""
    session_id: str = ""
    serial: int = -1
    sample_count: int = 0  # whole-lifetime observation count (trajectories may be trimmed)

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def identity_key(self) -> tuple:
        return (self.camera_id, self.session_id, self.serial)


class TrackAssignment(NamedTuple):
    """Which identity a detection of the current update belongs to."""

    track_id: str
    tracked: TrackedDetection
    state: TrackState


def identified_detections(assignments: list) -> list:
    """The frame's `TrackedDetection`s re-labelled with their manager-assigned
    identity (raw tracker IDs replaced). This is what every downstream
    consumer (overlay, per-box joins, exports) must use -- never the raw ID."""
    return [
        TrackedDetection(detection=a.tracked.detection, track_id=a.track_id) for a in assignments
    ]


class TrackManager:
    """Maintains per-track state across frames from a TrackedDetection stream."""

    def __init__(
        self,
        max_occlusion_s: float = DEFAULT_MAX_OCCLUSION_S,
        partial_after_s: float = DEFAULT_PARTIAL_AFTER_S,
        camera_id: str = "cam",
        session_id: str = "s000",
        history_s: float = None,
    ) -> None:
        if max_occlusion_s < partial_after_s:
            raise ValueError("max_occlusion_s must be >= partial_after_s")
        if partial_after_s < 0:
            raise ValueError("partial_after_s must be >= 0")
        if history_s is not None and history_s <= 0:
            raise ValueError("history_s must be > 0 when given")
        if not camera_id or not session_id:
            raise ValueError("camera_id and session_id must be non-empty")
        if ":" in session_id:
            raise ValueError("session_id must not contain ':' (track ids keep exactly one colon: camera:local-id)")
        self._max_occlusion_s = max_occlusion_s
        self._partial_after_s = partial_after_s
        self._camera_id = camera_id
        self._session_id = session_id
        self._used_sessions: set = {session_id}  # every session id this manager has ever used
        self._history_s = history_s
        self._next_serial = 0
        self._active: dict = {}  # track_id (identity) -> TrackState
        self._live_by_raw: dict = {}  # raw tracker ID -> live TrackState
        self._finished: list = []
        self.total_created = 0
        self.same_frame_raw_id_collisions = 0  # tracker emitted one raw ID twice in a frame (diagnostic)
        self.stale_raw_id_restarts = 0  # raw ID returned after > max_occlusion_s with state never finalized

    # -- read access -------------------------------------------------------

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def max_occlusion_s(self) -> float:
        return self._max_occlusion_s

    @property
    def used_sessions(self) -> frozenset:
        """Every session id this manager has used (current one included)."""
        return frozenset(self._used_sessions)

    @property
    def active_tracks(self) -> dict:
        """identity track_id -> live TrackState."""
        return dict(self._active)

    @property
    def finished_tracks(self) -> list:
        return list(self._finished)

    def all_tracks(self) -> list:
        return self.finished_tracks + list(self._active.values())

    def live_state_for_raw_id(self, raw_tracker_id: str):
        return self._live_by_raw.get(str(raw_tracker_id))

    # -- session control ---------------------------------------------------

    def begin_session(self, session_id: str) -> None:
        """Start a new identity namespace (e.g. after a stream reconnect or a
        tracker reset). All live tracks are finalized first -- raw IDs of the
        new tracker run are unrelated to the old run's -- and serials restart
        under the NEW session id, so keys cannot collide with the old
        session's."""
        if ":" in session_id or not session_id:
            raise ValueError("session_id must be non-empty and must not contain ':'")
        if session_id in self._used_sessions:
            # Serials restart at 0 in every session, so re-entering an earlier
            # session id (e.g. a reconnect epoch sequence 0 -> 1 -> 0) would
            # mint (camera, session, serial) keys that already exist.
            raise ValueError(
                f"session_id {session_id!r} was already used by this manager; a session id is never reused"
            )
        self.finalize()
        self._session_id = session_id
        self._used_sessions.add(session_id)
        self._next_serial = 0

    # -- update ------------------------------------------------------------

    def update(self, timestamp: float, tracked_detections: list) -> list:
        """Advance one frame/update. Returns the identity assignment of every
        tracked detection in this update, in input order."""
        seen_states: set = set()  # ids of states observed this update
        assignments: list = []

        for td in tracked_detections:
            raw = td.track_id
            state = self._live_by_raw.get(raw)

            if state is not None and (timestamp - state.last_seen) > self._max_occlusion_s + _TIME_EPS:
                # Raw ID came back after more time than this manager keeps a
                # track alive, without an intervening update having
                # finalized it (frames dropped / long stall). A different
                # vehicle may now carry this ID: close the old state, start a
                # new identity.
                self.stale_raw_id_restarts += 1
                self._finalize_track(state, OcclusionState.LOST)
                state = None

            if state is not None and id(state) in seen_states:
                # The tracker emitted the same raw ID twice in one update. Two
                # boxes cannot be one vehicle: the second gets its own identity.
                self.same_frame_raw_id_collisions += 1
                state = self._open_state(td, timestamp, register_raw=False)
            elif state is None:
                state = self._open_state(td, timestamp, register_raw=True)
            else:
                self._append_sample(state, td, timestamp)
                state.last_seen = timestamp
                state.consecutive_missed_frames = 0
                state.occlusion_state = OcclusionState.VISIBLE

            seen_states.add(id(state))
            assignments.append(TrackAssignment(track_id=state.track_id, tracked=td, state=state))

        for state in list(self._active.values()):
            if id(state) in seen_states:
                continue
            state.consecutive_missed_frames += 1
            gap = timestamp - state.last_seen
            if gap > self._max_occlusion_s + _TIME_EPS:
                self._finalize_track(state, OcclusionState.LOST)
            elif gap > self._partial_after_s + _TIME_EPS:
                state.occlusion_state = OcclusionState.PARTIAL

        return assignments

    def finalize(self) -> None:
        """Close out any tracks still active at end-of-stream."""
        for state in list(self._active.values()):
            self._finalize_track(state, state.occlusion_state)

    def evict_finished_before(self, t: float) -> int:
        """Drop finalized tracks whose `last_seen < t`; returns how many.
        Streaming callers use this to bound memory once no open window can
        still reference them."""
        keep = [s for s in self._finished if s.last_seen >= t]
        dropped = len(self._finished) - len(keep)
        self._finished = keep
        return dropped

    # -- internals ---------------------------------------------------------

    def _open_state(self, td: TrackedDetection, timestamp: float, register_raw: bool) -> TrackState:
        serial = self._next_serial
        self._next_serial += 1
        track_id = f"{self._session_id}-t{serial:05d}"
        centroid = td.detection.centroid
        ground_point = td.detection.ground_point
        state = TrackState(
            track_id=track_id,
            vehicle_class=td.detection.vehicle_class,
            first_seen=timestamp,
            last_seen=timestamp,
            occlusion_state=OcclusionState.VISIBLE,
            trajectory=[(centroid[0], centroid[1], timestamp)],
            ground_point_trajectory=[(ground_point[0], ground_point[1], timestamp)],
            raw_tracker_id=td.track_id,
            camera_id=self._camera_id,
            session_id=self._session_id,
            serial=serial,
            sample_count=1,
        )
        assert track_id not in self._active  # structural: serials are never reused
        self._active[track_id] = state
        if register_raw:
            self._live_by_raw[td.track_id] = state
        self.total_created += 1
        return state

    def _append_sample(self, state: TrackState, td: TrackedDetection, timestamp: float) -> None:
        c = td.detection.centroid
        g = td.detection.ground_point
        state.trajectory.append((c[0], c[1], timestamp))
        state.ground_point_trajectory.append((g[0], g[1], timestamp))
        state.sample_count += 1
        if self._history_s is not None:
            cutoff = timestamp - self._history_s
            for traj in (state.trajectory, state.ground_point_trajectory):
                drop = 0
                while drop < len(traj) - 1 and traj[drop][2] < cutoff:
                    drop += 1
                if drop:
                    del traj[:drop]

    def _finalize_track(self, state: TrackState, occlusion_state: OcclusionState) -> None:
        state.occlusion_state = occlusion_state
        state.is_finished = True
        self._finished.append(state)
        del self._active[state.track_id]
        if self._live_by_raw.get(state.raw_tracker_id) is state:
            del self._live_by_raw[state.raw_tracker_id]
