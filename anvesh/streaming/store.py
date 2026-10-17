"""Streaming track store: per-update vehicle state, maintained incrementally
instead of being constructed all at once at end-of-file.

The store wraps `TrackManager` (the identity/lifecycle owner) and adds:
  - per-frame `TrackUpdate` records (one per tracked detection, carrying the
    manager-assigned identity), delivered to an optional `on_update` hook;
  - window views: `snapshot(start, end, calibration_profile)` builds
    `VehicleTrack`s from only the samples inside `[start, end)` (see
    `clip_track_state`), so a window sees what happened IN that window, not a
    lifetime average (ANVESH_PRODUCT_GAP_AUDIT.md SS9.3);
  - bounded retention: trajectories are trimmed to `history_s` and finished
    tracks are evicted once no open window can reference them.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.perception.schema_conversion import build_vehicle_track
from anvesh.perception.track_manager import TrackManager, TrackState
from anvesh.storage.schemas import OcclusionState, VehicleClass


@dataclass(frozen=True)
class TrackUpdate:
    """One observation of one vehicle identity in one frame."""

    track_id: str  # session-scoped identity, e.g. "s000-t00012"
    raw_tracker_id: str  # the tracker's own ID (audit only)
    frame_index: int
    timestamp: float
    bbox_xyxy: tuple
    centroid: tuple
    ground_point: tuple
    vehicle_class: VehicleClass
    confidence: float
    occlusion_state: OcclusionState
    is_new: bool  # first sample of a new identity


def clip_track_state(state: TrackState, start: float, end: float):
    """The part of `state` observed within `[start, end)`, as a new
    TrackState, or None if it has no sample there. `first_seen`/`last_seen`
    are the first/last in-window sample times; `occlusion_state` is the
    state's current value (as of the moment of the snapshot)."""
    traj = [p for p in state.trajectory if start <= p[2] < end]
    if not traj:
        return None
    ground = [p for p in state.ground_point_trajectory if start <= p[2] < end]
    return TrackState(
        track_id=state.track_id,
        vehicle_class=state.vehicle_class,
        first_seen=traj[0][2],
        last_seen=traj[-1][2],
        occlusion_state=state.occlusion_state,
        trajectory=traj,
        ground_point_trajectory=ground,
        consecutive_missed_frames=state.consecutive_missed_frames,
        is_finished=state.is_finished,
        raw_tracker_id=state.raw_tracker_id,
        camera_id=state.camera_id,
        session_id=state.session_id,
        serial=state.serial,
        sample_count=len(traj),
    )


def build_window_tracks(states, start: float, end: float, camera_id: str, calibration_profile=None) -> list:
    """`VehicleTrack`s for every state with at least one sample in
    `[start, end)`, window-clipped (see `clip_track_state`) and ordered by
    (first in-window sample time, track id) -- deterministic. The ONE
    window-view builder: the streaming store and the recorded-file export both
    call it, so a window means the same thing on a live stream and a replay.
    (A batch caller passes FINAL states, so their `occlusion_state` is the
    track's eventual fate; a streaming snapshot sees the state as of window
    close -- the one documented difference between the two modes.)"""
    clipped = []
    for state in states:
        c = clip_track_state(state, start, end)
        if c is not None:
            clipped.append(c)
    clipped.sort(key=lambda s: (s.first_seen, s.track_id))
    return [build_vehicle_track(camera_id, c, calibration_profile=calibration_profile) for c in clipped]


class StreamingTrackStore:
    def __init__(
        self,
        camera_id: str,
        session_id: str = "s000",
        max_occlusion_s: float = 3.0,
        history_s: float = None,
        on_update=None,
    ) -> None:
        self._camera_id = camera_id
        self._manager = TrackManager(
            max_occlusion_s=max_occlusion_s,
            camera_id=camera_id,
            session_id=session_id,
            history_s=history_s,
        )
        self._on_update = on_update
        self.updates_emitted = 0

    @property
    def manager(self) -> TrackManager:
        return self._manager

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def update(self, frame_index: int, timestamp: float, tracked_detections: list) -> tuple:
        """Advance the store one frame. Returns `(assignments, updates)` --
        the manager's identity assignments (input order) and one
        `TrackUpdate` per assignment."""
        assignments = self._manager.update(timestamp, tracked_detections)
        updates = []
        for a in assignments:
            det = a.tracked.detection
            updates.append(
                TrackUpdate(
                    track_id=a.track_id,
                    raw_tracker_id=a.tracked.track_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    bbox_xyxy=det.bbox_xyxy,
                    centroid=det.centroid,
                    ground_point=det.ground_point,
                    vehicle_class=det.vehicle_class,
                    confidence=det.confidence,
                    occlusion_state=a.state.occlusion_state,
                    is_new=a.state.sample_count == 1,
                )
            )
        self.updates_emitted += len(updates)
        if self._on_update is not None:
            for u in updates:
                self._on_update(u)
        return assignments, updates

    def begin_session(self, session_id: str) -> None:
        self._manager.begin_session(session_id)

    def finish(self) -> None:
        self._manager.finalize()

    def snapshot(self, start: float, end: float, calibration_profile=None) -> list:
        """`VehicleTrack`s for every identity with at least one sample in
        `[start, end)`, ordered by (first sample time, track id) so the output
        is deterministic."""
        return build_window_tracks(self._manager.all_tracks(), start, end, self._camera_id, calibration_profile)

    def evict_before(self, t: float) -> int:
        return self._manager.evict_finished_before(t)

    def retained_track_count(self) -> int:
        return len(self._manager.all_tracks())
