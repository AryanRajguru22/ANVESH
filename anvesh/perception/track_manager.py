"""Track management (blueprint Part 3, module 4 -- lifecycle half): per-track
bookkeeping, deliberately independent of *how* track IDs are assigned.

Given a stream of `TrackedDetection`s (from any `Tracker` implementation),
this module owns:
  - trajectory accumulation (image-space (x, y, timestamp) samples)
  - track age (first_seen / last_seen)
  - the visible/partial/lost occlusion state, driven by consecutive missed
    frames -- this is the "handling of temporary missed detections":
    a track that misses a few frames is `partial`, not immediately dropped;
    only exceeding `max_missed_frames` finalizes it as `lost`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anvesh.perception.types import TrackedDetection
from anvesh.storage.schemas import OcclusionState, VehicleClass


@dataclass
class TrackState:
    track_id: str
    vehicle_class: VehicleClass
    first_seen: float
    last_seen: float
    occlusion_state: OcclusionState
    trajectory: list = field(default_factory=list)
    consecutive_missed_frames: int = 0
    is_finished: bool = False

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen


class TrackManager:
    """Maintains per-track state across frames from a TrackedDetection stream."""

    def __init__(self, max_missed_frames: int = 5, partial_after_missed_frames: int = 1) -> None:
        if max_missed_frames < partial_after_missed_frames:
            raise ValueError("max_missed_frames must be >= partial_after_missed_frames")
        if partial_after_missed_frames < 1:
            raise ValueError("partial_after_missed_frames must be >= 1")
        self._max_missed_frames = max_missed_frames
        self._partial_after_missed_frames = partial_after_missed_frames
        self._active: dict = {}
        self._finished: list = []

    @property
    def active_tracks(self) -> dict:
        return dict(self._active)

    @property
    def finished_tracks(self) -> list:
        return list(self._finished)

    def update(self, timestamp: float, tracked_detections: list) -> None:
        seen_ids = set()

        for td in tracked_detections:
            seen_ids.add(td.track_id)
            centroid = td.detection.centroid
            state = self._active.get(td.track_id)
            if state is None:
                state = TrackState(
                    track_id=td.track_id,
                    vehicle_class=td.detection.vehicle_class,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    occlusion_state=OcclusionState.VISIBLE,
                    trajectory=[(centroid[0], centroid[1], timestamp)],
                )
                self._active[td.track_id] = state
            else:
                state.last_seen = timestamp
                state.consecutive_missed_frames = 0
                state.occlusion_state = OcclusionState.VISIBLE
                state.trajectory.append((centroid[0], centroid[1], timestamp))

        for track_id, state in list(self._active.items()):
            if track_id in seen_ids:
                continue
            state.consecutive_missed_frames += 1
            if state.consecutive_missed_frames > self._max_missed_frames:
                self._finalize_track(track_id, state, OcclusionState.LOST)
            elif state.consecutive_missed_frames >= self._partial_after_missed_frames:
                state.occlusion_state = OcclusionState.PARTIAL

    def finalize(self) -> None:
        """Close out any tracks still active at end-of-stream."""
        for track_id, state in list(self._active.items()):
            self._finalize_track(track_id, state, state.occlusion_state)

    def all_tracks(self) -> list:
        return self.finished_tracks + list(self._active.values())

    def _finalize_track(self, track_id: str, state: TrackState, occlusion_state: OcclusionState) -> None:
        state.occlusion_state = occlusion_state
        state.is_finished = True
        self._finished.append(state)
        del self._active[track_id]
