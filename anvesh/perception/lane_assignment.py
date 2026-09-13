"""Per-vehicle lane assignment (see `ANVESH_LANE_SPEED_FORENSIC_AUDIT.md`
SS3.6/SS8.1/SS11 Commit 4): wires Commit 1's `RawDetection.ground_point`,
Commit 2's deterministic `world.lane_geometry.assign_lane`, and Commit 3's
per-camera lane configs together against real perception/tracking objects.

Deliberately a thin, additive layer:
  - Uses `RawDetection.ground_point` (bottom-center), never `.centroid`,
    per design requirement A and the Commit 1 rationale (a box-center point
    sits above the road plane for any vehicle taller than it is wide in
    screen space -- exactly wrong for a polygon test against lane paint on
    the road surface).
  - Never mutates `RawDetection`, `TrackedDetection`, `TrackState`, or the
    frozen `storage.schemas.VehicleTrack` -- lane assignment is a derived,
    non-schema companion result here, following the same pattern already
    established by `EvidenceItem`, `StoppedVehicleClassification`, and
    `CorridorStateAssembly` (the audit SS9 explicitly recommends against
    touching frozen Part-5 schemas for exactly this kind of diagnostic).
  - Computed PER OBSERVATION (per raw detection, at its own timestamp),
    never once-and-cached for a track's whole lifetime: `TrackState.
    ground_point_trajectory` (Commit 4 addition, `track_manager.py`) holds
    one ground point per frame, and `build_track_lane_history` assigns
    each sample independently. A vehicle that changes lanes mid-track
    shows that transition in its history; nothing here carries a
    first-seen lane forward as if it still applied later.

This module performs no lane-CHANGE detection (no comparison across
samples -- that is future work, audit SS11 Commit 5+), no speed
computation, and no candidate-vehicle ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.world.lane_geometry import LaneAssignmentResult, assign_lane


def assign_detection_lane(detection: RawDetection, regions: tuple) -> LaneAssignmentResult:
    """Assign one detection to a lane using its ground-contact point.

    `regions` is typically a camera's `LaneCameraConfig.regions`
    (`anvesh.world.lane_config.load_lane_config`). Pure and stateless: the
    same `(detection, regions)` pair always produces the same result
    (Commit 2's own determinism guarantee carries through unchanged), and
    calling this repeatedly never accumulates or caches anything -- there
    is nothing here for a caller to keep in sync as a track moves.
    """
    return assign_lane(detection.ground_point, regions)


@dataclass(frozen=True)
class LaneAssignmentSample:
    """One observation's lane assignment, at its own timestamp."""

    timestamp: float
    result: LaneAssignmentResult


@dataclass(frozen=True)
class TrackLaneHistory:
    """A track's full per-observation lane assignment history -- a
    non-schema companion object alongside (never inside)
    `storage.schemas.VehicleTrack`.

    `samples` is ordered by timestamp, one entry per ground-point
    observation the track accumulated, each independently assigned -- so a
    track that moves between lanes shows that transition across `samples`,
    rather than being described by a single whole-track-lifetime lane
    value that would go stale the moment the vehicle changes lanes.
    """

    track_id: str
    camera_id: str
    samples: tuple


def build_track_lane_history(camera_id: str, state: TrackState, regions: tuple) -> TrackLaneHistory:
    """Build the full per-observation lane history for one track, from its
    `ground_point_trajectory` (Commit 4 addition to `TrackState`, additive
    and parallel to the existing centroid-based `trajectory` -- see
    `track_manager.py`; nothing about `trajectory` or the existing
    `schema_conversion.build_vehicle_track` path is touched here).

    Mirrors `schema_conversion.build_vehicle_track`'s `track_id` naming
    convention (`f"{camera_id}:{state.track_id}"`) so the two outputs
    describing the same track are trivially joinable by ID.
    """
    samples = tuple(
        LaneAssignmentSample(timestamp=t, result=assign_lane((x, y), regions))
        for (x, y, t) in state.ground_point_trajectory
    )
    return TrackLaneHistory(
        track_id=f"{camera_id}:{state.track_id}",
        camera_id=camera_id,
        samples=samples,
    )
