"""Corridor-state assembly (blueprint Part 3, module 11): combine the
corridor's topology with each camera's per-window `TrafficState` into one
`anvesh.storage.schemas.CorridorState`.

Per blueprint Part 3's module 11 description this is "direct assembly --
no new modeling, mostly bookkeeping": no cross-camera cause-hypothesis
fusion (Part 6 Dempster-Shafer combination, milestone M4) happens here.
`fused_congestion_level` is assembled with a simple, documented severity-
ordering rule (worst-camera-wins), which is legitimate state bookkeeping,
not evidence fusion.

FLAGGED INCOMPATIBILITY (reported rather than silently patched):
`CorridorState.active_ranking_id` is a required, non-empty string FK to a
`CandidateCauseHypothesis` ranking (blueprint Part 5) -- but cause-
hypothesis ranking does not exist until M4. Every `CorridorState` built
here uses the documented placeholder `UNRANKED_PLACEHOLDER` for that
field, exactly analogous to M1's `UNCALIBRATED_PROFILE_ID` and M2's
`fit_planar_homography`/`uncalibrated_profile` pattern: no schema field
was added or changed, a sentinel value fills the gap, and this is called
out explicitly rather than silently invented. M4 must replace this
placeholder with a real ranking_id once `hypotheses/ranking.py` exists.

MISSING CAMERA DATA: `CorridorState.segment_states` has no minimum-length
validation in the frozen schema, so a window where only one camera
reported data is represented by omitting the missing camera from
`segment_states` entirely -- never by fabricating a zero-vehicle
`TrafficState` for a camera that simply did not report. `assemble_corridor_state`
requires at least one camera's data to exist per window (raises
`ValueError` if given none), since a corridor state needs *some* evidence
to be meaningful at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.storage.schemas import CongestionLevel, CorridorState
from anvesh.world.corridor import CorridorTopology

UNRANKED_PLACEHOLDER = "unranked-pending-M4"

_SEVERITY_ORDER = {
    CongestionLevel.FREE_FLOW: 0,
    CongestionLevel.DISSIPATING: 1,
    CongestionLevel.BUILDING: 2,
    CongestionLevel.CONGESTED: 3,
}


@dataclass(frozen=True)
class CorridorStateAssembly:
    """The frozen CorridorState plus provenance that doesn't belong in the
    Part 5 schema: which cameras actually contributed this window, and
    each contributing camera's ReliabilityScore."""

    corridor_state: CorridorState
    reliability_by_camera: dict
    cameras_with_data: tuple
    cameras_missing: tuple


def assemble_corridor_state(
    corridor_state_id: str,
    corridor_topology: CorridorTopology,
    window: tuple,
    traffic_states_by_camera: dict,
    reliability_by_camera: dict,
) -> CorridorStateAssembly:
    """Assemble one window's CorridorState from per-camera TrafficStates.

    `traffic_states_by_camera` and `reliability_by_camera` are keyed by
    camera_id; a camera present in the corridor's topology but absent
    from `traffic_states_by_camera` is treated as having reported no data
    this window (see module docstring) -- it is recorded in
    `cameras_missing`, not silently ignored.
    """
    all_camera_ids = tuple(corridor_topology.ordered_camera_ids())
    cameras_with_data = tuple(cid for cid in all_camera_ids if cid in traffic_states_by_camera)
    cameras_missing = tuple(cid for cid in all_camera_ids if cid not in traffic_states_by_camera)

    if not cameras_with_data:
        raise ValueError("assemble_corridor_state requires at least one camera's TrafficState for this window")

    segment_states = [traffic_states_by_camera[cid] for cid in cameras_with_data]
    fused_congestion_level = max(
        (ts.congestion_level for ts in segment_states),
        key=lambda level: _SEVERITY_ORDER[level],
    )

    corridor_state = CorridorState(
        corridor_state_id=corridor_state_id,
        window=window,
        segment_states=segment_states,
        fused_congestion_level=fused_congestion_level,
        active_ranking_id=UNRANKED_PLACEHOLDER,
    )

    return CorridorStateAssembly(
        corridor_state=corridor_state,
        reliability_by_camera={cid: reliability_by_camera[cid] for cid in cameras_with_data if cid in reliability_by_camera},
        cameras_with_data=cameras_with_data,
        cameras_missing=cameras_missing,
    )
