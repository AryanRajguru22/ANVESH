"""Deterministic end-to-end M3 scenario: two cameras' VehicleTracks ->
per-window TrafficState (with hysteresis) -> reliability -> CorridorState
assembly -> SQLite persistence -> reload, across a 3-window timeline.
"""

from pathlib import Path

from anvesh.evidence.reliability import compute_reliability
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.db import CorridorStateStore
from anvesh.storage.schemas import CameraRole, CongestionLevel, Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import assemble_corridor_state


def _track(track_id, camera_id, first_seen, last_seen, speed_value):
    return VehicleTrack(
        track_id=track_id,
        camera_id=camera_id,
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=0.0),
        motion_space=MotionSpace.WORLD,
    )


def _build_topology():
    return CorridorTopology(
        corridor_id="corridor-e2e",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
        ),
    )


def test_three_window_timeline_free_to_building_to_congested(tmp_path: Path):
    topology = _build_topology()
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")

    tracker_a = CongestionStateTracker()
    tracker_b = CongestionStateTracker()

    # Camera A: light -> heavy -> heavy (drives the corridor into congestion)
    cam_a_windows = [
        (0.0, 10.0, [_track(f"a{i}", "cam-A", 0.0, 10.0, speed_value=10.0) for i in range(2)]),
        (10.0, 20.0, [_track(f"a{i}", "cam-A", 10.0, 20.0, speed_value=1.0) for i in range(15)]),
        (20.0, 30.0, [_track(f"a{i}", "cam-A", 20.0, 30.0, speed_value=1.0) for i in range(15)]),
    ]
    # Camera B: light throughout -- never drives congestion on its own
    cam_b_windows = [
        (0.0, 10.0, [_track("b0", "cam-B", 0.0, 10.0, speed_value=9.0)]),
        (10.0, 20.0, [_track("b0", "cam-B", 10.0, 20.0, speed_value=9.0)]),
        (20.0, 30.0, [_track("b0", "cam-B", 20.0, 30.0, speed_value=9.0)]),
    ]

    expected_corridor_levels = [CongestionLevel.FREE_FLOW, CongestionLevel.BUILDING, CongestionLevel.CONGESTED]
    expected_cam_a_levels = [CongestionLevel.FREE_FLOW, CongestionLevel.BUILDING, CongestionLevel.CONGESTED]

    corridor_state_ids = []
    for i, ((ws, we, a_tracks), (_, _, b_tracks)) in enumerate(zip(cam_a_windows, cam_b_windows)):
        agg_a = aggregate_traffic_state("cam-A", ws, we, a_tracks, MotionSpace.WORLD, tracker_a)
        agg_b = aggregate_traffic_state("cam-B", ws, we, b_tracks, MotionSpace.WORLD, tracker_b)

        rel_a = compute_reliability("cam-A", ws, we, a_tracks, agg_a)
        rel_b = compute_reliability("cam-B", ws, we, b_tracks, agg_b)

        corridor_state_id = f"cs-window-{i}"
        assembly = assemble_corridor_state(
            corridor_state_id,
            topology,
            (ws, we),
            {"cam-A": agg_a.traffic_state, "cam-B": agg_b.traffic_state},
            {"cam-A": rel_a, "cam-B": rel_b},
        )

        assert agg_a.traffic_state.congestion_level == expected_cam_a_levels[i]
        assert assembly.corridor_state.fused_congestion_level == expected_corridor_levels[i]
        assert assembly.cameras_with_data == ("cam-A", "cam-B")
        assert len(assembly.corridor_state.segment_states) == 2

        store.save("corridor-e2e", assembly)
        corridor_state_ids.append(corridor_state_id)

    try:
        # reload the full timeline and confirm it reproduces the decisions exactly
        reloaded = store.list_for_corridor("corridor-e2e")
        assert [r.corridor_state.corridor_state_id for r in reloaded] == corridor_state_ids
        assert [r.corridor_state.fused_congestion_level for r in reloaded] == expected_corridor_levels

        by_camera_window2 = {s.camera_id: s for s in reloaded[1].corridor_state.segment_states}
        assert by_camera_window2["cam-A"].congestion_level == CongestionLevel.BUILDING
        assert by_camera_window2["cam-B"].congestion_level == CongestionLevel.FREE_FLOW
        assert reloaded[1].reliability_by_camera["cam-A"].score > 0.0
    finally:
        store.close()
