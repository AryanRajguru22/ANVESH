from pathlib import Path

from anvesh.evidence.reliability import ReliabilityScore
from anvesh.storage.db import CorridorStateStore
from anvesh.storage.schemas import CameraRole, CongestionLevel, Measurement, MotionSpace, TrafficState
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import assemble_corridor_state


def _traffic_state(camera_id, congestion_level=CongestionLevel.FREE_FLOW):
    return TrafficState(
        camera_id=camera_id,
        window_start=0.0,
        window_end=10.0,
        occupancy=0.3,
        mean_speed=Measurement(value=8.0, error=0.5),
        vehicle_count=3,
        flow_rate=0.3,
        density=3.0,
        congestion_level=congestion_level,
        motion_space=MotionSpace.WORLD,
    )


def _topology():
    return CorridorTopology(
        corridor_id="corridor-01",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
        ),
    )


def _assembly(corridor_state_id="cs-1", window=(0.0, 10.0)):
    topology = _topology()
    traffic_states = {
        "cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW),
        "cam-B": _traffic_state("cam-B", CongestionLevel.BUILDING),
    }
    reliability = {
        "cam-A": ReliabilityScore(camera_id="cam-A", window_start=0.0, window_end=10.0, score=0.9, factors={"x": 0.9}),
        "cam-B": ReliabilityScore(camera_id="cam-B", window_start=0.0, window_end=10.0, score=0.7, factors={"x": 0.7}),
    }
    return assemble_corridor_state(corridor_state_id, topology, window, traffic_states, reliability)


def test_save_and_get_round_trip(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        original = _assembly()
        store.save("corridor-01", original)

        loaded = store.get("cs-1")
        assert loaded is not None
        assert loaded.corridor_state.corridor_state_id == "cs-1"
        assert loaded.corridor_state.fused_congestion_level == CongestionLevel.BUILDING
        assert len(loaded.corridor_state.segment_states) == 2
        assert loaded.cameras_with_data == ("cam-A", "cam-B")
        assert loaded.cameras_missing == ()
        assert loaded.reliability_by_camera["cam-A"].score == 0.9
        assert loaded.reliability_by_camera["cam-B"].factors == {"x": 0.7}
    finally:
        store.close()


def test_get_missing_id_returns_none(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        assert store.get("does-not-exist") is None
    finally:
        store.close()


def test_list_for_corridor_orders_by_window_start(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        store.save("corridor-01", _assembly("cs-2", window=(10.0, 20.0)))
        store.save("corridor-01", _assembly("cs-1", window=(0.0, 10.0)))  # inserted second, earlier window

        results = store.list_for_corridor("corridor-01")
        assert [r.corridor_state.corridor_state_id for r in results] == ["cs-1", "cs-2"]
    finally:
        store.close()


def test_save_upserts_on_same_id(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        store.save("corridor-01", _assembly("cs-1"))
        updated = _assembly("cs-1")
        store.save("corridor-01", updated)

        results = store.list_for_corridor("corridor-01")
        assert len(results) == 1  # upsert, not a duplicate row
    finally:
        store.close()


def test_store_persists_across_reconnect(tmp_path: Path):
    db_path = tmp_path / "anvesh.sqlite3"
    store1 = CorridorStateStore(db_path)
    store1.save("corridor-01", _assembly())
    store1.close()

    store2 = CorridorStateStore(db_path)
    try:
        loaded = store2.get("cs-1")
        assert loaded is not None
        assert loaded.corridor_state.corridor_state_id == "cs-1"
    finally:
        store2.close()


def test_context_manager_closes_connection(tmp_path: Path):
    with CorridorStateStore(tmp_path / "anvesh.sqlite3") as store:
        store.save("corridor-01", _assembly())
        assert store.get("cs-1") is not None
