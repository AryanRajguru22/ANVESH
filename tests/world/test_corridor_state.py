import pytest

from anvesh.evidence.reliability import ReliabilityScore
from anvesh.storage.schemas import CameraRole, CongestionLevel, Measurement, MotionSpace, TrafficState
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import UNRANKED_PLACEHOLDER, assemble_corridor_state


def _traffic_state(camera_id, congestion_level):
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


def _reliability(camera_id, score=0.9):
    return ReliabilityScore(camera_id=camera_id, window_start=0.0, window_end=10.0, score=score, factors={})


def _topology():
    return CorridorTopology(
        corridor_id="corridor-01",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
        ),
    )


def test_two_camera_assembly_both_present():
    topology = _topology()
    traffic_states = {
        "cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW),
        "cam-B": _traffic_state("cam-B", CongestionLevel.BUILDING),
    }
    reliability = {"cam-A": _reliability("cam-A"), "cam-B": _reliability("cam-B")}

    assembly = assemble_corridor_state("cs-1", topology, (0.0, 10.0), traffic_states, reliability)

    assert len(assembly.corridor_state.segment_states) == 2
    assert assembly.cameras_with_data == ("cam-A", "cam-B")
    assert assembly.cameras_missing == ()
    # worse-case wins: BUILDING > FREE_FLOW
    assert assembly.corridor_state.fused_congestion_level == CongestionLevel.BUILDING


def test_fused_level_takes_the_worst_camera():
    topology = _topology()
    traffic_states = {
        "cam-A": _traffic_state("cam-A", CongestionLevel.CONGESTED),
        "cam-B": _traffic_state("cam-B", CongestionLevel.FREE_FLOW),
    }
    reliability = {"cam-A": _reliability("cam-A"), "cam-B": _reliability("cam-B")}

    assembly = assemble_corridor_state("cs-1", topology, (0.0, 10.0), traffic_states, reliability)
    assert assembly.corridor_state.fused_congestion_level == CongestionLevel.CONGESTED


def test_missing_camera_is_recorded_not_fabricated():
    topology = _topology()
    traffic_states = {"cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW)}
    reliability = {"cam-A": _reliability("cam-A")}

    assembly = assemble_corridor_state("cs-1", topology, (0.0, 10.0), traffic_states, reliability)

    assert assembly.cameras_with_data == ("cam-A",)
    assembly_missing = assembly.cameras_missing
    assert assembly_missing == ("cam-B",)
    assert len(assembly.corridor_state.segment_states) == 1
    assert assembly.corridor_state.segment_states[0].camera_id == "cam-A"


def test_all_cameras_missing_raises_value_error():
    topology = _topology()
    with pytest.raises(ValueError):
        assemble_corridor_state("cs-1", topology, (0.0, 10.0), {}, {})


def test_active_ranking_id_uses_documented_placeholder():
    topology = _topology()
    traffic_states = {"cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW)}
    assembly = assemble_corridor_state("cs-1", topology, (0.0, 10.0), traffic_states, {})
    assert assembly.corridor_state.active_ranking_id == UNRANKED_PLACEHOLDER


def test_provenance_preserves_per_camera_traffic_states_unmerged():
    """A downstream consumer must be able to tell which camera contributed
    which TrafficState -- segment_states must not be collapsed/averaged."""
    topology = _topology()
    traffic_states = {
        "cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW),
        "cam-B": _traffic_state("cam-B", CongestionLevel.CONGESTED),
    }
    assembly = assemble_corridor_state("cs-1", topology, (0.0, 10.0), traffic_states, {})

    by_camera = {s.camera_id: s for s in assembly.corridor_state.segment_states}
    assert by_camera["cam-A"].congestion_level == CongestionLevel.FREE_FLOW
    assert by_camera["cam-B"].congestion_level == CongestionLevel.CONGESTED


def test_reliability_map_only_includes_cameras_with_data():
    topology = _topology()
    traffic_states = {"cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW)}
    reliability = {"cam-A": _reliability("cam-A"), "cam-B": _reliability("cam-B")}  # cam-B has no traffic state

    assembly = assemble_corridor_state("cs-1", topology, (0.0, 10.0), traffic_states, reliability)
    assert set(assembly.reliability_by_camera.keys()) == {"cam-A"}
