import pytest

from anvesh.storage.schemas import CameraRole
from anvesh.world.corridor import CameraPlacement, CorridorTopology


def _topology():
    return CorridorTopology(
        corridor_id="corridor-01",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
        ),
    )


def test_ordered_camera_ids_upstream_first():
    topology = _topology()
    assert topology.ordered_camera_ids() == ["cam-A", "cam-B"]


def test_is_upstream_of():
    topology = _topology()
    assert topology.is_upstream_of("cam-A", "cam-B") is True
    assert topology.is_upstream_of("cam-B", "cam-A") is False


def test_distance_between_is_symmetric_and_correct():
    topology = _topology()
    assert topology.distance_between_m("cam-A", "cam-B") == 40.0
    assert topology.distance_between_m("cam-B", "cam-A") == 40.0


def test_placement_for_unknown_camera_raises_key_error():
    topology = _topology()
    with pytest.raises(KeyError):
        topology.placement_for("cam-Z")


def test_requires_exactly_two_placements():
    with pytest.raises(ValueError):
        CorridorTopology(
            corridor_id="corridor-01",
            placements=(CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),),
        )


def test_requires_one_upstream_and_one_downstream():
    with pytest.raises(ValueError):
        CorridorTopology(
            corridor_id="corridor-01",
            placements=(
                CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
                CameraPlacement(camera_id="cam-B", role=CameraRole.UPSTREAM, distance_from_upstream_m=40.0),
            ),
        )


def test_downstream_must_be_strictly_farther_than_upstream():
    with pytest.raises(ValueError):
        CorridorTopology(
            corridor_id="corridor-01",
            placements=(
                CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=50.0),
                CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
            ),
        )


def test_duplicate_camera_ids_rejected():
    with pytest.raises(ValueError):
        CorridorTopology(
            corridor_id="corridor-01",
            placements=(
                CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
                CameraPlacement(camera_id="cam-A", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
            ),
        )


def test_negative_distance_rejected():
    with pytest.raises(ValueError):
        CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=-1.0)
