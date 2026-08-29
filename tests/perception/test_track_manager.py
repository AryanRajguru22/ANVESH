from anvesh.perception.track_manager import TrackManager
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import OcclusionState, VehicleClass


def _tracked(track_id: str, x: float, y: float, timestamp: float, frame_index: int) -> TrackedDetection:
    detection = RawDetection(
        frame_index=frame_index,
        timestamp=timestamp,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.9,
        bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
    )
    return TrackedDetection(detection=detection, track_id=track_id)


def test_new_track_is_created_visible():
    manager = TrackManager(max_missed_frames=3, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])

    assert "1" in manager.active_tracks
    state = manager.active_tracks["1"]
    assert state.occlusion_state == OcclusionState.VISIBLE
    assert state.first_seen == 0.0
    assert state.last_seen == 0.0
    assert state.trajectory == [(10.0, 10.0, 0.0)]


def test_trajectory_accumulates_across_frames():
    manager = TrackManager(max_missed_frames=3, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [_tracked("1", 12.0, 10.0, 0.1, 1)])
    manager.update(0.2, [_tracked("1", 14.0, 10.0, 0.2, 2)])

    state = manager.active_tracks["1"]
    assert state.trajectory == [(10.0, 10.0, 0.0), (12.0, 10.0, 0.1), (14.0, 10.0, 0.2)]
    assert state.age_seconds == 0.2


def test_temporary_miss_marks_partial_not_lost():
    manager = TrackManager(max_missed_frames=3, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [])  # missed one frame

    assert "1" in manager.active_tracks
    assert manager.active_tracks["1"].occlusion_state == OcclusionState.PARTIAL
    assert manager.active_tracks["1"].consecutive_missed_frames == 1


def test_reappearing_track_resets_to_visible():
    manager = TrackManager(max_missed_frames=3, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [])
    manager.update(0.2, [_tracked("1", 11.0, 10.0, 0.2, 2)])

    state = manager.active_tracks["1"]
    assert state.occlusion_state == OcclusionState.VISIBLE
    assert state.consecutive_missed_frames == 0


def test_exceeding_max_missed_frames_finalizes_as_lost():
    manager = TrackManager(max_missed_frames=2, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [])
    manager.update(0.2, [])
    manager.update(0.3, [])  # 3rd consecutive miss > max_missed_frames=2

    assert "1" not in manager.active_tracks
    assert len(manager.finished_tracks) == 1
    finished = manager.finished_tracks[0]
    assert finished.occlusion_state == OcclusionState.LOST
    assert finished.is_finished is True


def test_finalize_closes_out_remaining_active_tracks():
    manager = TrackManager(max_missed_frames=5, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    assert "1" in manager.active_tracks

    manager.finalize()

    assert manager.active_tracks == {}
    assert len(manager.finished_tracks) == 1
    assert manager.finished_tracks[0].is_finished is True


def test_all_tracks_combines_finished_and_active():
    manager = TrackManager(max_missed_frames=2, partial_after_missed_frames=1)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0), _tracked("2", 50.0, 50.0, 0.0, 0)])
    manager.update(0.1, [])
    manager.update(0.2, [])
    manager.update(0.3, [])  # track "1" and "2" both exceed max_missed_frames

    assert len(manager.all_tracks()) == 2


def test_invalid_thresholds_raise_value_error():
    import pytest

    with pytest.raises(ValueError):
        TrackManager(max_missed_frames=1, partial_after_missed_frames=2)
    with pytest.raises(ValueError):
        TrackManager(max_missed_frames=3, partial_after_missed_frames=0)
