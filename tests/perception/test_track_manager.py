import random

import pytest

from anvesh.evaluation.legacy_track_manager import TrackManager as LegacyTrackManager
from anvesh.perception.track_manager import (
    TrackManager,
    identified_detections,
    occlusion_horizon_s,
)
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


def _manager(max_occlusion_s=0.35, **kw):
    # 0.1 s frames throughout; 0.35 s horizon == "3 missed frames survive,
    # the 4th finalizes" (kept small so tests stay short)
    return TrackManager(max_occlusion_s=max_occlusion_s, camera_id="cam", session_id="s000", **kw)


def _only_active(manager):
    tracks = list(manager.active_tracks.values())
    assert len(tracks) == 1
    return tracks[0]


# ---------------------------------------------------------------------------
# Lifecycle (existing M1 behaviours, now time-based and identity-keyed)
# ---------------------------------------------------------------------------


def test_new_track_is_created_visible_with_session_scoped_identity():
    manager = _manager()
    assignments = manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])

    state = _only_active(manager)
    assert state.track_id == "s000-t00000"
    assert state.raw_tracker_id == "1"
    assert state.identity_key == ("cam", "s000", 0)
    assert assignments[0].track_id == state.track_id
    assert state.occlusion_state == OcclusionState.VISIBLE
    assert state.first_seen == 0.0 and state.last_seen == 0.0
    assert state.trajectory == [(10.0, 10.0, 0.0)]


def test_trajectory_accumulates_across_frames():
    manager = _manager()
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [_tracked("1", 12.0, 10.0, 0.1, 1)])
    manager.update(0.2, [_tracked("1", 14.0, 10.0, 0.2, 2)])

    state = _only_active(manager)
    assert state.trajectory == [(10.0, 10.0, 0.0), (12.0, 10.0, 0.1), (14.0, 10.0, 0.2)]
    assert state.age_seconds == 0.2
    assert state.sample_count == 3


def test_ground_point_trajectory_accumulates_alongside_centroid_trajectory():
    """`_tracked` builds a 10x10 box around (x, y): centroid == (x, y),
    ground_point == (x, y + 5)."""
    manager = _manager()
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [_tracked("1", 12.0, 10.0, 0.1, 1)])

    state = _only_active(manager)
    assert state.trajectory == [(10.0, 10.0, 0.0), (12.0, 10.0, 0.1)]
    assert state.ground_point_trajectory == [(10.0, 15.0, 0.0), (12.0, 15.0, 0.1)]


def test_temporary_miss_marks_partial_not_lost():
    manager = _manager()
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [])  # missed one frame

    state = _only_active(manager)
    assert state.occlusion_state == OcclusionState.PARTIAL
    assert state.consecutive_missed_frames == 1


def test_reappearing_track_resets_to_visible_and_keeps_identity():
    manager = _manager()
    first = manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])[0].track_id
    manager.update(0.1, [])
    again = manager.update(0.2, [_tracked("1", 11.0, 10.0, 0.2, 2)])[0].track_id

    state = _only_active(manager)
    assert again == first  # revival inside the horizon continues the SAME identity
    assert state.occlusion_state == OcclusionState.VISIBLE
    assert state.consecutive_missed_frames == 0
    assert len(manager.all_tracks()) == 1


def test_exceeding_occlusion_horizon_finalizes_as_lost():
    manager = _manager(max_occlusion_s=0.25)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.update(0.1, [])
    manager.update(0.2, [])
    assert len(manager.active_tracks) == 1  # 0.2 s gap <= 0.25 s horizon
    manager.update(0.3, [])  # 0.3 s gap > 0.25 s

    assert manager.active_tracks == {}
    assert len(manager.finished_tracks) == 1
    finished = manager.finished_tracks[0]
    assert finished.occlusion_state == OcclusionState.LOST
    assert finished.is_finished is True


def test_lifecycle_is_time_based_not_frame_based():
    """The same real-time gap behaves identically at 10 fps and 25 fps; a
    frame-count rule would finalize 2.5x sooner (in seconds) at 25 fps."""
    for fps in (10.0, 25.0):
        dt = 1.0 / fps
        manager = TrackManager(max_occlusion_s=0.5, camera_id="cam")
        manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
        t = 0.0
        while t + dt <= 0.5:  # empty frames covering a 0.5 s gap
            t += dt
            manager.update(t, [])
        assert len(manager.active_tracks) == 1, f"fps={fps}: finalized inside the 0.5 s horizon"
        manager.update(0.5 + 2 * dt, [])
        assert manager.active_tracks == {}, f"fps={fps}: not finalized after the horizon"


def test_finalize_closes_out_remaining_active_tracks():
    manager = _manager()
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    manager.finalize()

    assert manager.active_tracks == {}
    assert len(manager.finished_tracks) == 1
    assert manager.finished_tracks[0].is_finished is True


def test_all_tracks_combines_finished_and_active():
    manager = _manager(max_occlusion_s=0.25)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0), _tracked("2", 50.0, 50.0, 0.0, 0)])
    manager.update(0.1, [_tracked("2", 50.0, 50.0, 0.1, 1)])
    manager.update(0.2, [_tracked("2", 50.0, 50.0, 0.2, 2)])
    manager.update(0.4, [_tracked("2", 50.0, 50.0, 0.4, 4)])  # "1" gap 0.4 s > 0.25 s

    assert len(manager.all_tracks()) == 2
    assert len(manager.finished_tracks) == 1


def test_invalid_thresholds_raise_value_error():
    with pytest.raises(ValueError):
        TrackManager(max_occlusion_s=0.1, partial_after_s=0.2)
    with pytest.raises(ValueError):
        TrackManager(max_occlusion_s=1.0, partial_after_s=-0.1)
    with pytest.raises(ValueError):
        TrackManager(session_id="a:b")  # ':' would break the one-colon track_id shape
    with pytest.raises(ValueError):
        TrackManager(history_s=0)


# ---------------------------------------------------------------------------
# IDENTITY (M-A, ANVESH_PRODUCT_GAP_AUDIT.md finding 4)
# ---------------------------------------------------------------------------


def _assert_no_duplicate_identity(manager):
    keys = [s.identity_key for s in manager.all_tracks()]
    ids = [s.track_id for s in manager.all_tracks()]
    assert len(keys) == len(set(keys)), "two distinct TrackStates share an identity key"
    assert len(ids) == len(set(ids))


def test_raw_tracker_id_reuse_after_termination_gets_a_new_identity():
    """The audit's duplicate-ID defect, exactly: the tracker (ByteTrack,
    longer buffer) re-emits raw id "7" AFTER this manager already finalized
    its state."""
    manager = _manager(max_occlusion_s=0.25)
    a = manager.update(0.0, [_tracked("7", 10.0, 10.0, 0.0, 0)])[0].track_id
    for i in range(1, 5):
        manager.update(i * 0.1, [])  # 0.4 s without observation -> finalized
    assert manager.active_tracks == {}

    b = manager.update(0.5, [_tracked("7", 400.0, 200.0, 0.5, 5)])[0].track_id  # raw id "7" is back

    assert a != b
    assert len(manager.all_tracks()) == 2
    assert {s.raw_tracker_id for s in manager.all_tracks()} == {"7"}  # same raw id...
    _assert_no_duplicate_identity(manager)  # ...never the same identity
    assert manager.active_tracks[b].serial == 1


def test_legacy_manager_creates_the_duplicate_state_this_test_guards_against():
    """Contrast: the frozen pre-fix manager, fed the identical sequence,
    ends with two TrackStates carrying the same `track_id` -- proving the
    regression test above can fail."""
    legacy = LegacyTrackManager(max_missed_frames=2, partial_after_missed_frames=1)
    legacy.update(0.0, [_tracked("7", 10.0, 10.0, 0.0, 0)])
    for i in range(1, 5):
        legacy.update(i * 0.1, [])
    legacy.update(0.5, [_tracked("7", 400.0, 200.0, 0.5, 5)])
    ids = [s.track_id for s in legacy.all_tracks()]
    assert ids == ["7", "7"]


def test_raw_id_returning_after_excessive_gap_without_intervening_updates_is_a_new_identity():
    """Frames dropped / stalled stream: no update ran between last sight and
    return, so nothing finalized the state; the time gap alone must split it."""
    manager = _manager(max_occlusion_s=0.35)
    a = manager.update(0.0, [_tracked("3", 10.0, 10.0, 0.0, 0)])[0].track_id
    b = manager.update(30.0, [_tracked("3", 11.0, 10.0, 30.0, 300)])[0].track_id

    assert a != b
    assert manager.stale_raw_id_restarts == 1
    assert len(manager.finished_tracks) == 1 and len(manager.active_tracks) == 1
    assert manager.finished_tracks[0].occlusion_state == OcclusionState.LOST
    _assert_no_duplicate_identity(manager)


def test_revival_inside_horizon_is_not_split():
    """ByteTrack revives a lost id within its buffer; with a horizon aligned
    to that buffer the manager must NOT have given up, so the vehicle keeps
    one identity (this is what removes the pre-fix duplicates at the source)."""
    horizon = occlusion_horizon_s(30, fps=10.0)  # 3.15 s
    manager = TrackManager(max_occlusion_s=horizon, camera_id="cam")
    a = manager.update(0.0, [_tracked("9", 10.0, 10.0, 0.0, 0)])[0].track_id
    for i in range(1, 29):  # 2.8 s unseen = 28 frames
        manager.update(i * 0.1, [])
    b = manager.update(2.9, [_tracked("9", 12.0, 10.0, 2.9, 29)])[0].track_id
    assert a == b
    assert len(manager.all_tracks()) == 1


def test_same_raw_id_twice_in_one_frame_cannot_share_an_identity():
    manager = _manager()
    out = manager.update(0.0, [_tracked("5", 10.0, 10.0, 0.0, 0), _tracked("5", 300.0, 10.0, 0.0, 0)])
    assert out[0].track_id != out[1].track_id
    assert manager.same_frame_raw_id_collisions == 1
    _assert_no_duplicate_identity(manager)


def test_identity_assignment_is_deterministic():
    def run():
        m = _manager()
        ids = []
        for i in range(6):
            raws = ["a", "b"] if i != 3 else ["b"]
            ids.append([x.track_id for x in m.update(i * 0.1, [_tracked(r, 10.0 * k, 5.0, i * 0.1, i) for k, r in enumerate(raws)])])
        return ids

    assert run() == run()


def test_randomized_raw_id_churn_never_produces_a_duplicate_identity():
    """Property-style stress: a tracker that drops, revives and re-issues a
    small pool of raw ids at random for 400 frames."""
    rng = random.Random(1234)
    manager = TrackManager(max_occlusion_s=0.35, camera_id="cam")
    pool = ["1", "2", "3", "4"]
    for i in range(400):
        present = [r for r in pool if rng.random() < 0.6]
        manager.update(i * 0.1, [_tracked(r, 10.0 * int(r), 5.0, i * 0.1, i) for r in present])
    manager.finalize()
    states = manager.all_tracks()
    assert len(states) > len(pool)  # ids really were re-used across several lifetimes
    _assert_no_duplicate_identity(manager)
    assert manager.total_created == len(states)


def test_begin_session_fences_identity_and_restarts_serials():
    manager = _manager()
    a = manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])[0].track_id
    manager.begin_session("s001")  # e.g. reconnect + tracker reset: raw ids restart at "1"
    b = manager.update(0.1, [_tracked("1", 10.0, 10.0, 0.1, 1)])[0].track_id

    assert a == "s000-t00000" and b == "s001-t00000"
    assert a != b
    assert len(manager.finished_tracks) == 1 and len(manager.active_tracks) == 1
    _assert_no_duplicate_identity(manager)
    with pytest.raises(ValueError):
        manager.begin_session("s001")


def test_identified_detections_replace_raw_ids():
    manager = _manager()
    tds = [_tracked("77", 10.0, 10.0, 0.0, 0)]
    out = identified_detections(manager.update(0.0, tds))
    assert [t.track_id for t in out] == ["s000-t00000"]
    assert out[0].detection is tds[0].detection


def test_occlusion_horizon_helper():
    assert occlusion_horizon_s(30, 10.0) == pytest.approx(3.15)
    assert occlusion_horizon_s(30, 25.0) == pytest.approx(1.26)
    with pytest.raises(ValueError):
        occlusion_horizon_s(30, 0)


# ---------------------------------------------------------------------------
# Bounded memory
# ---------------------------------------------------------------------------


def test_history_trimming_bounds_trajectory_but_keeps_lifetime_facts():
    manager = TrackManager(max_occlusion_s=1.0, camera_id="cam", history_s=2.0)
    for i in range(200):
        t = i * 0.1
        manager.update(t, [_tracked("1", float(i), 10.0, t, i)])
    state = _only_active(manager)
    assert len(state.trajectory) <= 22 and len(state.ground_point_trajectory) <= 22
    assert state.first_seen == 0.0
    assert state.sample_count == 200
    assert state.trajectory[-1][2] == pytest.approx(19.9)


def test_evict_finished_before():
    manager = _manager(max_occlusion_s=0.15)
    manager.update(0.0, [_tracked("1", 10.0, 10.0, 0.0, 0)])
    for i in range(1, 4):
        manager.update(i * 0.1, [])
    manager.update(5.0, [_tracked("2", 10.0, 10.0, 5.0, 50)])
    assert len(manager.finished_tracks) == 1
    assert manager.evict_finished_before(0.0) == 0
    assert manager.evict_finished_before(1.0) == 1
    assert manager.finished_tracks == []
    assert len(manager.active_tracks) == 1
