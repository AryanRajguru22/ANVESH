import pytest

from anvesh.perception.traffic_state import CongestionStateTracker
from anvesh.perception.windowing import (
    Window,
    build_traffic_state_timeline,
    generate_windows,
    tracks_in_window,
    video_duration_from_tracks,
)
from anvesh.storage.schemas import Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack


def _track(track_id, first_seen, last_seen, speed_value=5.0):
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-A",
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=0.0),
        motion_space=MotionSpace.WORLD,
    )


# ---------------------------------------------------------------------------
# generate_windows
# ---------------------------------------------------------------------------


def test_normal_windows_evenly_divide_duration():
    windows = generate_windows(30.0, 10.0)
    assert windows == (
        Window(0, 0.0, 10.0),
        Window(1, 10.0, 20.0),
        Window(2, 20.0, 30.0),
    )


def test_final_window_is_clipped_not_dropped_or_overrun():
    windows = generate_windows(25.0, 10.0)
    assert len(windows) == 3
    assert windows[-1] == Window(2, 20.0, 25.0)  # clipped to 5s, not 10s


def test_zero_duration_yields_no_windows():
    assert generate_windows(0.0, 10.0) == ()


def test_negative_duration_rejected():
    with pytest.raises(ValueError):
        generate_windows(-1.0, 10.0)


def test_non_positive_window_seconds_rejected():
    with pytest.raises(ValueError):
        generate_windows(10.0, 0.0)
    with pytest.raises(ValueError):
        generate_windows(10.0, -5.0)


def test_window_rejects_end_before_start():
    with pytest.raises(ValueError):
        Window(0, 10.0, 5.0)


def test_generate_windows_is_deterministic():
    a = generate_windows(37.3, 10.0)
    b = generate_windows(37.3, 10.0)
    assert a == b


# ---------------------------------------------------------------------------
# tracks_in_window
# ---------------------------------------------------------------------------


def test_track_fully_inside_window():
    window = Window(0, 0.0, 10.0)
    t = _track("1", 2.0, 8.0)
    assert tracks_in_window([t], window) == [t]


def test_track_overlapping_multiple_windows_appears_in_each():
    long_track = _track("1", 5.0, 25.0)
    windows = generate_windows(30.0, 10.0)
    matches = [w.index for w in windows if tracks_in_window([long_track], w)]
    assert matches == [0, 1, 2]  # overlaps [0,10), [10,20), and touches [20,30)


def test_track_entirely_outside_window_excluded():
    window = Window(0, 0.0, 10.0)
    before = _track("before", -5.0, -1.0)
    after = _track("after", 20.0, 25.0)
    assert tracks_in_window([before, after], window) == []


def test_empty_track_list_yields_empty_window_result():
    window = Window(0, 0.0, 10.0)
    assert tracks_in_window([], window) == []


def test_boundary_timestamps_use_strict_inequality_matching_aggregate_traffic_state():
    """A track ending EXACTLY at a window's start, or starting EXACTLY at
    a window's end, must not count as overlapping -- must mirror
    perception.traffic_state.aggregate_traffic_state's own predicate."""
    window = Window(0, 10.0, 20.0)
    ends_at_start = _track("a", 5.0, 10.0)  # last_seen == window.start
    starts_at_end = _track("b", 20.0, 25.0)  # first_seen == window.end
    touches_start = _track("c", 5.0, 10.0001)  # just barely overlaps
    assert tracks_in_window([ends_at_start, starts_at_end], window) == []
    assert tracks_in_window([touches_start], window) == [touches_start]


def test_tracks_in_window_preserves_order_and_identity():
    t1, t2, t3 = _track("1", 0.0, 5.0), _track("2", 1.0, 6.0), _track("3", 2.0, 7.0)
    window = Window(0, 0.0, 10.0)
    result = tracks_in_window([t1, t2, t3], window)
    assert result == [t1, t2, t3]
    assert [t.track_id for t in result] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# video_duration_from_tracks
# ---------------------------------------------------------------------------


def test_video_duration_is_the_max_last_seen():
    tracks = [_track("1", 0.0, 5.0), _track("2", 3.0, 12.5), _track("3", 1.0, 9.0)]
    assert video_duration_from_tracks(tracks) == 12.5


def test_video_duration_zero_for_empty_tracks():
    assert video_duration_from_tracks([]) == 0.0


# ---------------------------------------------------------------------------
# build_traffic_state_timeline
# ---------------------------------------------------------------------------


def test_build_timeline_produces_one_aggregation_per_window():
    tracks = [_track("1", 0.0, 25.0)]
    timeline = build_traffic_state_timeline(
        "cam-A", tracks, MotionSpace.WORLD, window_seconds=10.0, congestion_tracker=CongestionStateTracker()
    )
    assert len(timeline) == 3  # 0-10, 10-20, 20-25
    for agg in timeline:
        assert agg.traffic_state.camera_id == "cam-A"


def test_build_timeline_shares_one_tracker_across_windows_for_hysteresis():
    """Sustained heavy traffic across windows must be able to reach
    CONGESTED via the shared tracker's hysteresis -- a fresh tracker per
    window would never accumulate BUILDING -> CONGESTED."""
    from anvesh.storage.schemas import CongestionLevel

    heavy_tracks = [_track(str(i), 0.0, 20.0, speed_value=1.0) for i in range(15)]
    timeline = build_traffic_state_timeline(
        "cam-A", heavy_tracks, MotionSpace.WORLD, window_seconds=10.0, congestion_tracker=CongestionStateTracker()
    )
    assert len(timeline) == 2
    assert timeline[0].traffic_state.congestion_level == CongestionLevel.BUILDING
    assert timeline[1].traffic_state.congestion_level == CongestionLevel.CONGESTED


def test_build_timeline_empty_tracks_yields_empty_timeline():
    timeline = build_traffic_state_timeline(
        "cam-A", [], MotionSpace.WORLD, window_seconds=10.0, congestion_tracker=CongestionStateTracker()
    )
    assert timeline == ()


def test_build_timeline_is_deterministic():
    tracks = [_track("1", 0.0, 25.0), _track("2", 5.0, 15.0)]
    t1 = build_traffic_state_timeline(
        "cam-A", tracks, MotionSpace.WORLD, window_seconds=10.0, congestion_tracker=CongestionStateTracker()
    )
    t2 = build_traffic_state_timeline(
        "cam-A", tracks, MotionSpace.WORLD, window_seconds=10.0, congestion_tracker=CongestionStateTracker()
    )
    assert [(a.traffic_state.vehicle_count, a.traffic_state.mean_speed.value) for a in t1] == [
        (a.traffic_state.vehicle_count, a.traffic_state.mean_speed.value) for a in t2
    ]


def test_build_timeline_respects_explicit_total_duration_override():
    tracks = [_track("1", 0.0, 5.0)]
    timeline = build_traffic_state_timeline(
        "cam-A", tracks, MotionSpace.WORLD, window_seconds=10.0,
        congestion_tracker=CongestionStateTracker(), total_duration=30.0,
    )
    assert len(timeline) == 3  # overridden duration, not the track-derived 5.0s
