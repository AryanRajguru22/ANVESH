import pytest

from anvesh.perception.types import Frame
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.windows import SlidingWindowScheduler, WindowSpec
from tests.streaming.support import TINY, ScriptedTracker, make_frames

FPS = 10.0


def _stationary(frame_index):
    return [("a", (50.0, 50.0)), ("b", (100.0, 80.0))]


def _run(engine, frames):
    snaps = []
    for f in frames:
        snaps.extend(engine.process_frame(f))
    snaps.extend(engine.finish())
    return snaps


# ---------------------------------------------------------------------------
# WindowSpec / scheduler
# ---------------------------------------------------------------------------


def test_window_spec_defaults_and_validation():
    spec = WindowSpec(window_s=10.0)
    assert spec.stride_s == 10.0 and not spec.overlapping
    assert spec.bounds(2) == (20.0, 30.0)
    assert WindowSpec(window_s=10.0, stride_s=5.0).overlapping
    for kwargs in ({"window_s": 0}, {"window_s": 1, "stride_s": 0}, {"window_s": 1, "lateness_s": -1}):
        with pytest.raises(ValueError):
            WindowSpec(**kwargs)


def test_scheduler_returns_each_due_window_exactly_once_and_in_order():
    sched = SlidingWindowScheduler(WindowSpec(window_s=10.0, lateness_s=1.0))
    assert sched.due(10.5) == []  # closes at 11.0 (end + lateness)
    assert sched.due(11.0) == [0]
    assert sched.due(11.0) == []
    assert sched.due(31.0) == [1, 2]  # a stall closes several at once
    assert sched.remaining_until(35.0) == [3]


# ---------------------------------------------------------------------------
# Engine: windows
# ---------------------------------------------------------------------------


def test_windows_close_on_stream_time_without_reaching_eof():
    engine = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0))
    closed_at = {}
    for f in make_frames(250, FPS):  # 25 s; we never call finish()
        for s in engine.process_frame(f):
            closed_at[s.window_index] = f.index
    assert closed_at == {0: 100, 1: 200}  # window k closes on the first frame with t >= end
    # deterministic, camera-scoped ids
    snaps = _run(StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0)), make_frames(250, FPS))
    assert [s.window_id for s in snaps] == ["cam:w000000", "cam:w000001", "cam:w000002"]


def test_window_contains_only_in_window_samples_and_final_window_is_clipped():
    engine = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0))
    snaps = _run(engine, make_frames(250, FPS))
    w0, w1, w2 = snaps
    assert (w0.start, w0.end, w0.final_partial) == (0.0, 10.0, False)
    assert (w2.start, w2.end, w2.final_partial) == (20.0, 25.0, True)  # clipped to the 25.0 s of stream
    for w in snaps:
        for t in w.tracks:
            assert t.first_seen >= w.start and t.last_seen < w.end + 1e-9  # clipped view, not lifetime
    assert w0.frames_observed == 100 and w1.frames_observed == 100 and w2.frames_observed == 50
    assert all(not w.degraded for w in snaps)


def test_in_window_speed_is_not_a_lifetime_average():
    """A vehicle that is still for 10 s then drives off: window 0 must see it
    stopped (the legacy lifetime-mean speed would have smeared the later
    motion into every window it overlaps)."""
    def script(i):
        x = 100.0 if i < 100 else 100.0 + 8.0 * (i - 100)
        return [("v", (x, 60.0))]

    snaps = _run(StreamingPerceptionEngine("cam", ScriptedTracker(script), FPS, windows=WindowSpec(10.0)), make_frames(200, FPS))
    (t0,) = snaps[0].tracks
    (t1,) = snaps[1].tracks
    assert t0.speed_estimate.value == pytest.approx(0.0)
    assert t1.speed_estimate.value == pytest.approx(80.0, rel=0.05)  # 8 px / 0.1 s


def test_overlapping_windows_share_samples():
    engine = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0, stride_s=5.0))
    snaps = _run(engine, make_frames(200, FPS))
    assert [(s.start, s.end) for s in snaps][:3] == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0)]
    # the same identity appears in consecutive overlapping windows
    ids = [{t.track_id for t in s.tracks} for s in snaps[:3]]
    assert ids[0] == ids[1] == ids[2]


def test_lateness_delays_window_close():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0, lateness_s=2.0))
    closed = {}
    for f in make_frames(150, FPS):
        for s in eng.process_frame(f):
            closed[s.window_index] = f.index
    assert closed == {0: 120}


# ---------------------------------------------------------------------------
# Engine: identity, sessions, drops
# ---------------------------------------------------------------------------


def test_tracker_id_reuse_never_collides_through_the_whole_engine():
    """raw id "1" disappears for longer than the horizon and is re-issued."""
    def script(i):
        return [("1", (50.0 + i, 50.0))] if (i < 40 or i >= 120) else []

    eng = StreamingPerceptionEngine("cam", ScriptedTracker(script, lost_buffer_frames=5), FPS, windows=WindowSpec(10.0))
    snaps = _run(eng, make_frames(160, FPS))
    all_ids = [t.track_id for s in snaps for t in s.tracks]
    # finished states are evicted once no open window needs them (bounded
    # memory), so count what was ever created and what the windows saw
    assert eng.store.manager.total_created == 2
    assert eng.stats()["tracks_created"] == 2
    for s in snaps:  # no duplicate identity inside any window
        ids = [t.track_id for t in s.tracks]
        assert len(ids) == len(set(ids))
    assert len(set(all_ids)) == 2


def test_epoch_change_starts_a_new_session_and_resets_the_tracker():
    tracker = ScriptedTracker(lambda i: [("1", (50.0, 50.0))])
    eng = StreamingPerceptionEngine("cam", tracker, FPS, windows=WindowSpec(10.0), run_id=None)  # explicit session-only scope
    frames = make_frames(50, FPS) + make_frames(50, FPS, epoch=1, start_index=50)
    snaps = _run(eng, frames)
    assert tracker.reset_calls == 1
    assert eng.stats()["sessions_started"] == 2
    ids = {t.track_id for s in snaps for t in s.tracks}
    assert ids == {"cam:s000-t00000", "cam:s001-t00000"}  # same raw id "1", different sessions


def test_dropped_frames_are_counted_and_mark_the_window_degraded():
    frames = make_frames(100, FPS)
    kept = [f for f in frames if not (30 <= f.index < 40)]  # 10 frames lost
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0))
    snaps = _run(eng, kept)
    assert eng.stats()["frames_dropped"] == 10
    assert snaps[0].dropped_frames == 10 and snaps[0].degraded
    assert snaps[0].frames_observed == 90


def test_source_reported_gaps_count_as_dropped():
    frames = make_frames(20, FPS)
    frames[5] = Frame(index=5, timestamp=0.5, image=TINY, gap_before=3)
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0))
    _run(eng, frames)
    assert eng.stats()["frames_dropped"] == 3


def test_engine_horizon_is_lifted_to_the_trackers_own_buffer(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="anvesh.streaming.engine")
    eng = StreamingPerceptionEngine(
        "cam", ScriptedTracker(_stationary, lost_buffer_frames=30), FPS, min_occlusion_s=0.5
    )
    assert eng.max_occlusion_s == pytest.approx(3.15)
    assert any("shorter than the tracker" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Bounded memory
# ---------------------------------------------------------------------------


def test_memory_is_bounded_over_a_long_churning_stream():
    """20 minutes of 10 fps with vehicles that continuously enter/leave and a
    long-lived one: tracked state, drop events and frame times must plateau."""
    def script(i):
        out = [("parked", (30.0, 30.0))]
        k = i // 15  # a new vehicle every 1.5 s, visible for 4 s
        for j in range(max(0, k - 2), k + 1):
            if 0 <= i - j * 15 < 40:
                out.append((f"v{j}", (60.0 + (i - j * 15), 40.0 + j % 7)))
        return out

    eng = StreamingPerceptionEngine("cam", ScriptedTracker(script, lost_buffer_frames=10), FPS, windows=WindowSpec(10.0))
    retained = []
    n = 12000
    for f in make_frames(n, FPS):
        eng.process_frame(f)
        if f.index % 1000 == 999:
            retained.append(
                (eng.store.retained_track_count(), len(eng._frame_times), len(eng._drop_events),
                 len(eng.store.manager.active_tracks["s000-t00000"].trajectory) if "s000-t00000" in eng.store.manager.active_tracks else 0)
            )
    assert eng.stats()["tracks_created"] > 600  # lots of churn really happened
    assert max(r[0] for r in retained) <= 12, retained
    assert max(r[1] for r in retained) <= 140, retained  # ~window + margin, not 12000
    assert max(r[3] for r in retained) <= 140, retained  # long-lived track's history is trimmed


def test_archive_mode_retains_everything_like_the_legacy_batch():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, archive=True, run_id=None)
    for f in make_frames(300, FPS):
        eng.process_frame(f)
    assert len(eng.camera_observations) == 300
    assert len(eng.store.manager.active_tracks["s000-t00000"].trajectory) == 300


def test_run_id_makes_identities_unique_across_runs_and_is_deterministic():
    from anvesh.perception.track_manager import derive_run_id

    assert derive_run_id("cam-1", "/v/a.mp4", 123) == derive_run_id("cam-1", "/v/a.mp4", 123)
    assert derive_run_id("cam-1", "/v/a.mp4", 123) != derive_run_id("cam-1", "/v/a.mp4", 124)
    assert len(derive_run_id("x")) == 8 and ":" not in derive_run_id("x")

    def ids(run_id):
        eng = StreamingPerceptionEngine("cam", ScriptedTracker(_stationary), FPS, windows=WindowSpec(10.0), run_id=run_id)
        return {t.track_id for s in _run(eng, make_frames(120, FPS)) for t in s.tracks}

    a, b = ids(derive_run_id("run-A")), ids(derive_run_id("run-B"))
    assert a.isdisjoint(b)  # same camera, same raw tracker ids, two runs -> no key collision
    assert ids(derive_run_id("run-A")) == a  # deterministic
    assert all(i.count(":") == 1 for i in a)  # still the dashboard-parsable single-colon shape
    assert ids(None) == {"cam:s000-t00000", "cam:s000-t00001"}  # default unchanged
