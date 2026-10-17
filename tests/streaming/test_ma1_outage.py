"""M-A.1 Fix 3: camera outages, silence and dropped frames.

Core rule under test: a camera that delivered nothing is UNAVAILABLE -- never an
"empty road / free flow" observation -- and lost frames are charged to the windows
they actually fell in.
"""

from __future__ import annotations

import threading
import time

import pytest

from anvesh.perception.types import Frame
from anvesh.storage.schemas import (
    CongestionLevel, Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack,
)
from anvesh.streaming.engine import StreamHealth, StreamingPerceptionEngine
from anvesh.streaming.pipeline import StreamingPipeline
from anvesh.streaming.queues import DropPolicy
from anvesh.streaming.reasoning import AnalysisStatus, WindowReasoner
from anvesh.streaming.rtsp import RtspFrameSource
from anvesh.streaming.windows import WindowAvailability, WindowSnapshot, WindowSpec
from tests.streaming.support import FakeClock, ListFrameSource, ScriptedTracker, TINY, make_frames

FPS = 10.0


def _car(i):
    return [(1, (50.0 + i, 50.0))]


def _empty(i):
    return []


def _engine(script=_car, **kw):
    return StreamingPerceptionEngine("cam", ScriptedTracker(script), FPS, windows=WindowSpec(10.0), run_id=None, **kw)


def _feed(eng, frames):
    snaps = []
    for f in frames:
        snaps.extend(eng.process_frame(f))
    return snaps


def _reconnect_frames():
    """0..4.9 s of video, a 35 s outage, then reconnect (epoch 1) at t=40 s with 15 s more."""
    gap = int(round(35.0 * FPS)) - 1
    frames = [Frame(i, i / FPS, TINY, epoch=0) for i in range(50)]
    frames += [Frame(50 + j, 40.0 + j / FPS, TINY, epoch=1, gap_before=(gap if j == 0 else 0)) for j in range(150)]
    return frames


# ------------------------------------------------------ 1. 35 s reconnect gap --

def test_a_35s_reconnect_gap_yields_unavailable_windows_never_free_flow():
    eng = _engine()
    snaps = _feed(eng, _reconnect_frames()) + eng.finish()
    reasoner = WindowReasoner("cam")
    analyses = [reasoner.process(s) for s in snaps]
    by_idx = {a.snapshot.window_index: a for a in analyses}

    for k in (1, 2, 3):  # windows lying wholly inside the outage
        s = by_idx[k].snapshot
        assert s.frames_observed == 0 and s.availability == WindowAvailability.UNAVAILABLE and s.coverage == 0.0
        assert by_idx[k].status == AnalysisStatus.UNAVAILABLE
        assert by_idx[k].aggregation is None and by_idx[k].result is None  # no traffic state, no ranking
    assert by_idx[0].status == AnalysisStatus.UNAVAILABLE  # only 4.9 s of its 10 s were observed (< 50%)
    assert by_idx[4].status == AnalysisStatus.OBSERVED and by_idx[5].status == AnalysisStatus.OBSERVED


def test_dropped_frames_of_the_outage_are_charged_to_the_windows_it_spans_not_the_recovery_window():
    eng = _engine()
    snaps = _feed(eng, _reconnect_frames()) + eng.finish()
    dropped = {s.window_index: s.dropped_frames for s in snaps}
    assert sum(dropped.values()) == 349 == eng.stats()["frames_dropped"]
    assert dropped[4] == 0 and dropped[5] == 0  # the recovery window really did see all its frames
    assert dropped[0] == pytest.approx(50, abs=1) and dropped[1] == pytest.approx(100, abs=1)
    assert dropped[2] == pytest.approx(100, abs=1) and dropped[3] == pytest.approx(99, abs=1)


def test_reconnect_starts_a_new_identity_session_and_recovers():
    eng = _engine()
    snaps = _feed(eng, _reconnect_frames()) + eng.finish()
    sessions = {sess for s in snaps for sess in s.sessions}
    assert sessions == {"s000", "s001"}
    assert eng.health == StreamHealth.LIVE
    assert [e.current for e in eng.health_events] == [StreamHealth.LIVE]  # LIVE at the first frame only


def test_recovery_resets_hysteresis_so_a_stale_level_does_not_leak_across_the_outage():
    reasoner = WindowReasoner("cam")
    n = 30  # > max_capacity_vehicles => occupancy clamps at 1.0 => "high" pressure in image space
    for k in range(3):
        reasoner.process(_snapshot(k, n_tracks=n))
    assert reasoner.congestion_state == CongestionLevel.CONGESTED
    a = reasoner.process(_snapshot(3, n_tracks=0, availability=WindowAvailability.UNAVAILABLE, coverage=0.0, frames=0))
    assert a.unavailable and reasoner.congestion_state == CongestionLevel.FREE_FLOW  # memory cleared, not "dissipating"


# ------------------------------------------------------ 2. total source silence --

def test_total_silence_is_detected_without_another_frame_via_tick():
    transitions = []
    eng = _engine(on_health=transitions.append)
    _feed(eng, make_frames(50, FPS))  # 0..4.9 s then the camera goes dark
    assert eng.health == StreamHealth.LIVE

    assert eng.tick(5.5) == [] and eng.health == StreamHealth.LIVE  # shorter than the stall threshold
    assert eng.tick(8.0) == [] and eng.health == StreamHealth.STALLED  # > 2 s, no window due yet

    closed = eng.tick(25.0)  # no frame ever arrived; the clock alone closes w0 and w1
    assert eng.health == StreamHealth.DOWN
    assert [s.window_index for s in closed] == [0, 1]
    assert closed[1].frames_observed == 0 and closed[1].availability == WindowAvailability.UNAVAILABLE
    assert [(e.previous, e.current) for e in transitions] == [
        (StreamHealth.NO_SIGNAL, StreamHealth.LIVE), (StreamHealth.LIVE, StreamHealth.STALLED),
        (StreamHealth.STALLED, StreamHealth.DOWN),
    ]
    assert "silence" in transitions[1].reason or "no frame" in transitions[1].reason


def test_silence_ages_out_the_tracks_that_can_no_longer_be_observed():
    eng = _engine()
    _feed(eng, make_frames(30, FPS))
    assert len(eng.store.manager.active_tracks) == 1
    closed = eng.tick(60.0)
    assert len(eng.store.manager.active_tracks) == 0  # finalized (LOST), then evicted once no open window needs it
    assert eng.store.manager.total_created == 1
    assert closed and closed[0].tracks  # the window that closed still saw the track before it was aged out


def test_source_reported_reconnecting_state_is_down_immediately():
    eng = _engine()
    _feed(eng, make_frames(30, FPS))
    eng.tick(3.05, source_state="reconnecting")  # only 0.1 s since the last frame
    assert eng.health == StreamHealth.DOWN
    assert "reconnecting" in eng.health_events[-1].reason


def test_no_signal_before_the_first_frame_is_reported_not_treated_as_live():
    eng = _engine()
    assert eng.health == StreamHealth.NO_SIGNAL
    assert eng.tick(5.0) == [] and eng.health == StreamHealth.NO_SIGNAL
    eng.tick(5.0, source_state="failed")
    assert eng.health == StreamHealth.DOWN


def test_tick_is_harmless_with_a_clock_that_does_not_advance_and_never_goes_backwards():
    eng = _engine()
    _feed(eng, make_frames(30, FPS))
    assert eng.tick(2.9) == [] and eng.tick(1.0) == []
    assert eng.stream_time == pytest.approx(2.9)


# ------------------------------------------- 3. healthy camera, empty road --

def test_a_healthy_camera_observing_an_empty_road_is_observed_free_flow_not_unavailable():
    eng = _engine(script=_empty)
    snaps = _feed(eng, make_frames(250, FPS)) + eng.finish()
    reasoner = WindowReasoner("cam")
    analyses = [reasoner.process(s) for s in snaps]
    assert len(analyses) == 3
    for a in analyses:
        assert a.status == AnalysisStatus.OBSERVED and not a.degraded
        assert a.snapshot.availability == WindowAvailability.OBSERVED and a.snapshot.frames_observed > 0
        assert len(a.snapshot.tracks) == 0
        assert a.aggregation.traffic_state.vehicle_count == 0
        assert a.aggregation.traffic_state.congestion_level == CongestionLevel.FREE_FLOW  # genuinely free


# ------------------------------------------- 4. drops spanning a boundary --

def test_dropped_frames_spanning_a_window_boundary_are_split_between_both_windows():
    eng = _engine()
    frames = [f for f in make_frames(250, FPS) if not (95 <= f.index <= 104)]  # 9.5 s .. 10.4 s lost
    snaps = _feed(eng, frames) + eng.finish()
    assert [s.dropped_frames for s in snaps[:3]] == [5, 5, 0]
    assert [s.availability for s in snaps[:3]] == [
        WindowAvailability.DEGRADED, WindowAvailability.DEGRADED, WindowAvailability.OBSERVED,
    ]
    assert sum(s.dropped_frames for s in snaps) == eng.stats()["frames_dropped"] == 10


def test_source_reported_gap_is_attributed_by_time_too():
    eng = _engine()
    frames = make_frames(95, FPS) + [Frame(95 + j, 10.5 + j / FPS, TINY, gap_before=(10 if j == 0 else 0)) for j in range(100)]  # a live source: index is per DELIVERED frame
    snaps = _feed(eng, frames) + eng.finish()
    assert [s.dropped_frames for s in snaps[:2]] == [5, 5]


def test_normal_complete_recording_is_fully_observed_and_eof_is_not_an_outage():
    eng = _engine()
    snaps = _feed(eng, make_frames(250, FPS)) + eng.finish()
    assert [s.availability for s in snaps] == [WindowAvailability.OBSERVED] * 3
    assert snaps[2].final_partial and snaps[2].coverage == pytest.approx(1.0)
    assert all(s.dropped_frames == 0 and s.silent_s == 0.0 for s in snaps)


def test_brief_gaps_below_the_silence_threshold_do_not_make_a_window_unavailable():
    eng = _engine()
    frames = [f for f in make_frames(100, FPS) if not (40 <= f.index < 45)]  # a 0.6 s hole
    s = (_feed(eng, frames) + eng.finish())[0]
    assert s.silent_s == 0.0 and s.dropped_frames == 5 and s.availability == WindowAvailability.DEGRADED


def test_a_partially_silent_window_is_degraded_with_time_based_coverage():
    eng = _engine()
    frames = [f for f in make_frames(100, FPS) if not (30 <= f.index < 60)]  # 3 s silence (2.9 .. 6.0 -> 3.1 s gap)
    s = (_feed(eng, frames) + eng.finish())[0]
    assert s.availability == WindowAvailability.DEGRADED and 0.6 < s.coverage < 0.75
    assert s.silent_s == pytest.approx(3.1, abs=0.05)


# ------------------------------------------- 6. reasoner on degraded windows --

def _snapshot(k, n_tracks, availability=WindowAvailability.OBSERVED, coverage=1.0, frames=100, dropped=0):
    start, end = 10.0 * k, 10.0 * (k + 1)
    tracks = tuple(
        VehicleTrack(
            track_id=f"cam:s000-t{k:03d}{i:03d}", camera_id="cam", first_seen=start, last_seen=end - 0.1,
            vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE,
            position_history=[(1.0, 1.0, start), (2.0, 2.0, end - 0.1)],
            speed_estimate=Measurement(value=20.0, error=1.0), motion_space=MotionSpace.IMAGE,
        )
        for i in range(n_tracks)
    )
    return WindowSnapshot(
        window_index=k, window_id=f"cam:w{k:06d}", camera_id="cam", start=start, end=end, tracks=tracks,
        frames_observed=frames, dropped_frames=dropped, sessions=("s000",), closed_at_stream_time=end,
        expected_frames=100, silent_s=(1 - coverage) * 10.0, coverage=coverage, availability=availability,
    )


def test_reasoner_unavailable_window_reasons_about_nothing():
    reasoner = WindowReasoner("cam")
    a = reasoner.process(_snapshot(0, 12, WindowAvailability.UNAVAILABLE, coverage=0.0, frames=0))
    assert a.unavailable and a.degraded and a.aggregation is None and a.result is None and not a.congestion_advanced
    assert "unavailable" in a.reason


def test_reasoner_zero_observed_frames_is_unavailable_even_if_the_snapshot_claims_observed():
    a = WindowReasoner("cam").process(_snapshot(0, 0, WindowAvailability.OBSERVED, frames=0))
    assert a.unavailable


def test_reasoner_lightly_degraded_window_is_flagged_and_still_advances_hysteresis():
    reasoner = WindowReasoner("cam")
    a = reasoner.process(_snapshot(0, 30, WindowAvailability.DEGRADED, coverage=0.95, dropped=3))
    assert a.status == AnalysisStatus.DEGRADED and a.congestion_advanced and a.result is not None
    assert any("lower bounds" in lim for lim in a.result.limitations)
    assert reasoner.congestion_state == CongestionLevel.BUILDING


def test_reasoner_heavily_degraded_window_does_not_advance_hysteresis():
    reasoner = WindowReasoner("cam")
    for k in range(2):
        reasoner.process(_snapshot(k, 30))
    assert reasoner.congestion_state == CongestionLevel.CONGESTED
    # a thin window (60% observed) that happens to look empty must not step CONGESTED -> DISSIPATING
    a = reasoner.process(_snapshot(2, 0, WindowAvailability.DEGRADED, coverage=0.6, frames=60, dropped=40))
    assert a.status == AnalysisStatus.DEGRADED and not a.congestion_advanced
    assert reasoner.congestion_state == CongestionLevel.CONGESTED
    assert any("NOT advanced" in lim for lim in a.result.limitations)


def test_reasoner_observed_window_is_unchanged():
    a = WindowReasoner("cam").process(_snapshot(0, 5))
    assert a.status == AnalysisStatus.OBSERVED and not a.degraded and a.congestion_advanced


# ------------------------------------- liveness through the threaded pipeline --

class _StallingSource:
    """Delivers `n` frames then goes silent until closed; exposes a settable stream clock."""

    def __init__(self, n=50):
        self._frames = make_frames(n, FPS)
        self.now = (n - 1) / FPS
        self.state = "connected"
        self._closed = threading.Event()

    def fps(self):
        return FPS

    def __iter__(self):
        yield from self._frames
        self._closed.wait(timeout=10.0)  # silent camera: no more frames, no end of stream

    def stream_time(self):
        return self.now

    def health(self):
        from anvesh.perception.ingestion import SourceHealth
        return SourceHealth(self.state, 0, len(self._frames), 0, 0, FPS, 0.0)

    def close(self):
        self._closed.set()


def test_pipeline_idle_polling_reports_a_dead_camera_without_waiting_for_another_frame():
    src = _StallingSource()
    eng = _engine()
    got = []
    pipe = StreamingPipeline(src, eng, on_window=got.append, frame_queue_size=4,
                             frame_policy=DropPolicy.DROP_OLDEST, idle_poll_s=0.01)
    t = threading.Thread(target=pipe.run, daemon=True)
    t.start()
    try:
        src.now = 30.0  # the camera has been dark for ~25 s of stream time
        deadline = time.time() + 5.0
        while time.time() < deadline and len(got) < 3:
            time.sleep(0.01)
        assert len(got) == 3 and all(s.unavailable or s.window_index == 0 for s in got)
        assert got[1].frames_observed == 0 and got[1].availability == WindowAvailability.UNAVAILABLE
        assert eng.health == StreamHealth.DOWN
        src.state = "reconnecting"
    finally:
        pipe.stop()
        t.join(timeout=5.0)
    assert not t.is_alive()


def test_pipeline_without_idle_polling_behaves_exactly_as_before_for_files():
    eng = _engine()
    got = []
    pipe = StreamingPipeline(ListFrameSource(make_frames(250, FPS)), eng, on_window=got.append)
    pipe.run()
    assert [s.availability for s in got] == [WindowAvailability.OBSERVED] * 3


def test_rtsp_source_exposes_its_stream_clock():
    clock = FakeClock()

    class Cap:
        def __init__(self):
            self.n = 0

        def isOpened(self):
            return True

        def read(self):
            self.n += 1
            clock.t += 0.1
            return (self.n <= 3), TINY

        def release(self):
            pass

        def get(self, prop):
            return FPS

    src = RtspFrameSource("rtsp://x", nominal_fps=FPS, capture_factory=lambda u: Cap(), clock=clock,
                          sleep=clock.sleep, max_reconnects=0)
    assert src.stream_time() is None  # before the first frame
    it = iter(src)
    first = next(it)
    assert first.timestamp == 0.0 and src.stream_time() == pytest.approx(0.0)
    clock.t += 5.0
    assert src.stream_time() == pytest.approx(5.0)  # advances with the clock even while no frame arrives


# ---------------------------------------- idle polling must not alter normal flow --

def test_idle_ticks_between_ordinary_frames_change_nothing():
    """The pipeline polls whenever its queue is empty -- most of the time in normal live flow.
    Ticks at last-frame + 0.01 s / + 0.5 s (below the silence threshold) must leave windows,
    track occlusion states and stop episodes EXACTLY as a run with no ticks."""
    def run(ticks):
        eng = _engine()
        snaps = []
        for f in make_frames(250, FPS):
            snaps.extend(eng.process_frame(f))
            if ticks:
                snaps.extend(eng.tick(f.timestamp + 0.01))
                snaps.extend(eng.tick(f.timestamp + 0.5))
        snaps.extend(eng.finish())
        return snaps, eng

    plain, e_plain = run(False)
    ticked, e_ticked = run(True)
    assert ticked == plain  # frozen-dataclass equality: tracks (incl. occlusion), coverage, drops, stop episodes
    assert [s.availability for s in ticked] == [WindowAvailability.OBSERVED] * 3
    assert all(t.occlusion_state.value == "visible" for s in ticked[:2] for t in s.tracks)  # never aged to PARTIAL
    assert e_ticked.health == StreamHealth.LIVE and len(e_ticked.health_events) == 1


def test_a_tick_past_the_silence_threshold_does_act():
    eng = _engine()
    _feed(eng, make_frames(55, FPS))
    assert eng.tick(5.4 + 0.9) == [] and eng.store.manager.active_tracks  # 0.9 s < 1 s threshold: untouched
    eng.tick(5.4 + 1.2)  # 1.2 s of silence: now genuine
    assert eng.store.manager.active_tracks == {} or all(
        t.occlusion_state.value != "visible" for t in eng.store.manager.active_tracks.values()
    )
