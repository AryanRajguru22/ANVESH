"""M-A.1 Fix 5: stopped-vehicle episodes across window boundaries.

The reproduced defect: a vehicle stopped from 7.5 s to 12.5 s (5 s >= the 3 s dwell
threshold) was reported by NEITHER window [0,10) nor [10,20), because each window
clipped the track to 2.5 s.
"""

from __future__ import annotations

import pytest

from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.safety.stop_episodes import (
    IMAGE_BASIS, METRIC_BASIS, EpisodePhase, StopEpisodeTracker, build_stop_event, episode_views,
    stop_episodes_from_samples,
)
from anvesh.safety.stopped_vehicle import StoppedVehicleThresholds, classify_stopped_vehicle
from anvesh.perception.schema_conversion import build_vehicle_track
from anvesh.storage.schemas import CameraRole, ConfidenceTier, MotionSpace
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.reasoning import WindowReasoner
from anvesh.streaming.windows import WindowAvailability, WindowSpec
from tests.streaming.support import ScriptedTracker, make_frames

FPS = 10.0
TID = "cam:s000-t00001"


def _drive(stop_from, stop_to, x0=100.0, speed=50.0):
    """x(t): moves at `speed` px/s, is stationary on [stop_from, stop_to), then moves on."""
    def x_at(t):
        if t < stop_from:
            return x0 + speed * t
        if t < stop_to:
            return x0 + speed * stop_from
        return x0 + speed * stop_from + speed * (t - stop_to)
    return x_at


def _run_engine(stop_from, stop_to, seconds=40.0, **engine_kw):
    x_at = _drive(stop_from, stop_to)
    eng = StreamingPerceptionEngine(
        "cam", ScriptedTracker(lambda i: [(1, (x_at(i / FPS), 50.0))]), FPS, windows=WindowSpec(10.0), run_id=None, **engine_kw
    )
    reasoner = WindowReasoner("cam")
    snaps = []
    for f in make_frames(int(seconds * FPS), FPS):
        snaps.extend(eng.process_frame(f))
    snaps.extend(eng.finish())
    return eng, [reasoner.process(s) for s in snaps]


def _events(analyses):
    return [e for a in analyses for e in a.result.safety_events]


def _phases(analyses, k):
    return [(v.phase, v.episode.start, v.episode.end) for v in analyses[k].stop_episodes]


# ------------------------------------------------ the reproduced regression --

def test_a_stop_crossing_a_window_boundary_is_detected_exactly_once():
    _, analyses = _run_engine(7.5, 12.5)
    events = _events(analyses)
    assert len(events) == 1  # was 0 (missed in BOTH windows)
    # the result itself carries the lifecycle: opened in w1, nothing yet in w0
    assert [v.phase for v in analyses[1].result.stop_episodes] == [EpisodePhase.OPENED]
    assert analyses[0].result.stop_episodes == ()
    e = events[0]
    assert e.event_type == "stopped_vehicle" and e.window == (7.5, 12.5)
    assert analyses[0].result.safety_events == ()  # w0 alone saw only 2.5 s of the stop ...
    assert analyses[1].result.safety_events == (e,)  # ... and the episode opens in w1, once


def test_neither_half_of_a_boundary_stop_is_long_enough_alone_yet_the_episode_qualifies():
    eng, analyses = _run_engine(7.5, 12.5)
    assert analyses[0].snapshot.tracks and analyses[1].snapshot.tracks
    # the clipped per-window classifier (legacy path) still sees < 3 s in each window ...
    for a in analyses[:2]:
        for tr in a.snapshot.tracks:
            assert classify_stopped_vehicle(tr).status.value != "stopped"
    # ... but the episode follows the run across the boundary
    assert len(_events(analyses)) == 1


def test_the_episode_agrees_with_the_unclipped_whole_track_classification():
    x_at = _drive(7.5, 12.5)
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(lambda i: [(1, (x_at(i / FPS), 50.0))]), FPS, archive=True, run_id=None)
    for f in make_frames(400, FPS):
        eng.process_frame(f)
    eng.finish()
    whole = classify_stopped_vehicle(build_vehicle_track("cam", eng.store.manager.all_tracks()[0]))
    assert whole.status.value == "stopped"
    (ep,) = eng._stops.episodes()
    assert (ep.start, ep.end) == whole.dwell_window


# ------------------------------------------------ lifecycle across windows --

def test_stop_beginning_before_a_window_and_persisting_across_several_windows_is_one_incident():
    _, analyses = _run_engine(2.0, 33.0, seconds=45.0)  # confirmed in w0; stationary through w1..w3; moves in w3
    events = _events(analyses)
    assert len(events) == 1  # NOT one per window
    assert _phases(analyses, 0)[0][0] == EpisodePhase.OPENED
    assert _phases(analyses, 1)[0][0] == EpisodePhase.CONTINUING
    assert _phases(analyses, 2)[0][0] == EpisodePhase.CONTINUING
    assert _phases(analyses, 3)[0][0] == EpisodePhase.CLOSED and _phases(analyses, 3)[0][2] == pytest.approx(33.0, abs=0.2)
    assert analyses[1].result.safety_events == () and analyses[3].result.safety_events == ()
    # the same stable id reaches every window
    assert len({v.episode.episode_id for a in analyses[:4] for v in a.stop_episodes}) == 1
    assert events[0].event_id == analyses[0].stop_episodes[0].episode.episode_id


def test_stop_ending_after_a_window_stays_open_in_that_window():
    _, analyses = _run_engine(5.0, 18.0)
    assert _phases(analyses, 0)[0][0] == EpisodePhase.OPENED and _phases(analyses, 0)[0][2] is None  # still open at w0's close
    assert _phases(analyses, 1)[0][0] == EpisodePhase.CLOSED
    assert len(_events(analyses)) == 1


def test_a_stop_that_never_reaches_the_dwell_threshold_is_not_an_incident():
    _, analyses = _run_engine(7.0, 9.0)  # 2 s
    assert _events(analyses) == [] and all(a.stop_episodes == () for a in analyses)


def test_two_separate_stops_by_the_same_tracklet_are_two_incidents_with_distinct_ids():
    t = StopEpisodeTracker("cam")
    xs = lambda s: (100.0 + 50.0 * s) if s < 4 else (300.0 if s < 8 else 300.0 + 50.0 * (s - 8)) if s < 11 else (450.0 if s < 15 else 450.0 + 50.0 * (s - 15))  # noqa: E731
    for i in range(0, 200):
        s = i / FPS
        t.observe(TID, MotionSpace.IMAGE, s, xs(s), 0.0)
    t.finish()
    eps = t.episodes()
    assert len(eps) == 2 and len({e.episode_id for e in eps}) == 2


def test_a_silence_longer_than_max_gap_breaks_the_run():
    t = StopEpisodeTracker("cam")
    for i in range(0, 25):  # stationary 0..2.4 s
        t.observe(TID, MotionSpace.IMAGE, i / FPS, 10.0, 10.0)
    for i in range(40, 75):  # 1.5 s silence, then stationary again 4.0..7.4 s
        t.observe(TID, MotionSpace.IMAGE, i / FPS, 10.0, 10.0)
    t.finish()
    eps = t.episodes()
    assert len(eps) == 1 and eps[0].start == pytest.approx(4.0)  # the 2.4 s run never qualified; the later 3 s one did


def test_no_stop_for_a_moving_vehicle():
    _, analyses = _run_engine(100.0, 101.0)  # never stops within the clip
    assert _events(analyses) == []


# ------------------------------------------------ image vs metric distinction --

def test_image_space_episodes_are_labelled_as_a_pixel_heuristic_never_a_real_world_speed():
    _, analyses = _run_engine(7.5, 12.5)
    (view,) = [v for a in analyses for v in a.stop_episodes if v.phase == EpisodePhase.OPENED]
    ep = view.episode
    assert ep.motion_space == MotionSpace.IMAGE and ep.speed_basis == IMAGE_BASIS
    assert ep.speed_threshold == StoppedVehicleThresholds().image_speed_threshold_px_per_s
    assert build_stop_event(view).motion_space == MotionSpace.IMAGE


def test_metric_and_image_thresholds_are_applied_to_their_own_space_only():
    th = StoppedVehicleThresholds()
    slow = 2.0  # units/s: under the 5 px/s image heuristic, but OVER the 0.5 m/s metric limit
    img, wrd = StopEpisodeTracker("cam"), StopEpisodeTracker("cam")
    for i in range(80):
        s = i / FPS
        img.observe("c:a", MotionSpace.IMAGE, s, slow * s, 0.0)
        wrd.observe("c:b", MotionSpace.WORLD, s, slow * s, 0.0)
    img.finish(); wrd.finish()
    assert len(img.episodes()) == 1 and img.episodes()[0].speed_basis == IMAGE_BASIS   # 2 px/s <= 5 px/s heuristic
    assert wrd.episodes() == []                                                        # 2 m/s  >  0.5 m/s metric limit
    wrd2 = StopEpisodeTracker("cam")
    for i in range(80):
        wrd2.observe("c:b", MotionSpace.WORLD, i / FPS, 0.3 * i / FPS, 0.0)  # 0.3 m/s
    wrd2.finish()
    (ep,) = wrd2.episodes()
    assert ep.speed_basis == METRIC_BASIS and ep.speed_threshold == th.world_speed_threshold_m_per_s
    assert build_stop_event(episode_views([ep], 0.0, 10.0)[0]).motion_space == MotionSpace.WORLD


def test_confidence_grows_with_dwell_and_is_never_high_for_a_just_qualifying_stop():
    short = stop_episodes_from_samples("cam", {TID: (MotionSpace.IMAGE, [(i / FPS, 0.0, 0.0) for i in range(35)])})[0]
    long = stop_episodes_from_samples("cam", {TID: (MotionSpace.IMAGE, [(i / FPS, 0.0, 0.0) for i in range(80)])})[0]
    assert short.confidence_tier != ConfidenceTier.HIGH and long.confidence_tier == ConfidenceTier.HIGH


# ------------------------------------------------ events, reasoner, parity --

def test_only_the_opening_window_yields_a_safety_event():
    ep = stop_episodes_from_samples("cam", {TID: (MotionSpace.IMAGE, [(i / FPS, 0.0, 0.0) for i in range(300)])})[0]
    phases = [(w, build_stop_event(v)) for w in (0.0, 10.0, 20.0) for v in episode_views([ep], w, w + 10.0)]
    assert [p[0] for p in phases] == [0.0, 10.0, 20.0]
    assert phases[0][1] is not None and phases[1][1] is None and phases[2][1] is None


def test_streaming_episodes_equal_post_hoc_episodes_from_the_same_samples():
    x_at = _drive(7.5, 12.5)
    samples = [(i / FPS, x_at(i / FPS), 50.0) for i in range(400)]
    posthoc = stop_episodes_from_samples("cam", {"cam:s000-t00000": (MotionSpace.IMAGE, samples)})
    _, analyses = _run_engine(7.5, 12.5)
    # closed episodes are evicted from the live tracker once no open window needs them (bounded
    # memory), so compare with what the windows REPORTED: the episode as it stood when it closed
    live = [v.episode for a in analyses for v in a.stop_episodes if v.phase == EpisodePhase.OPENED]
    key = lambda e: (e.start, e.confirmed_at, e.last_low_speed, e.motion_space)  # noqa: E731
    assert [key(e) for e in posthoc] == [key(e) for e in live]
    assert posthoc[0].end == 12.5 and any(
        v.episode.end == 12.5 for a in analyses for v in a.stop_episodes if v.phase == EpisodePhase.OPENED
    )


def test_legacy_per_window_classification_is_unchanged_when_no_episodes_are_supplied():
    _, analyses = _run_engine(5.0, 30.0)  # stationary for the whole of w1 = [10, 20)
    snap = analyses[1].snapshot
    ci = CameraWindowInput(camera_id="cam", role=CameraRole.UPSTREAM, aggregation=analyses[1].aggregation, tracks=list(snap.tracks))
    res = run_corridor_window(snap.window, (ci,))  # no stop_episodes -> per-window classification, as before M-A.1
    assert any(e.event_type == "stopped_vehicle" for e in res.safety_events)  # a stop wholly inside the window is still found
    assert res.stop_episodes == ()


def test_a_stop_confirmed_in_an_unavailable_window_is_not_lost():
    from anvesh.perception.types import Frame
    from tests.streaming.support import TINY
    x_at = _drive(1.0, 30.0)
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(lambda i: [(1, (x_at(i / FPS), 50.0))]), FPS,
                                    windows=WindowSpec(10.0), run_id=None)
    snaps = []
    for i in range(50):  # observe 0..4.9 s: the stop is confirmed at 4.0 s, then the camera dies
        snaps.extend(eng.process_frame(Frame(i, i / FPS, TINY)))
    snaps.extend(eng.tick(25.0))
    reasoner = WindowReasoner("cam")
    a0 = reasoner.process(snaps[0])
    assert a0.snapshot.availability == WindowAvailability.UNAVAILABLE  # only 4.9 s of the 10 s window were observed
    assert a0.result is None and len(a0.stop_events) == 1  # the observed stop survives the outage


# ------------------------------------------------ bounded state --

def test_new_streaming_state_stays_bounded_over_a_long_churning_stream():
    """Every 8 s a new vehicle appears, stops for 4 s (an episode) and leaves; 1000 s, with a
    dropped frame every 50 frames. Nothing the M-A.1 changes added may grow with stream length."""
    def script(i):
        t = i / FPS
        k, phase = int(t // 8), t % 8
        if phase < 2:
            return [(k + 1, (100.0 + 50.0 * phase, 50.0))]
        if phase < 6:
            return [(k + 1, (200.0, 50.0))]          # stationary 4 s -> an episode
        return [(k + 1, (200.0 + 50.0 * (phase - 6), 50.0))]

    eng = StreamingPerceptionEngine("cam", ScriptedTracker(script), FPS, windows=WindowSpec(10.0), run_id=None)
    sizes = []
    total_episodes_opened = 0
    for f in make_frames(10000, FPS):
        if f.index % 50 == 49:
            continue  # a lost frame
        for snap in eng.process_frame(f):
            total_episodes_opened += sum(1 for v in snap.stop_episodes if v.phase == EpisodePhase.OPENED)
        if f.index % 1000 == 0:
            sizes.append((len(eng._stops._closed), eng._stops.tracked_count, len(eng._frame_times), len(eng._drop_events)))
    assert total_episodes_opened >= 100  # the scenario really did produce many episodes
    for closed, tracked, frame_times, drops in sizes:
        assert closed <= 10 and tracked <= 5 and frame_times <= 250 and drops <= 40
    assert max(s[0] for s in sizes[len(sizes) // 2:]) <= max(s[0] for s in sizes[: len(sizes) // 2]) + 3  # no upward trend


# ------------------------------------------------ a closure learned one window late ----

def test_an_episode_that_ended_in_one_window_but_is_closed_in_the_next_is_reported_closed_once():
    """The track vanishes at 19.4 s, 0.6 s before window [10,20) ends: not yet silent for max_gap (1 s), so the episode is still open in
    that window. It closes while window [20,30) is processed with end 19.4 < 20, which `episode_views` alone would drop: the incident raised for it
    would stay 'still open' for the rest of the stream (seen on the demo footage)."""
    tr = StopEpisodeTracker("cam")
    for i in range(195):  # 0.0 .. 19.4 s at 10 fps; moving for 5 s, then stationary
        t = i / FPS
        tr.observe(TID, MotionSpace.IMAGE, t, 100.0 + 50.0 * min(t, 5.0), 50.0)
    tr.advance(20.0)
    w1 = tr.views(10.0, 20.0)
    assert [(v.phase, v.episode.end) for v in w1] == [(EpisodePhase.CONTINUING, None)]
    tr.advance(30.0)  # silent for 10.6 s: closed, end = the last low-speed sample
    tr.evict_before(20.0)  # what the engine does after every frame (floor = the oldest open window's start); must not drop the owed report
    w2 = tr.views(20.0, 30.0)
    assert len(w2) == 1 and w2[0].phase == EpisodePhase.CLOSED and w2[0].episode.closed_reason == "track_ended"
    assert w2[0].episode.end == pytest.approx(19.4) and w2[0].episode.end < 20.0
    assert tr.views(30.0, 40.0) == ()  # reported once, not forever
    assert tr.evict_before(30.0) == 1 and tr.episodes() == []  # and only then is it evicted


def test_an_episode_reported_closed_in_its_own_window_is_not_reported_again():
    tr = StopEpisodeTracker("cam")
    for i in range(100):  # stationary 0..9.9 s, then it moves off
        tr.observe(TID, MotionSpace.IMAGE, i / FPS, 100.0, 50.0)
    for i in range(100, 120):
        tr.observe(TID, MotionSpace.IMAGE, i / FPS, 100.0 + 80.0 * (i / FPS - 10.0), 50.0)
    tr.advance(10.0)
    w1 = tr.views(0.0, 10.0)
    assert [v.phase for v in w1] == [EpisodePhase.OPENED] or [v.phase for v in w1] == [EpisodePhase.CONTINUING]
    tr.advance(20.0)
    w2 = tr.views(10.0, 20.0)
    assert len(w2) <= 1 and all(v.phase == EpisodePhase.CLOSED for v in w2)
    assert tr.views(20.0, 30.0) == ()
