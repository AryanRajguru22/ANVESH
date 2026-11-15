"""Hardening C: the OPT-IN displacement stop criterion (jitter-robust, gap-bridging); default behaviour untouched."""

import math

import pytest

from anvesh.safety.stop_episodes import (
    IMAGE_BASIS,
    IMAGE_DISPLACEMENT_BASIS,
    METRIC_DISPLACEMENT_BASIS,
    DisplacementCriterion,
    EpisodePhase,
    StopEpisodeTracker,
    build_stop_event,
    stop_episodes_from_samples,
)
from anvesh.safety.stopped_vehicle import StoppedVehicleThresholds
from anvesh.storage.schemas import MotionSpace
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.reasoning import WindowReasoner
from anvesh.streaming.windows import WindowSpec
from tests.streaming.support import ScriptedTracker, make_frames

IMG, WORLD = MotionSpace.IMAGE, MotionSpace.WORLD
TID = "cam:s0-t1"
DISP = DisplacementCriterion()  # R=7.5 px over W=3 s (== 5 px/s), exit 1.0 s, bridge 3.0 s; dwell 3 s, max_gap 1 s


def _run(samples, displacement=DISP, space=IMG, finish=True, advance_to=None):
    tr = StopEpisodeTracker("cam", None, displacement)
    for t, x, y in samples:
        tr.observe(TID, space, t, x, y)
        tr.advance(t)
    if advance_to is not None:
        tr.advance(advance_to)
    if finish:
        tr.finish()
    return tr.episodes()


def _still(fps, seconds, jitter=0.0, x0=200.0, y0=150.0, t0=0.0):
    n = int(round(seconds * fps))
    return [(t0 + i / fps, x0 + (jitter if i % 2 == 0 else -jitter), y0) for i in range(n)]


# ------------------------------------------------------------------ default behaviour is untouched ----


def test_default_tracker_is_still_the_step_speed_rule():
    eps = _run(_still(10, 20), displacement=None)
    assert len(eps) == 1 and eps[0].criterion == "step_speed" and eps[0].speed_basis == IMAGE_BASIS
    assert eps[0].bridged_s == 0.0 and eps[0].bridged_gaps == 0


@pytest.mark.parametrize("fps, jitter", [(10, 0.3), (30, 0.1), (30, 0.2)])
def test_the_repro_default_rule_misses_a_jittery_parked_car_displacement_does_not(fps, jitter):
    samples = _still(fps, 20, jitter)
    assert _run(samples, displacement=None) == []  # the confirmed defect, still present by default
    eps = _run(samples)
    assert len(eps) == 1 and eps[0].start == 0.0 and eps[0].criterion == "displacement"
    assert eps[0].speed_basis == IMAGE_DISPLACEMENT_BASIS and eps[0].speed_threshold == DISP.image_radius_px


# ------------------------------------------------------------------ frame-rate independence ----------


@pytest.mark.parametrize("fps", [5, 10, 15, 25, 30, 60])
def test_jitter_does_not_change_the_outcome_with_frame_rate(fps):
    eps = _run(_still(fps, 20, jitter=2.0))
    assert len(eps) == 1 and eps[0].start == 0.0 and eps[0].confirmed_at == pytest.approx(3.0, abs=1.0 / fps + 1e-9)


# ------------------------------------------------------------------ radius / dwell boundary -----------


def _crawl(speed_px_s, seconds=20.0, fps=10):
    return [(i / fps, 100.0 + speed_px_s * i / fps, 100.0) for i in range(int(seconds * fps))]


def test_defaults_are_derived_from_the_default_rules_thresholds():
    th = StoppedVehicleThresholds()
    assert DISP.anchor_window_s == th.dwell_seconds
    # constant speed v sits v*W/2 from the trailing mean => stationary iff v <= 2R/W: the same speeds as the default rule
    assert 2 * DISP.image_radius_px / DISP.anchor_window_s == pytest.approx(th.image_speed_threshold_px_per_s)
    assert 2 * DISP.world_radius_m / DISP.anchor_window_s == pytest.approx(th.world_speed_threshold_m_per_s)


def test_movers_are_never_a_stop_and_the_boundary_is_2r_over_window():
    assert _run(_crawl(20.0)) == []
    assert _run(_crawl(10.0)) == []
    assert _run(_crawl(6.0)) == []
    assert _run(_crawl(5.1)) == []      # just above 2R/W = 5 px/s: never confirms
    crawl = _run(_crawl(4.9))           # just below: ONE continuous episode for the whole crawl, not a chain
    assert len(crawl) == 1 and crawl[0].start == 0.0 and crawl[0].end == pytest.approx(19.9)


@pytest.mark.parametrize("fps", [5, 10, 30, 60])
def test_the_speed_boundary_does_not_depend_on_the_frame_rate(fps):
    # testing the new sample against the window that includes it makes the boundary exactly 2R/W = 5 px/s
    assert _run(_crawl(5.1, fps=fps)) == []
    eps = _run(_crawl(4.9, fps=fps))
    assert len(eps) == 1 and eps[0].start == 0.0


def test_a_slow_crawl_is_one_episode_not_a_chain_of_incidents():
    eps = _run(_crawl(3.0, seconds=60.0))
    assert len(eps) == 1 and eps[0].episode_id.endswith(":0.0") and eps[0].end == pytest.approx(59.9)


def test_world_space_uses_metres_and_its_own_basis():
    d = DisplacementCriterion(world_radius_m=0.5)
    parked = _run([(i / 10, 5.0 + (0.1 if i % 2 else -0.1), 2.0) for i in range(200)], d, WORLD)
    assert len(parked) == 1 and parked[0].speed_basis == METRIC_DISPLACEMENT_BASIS and parked[0].speed_threshold == 0.5
    assert parked[0].motion_space == WORLD
    assert _run([(i / 10, 5.0 + 1.0 * i / 10, 2.0) for i in range(200)], d, WORLD) == []  # 1 m/s is moving


# ------------------------------------------------------------------ outliers vs real departure ---------


def _with_excursion(duration_s, away_px=60.0, at=10.0, total=30.0, fps=10):
    out = []
    for i in range(int(total * fps)):
        t = i / fps
        x = 200.0 + (away_px if at <= t < at + duration_s else 0.0)
        out.append((t, x, 150.0))
    return out


def test_a_short_noise_excursion_does_not_split_a_long_stop():
    eps = _run(_with_excursion(0.3))  # 3 samples far away, shorter than exit_seconds
    assert len(eps) == 1 and eps[0].start == 0.0 and eps[0].closed_reason == "stream_ended"


def test_a_sustained_departure_closes_the_run_and_a_new_stop_elsewhere_is_a_new_episode():
    eps = _run(_with_excursion(15.0))  # parked, moved 60 px away for 15 s (stops THERE), came back
    assert len(eps) == 3
    first, second, third = eps
    assert first.start == 0.0 and first.closed_reason == "moved" and first.end == pytest.approx(9.9, abs=0.11)
    assert second.start == pytest.approx(10.0, abs=0.01) and second.start != first.start
    assert len({e.episode_id for e in eps}) == 3


@pytest.mark.parametrize("excursion, closes", [(0.8, False), (0.9, False), (1.0, True), (1.5, True)])
def test_the_exit_threshold_boundary_is_exit_seconds(excursion, closes):
    # excursion samples span [10.0, 10.0 + excursion); the run closes iff the out-of-radius time since the last
    # in-radius sample (9.9) reaches exit_seconds (1.0 s) before the vehicle is back
    eps = _run(_with_excursion(excursion))
    assert (len(eps) > 1) is closes


# ------------------------------------------------------------------ gap bridging / id preservation -----


def _with_gap(gap_s, x_after=200.0, fps=10, before=8.0, after=10.0):
    pre = [(i / fps, 200.0, 150.0) for i in range(int(before * fps))]
    t_resume = pre[-1][0] + gap_s
    post = [(round(t_resume + i / fps, 9), x_after, 150.0) for i in range(int(after * fps))]
    return pre + post


def test_a_short_silence_over_an_unmoved_vehicle_is_bridged_and_the_id_survives():
    eps = _run(_with_gap(1.2))  # 1.2 s > max_gap 1.0 s, <= bridge 3.0 s
    assert len(eps) == 1
    e = eps[0]
    assert e.start == 0.0 and e.episode_id.endswith("stopped_vehicle:0.0")
    assert e.bridged_gaps == 1 and e.bridged_s == pytest.approx(1.2)
    # the default rule would have closed at the gap and minted a second id for the same stop
    assert len(_run(_with_gap(1.2), displacement=None)) == 2


def test_without_bridging_the_gap_splits_the_stop():
    eps = _run(_with_gap(1.2), displacement=DisplacementCriterion(bridge_gap_s=0.0))
    assert len(eps) == 2 and eps[0].closed_reason == "gap" and eps[1].episode_id != eps[0].episode_id


@pytest.mark.parametrize("gap, bridged", [(1.0, True), (3.0, True), (3.1, False)])
def test_the_bridge_limit_boundary(gap, bridged):
    # gap == max_gap (1.0 s) is not a silence at all; gap == bridge limit (3.0 s) is bridged; just over is not
    eps = _run(_with_gap(gap))
    assert (len(eps) == 1) is bridged
    if bridged and gap > 1.0:
        assert eps[0].bridged_gaps == 1


def test_a_vehicle_that_reappears_elsewhere_after_a_gap_is_not_bridged():
    eps = _run(_with_gap(1.5, x_after=300.0))
    assert len(eps) == 2 and eps[0].closed_reason == "gap" and eps[0].end == pytest.approx(7.9)


def test_bridged_time_is_never_counted_as_observed_dwell():
    # 1.4 s seen, 2.0 s dark, then 1.4 s seen: 2.8 s observed < 3 s dwell, although the wall span is 4.8 s
    s = [(i / 10, 100.0, 100.0) for i in range(15)] + [(3.4 + i / 10, 100.0, 100.0) for i in range(15)]
    assert _run(s) == []
    s2 = s + [(4.8 + i / 10, 100.0, 100.0) for i in range(1, 4)]  # a little more observed time
    eps = _run(s2)
    assert len(eps) == 1 and eps[0].confirmed_at > 3.4 and eps[0].bridged_s == pytest.approx(2.0)


def test_advance_keeps_a_candidate_alive_over_the_bridge_horizon_then_drops_it():
    tr = StopEpisodeTracker("cam", None, DISP)
    for t, x, y in _still(10, 8):
        tr.observe(TID, IMG, t, x, y)
    last = 7.9
    tr.advance(last + 2.9)
    assert tr.tracked_count == 1 and tr.episodes()[0].is_open
    tr.advance(last + 3.2)
    assert tr.tracked_count == 0 and tr.episodes()[0].closed_reason == "track_ended"


def test_without_a_displacement_criterion_advance_still_uses_max_gap():
    tr = StopEpisodeTracker("cam", None, None)
    for t, x, y in _still(10, 8):
        tr.observe(TID, IMG, t, x, y)
    tr.advance(7.9 + 1.2)
    assert tr.tracked_count == 0


# ------------------------------------------------------------------ parameters -----------------------


@pytest.mark.parametrize("kw", [
    dict(image_radius_px=0), dict(image_radius_px=-1), dict(image_radius_px=float("nan")), dict(image_radius_px=float("inf")),
    dict(world_radius_m=0), dict(exit_seconds=0), dict(exit_seconds=-0.1), dict(bridge_gap_s=-0.1),
    dict(bridge_gap_s=float("nan")), dict(image_radius_px=True), dict(bridge_gap_s=None),
])
def test_invalid_criterion_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        DisplacementCriterion(**kw)


def test_bridge_gap_zero_is_valid_and_disables_bridging():
    assert DisplacementCriterion(bridge_gap_s=0).bridge_gap_s == 0


# ------------------------------------------------------------------ streaming windows + parity -------


def _jittery_script(i):
    return [(1, (200.0 + (0.3 if i % 2 else -0.3), 150.0))]  # a parked car, ±0.3 px at 10 fps


def _stream(displacement):
    eng = StreamingPerceptionEngine("camS", ScriptedTracker(_jittery_script), fps=10.0, windows=WindowSpec(window_s=10.0),
                                    stop_displacement=displacement)
    snaps = []
    for f in make_frames(400, 10.0):
        snaps += eng.process_frame(f)
    return eng, snaps + eng.finish()


def test_a_parked_car_is_one_incident_across_windows_with_the_criterion_and_missed_without():
    _, off = _stream(None)
    assert all(not s.stop_episodes for s in off)  # the default rule: no episode at all (confirmed defect)

    eng, on = _stream(DISP)
    phases = [[v.phase for v in s.stop_episodes] for s in on]
    flat = [p for ps in phases for p in ps]
    assert flat.count(EpisodePhase.OPENED) == 1 and flat.count(EpisodePhase.CLOSED) <= 1  # the same incident throughout
    assert set(flat) <= {EpisodePhase.OPENED, EpisodePhase.CONTINUING, EpisodePhase.CLOSED}
    ids = {v.episode.episode_id for s in on for v in s.stop_episodes}
    assert len(ids) == 1
    reasoner = WindowReasoner("camS", calibration_profile=None)
    events = [e for s in on for e in reasoner.process(s).stop_events]
    assert len(events) == 1 and events[0].event_id in ids


def test_stream_and_post_hoc_episodes_agree_with_jitter_a_gap_and_an_outlier():
    samples = []
    for i in range(0, 400):
        t = i / 10
        if 15.0 <= t < 16.3:
            continue  # a 1.3 s dropout
        x = 200.0 + (0.4 if i % 2 else -0.4) + (80.0 if i == 250 else 0.0)  # one wild sample
        samples.append((t, x, 150.0))
    stream = StopEpisodeTracker("cam", None, DISP)
    for t, x, y in samples:
        stream.observe(TID, IMG, t, x, y)
        stream.advance(t)
    stream.finish()
    post = stop_episodes_from_samples("cam", {TID: (IMG, samples)}, None, DISP)

    def sig(eps):
        return sorted((e.episode_id, e.start, e.confirmed_at, e.end, e.bridged_gaps, round(e.bridged_s, 9)) for e in eps)

    assert sig(stream.episodes()) == sig(post) and len(post) == 1
    assert post[0].bridged_gaps == 1 and post[0].start == 0.0


def test_a_safety_event_is_built_once_and_keeps_the_original_id_after_a_bridged_gap():
    eps = _run(_with_gap(1.2))
    from anvesh.safety.stop_episodes import StopEpisodeView
    event = build_stop_event(StopEpisodeView(eps[0], EpisodePhase.OPENED))
    assert event is not None and event.event_id == eps[0].episode_id and event.motion_space == IMG


def test_thresholds_for_dwell_and_gap_still_come_from_the_shared_thresholds_object():
    th = StoppedVehicleThresholds(dwell_seconds=5.0, max_gap_seconds=0.5)
    tr = StopEpisodeTracker("cam", th, DISP)
    for t, x, y in _still(10, 4.9):
        tr.observe(TID, IMG, t, x, y)
    tr.finish()
    assert tr.episodes() == []  # 4.8 s observed < 5 s dwell
    tr = StopEpisodeTracker("cam", th, DISP)
    for t, x, y in _still(10, 6.0):
        tr.observe(TID, IMG, t, x, y)
    tr.finish()
    assert len(tr.episodes()) == 1
