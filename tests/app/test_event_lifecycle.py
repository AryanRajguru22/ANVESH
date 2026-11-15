"""Congestion-episode lifecycle (close grace) and queue-stop latching, driven with scripted window analyses.

The windows are SIMULATED level sequences, not footage: they show how the lifecycle rules behave, not how often a level is right on a road.
"""

from types import SimpleNamespace

import pytest

from anvesh.app.cameras import ConfigError, parse_camera_config
from anvesh.app.detectors import CongestionDetector, DetectorContext, StoppedVehicleDetector, TrackHistory
from anvesh.safety.stop_episodes import EpisodePhase
from anvesh.storage.schemas import CongestionLevel, MotionSpace

L = {"." : CongestionLevel.FREE_FLOW, "b": CongestionLevel.BUILDING, "C": CongestionLevel.CONGESTED, "d": CongestionLevel.DISSIPATING}


def _ctx(config=None, camera="cam"):
    return DetectorContext(camera, 10.0, parse_camera_config(config or {}), TrackHistory())


def _window(i, level, mean_speed=None, stops=()):
    ts = SimpleNamespace(congestion_level=level, vehicle_count=10, occupancy=0.5, flow_rate=0.2, motion_space=MotionSpace.IMAGE,
                         measured_mean_speed=lambda: None if mean_speed is None else SimpleNamespace(value=mean_speed))
    if mean_speed is not None:
        ts.motion_space = MotionSpace.WORLD
    agg = SimpleNamespace(traffic_state=ts, mean_concurrency=9.0, class_counts={})
    snap = SimpleNamespace(window_id=f"w{i}", start=10.0 * i, end=10.0 * (i + 1))
    return SimpleNamespace(unavailable=False, aggregation=agg, snapshot=snap, degraded=False, result=None, stop_episodes=tuple(stops))


def _run_congestion(pattern, config=None, speeds=None):
    det = CongestionDetector(_ctx(config))
    events = []
    for i, ch in enumerate(pattern):
        events += det.on_window(_window(i, L[ch], None if speeds is None else speeds[i]))
    return det, events


def _by_key(events):
    last = {}
    for e in events:
        last[e.natural_key] = e  # the incident service keeps the latest emission of a natural key
    return list(last.values())


def test_a_short_free_flow_gap_inside_stop_and_go_traffic_is_one_episode_not_two():
    _, ev = _run_congestion(".bCCd..bCCd.....")  # 20 s of free flow between two waves, grace 30 s
    eps = _by_key(ev)
    assert len(eps) == 1
    e = eps[0]
    assert e.closed and e.indicators["interruptions"] == 1
    assert e.start_offset_s == 10.0 and e.end_offset_s == 110.0  # back-dated to the first free-flow window after the last wave
    assert any("merged as one stop-and-go episode" in u for u in e.uncertainty)
    levels = [p["level"] for p in e.indicators["trend"]]
    assert "free_flow" in levels  # the gap is kept in the trend, not hidden


def test_a_free_flow_gap_longer_than_the_grace_closes_the_episode_and_starts_a_new_one():
    _, ev = _run_congestion(".bCd.....bCd....")
    eps = _by_key(ev)
    assert len(eps) == 2 and all(e.closed for e in eps)
    assert [e.end_offset_s for e in eps] == [40.0, 120.0]
    assert all(e.indicators["interruptions"] == 0 for e in eps)


def test_the_grace_is_configurable_and_validated():
    _, ev = _run_congestion(".bCd...bCd.....", {"thresholds": {"congestion_close_grace_s": 60}})
    assert len(_by_key(ev)) == 1
    _, ev = _run_congestion(".bCd...bCd.....", {"thresholds": {"congestion_close_grace_s": 10}})
    assert len(_by_key(ev)) == 2
    for bad in (0, -5, True, "30", None):
        with pytest.raises(ConfigError):
            parse_camera_config({"thresholds": {"congestion_close_grace_s": bad}})


def test_stream_end_inside_the_grace_closes_with_the_back_dated_end():
    det, ev = _run_congestion(".bCd.")  # one free-flow window, then the stream ends
    assert det._open is not None and not ev[-1].closed
    assert ev[-1].summary.endswith("(clearing)")
    fin = det.finish(50.0)
    assert len(fin) == 1 and fin[0].closed and fin[0].end_offset_s == 40.0
    assert "before the close grace period elapsed" in fin[0].indicators["closed_reason"]


def test_stream_end_while_still_congested_closes_at_the_stream_end():
    det, _ = _run_congestion(".bCC")
    fin = det.finish(40.0)
    assert fin[0].end_offset_s == 40.0 and "still open" in fin[0].indicators["closed_reason"]


def test_severity_counts_only_the_time_not_free_flowing_never_the_span():
    # 40 s of congestion, 20 s gap, 40 s of congestion, 3 windows free: span 100 s... made long by gaps must not become "sustained"
    pattern = "bCCC" + ".." + "CCC" * 30 + "..."
    _, ev = _run_congestion(pattern, {"thresholds": {"congestion_close_grace_s": 30}})
    e = _by_key(ev)[0]
    assert e.indicators["non_free_s"] == 10.0 * (4 + 90)
    assert e.severity == "high" and e.indicators["non_free_s"] >= 300
    _, ev = _run_congestion("bCCC" + ".." + "CCC" + "...")
    e = _by_key(ev)[0]
    assert e.severity == "medium" and e.indicators["non_free_s"] == 70.0 and e.indicators["duration_s"] == 90.0


def test_a_congested_label_reached_on_occupancy_while_traffic_moves_fast_says_so():
    speeds = [None] + [17.0] * 5 + [None] * 4
    _, ev = _run_congestion(".bCCCC....", speeds=speeds)
    e = _by_key(ev)[0]
    note = [u for u in e.uncertainty if "occupancy proxy" in u and "not necessarily slow traffic" in u]
    assert note and "61" in note[0]  # 17 m/s ~ 61 km/h, stated as approximate
    # a genuinely slow queue gets no such note
    _, ev = _run_congestion(".bCCCC....", speeds=[None] + [2.0] * 5 + [None] * 4)
    assert not any("not necessarily slow traffic" in u for u in _by_key(ev)[0].uncertainty)


# -- stopped vehicle: the queue reading is latched, and queue stops share a group ---------------------------------------------


def _view(n, start=0.0, end=None, phase=EpisodePhase.CONTINUING):
    ep = SimpleNamespace(episode_id=f"ep{n}", track_id=f"cam:s0-t{n:05d}", motion_space=MotionSpace.IMAGE, start=start, end=end, last_low_speed=start + 20,
                         confirmed_at=start + 6, criterion="displacement", speed_basis="px", speed_threshold=1.0, bridged_gaps=0, is_open=end is None,
                         closed_reason=None if end is None else "moved", confidence_tier=SimpleNamespace(value="medium"), dwell_required=5.0, score=0.8)
    return SimpleNamespace(episode=ep, phase=phase)


def _stop_events(levels, stops_per_window):
    det = StoppedVehicleDetector(_ctx())
    out = []
    for i, (lv, stops) in enumerate(zip(levels, stops_per_window)):
        out += det.on_window(_window(i, L[lv], stops=stops))
    return _by_key(out), out


def test_a_queued_stop_is_not_re_rated_as_an_isolated_obstruction_when_the_queue_clears():
    """The incident is re-emitted every window and the LAST emission is stored. Before the latch, a stop that began in a CONGESTED window
    was stored as 'medium, outside congested traffic' because its closing window read DISSIPATING/FREE_FLOW."""
    v_open = _view(1, start=5.0)
    v_closed = _view(1, start=5.0, end=45.0, phase=EpisodePhase.CLOSED)
    final, out = _stop_events("CCd.", [[v_open], [v_open], [v_closed], [v_closed]])
    assert len(final) == 1
    inc = final[0]
    assert inc.severity == "low" and inc.indicators["queue_member"] and inc.indicators["queue_signal"] == "congestion_level"
    assert "congested" in inc.severity_basis  # the level at which it was judged, not the level of the closing window
    assert all(e.severity == "low" for e in out)


def test_a_lone_stop_in_free_flow_stays_the_obstruction_candidate():
    final, _ = _stop_events("...", [[_view(1, start=2.0)]] * 3)
    inc = final[0]
    assert inc.severity == "medium" and not inc.indicators["queue_member"] and inc.indicators["queue_group_id"] is None


def test_a_stop_while_the_queue_is_still_clearing_is_queue_context():
    final, _ = _stop_events("d", [[_view(1)]])
    assert final[0].severity == "low" and final[0].indicators["queue_signal"] == "congestion_level"


def test_queue_stops_close_in_time_share_one_group_and_a_much_later_one_starts_another():
    views = [_view(i) for i in range(4)]
    final, _ = _stop_events("CCC", [views, views, views])
    ids = {e.indicators["queue_group_id"] for e in final}
    assert len(final) == 4 and len(ids) == 1 and ids != {None}
    assert {e.indicators["queue_group_peak_stationary"] for e in final} == {4}
    det = StoppedVehicleDetector(_ctx())
    det.on_window(_window(0, CongestionLevel.CONGESTED, stops=[_view(1)]))
    later = det.on_window(_window(20, CongestionLevel.CONGESTED, stops=[_view(2, start=190.0)]))
    first_id = det._latch["ep1"]["group"]["id"]
    assert later[0].indicators["queue_group_id"] != first_id  # > 30 s later: a different queue, not the same one
    assert len(det._latch) == 2
