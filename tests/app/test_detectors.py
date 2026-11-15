"""Detector behaviour. Frame-level detectors are driven directly with scripted tracks; the engine-level
tests run the REAL streaming engine on SIMULATED scenarios (never presented as real footage)."""

import numpy as np
import pytest

from anvesh.app.cameras import parse_camera_config
from anvesh.app.capabilities import CAPABILITIES, all_category_status, category_status
from anvesh.app.detectors import (BrakingDetector, OverspeedDetector, CollisionCandidateDetector, DetectorContext, IllegalTurnDetector, Observation, RapidLaneChangeDetector,
                                  RedLightDetector, SignalState, TrackHistory, WrongLaneDetector, WrongWayDetector, build_detectors, iou,
                                  point_in_polygon)
from anvesh.storage.schemas import GeometryValidity, MotionSpace

from .test_session_synthetic import run_scenario

W = 60.0


def box(cx, cy, w=W, h=36.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def obs(tid, cx, cy, cls="car", w=W):
    b = box(cx, cy, w)
    return Observation(tid, cls, b, ((b[0] + b[2]) / 2, b[3]), 0.9)


def make(detector_cls, config=None, signal=None, **ctx_kw):
    cfg = parse_camera_config(config or {})
    ctx = DetectorContext("cam", 10.0, cfg, TrackHistory(), signal=signal, **ctx_kw)
    return detector_cls(ctx), ctx


def drive(det, ctx, frames):
    """frames: list of (t, [Observation]). Returns all events."""
    out = []
    for i, (t, observations) in enumerate(frames):
        ctx.history.update(t, i, observations)
        out += det.on_frame(t, i, observations)
    return out


# -- geometry helpers ----------------------------------------------------------------------

def test_point_in_polygon_and_iou():
    sq = ((0, 0), (10, 0), (10, 10), (0, 10))
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# -- wrong way ------------------------------------------------------------------------------

ZONE = {"direction_zones": [{"zone_id": "east", "polygon": [[0, 100], [640, 100], [640, 340], [0, 340]], "expected": [1, 0]}]}


def test_wrong_way_fires_for_the_vehicle_going_against_the_zone_only():
    det, ctx = make(WrongWayDetector, ZONE)
    frames = []
    for i in range(60):
        t = i / 10
        frames.append((t, [obs("ok", 50 + 60 * t, 200), obs("bad", 500 - 60 * t, 260)]))
    ev = drive(det, ctx, frames)
    assert [e.track_ids for e in ev] == [["bad"]]
    e = ev[0]
    assert e.event_type == "wrong_way" and e.indicators["cos_to_expected"] < -0.9 and e.indicators["space"] == "image"
    assert e.involved[0]["role"] == "observed" and any("image space" in u for u in e.uncertainty)


def test_wrong_way_ignores_short_or_tiny_movement_and_stationary():
    det, ctx = make(WrongWayDetector, ZONE)
    frames = [(i / 10, [obs("jit", 300 - (i % 3), 200), obs("brief", 500 - 40 * (i / 10), 260)]) for i in range(8)]
    assert drive(det, ctx, frames) == []


def test_wrong_way_suppresses_duplicate_after_identity_switch():
    det, ctx = make(WrongWayDetector, ZONE)
    frames = []
    for i in range(80):
        t = i / 10
        tid = "bad-a" if i < 40 else "bad-b"  # the tracker re-labels the same vehicle mid-way
        frames.append((t, [obs(tid, 600 - 60 * t, 260)]))
    ev = drive(det, ctx, frames)
    assert len(ev) == 1


def test_wrong_way_needs_configured_zone_and_capability_says_so():
    cfg = parse_camera_config({})
    assert "direction_zones" in category_status("wrong_way", cfg)["missing"]
    assert category_status("wrong_way", parse_camera_config(ZONE))["available"]
    assert all(d.name != "wrong_way" for d in build_detectors(DetectorContext("c", 10, cfg, TrackHistory())))


# -- collision candidate ----------------------------------------------------------------------

def _collision_frames(stop_after=True, approach=True):
    frames = []
    for i in range(160):
        t = i / 10
        if approach and t < 4.0:
            a, b = -60 + 90 * t, 700 - 90 * t
        else:
            a, b = (300, 340) if approach else (300, 340)
        frames.append((t, [obs("a", a, 250), obs("b", b, 250)]))
    return frames


def test_collision_candidate_requires_approach_contact_and_joint_stop():
    det, ctx = make(CollisionCandidateDetector)
    ev = drive(det, ctx, _collision_frames())
    assert len(ev) == 1
    e = ev[0]
    assert e.event_type == "possible_collision" and sorted(e.track_ids) == ["a", "b"]
    assert {v["role"] for v in e.involved} == {"potentially_involved"}  # never "cause"
    assert e.confidence_tier == "low" and any("not validated" in e.confidence_basis for _ in [0])
    assert any("causation" in u for u in e.uncertainty) and e.indicators["inference"].startswith("2-D box overlap")


def test_parked_overlapping_vehicles_are_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    frames = [(i / 10, [obs("a", 300, 250), obs("b", 340, 250)]) for i in range(160)]  # static, overlapping boxes
    assert drive(det, ctx, frames) == []


def test_contact_that_does_not_end_in_a_stop_is_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    frames = []
    for i in range(160):
        t = i / 10
        frames.append((t, [obs("a", -60 + 90 * t, 250), obs("b", -20 + 90 * t, 252)]))  # one overtakes the other, both keep moving
    assert drive(det, ctx, frames) == []


def test_creeping_queue_with_touching_boxes_is_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    frames = [(i / 10, [obs("a", 300 + 0.5 * i / 10, 250), obs("b", 345 + 0.5 * i / 10, 250)]) for i in range(160)]
    assert drive(det, ctx, frames) == []


def _crash_frames(others=()):
    """Head-on approach to a contact at x=300, then both vehicles stationary. `others`: callables t -> list of extra Observations."""
    frames = []
    for i in range(200):
        t = i / 10
        if t < 4.0:
            a, b = -60 + 90 * t, 700 - 90 * t
        else:
            a, b = 300, 340
        extra = []
        for fn in others:
            extra += fn(t)
        frames.append((t, [obs("a", a, 250), obs("b", b, 250)] + extra))
    return frames


def _parked(n, *, x0=60, y=120):
    return lambda t: [obs(f"p{k}", x0 + 90 * k, y) for k in range(n)]


def _flowing(n):
    return lambda t: [obs(f"f{k}", -80 + 120 * t + 70 * k, 330 + 4 * k) for k in range(n)]


def test_collision_still_fires_when_neighbouring_traffic_keeps_flowing():
    det, ctx = make(CollisionCandidateDetector)
    ev = drive(det, ctx, _crash_frames(others=(_flowing(3),)))
    assert len(ev) == 1
    ind = ev[0].indicators
    assert "not_a_stationary_queue" in ind["gates_passed"] and ind["surroundings"]["others_stationary"] == 0
    assert ind["closing_before_contact_widths"] > 0.15 and ind["time_contact_to_stop_s"] <= 2.5


def test_pair_stopping_inside_a_stationary_queue_is_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    # three vehicles were already stationary right beside the pair before contact: this is a queue
    near_queue = lambda t: [obs(f"q{k}", 380 + 70 * k, 250 + 30) for k in range(3)] + [obs("q3", 220, 250 + 30)]
    ev = drive(det, ctx, _crash_frames(others=(near_queue,)))
    assert ev == [] and det.suppressed["stationary_queue"] == 1


def test_whole_scene_standstill_is_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    # five vehicles far from the pair (outside the neighbourhood radius) are all stationary: a jam, not a crash
    ev = drive(det, ctx, _crash_frames(others=(_parked(5, x0=40, y=-400),)))
    assert ev == [] and det.suppressed["standstill"] == 1


def test_two_vehicles_moving_side_by_side_then_stopping_together_is_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    frames = []
    for i in range(200):
        t = i / 10
        x = 100 + 90 * min(t, 4.0)  # same speed, boxes overlapping the whole time, then both stop: no approach
        frames.append((t, [obs("a", x, 250), obs("b", x + 45, 252)]))
    assert drive(det, ctx, frames) == []  # rejected already: no pre-contact approach history; the closing gate is a second line of defence


def test_gradual_slowdown_in_contact_is_not_a_collision():
    det, ctx = make(CollisionCandidateDetector)
    frames, xa, xb = [], 200.0, 120.0
    for i in range(160):
        t = i / 10
        v = 90.0 if t < 3.0 else max(0.0, 90.0 * (1 - (t - 3.0) / 6.0))  # cruising, then a 6 s gentle stop
        xa += v * 0.1
        xb += (v + (15.0 if t < 3.0 else 0.0)) * 0.1  # b closes slowly on a, touches, then both brake gently
        frames.append((t, [obs("a", xa, 250), obs("b", xb, 250)]))
    assert drive(det, ctx, frames) == [] and det.suppressed["gradual_stop"] == 1


# -- sudden braking (metric speed, smoothed; marking-standard calibration at 0.1 m / px) --------------------

def _metric_profile():
    from anvesh.world.calibration import CalibrationPoint, MetricValidationRecord, fit_planar_homography
    pts = [CalibrationPoint((x, y), (x * 0.1, y * 0.1)) for x, y in ((0, 0), (1000, 0), (1000, 800), (0, 800), (500, 400))]
    rec = MetricValidationRecord("marking_standard_holdout", "test", 4, 0.1, 1.0)
    return fit_planar_homography("p", "cam", pts, validation=rec)


def _braking_run(speed_of_t, seconds=14.0, jitter_px=0.0, x0=20.0, frame_size=None):
    """A vehicle driving along +x at speed_of_t(t) m/s (0.1 m per pixel, 25 fps)."""
    import random
    rnd = random.Random(7)
    det, ctx = make(BrakingDetector, {}, profile=_metric_profile(), geometry_validity=GeometryValidity.WORLD_VALIDATED, motion_space=MotionSpace.WORLD)
    ctx.frame_size = frame_size
    x, frames = x0, []
    for i in range(int(seconds * 25)):
        t = i / 25
        x += speed_of_t(t) / 0.1 / 25
        frames.append((t, [obs("v", x + rnd.uniform(-jitter_px, jitter_px), 400 + rnd.uniform(-jitter_px, jitter_px))]))
    return det, drive(det, ctx, frames)


def test_hard_braking_from_motorway_speed_is_reported_with_its_numbers():
    # 28 m/s (100 km/h) -> 14 m/s over 2 s (7 m/s^2), cruising before
    det, ev = _braking_run(lambda t: 28.0 if t < 5 else max(14.0, 28.0 - 7.0 * (t - 5)), seconds=12)
    assert len(ev) == 1
    ind = ev[0].indicators
    assert ind["from_kmh"] > 90 and ind["drop_kmh"] >= 25 and ind["decel_mps2"] >= 3.5 and ind["approximate_calibration"] is True
    assert ev[0].confidence_tier == "low" and "not a violation by itself" in ev[0].uncertainty[1]


def test_box_clipped_by_the_frame_edge_is_never_read_as_braking():
    brake = lambda t: 28.0 if t < 5 else max(14.0, 28.0 - 7.0 * (t - 5))
    # box bottom (y=418) touches the bottom of a 420 px frame: a lorry leaving the picture, its ground point is not real
    det, ev = _braking_run(brake, seconds=12, frame_size=(5000, 420))
    assert ev == []
    det, ev = _braking_run(brake, seconds=12, frame_size=(5000, 2000))  # same motion, not at an edge: reported
    assert len(ev) == 1


def test_steady_speed_and_box_jitter_never_report_braking():
    det, ev = _braking_run(lambda t: 28.0, jitter_px=2.0)
    assert ev == []


def test_gentle_slowdown_is_not_sudden_braking():
    det, ev = _braking_run(lambda t: 28.0 if t < 3 else max(5.0, 28.0 - 1.2 * (t - 3)), seconds=20)  # 1.2 m/s^2
    assert ev == []


def test_ordinary_stop_from_walking_pace_is_not_sudden_braking():
    det, ev = _braking_run(lambda t: 8.0 if t < 3 else max(0.0, 8.0 - 4.0 * (t - 3)), seconds=10)  # starts below MIN_PRE_KMH
    assert ev == []


def test_braking_detector_needs_validated_calibration():
    cfg = parse_camera_config({})
    ctx = DetectorContext("cam", 25.0, cfg, TrackHistory(), geometry_validity=GeometryValidity.IMAGE)
    assert "sudden_braking" not in [d.name for d in build_detectors(ctx)]


# -- overspeed (smoothed, sustained, inside the speed zone) -------------------------------------------------

def _overspeed_run(speed_of_t, limit=100.0, zone=None, seconds=12.0, jitter_px=0.0):
    import random
    rnd = random.Random(3)
    cfg = {"speed_limit_kmh": limit}
    if zone:
        cfg["speed_zone"] = zone
    det, ctx = make(OverspeedDetector, cfg, profile=_metric_profile(), geometry_validity=GeometryValidity.WORLD_VALIDATED, motion_space=MotionSpace.WORLD)
    x, frames = 20.0, []
    for i in range(int(seconds * 25)):
        t = i / 25
        x += speed_of_t(t) / 0.1 / 25
        frames.append((t, [obs("v", x + rnd.uniform(-jitter_px, jitter_px), 400 + rnd.uniform(-jitter_px, jitter_px))]))
    return det, drive(det, ctx, frames)


def test_sustained_speed_over_the_limit_is_reported_once_with_error_band():
    det, ev = _overspeed_run(lambda t: 36.0, jitter_px=1.0)  # 129.6 km/h vs 100
    assert len(ev) == 1
    ind = ev[0].indicators
    assert 120 < ind["speed_kmh"] < 140 and ind["sustained_s"] >= 2.0 and ind["speed_error_kmh"] > 0 and ind["approximate_calibration"] is True
    assert ev[0].confidence_tier == "low" and "approximate" in ev[0].confidence_basis


def test_a_brief_spike_over_the_limit_is_not_overspeed():
    det, ev = _overspeed_run(lambda t: 36.0 if 5.0 <= t < 5.8 else 25.0)  # 0.8 s above 100 km/h
    assert ev == []


def test_speed_below_limit_plus_margin_is_not_overspeed():
    det, ev = _overspeed_run(lambda t: 28.5)  # 102.6 km/h: inside the 5 km/h margin over a 100 km/h limit
    assert ev == []


def test_speed_outside_the_speed_zone_is_not_judged():
    # the vehicle drives along y=400; the zone covers y<300 only
    det, ev = _overspeed_run(lambda t: 36.0, zone=[[0, 0], [2000, 0], [2000, 300], [0, 300]])
    assert ev == [] and det.skipped_outside_zone > 0


# -- red light -------------------------------------------------------------------------------

SIG = {"signal": {"stop_line": [[100, 150], [540, 150]], "approach": [0, 1], "mode": "schedule",
                  "schedule": {"red_s": 5, "green_s": 5, "amber_s": 2, "offset_s": 0}}}


def _runner_frames(speed=30.0, start_y=40):
    return [(i / 10, [obs("r", 320, start_y + speed * i / 10)]) for i in range(100)]


def test_red_light_fires_only_when_signal_has_been_red_and_crossing_is_in_approach_direction():
    sig = SignalState("schedule", SIG["signal"]["schedule"])
    det, ctx = make(RedLightDetector, SIG, signal=sig)
    ev = drive(det, ctx, _runner_frames())  # crosses the line at ~3.1 s, red during 0-5 s
    assert len(ev) == 1 and ev[0].indicators["signal_state"] == "red"
    assert ev[0].indicators["signal_source"].startswith("configured_schedule") and ev[0].confidence_tier == "low"
    assert any("not read from the video" in u for u in ev[0].uncertainty)


def test_red_light_not_on_green_or_unknown_or_wrong_direction():
    green_cfg = {"red_s": 2, "green_s": 20, "amber_s": 2, "offset_s": 0}
    sig = SignalState("schedule", green_cfg)
    det, ctx = make(RedLightDetector, SIG, signal=sig)
    assert drive(det, ctx, _runner_frames()) == []  # light turned green at 2 s, crossing at 3.1 s
    ext = SignalState("external")
    det2, ctx2 = make(RedLightDetector, {"signal": dict(SIG["signal"], mode="external")}, signal=ext)
    assert drive(det2, ctx2, _runner_frames()) == [] and det2.unjudged_crossings == 1  # no feed: never assumed red
    sig3 = SignalState("schedule", SIG["signal"]["schedule"])
    det3, ctx3 = make(RedLightDetector, SIG, signal=sig3)
    back = [(i / 10, [obs("r", 320, 340 - 30 * i / 10)]) for i in range(100)]  # leaving, against the approach direction
    assert drive(det3, ctx3, back) == []


def test_red_light_external_feed_gives_medium_confidence_and_requires_grace():
    ext = SignalState("external")
    ext.set("red", 0.0)
    det, ctx = make(RedLightDetector, {"signal": dict(SIG["signal"], mode="external")}, signal=ext)
    ev = drive(det, ctx, _runner_frames())
    assert len(ev) == 1 and ev[0].confidence_tier == "medium"
    late = SignalState("external")
    late.set("red", 3.0)  # turned red only 0.1 s before the crossing: inside the grace period
    det2, ctx2 = make(RedLightDetector, {"signal": dict(SIG["signal"], mode="external")}, signal=late)
    assert drive(det2, ctx2, _runner_frames()) == []


def test_signal_state_schedule_phases():
    s = SignalState("schedule", {"red_s": 5, "green_s": 5, "amber_s": 2, "offset_s": 0})
    assert [s.at(t) for t in (0, 4.9, 5.1, 9.9, 10.5, 12.1)] == ["red", "red", "green", "green", "amber", "red"]
    with pytest.raises(ValueError):
        SignalState("manual").set("purple", 0)


# -- illegal turn / lanes ---------------------------------------------------------------------

def test_illegal_turn_entry_then_exit():
    cfg = {"turn_rules": [{"rule_id": "no-left", "label": "No left turn", "entry": [[300, 300], [400, 300], [400, 340], [300, 340]],
                           "exit": [[0, 150], [100, 150], [100, 200], [0, 200]]}]}
    det, ctx = make(IllegalTurnDetector, cfg)
    path = [(350, 320)] * 3 + [(300, 250), (200, 200), (100, 180), (50, 170)]
    frames = [(i / 2, [obs("t", x, y - 18)]) for i, (x, y) in enumerate(path)]  # ground y = cy + 18
    ev = drive(det, ctx, frames)
    assert len(ev) == 1 and ev[0].indicators["rule_id"] == "no-left"
    other = [(i / 2, [obs("s", x, y - 18)]) for i, (x, y) in enumerate([(350, 320), (350, 250), (350, 160)])]
    det2, ctx2 = make(IllegalTurnDetector, cfg)
    assert drive(det2, ctx2, other) == []


LANES = {"lanes": [{"lane_id": "L1", "vertices": [[0, 100], [300, 100], [300, 340], [0, 340]], "allowed_classes": ["car", "motorcycle"]},
                   {"lane_id": "L2", "vertices": [[300, 100], [640, 100], [640, 340], [300, 340]], "allowed_classes": None}]}


def test_wrong_lane_by_vehicle_class_after_dwell():
    det, ctx = make(WrongLaneDetector, LANES)
    frames = [(i / 10, [obs("truck", 150, 200, cls="truck"), obs("car", 150, 260, cls="car"), obs("truck2", 450, 200, cls="truck")]) for i in range(50)]
    ev = drive(det, ctx, frames)
    assert [e.track_ids for e in ev] == [["truck"]] and ev[0].indicators["lane_id"] == "L1"


def test_rapid_lane_changes_candidate_and_boundary_jitter_is_debounced():
    det, ctx = make(RapidLaneChangeDetector, LANES)
    frames = []
    for i in range(100):
        t = i / 10
        lane_x = 150 if int(t / 1.5) % 2 == 0 else 450  # alternate lanes every 1.5 s
        frames.append((t, [obs("weave", lane_x, 220)]))
    ev = drive(det, ctx, frames)
    assert len(ev) == 1 and ev[0].event_type == "rash_driving" and ev[0].indicators["lane_changes"] >= 3
    assert any("CANDIDATE" in u for u in ev[0].uncertainty)
    det2, ctx2 = make(RapidLaneChangeDetector, LANES)
    jitter = [(i / 10, [obs("steady", 298 if i % 2 else 302, 220)]) for i in range(100)]  # flickers across the boundary each frame
    assert drive(det2, ctx2, jitter) == []


# -- engine-level, SIMULATED scenarios (real streaming engine) ---------------------------------

def test_engine_wrong_way_scenario_flags_only_the_wrong_car(services):
    cfg = {"direction_zones": [{"zone_id": "e", "polygon": [[0, 150], [640, 150], [640, 340], [0, 340]], "expected": [1, 0]}]}
    _, runner = run_scenario(services, "wrong_way", cfg)
    rows, _ = services.incidents.list({"session_id": runner.session_id, "event_type": "wrong_way"})
    assert len(rows) == 1
    inc = rows[0]
    assert len(inc["track_ids"]) == 1 and inc["involved"][0]["role"] == "observed"
    assert inc["data_class"] == "simulated" and inc["evidence_completeness"] in ("full", "partial")
    assert "wrong_way" in runner.runtime()["detectors"]


def test_engine_collision_scenario_creates_candidate_incident_with_both_vehicles(services):
    _, runner = run_scenario(services, "collision", {})
    rows, _ = services.incidents.list({"session_id": runner.session_id, "event_type": "possible_collision"})
    assert len(rows) == 1
    inc = rows[0]
    assert len(inc["track_ids"]) == 2 and inc["severity"] == "high" and inc["confidence_tier"] == "low"
    assert inc["established_cause"] is None and {v["role"] for v in inc["involved"]} == {"potentially_involved"}
    kinds = {a["kind"] for a in services.evidence.list(inc["incident_id"])}
    assert "clip" in kinds and "frame" in kinds
    # a high-severity candidate is routed to the control-room TEST adapter
    alerts = services.alerts.list(incident_id=inc["incident_id"])
    assert len(alerts) == 1 and alerts[0]["is_test"] == 1


def test_engine_red_light_scenario_with_schedule(services):
    cfg = dict(SIG)
    _, runner = run_scenario(services, "red_light", cfg)
    rows, _ = services.incidents.list({"session_id": runner.session_id, "event_type": "red_light_violation"})
    assert len(rows) == 1 and rows[0]["indicators"]["signal_source"].startswith("configured_schedule")


def test_engine_congestion_incident_from_a_queue(services):
    _, runner = run_scenario(services, "queue", {"thresholds": {"max_capacity_vehicles": 6}})
    rows, _ = services.incidents.list({"session_id": runner.session_id, "event_type": "congestion"})
    assert rows, runner.runtime()["congestion_trend"]
    inc = rows[0]
    assert inc["motion_space"] == "image" and any("count/occupancy proxy" in u for u in inc["uncertainty"])
    assert inc["indicators"]["trend"] and inc["established_cause"] is None
    for cause in inc["candidate_causes"]:
        assert "not an established cause" in cause.get("status", "not an established cause")


# -- speed safety gating (M-A invariants must hold for overspeed) ------------------------------

def _grid_points():
    from anvesh.world.calibration import CalibrationPoint
    pts = [((0, 0), (0, 0)), ((640, 0), (64, 0)), ((640, 360), (64, 36)), ((0, 360), (0, 36)), ((320, 180), (32, 18))]
    return [CalibrationPoint(image_xy=i, world_xy=w) for i, w in pts]


def test_overspeed_is_unavailable_for_unvalidated_homography_and_fires_for_validated(services):
    cal = {"reference_points": [{"image_xy": list(p.image_xy), "world_xy": list(p.world_xy)} for p in _grid_points()]}
    # 1) fitted but UNVALIDATED: tracks stay image-space, nothing metric, no overspeed detector at all
    services.cameras.create("admin", {"camera_id": "sim-unval", "name": "u", "source_type": "synthetic", "source_uri": "synthetic:stopped_car",
                                      "data_class": "simulated", "config": {"calibration": cal, "speed_limit_kmh": 1}})
    from anvesh.app.runner import SessionManager
    from .test_session_synthetic import _tracker_factory
    mgr = SessionManager(services, tracker_factory=_tracker_factory)
    s = mgr.start("sim-unval", "admin")
    r = mgr.get(s["session_id"])
    r._thread.join(60)
    assert r.profile.geometry_validity == GeometryValidity.WORLD_UNVALIDATED
    assert "overspeed" not in [d.name for d in r.detectors] and "sudden_braking" not in [d.name for d in r.detectors]
    rows, _ = services.incidents.list({"session_id": r.session_id, "event_type": "overspeed"})
    assert rows == []
    assert all(i["motion_space"] == "image" for i in services.incidents.list({"session_id": r.session_id})[0])
    cap = category_status("overspeed", parse_camera_config({"speed_limit_kmh": 50}), geometry_validated=False)
    assert cap["available"] is False and any("validated metric geometry" in m for m in cap["missing"])
    # 2) SYNTHETIC-validated opt-in: the detector exists, the event is flagged synthetic and low confidence
    cal2 = dict(cal, validation={"kind": "synthetic_test_only", "evidence_ref": "synthetic test fixture", "n_holdout_points": 0,
                                 "holdout_rmse_m": 0.0, "tolerance_m": 1e-9})
    services.cameras.create("admin", {"camera_id": "sim-val", "name": "v", "source_type": "synthetic", "source_uri": "synthetic:stopped_car",
                                      "data_class": "simulated", "config": {"calibration": cal2, "speed_limit_kmh": 1}})
    s2 = mgr.start("sim-val", "admin")
    r2 = mgr.get(s2["session_id"])
    r2._thread.join(60)
    assert r2.profile.geometry_validity == GeometryValidity.WORLD_VALIDATED and r2.profile.is_synthetic_validated
    assert "overspeed" in [d.name for d in r2.detectors]
    rows, _ = services.incidents.list({"session_id": r2.session_id, "event_type": "overspeed"})
    assert rows, "cars crossing at ~5 m/s must exceed a 1 km/h limit under validated geometry"
    e = rows[0]
    assert e["motion_space"] == "world" and e["geometry_validity"] == "world_validated" and e["indicators"]["synthetic_calibration"] is True
    assert e["confidence_tier"] == "low" and "SYNTHETIC" in e["confidence_basis"]


def test_capability_registry_is_honest_about_every_category():
    cams = all_category_status()
    by = {c["event_type"]: c for c in cams}
    assert set(by) == set(CAPABILITIES)
    for t in ("helmet_violation", "seatbelt_violation"):
        assert by[t]["status"] == "needs_model" and by[t]["available"] is False and by[t]["detector"] is None
    assert by["stopped_vehicle"]["available"] and by["congestion"]["available"]
    assert by["possible_collision"]["status"] == "experimental" and not by["possible_collision"]["validated_on_real_footage"]
    for t in ("wrong_way", "wrong_lane", "red_light_violation", "illegal_turn", "overspeed", "rash_driving"):
        assert by[t]["available"] is False and by[t]["missing"], t
        assert not by[t]["validated_on_real_footage"]
    full = parse_camera_config({**ZONE, **SIG, **LANES, "turn_rules": [{"rule_id": "x", "entry": [[0, 0], [1, 0], [1, 1]], "exit": [[5, 5], [6, 5], [6, 6]]}]})
    st = {c["event_type"]: c for c in all_category_status(full)}
    assert st["wrong_way"]["available"] and st["red_light_violation"]["available"] and st["illegal_turn"]["available"] and st["rash_driving"]["available"]
    assert not st["overspeed"]["available"]
