"""The officer layer: jam-cause candidate, lane labels, alert board, the three demonstration scenes end to end, the officer API and the
one-click sign-in safety rules."""
import json
import math
import time

import pytest

from anvesh.app.detectors import Observation, TrackHistory
from anvesh.app.jamcause import JamCauseAnalyzer
from anvesh.app.officer import AlertBoard, LaneMap, lane_label
from anvesh.app.runner import SessionManager
from anvesh.app.synthetic import ScriptedTracker


# ---------------------------------------------------------------------------------------------------------------------------
# jam cause, driven by hand-made tracks
def _feed(history, t, boxes):
    """boxes: {track_id: (cx, cy)} ground = bottom centre of a 60x36 box."""
    obs = [Observation(tid, "car", (cx - 30, cy - 36, cx + 30, cy), (cx, cy), 0.9) for tid, (cx, cy) in boxes.items()]
    history.update(t, int(t * 10), obs)


def _run_queue(front_stops_first=True, n_followers=4, stop_gap_s=1.6):
    hist = TrackHistory()
    an = JamCauseAnalyzer(hist)
    x_stop = 500.0
    paths = {"s-t1": lambda t: min(-60 + 100 * t, x_stop) if front_stops_first else min(-60 + 100 * t, x_stop)}
    # followers enter later and stop 85 px behind each other
    starts = {}
    for i in range(1, n_followers + 1):
        enter = 1.0 + 2.0 * i
        stop_x = x_stop - 85.0 * i
        starts[f"s-t{i + 1}"] = (enter, stop_x)
    states = []
    for k in range(0, 300):
        t = k * 0.1
        boxes = {"s-t1": (min(-60 + 110 * t, x_stop), 250.0)}
        for tid, (enter, stop_x) in starts.items():
            if t >= enter:
                boxes[tid] = (min(-60 + 110 * (t - enter), stop_x), 250.0)
        _feed(hist, t, boxes)
        states.append(an.update(t))
    return an, states


def test_front_vehicle_is_the_jam_cause_candidate():
    an, states = _run_queue()
    final = states[-1]
    assert final["active"] and final["vehicles"] >= 4
    assert final["cause_track"] == "s-t1"
    assert final["stopped_first"] is True and final["rating"] == "likely"
    assert final["order_correlation"] is not None and final["order_correlation"] >= 0.5
    assert "candidate" in final["text"].lower() and "Candidate only" in final["disclaimer"]


def test_no_jam_when_everything_moves():
    hist, an = TrackHistory(), JamCauseAnalyzer(TrackHistory())
    an = JamCauseAnalyzer(hist)
    for k in range(200):
        t = k * 0.1
        _feed(hist, t, {f"m-t{i}": (-60 + 100 * t + 70 * i, 250.0) for i in range(6)})
        st = an.update(t)
    assert st["active"] is False and st["cause_track"] is None


def test_two_stopped_vehicles_are_not_a_jam():
    hist = TrackHistory()
    an = JamCauseAnalyzer(hist)
    for k in range(250):
        t = k * 0.1
        _feed(hist, t, {"a-t1": (min(-60 + 100 * t, 300), 250.0), "a-t2": (min(-60 + 100 * t, 215), 250.0)})
        st = an.update(t)
    assert st["active"] is False


def test_a_jam_is_closed_after_the_queue_clears():
    hist = TrackHistory()
    an = JamCauseAnalyzer(hist)
    last = None
    for k in range(600):
        t = k * 0.1
        boxes = {}
        for i in range(4):
            stop_x = 400 - 80 * i
            x = min(-60 + 110 * max(0.0, t - 0.5 * i), stop_x)
            if t > 25:
                x = stop_x + 140 * (t - 25)  # the queue drives off
            boxes[f"c-t{i}"] = (x, 250.0)
        _feed(hist, t, boxes)
        last = an.update(t)
    assert last["active"] is False
    assert an.closed and an.closed[-1]["peak_vehicles"] >= 3


# ---------------------------------------------------------------------------------------------------------------------------
def test_lane_labels_and_membership():
    assert lane_label("TOWARD-3-offside") == "TOWARD 3" and lane_label("L2") == "L2" and lane_label("LANE-1") == "LANE 1"
    lm = LaneMap([{"lane_id": "LANE-1", "vertices": [[0, 0], [100, 0], [100, 50], [0, 50]]}, {"lane_id": "LANE-2", "vertices": [[0, 50], [100, 50], [100, 100], [0, 100]]}])
    assert lm.lane_of((50, 20))["label"] == "LANE 1" and lm.lane_of((50, 70))["id"] == "LANE-2" and lm.lane_of((500, 500)) is None


def test_alert_board_keeps_thumbnails_and_active_window():
    b = AlertBoard()
    a = b.add("wrong_way", "x", 10.0, track_ids=["t1"], thumb=b"jpg")
    assert b.thumb(a["id"]) == b"jpg" and b.recent(5)[0]["id"] == a["id"]
    assert b.active(12.0, ("wrong_way",)) and not b.active(30.0, ("wrong_way",))
    assert b.counts["wrong_way"] == 1


# ---------------------------------------------------------------------------------------------------------------------------
def _tracker_factory(camera, config):
    return ScriptedTracker(camera["source_uri"].split(":", 1)[-1])


def _lanes():
    return [{"lane_id": f"LANE-{i + 1}", "vertices": [[0, 172 + 52 * i], [640, 172 + 52 * i], [640, 172 + 52 * (i + 1)], [0, 172 + 52 * (i + 1)]]} for i in range(3)]


def _run(services, scenario, config):
    services.cameras.create("admin", {"camera_id": f"sim-{scenario}", "name": scenario, "source_type": "synthetic", "source_uri": f"synthetic:{scenario}",
                                      "data_class": "simulated", "config": config})
    mgr = SessionManager(services, tracker_factory=_tracker_factory)
    sess = mgr.start(f"sim-{scenario}", "admin")
    runner = mgr.get(sess["session_id"])
    runner._thread.join(120)
    assert not runner._thread.is_alive()
    return runner


def test_wrong_way_scene_raises_the_wrong_way_alert(services):
    zone = [{"zone_id": "east", "polygon": [[0, 100], [640, 100], [640, 340], [0, 340]], "expected": [1, 0]}]
    r = _run(services, "show_wrong_way", {"direction_zones": zone, "lanes": _lanes()})
    st = r.officer_state()
    assert st["state"] == "completed" and st["simulated"] is True
    kinds = [a["kind"] for a in st["alerts"]]
    assert "wrong_way" in kinds, kinds
    tiles = {t["key"]: t for t in st["tiles"]}
    assert tiles["lanes"]["state"] == "ok" and tiles["speed"]["state"] == "off"  # no calibration -> no invented speed
    assert tiles["plate"]["state"] == "off"  # simulated scene: no plate reader


def test_accident_scene_raises_a_possible_accident_alert(services):
    r = _run(services, "show_accident", {"lanes": _lanes()})
    st = r.officer_state()
    assert "possible_collision" in [a["kind"] for a in st["alerts"]], st["alerts"]
    assert st["alert_counts"].get("possible_collision") == 1


def test_jam_scene_names_the_front_vehicle_as_candidate_cause(services):
    r = _run(services, "show_jam", {"lanes": _lanes(), "thresholds": {"max_capacity_vehicles": 12}})
    st = r.officer_state()
    jam_alerts = [a for a in st["alerts"] if a["kind"] == "jam"]
    assert jam_alerts, [a["kind"] for a in st["alerts"]]
    assert st["jam"]["active"] and st["jam"]["cause_label"].startswith("truck")
    assert st["jam"]["rating"] in ("likely", "possible")
    jt = {t["key"]: t for t in st["tiles"]}["jam"]
    assert jt["state"] == "bad" and "candidate" in jt["detail"].lower()
    # jam alerts carry a picture
    assert r.officer.alerts.thumb(jam_alerts[0]["id"]).startswith(b"\xff\xd8")
    # queue members are not repeated as separate "vehicle stopped" alerts
    assert sum(1 for a in st["alerts"] if a["kind"] == "stopped_vehicle") <= 2


def test_preview_frame_is_upscaled_for_the_projector(services):
    import cv2
    import numpy as np

    r = _run(services, "show_wrong_way", {"lanes": _lanes()})
    # the last preview was released with the session; render one frame directly to check the size contract
    img = np.zeros((360, 640, 3), np.uint8)
    out = r.officer.render(img, 3.0, [], [], {"active": False})
    assert out.shape[1] == 1280 and out.shape[0] == 720


# ---------------------------------------------------------------------------------------------------------------------------
# API: officer endpoint and one-click sign-in
@pytest.fixture()
def server(services):
    from anvesh.app.server import serve

    services.auth.create_user("inv-demo", "Inv", "investigator", "pw-inv-demo-long", is_demo=True)
    srv = serve(services, port=0, tracker_factory=_tracker_factory)
    yield srv
    srv.stop()


def _http(srv, method, path, body=None, cookie=None):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
    headers = {"Content-Type": "application/json", "X-Requested-With": "anvesh"}
    if cookie:
        headers["Cookie"] = cookie
    conn.request(method, "/api/v1" + path, body=json.dumps(body) if body is not None else None, headers=headers)
    r = conn.getresponse()
    data = r.read()
    return r.status, data, r.getheader("Set-Cookie")


def test_one_click_login_does_not_exist_unless_enabled(server):
    st, _data, _c = _http(server, "POST", "/auth/demo")
    assert st == 404
    st, data, _c = _http(server, "GET", "/auth/demo")
    assert st == 200 and json.loads(data) == {"one_click": False}


def test_one_click_login_works_only_when_enabled_on_loopback(server):
    server.demo_login = ("inv-demo", "pw-inv-demo-long")
    st, data, cookie = _http(server, "GET", "/auth/demo")
    assert json.loads(data) == {"one_click": True}
    st, data, cookie = _http(server, "POST", "/auth/demo")
    assert st == 200 and cookie and "HttpOnly" in cookie
    assert json.loads(data)["user"]["username"] == "inv-demo"
    server.host = "0.0.0.0"  # a server reachable from other machines must not offer the shortcut
    st, _d, _c = _http(server, "POST", "/auth/demo")
    assert st == 404


def test_home_page_is_the_simple_screen_only_when_asked(server):
    import http.client

    def get(path):
        c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read().decode("utf-8", "replace")

    st, body = get("/")
    assert st == 200 and "ANVESH - Traffic Intelligence" in body
    server.simple_home = True
    st, body = get("/")
    assert "Live traffic watch" in body
    st, body = get("/advanced")
    assert "ANVESH - Traffic Intelligence" in body


def test_officer_endpoint_requires_sign_in_and_returns_state(server, services):
    st, _d, _c = _http(server, "GET", "/sessions/SES-NOPE/officer")
    assert st == 401
    server.demo_login = ("inv-demo", "pw-inv-demo-long")
    _st, _d, cookie = _http(server, "POST", "/auth/demo")
    cookie = cookie.split(";")[0]
    services.cameras.create("admin", {"camera_id": "sim-jam", "name": "j", "source_type": "synthetic", "source_uri": "synthetic:show_jam",
                                      "data_class": "simulated", "config": {"lanes": _lanes()}})
    st, data, _c = _http(server, "POST", "/cameras/sim-jam/sessions", cookie=cookie)
    assert st == 202
    sid = json.loads(data)["session_id"]
    deadline = time.time() + 60
    body = None
    while time.time() < deadline:
        st, data, _c = _http(server, "GET", f"/sessions/{sid}/officer", cookie=cookie)
        if st == 200:
            body = json.loads(data)
            if body["state"] in ("completed", "failed"):
                break
        time.sleep(0.5)
    assert body and body["state"] == "completed", body
    assert [t["key"] for t in body["tiles"]] == ["lanes", "speed", "wrongway", "accident", "plate", "jam"]
    jam = [a for a in body["alerts"] if a["kind"] == "jam"]
    assert jam
    st, data, _c = _http(server, "GET", f"/sessions/{sid}/alerts/{jam[0]['id']}/thumb.jpg", cookie=cookie)
    assert st == 200 and data[:2] == b"\xff\xd8"


def test_vehicles_already_standing_when_first_tracked_use_the_configured_direction_and_never_claim_to_have_stopped_first():
    hist = TrackHistory()
    zones = [{"polygon": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]], "expected": [1, 0]}]
    an = JamCauseAnalyzer(hist, zones=zones)
    st = None
    for k in range(120):
        t = k * 0.1
        _feed(hist, t, {f"b-t{i}": (500 - 80 * i, 250.0) for i in range(4)})
        st = an.update(t)
    assert st["active"] and st["cause_track"] == "b-t0"
    assert st["stopped_first"] is not True and st["rating"] == "possible"
    assert "configured direction" in st["direction_source"]
    assert any("could not be measured" in e for e in st["evidence"])


def test_without_any_direction_the_jam_is_reported_but_no_cause_is_named():
    hist = TrackHistory()
    an = JamCauseAnalyzer(hist)
    st = None
    for k in range(120):
        t = k * 0.1
        _feed(hist, t, {f"d-t{i}": (500 - 80 * i, 250.0) for i in range(4)})
        st = an.update(t)
    assert st["active"] and st["cause_track"] is None and "cannot be told" in st["text"]
