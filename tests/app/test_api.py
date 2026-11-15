"""The HTTP contract: authentication, authorisation, CSRF, validation, pagination, evidence serving, upload -> session -> incident
-> case -> hand-off through the real server on an ephemeral port."""

import json
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from anvesh.app.db import SCHEMA_VERSION
from anvesh.app.runner import SessionManager
from anvesh.app.server import AppServer
from anvesh.app.synthetic import ScriptedTracker

from .conftest import seed_incident


class Client:
    def __init__(self, base):
        self.base, self.token, self.cookie = base, None, None

    def call(self, method, path, body=None, headers=None, raw=None, use_cookie=False, token=True):
        url = self.base + "/api/v1" + path if path.startswith("/") and not path.startswith("/api") else self.base + path
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        h = {"Content-Type": "application/json"} if raw is None else {"Content-Type": "application/octet-stream"}
        if self.token and token and not use_cookie:
            h["Authorization"] = "Bearer " + self.token
        if use_cookie and self.cookie:
            h["Cookie"] = self.cookie
        h.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = r.read()
                return r.status, (json.loads(payload) if r.headers.get("Content-Type", "").startswith("application/json") else payload), dict(r.headers)
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                return e.code, json.loads(payload), dict(e.headers)
            except ValueError:
                return e.code, payload, dict(e.headers)

    def login(self, username):
        st, body, headers = self.call("POST", "/auth/login", {"username": username, "password": "pw-" + username + "-long"})
        assert st == 200, body
        self.token = body["token"]
        self.cookie = headers["Set-Cookie"].split(";")[0]
        return body


@pytest.fixture()
def api(services, users):
    mgr = SessionManager(services, tracker_factory=lambda cam, cfg: ScriptedTracker("stopped_car"))
    server = AppServer(services, mgr, host="127.0.0.1", port=0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    yield base, services
    server.stop()


def client(base, user=None):
    c = Client(base)
    if user:
        c.login(user)
    return c


def test_health_ready_and_unauthenticated_access(api):
    base, _ = api
    c = client(base)
    assert c.call("GET", "/health")[0] == 200
    st, body, _ = c.call("GET", "/ready")
    assert st == 200 and body["checks"]["database"] and body["checks"]["schema_version"] == SCHEMA_VERSION
    assert body["detector"] == {"source": "pretrained", "weights": "yolov8n.pt"}  # which detector is configured is visible
    for path in ("/overview", "/incidents", "/cases", "/cameras", "/audit", "/users", "/system/status"):
        st, body, _ = c.call("GET", path)
        assert st == 401 and body["error"]["code"] == "unauthenticated", path
    assert c.call("GET", "/nope", token=False)[0] in (401, 404)


def test_login_cookie_flags_logout_and_bad_credentials(api):
    base, _ = api
    c = Client(base)
    st, body, _ = c.call("POST", "/auth/login", {"username": "off1", "password": "wrong-wrong-wrong"})
    assert st == 401 and body["error"]["message"] == "invalid credentials"
    st, body, headers = c.call("POST", "/auth/login", {"username": "off1", "password": "pw-off1-long"})
    assert st == 200 and body["user"]["role"] == "officer" and "case.approve" in body["user"]["permissions"]
    cookie = headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    c.token = body["token"]
    assert c.call("GET", "/auth/me")[1]["user"]["username"] == "off1"
    c.call("POST", "/auth/logout")
    assert c.call("GET", "/auth/me")[0] == 401


def test_cookie_auth_requires_csrf_header_for_state_changes(api):
    base, svc = api
    c = client(base, "admin1")
    body = {"camera_id": "c-csrf", "name": "x", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "simulated"}
    st, resp, _ = c.call("POST", "/cameras", body, use_cookie=True)  # cookie, no X-Requested-With
    assert st == 403 and resp["error"]["code"] == "csrf"
    st, resp, _ = c.call("POST", "/cameras", body, use_cookie=True, headers={"X-Requested-With": "anvesh", "Origin": "http://evil.example"})
    assert st == 403
    st, resp, _ = c.call("POST", "/cameras", body, use_cookie=True, headers={"X-Requested-With": "anvesh"})
    assert st == 201 and resp["camera_id"] == "c-csrf"
    assert c.call("GET", "/cameras", use_cookie=True)[0] == 200  # reads work with the cookie (img/video tags)


def test_rbac_matrix_over_http(api):
    base, svc = api
    inc = seed_incident(svc)
    iid = inc["incident_id"]
    inv, rev, off, adm = (client(base, u) for u in ("inv1", "rev1", "off1", "admin1"))
    # camera management: administrator only
    body = {"camera_id": "c-rbac", "name": "x", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "simulated"}
    assert inv.call("POST", "/cameras", body)[0] == 403 and rev.call("POST", "/cameras", body)[0] == 403
    assert adm.call("POST", "/cameras", body)[0] == 201
    # users and audit: administrator (and audit.view officer) only
    assert inv.call("GET", "/users")[0] == 403 and adm.call("GET", "/users")[0] == 200
    assert rev.call("GET", "/audit")[0] == 403 and off.call("GET", "/audit")[0] == 200 and adm.call("GET", "/audit/verify")[0] == 200
    # case creation, verification, approval
    st, case, _ = inv.call("POST", f"/incidents/{iid}/cases", {"offence_code": "STOP-OBS"})
    assert st == 201
    cid = case["case_id"]
    assert rev.call("POST", f"/incidents/{iid}/cases", {"offence_code": "STOP-OBS"})[0] in (403, 409)
    assert inv.call("POST", f"/cases/{cid}/actions", {"action": "submit_for_evidence_verification"})[0] == 200
    assert inv.call("POST", f"/cases/{cid}/actions", {"action": "verify_evidence"})[0] == 403
    assert rev.call("POST", f"/cases/{cid}/actions", {"action": "verify_evidence"})[0] == 200
    st, body, _ = rev.call("POST", f"/cases/{cid}/actions", {"action": "approve"})
    assert st == 403
    assert adm.call("POST", f"/cases/{cid}/actions", {"action": "approve"})[0] == 403
    assert inv.call("POST", f"/cases/{cid}/actions", {"action": "bogus"})[0] == 422


def test_validation_and_pagination_and_error_shape(api):
    base, svc = api
    for i in range(5):
        seed_incident(svc, key=f"k{i}", severity="low")
    c = client(base, "inv1")
    st, body, _ = c.call("GET", "/incidents?limit=2&offset=3&order=priority")
    assert st == 200 and body["total"] == 5 and len(body["items"]) == 2 and body["limit"] == 2 and body["offset"] == 3
    assert c.call("GET", "/incidents?limit=abc")[0] == 422
    assert c.call("GET", "/incidents?severity=critical")[1]["total"] == 0
    st, body, _ = c.call("GET", "/incidents/INC-DOES-NOT-EXIST")
    assert st == 404 and set(body["error"]) == {"code", "message", "details"}
    adm = client(base, "admin1")
    st, body, _ = adm.call("POST", "/cameras", {"name": "bad", "source_type": "rtsp", "source_uri": "http://not-rtsp", "data_class": "demo"})
    assert st == 422 and "rtsp://" in body["error"]["message"]
    st, body, _ = adm.call("POST", "/cameras", {"name": "bad", "source_type": "rtsp", "source_uri": "rtsp://h/s", "data_class": "demo"})
    assert st == 422 and "nominal_fps" in body["error"]["message"]  # a live camera must be able to reconnect from a cold start
    st, body, _ = adm.call("POST", "/cameras", {"name": "bad", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "demo"})
    assert st == 422 and "simulated" in body["error"]["message"]
    st, body, _ = adm.call("POST", "/cameras", {"name": "x", "source_type": "file", "source_uri": "C:/Windows/system32/drivers/etc/hosts", "data_class": "demo"})
    assert st == 422 and "media root" in body["error"]["message"]
    st, body, _ = adm.call("POST", "/cameras", {"name": "x", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "simulated",
                                                "config": {"direction_zones": [{"polygon": [[0, 0]], "expected": [1, 0]}]}})
    assert st == 422 and "polygon" in body["error"]["message"]
    assert client(base, "inv1").call("POST", "/incidents/x/notes", {"note": " "})[0] in (404, 422)


def test_stream_credentials_are_never_returned(api):
    base, svc = api
    adm = client(base, "admin1")
    st, cam, _ = adm.call("POST", "/cameras", {"camera_id": "c-rtsp", "name": "r", "source_type": "rtsp", "nominal_fps": 25,
                                              "source_uri": "rtsp://user:s3cret@10.0.0.9:554/live", "data_class": "demo"})
    assert st == 201 and "s3cret" not in json.dumps(cam) and "source_uri" not in cam and "***" in cam["source_uri_display"]
    assert "s3cret" not in json.dumps(adm.call("GET", "/cameras")[1])
    assert "s3cret" not in json.dumps(adm.call("GET", "/audit?limit=50")[1])


def test_evidence_served_with_auth_and_range_and_integrity_endpoint(api):
    base, svc = api
    inc = seed_incident(svc)
    clip = next(a for a in svc.evidence.list(inc["incident_id"]) if a["kind"] == "clip")
    frame = next(a for a in svc.evidence.list(inc["incident_id"]) if a["kind"] == "frame")
    anon = Client(base)
    assert anon.call("GET", f"/evidence/{frame['asset_id']}/file")[0] == 401
    c = client(base, "rev1")
    st, data, h = c.call("GET", f"/evidence/{frame['asset_id']}/file")
    assert st == 200 and h["Content-Type"] == "image/jpeg" and data[:2] == b"\xff\xd8"
    st, part, h = c.call("GET", f"/evidence/{clip['asset_id']}/file", headers={"Range": "bytes=0-99"})
    assert st == 206 and len(part) == 100 and h["Content-Range"].startswith("bytes 0-99/")
    assert c.call("GET", f"/evidence/{clip['asset_id']}/file", headers={"Range": "bytes=999999999-"})[0] == 416
    assert c.call("GET", f"/evidence/{clip['asset_id']}/verify")[1]["ok"] is True
    # tampering is surfaced by the verify endpoint
    svc.evidence.path_of(frame).write_bytes(b"x")
    assert c.call("GET", f"/evidence/{frame['asset_id']}/verify")[1] == {"asset_id": frame["asset_id"], "ok": False, "reason": "sha256 mismatch"}
    assert any(r["action"] == "evidence.view" for r in svc.audit.query(entity_type="incident", entity_id=inc["incident_id"])[0])


def test_static_dashboard_only_and_no_path_traversal(api):
    base, _ = api
    c = Client(base)
    st, body, h = c.call("GET", "/")
    assert st in (200, 404)  # 200 once the dashboard exists
    for evil in ("/../config.py", "/..%2f..%2fREADME.md", "/%2e%2e/%2e%2e/pyproject.toml", "/.git/config", "/anvesh_app.sqlite3"):
        st, body, _ = c.call("GET", evil, token=False)
        assert st == 404, evil
    st, _, h = c.call("GET", "/health")
    assert h["X-Content-Type-Options"] == "nosniff" and "frame-ancestors 'none'" in h["Content-Security-Policy"]


def _make_video(path, n=300, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for i in range(n):
        w.write(np.full((360, 640, 3), 80 + (i % 50), np.uint8))
    w.release()


def test_upload_camera_session_incident_case_handoff_over_http(api, tmp_path):
    base, svc = api
    vid = tmp_path / "clip.mp4"
    _make_video(vid)
    adm, inv, rev, off = (client(base, u) for u in ("admin1", "inv1", "rev1", "off1"))
    st, up, _ = adm.call("POST", "/uploads?filename=clip.mp4", raw=vid.read_bytes())
    assert st == 201 and up["frames"] == 300 and up["fps"] == 10
    assert adm.call("POST", "/uploads?filename=evil.exe", raw=b"MZ")[0] == 422
    assert adm.call("POST", "/uploads?filename=junk.mp4", raw=b"not a video at all")[0] == 422
    st, cam, _ = adm.call("POST", "/cameras", {"camera_id": "up-cam", "name": "Upload cam", "source_type": "file", "source_uri": up["path"], "data_class": "demo",
                                              "config": {"stop_criterion": "displacement", "recording_start_utc": "2026-10-01T10:00:00.000Z"},
                                              "location_label": "Test road", "latitude": 12.97, "longitude": 77.59})
    assert st == 201, cam
    st, sess, _ = inv.call("POST", "/cameras/up-cam/sessions")
    assert st == 202
    sid = sess["session_id"]
    for _ in range(300):
        rt = adm.call("GET", f"/sessions/{sid}")[1]["runtime"]
        if rt and rt["state"] in ("completed", "failed", "stopped"):
            break
        time.sleep(0.1)
    rt = adm.call("GET", f"/sessions/{sid}")[1]
    assert rt["state"] == "completed", rt
    assert rt["time_basis"] == "declared_recording_start" and rt["recording_start_utc"] == "2026-10-01T10:00:00.000Z"
    st, listing, _ = inv.call("GET", f"/incidents?session_id={sid}&event_type=stopped_vehicle")
    assert st == 200 and listing["total"] >= 1
    inc = listing["items"][0]
    assert inc["thumbnail_asset_id"] and inc["start_utc"].startswith("2026-10-01T10:00:0")
    st, detail, _ = inv.call("GET", f"/incidents/{inc['incident_id']}")
    assert detail["source_video"] is True and {"frame", "clip", "frame_annotated"} <= {a["kind"] for a in detail["evidence"]}
    st, vid_bytes, h = inv.call("GET", f"/sessions/{sid}/video", headers={"Range": "bytes=0-1023"})
    assert st == 206 and len(vid_bytes) == 1024
    # investigator -> reviewer -> officer
    st, case, _ = inv.call("POST", f"/incidents/{inc['incident_id']}/cases", {"offence_code": "STOP-OBS"})
    cid = case["case_id"]
    inv.call("POST", f"/cases/{cid}/actions", {"action": "submit_for_evidence_verification"})
    assert rev.call("POST", f"/cases/{cid}/actions", {"action": "verify_evidence", "note": "clip and frames reviewed"})[0] == 200
    st, pl, _ = inv.call("POST", f"/incidents/{inc['incident_id']}/plates", {"text": "mh12 ab 1234", "track_id": inc["track_ids"][0]})
    assert st == 201 and pl["source"] == "manual_entry"
    assert inv.call("POST", f"/incidents/{inc['incident_id']}/plates/ocr", {"track_id": inc["track_ids"][0]})[1]["status"] == "unavailable"
    assert inv.call("POST", f"/plates/{pl['reading_id']}/verify", {})[0] == 403  # investigators cannot verify
    assert rev.call("POST", f"/cases/{cid}/actions", {"action": "verify_plate"})[0] == 200
    st, look, _ = off.call("POST", f"/cases/{cid}/registry-lookup", {})
    assert st == 201 and look["status"] == "found" and "SYNTHETIC" in look["data"]["owner_name"] and look["is_simulated"]
    st, revcase, _ = rev.call("GET", f"/cases/{cid}")
    assert "owner_name" not in revcase["registry_lookups"][0]["data"] and revcase["registry_lookups"][0]["owner_fields_hidden"]
    st, body, _ = off.call("POST", f"/cases/{cid}/actions", {"action": "approve"})
    assert st == 200 and body["state"] == "approved_for_submission"
    st, ho, _ = off.call("POST", f"/cases/{cid}/handoff", {"adapter": "sandbox"})
    assert st == 201 and ho["external_reference"].startswith("TEST-CHALLAN-") and ho["notice"].startswith("SANDBOX")
    st, again, _ = off.call("POST", f"/cases/{cid}/handoff", {"adapter": "sandbox"})
    assert st == 409  # state already moved on: no double submission
    st, html, h = off.call("GET", f"/cases/{cid}/report.html")
    assert st == 200 and h["Content-Type"].startswith("text/html") and b"NON-PRODUCTION DATA" in html and b"TEST-CHALLAN" in html
    st, pkg, h = off.call("GET", f"/cases/{cid}/package")
    assert st == 200 and pkg[:2] == b"PK" and "attachment" in h["Content-Disposition"]
    # the audit trail names who did what
    st, au, _ = off.call("GET", f"/audit?entity_type=case&entity_id={cid}&limit=100")
    who = {(a["action"], a["actor"]) for a in au["items"]}
    assert ("case.verify_evidence", "rev1") in who and ("case.approve", "off1") in who and ("case.create", "inv1") in who
    assert adm.call("GET", "/audit/verify")[1]["ok"] is True


def test_overview_congestion_stats_and_system_status_are_real_data(api):
    base, svc = api
    seed_incident(svc, key="a", severity="high", event_type="possible_collision")
    c = client(base, "off1")
    ov = c.call("GET", "/overview")[1]
    assert ov["incidents"]["open_total"] == 1 and ov["incidents"]["by_type"] == {"possible_collision": 1}
    assert ov["cameras"]["total"] == 1 and ov["alerts"]["pending"] in (0, 1) and ov["recent_activity"]  # the alert worker may have run already
    assert len(c.call("GET", "/alerts")[1]["items"]) == 1
    sysst = c.call("GET", "/system/status")[1]
    ints = sysst["integrations"]
    assert ints["ocr"]["available"] is False and ints["registry"][0]["state"] == "not_connected"
    assert [x for x in ints["challan"] if x["key"] == "official"][0]["state"] == "not_connected"
    assert [a for a in ints["alerts"] if a["adapter"] == "test"][0]["is_test"] is True
    assert c.call("GET", "/congestion")[0] == 200 and c.call("GET", "/stats")[0] == 200
    caps = c.call("GET", "/capabilities")[1]["events"]
    assert {e["event_type"]: e["status"] for e in caps}["helmet_violation"] == "needs_model"


def test_settings_admin_only_and_validated(api):
    base, svc = api
    adm, off = client(base, "admin1"), client(base, "off1")
    assert off.call("GET", "/settings")[0] == 403
    st, body, _ = adm.call("PUT", "/settings", {"retention": {"evidence_days": 90, "audit_days": 3650}})
    assert st == 200 and body["retention"]["evidence_days"] == 90
    assert adm.call("PUT", "/settings", {"retention": {"evidence_days": 0, "audit_days": 1}})[0] == 422
    assert adm.call("PUT", "/settings", {"alert_rules": [{"adapter": "webhook", "min_severity": "high", "recipient": "x"}]})[0] == 422
    assert adm.call("PUT", "/settings", {"detection_defaults": {"bogus": 1}})[0] == 422
    rep = adm.call("GET", "/retention/report")[1]
    assert rep["purge_implemented"] is True and rep["purge_scope"] == "evidence files only" and rep["audit_purge_implemented"] is False
    self_lockout = adm.call("PATCH", f"/users/{svc.auth.list_users()[0]['user_id']}", {"active": False})
    assert self_lockout[0] in (200, 409)


def test_unread_request_bodies_do_not_corrupt_keepalive_connections(api):
    """Regression: POST handlers that ignore their body (e.g. starting a session) used to leave the bytes on the
    connection, so the next request on it parsed as '{}GET ...' and failed with 501."""
    import http.client
    base, _ = api
    host, port = base.replace("http://", "").split(":")
    c = Client(base)
    c.login("inv1")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    hdr = {"Authorization": "Bearer " + c.token, "Content-Type": "application/json", "X-Requested-With": "anvesh"}
    conn.request("POST", "/api/v1/auth/logout", body=b'{"ignored": "body"}', headers=hdr)
    r = conn.getresponse()
    r.read()
    assert r.status == 200
    conn.request("GET", "/api/v1/health")  # same connection, must still parse
    r2 = conn.getresponse()
    assert r2.status == 200 and json.loads(r2.read())["status"] == "ok"
    conn.request("POST", "/api/v1/cameras", body=b'{"x": 1}', headers={"Content-Type": "application/json"})  # 401 before the body is read
    r3 = conn.getresponse()
    r3.read()
    assert r3.status == 401
    conn.request("GET", "/api/v1/health")
    assert conn.getresponse().status == 200
    conn.close()
