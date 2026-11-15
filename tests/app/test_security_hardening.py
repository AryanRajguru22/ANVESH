"""Security / reliability regressions found by the 2026-10-20 audit (docs/ANVESH_EXECUTION_REPORT.md, section 9).

Every test here failed against the code as it stood before the audit and passes after the fix. They use the real
HTTP server on an ephemeral port; raw sockets are used where the standard client would hide the defect.
"""

import http.client
import json
import re
import socket
import time

import pytest

from anvesh.app.cameras import ConfigError
from anvesh.app.config import AppConfig
from anvesh.app.plates import UnavailableOcr, UnavailablePlateDetector
from anvesh.app.runner import SessionManager
from anvesh.app.server import AppServer
from anvesh.app.services import build_services
from anvesh.app.synthetic import ScriptedTracker

from .conftest import seed_incident
from .test_api import Client, api, client  # noqa: F401  (api is a fixture)


def _raw(base: str, payload: bytes, timeout: float = 4.0):
    """Send raw bytes; return (everything the server sent, whether it then closed the connection). Data is None if the server sent
    nothing at all within `timeout`. The response is read until EOF/reset/idle because headers and body can arrive in separate segments."""
    host, port = base.replace("http://", "").split(":")
    s = socket.create_connection((host, int(port)), timeout=timeout)
    try:
        s.sendall(payload)
        s.settimeout(timeout)
        try:
            data = s.recv(65536)
        except socket.timeout:
            return None, False
        s.settimeout(1.5)
        closed = False
        while True:
            try:
                more = s.recv(65536)
            except socket.timeout:
                break  # still open and idle
            except OSError:
                closed = True  # reset by the peer: the connection is gone
                break
            if not more:
                closed = True
                break
            data += more
        return data, closed
    finally:
        s.close()


def _login_request(extra_headers: str, body: bytes = b"") -> bytes:
    return (b"POST /api/v1/auth/login HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n" + extra_headers.encode() + b"\r\n" + body)


# -- request framing -------------------------------------------------------------------------------------------


def test_negative_content_length_is_rejected_instead_of_blocking_the_handler_thread(api):
    base, _ = api
    started = time.time()
    first, _closed = _raw(base, _login_request("Content-Length: -1\r\n"), timeout=3.0)
    assert first is not None, "server did not answer a negative Content-Length (handler thread is blocked in rfile.read(-1))"
    assert first.startswith(b"HTTP/1.1 4"), first[:80]
    assert time.time() - started < 3.0


def test_non_numeric_content_length_is_a_client_error_not_a_500(api):
    base, _ = api
    first, _ = _raw(base, _login_request("Content-Length: banana\r\n"), timeout=3.0)
    assert first is not None and first.startswith(b"HTTP/1.1 4"), first


def test_chunked_request_bodies_are_refused_and_the_connection_is_closed(api):
    """The server frames bodies by Content-Length only. An unrefused chunked body would stay on the keep-alive connection
    and be parsed as the next request."""
    base, _ = api
    smuggled = b"0\r\n\r\nGET /api/v1/audit HTTP/1.1\r\nHost: x\r\n\r\n"
    first, closed = _raw(base, _login_request("Transfer-Encoding: chunked\r\n", b"5\r\n{}   \r\n" + smuggled), timeout=3.0)
    assert first is not None and first.startswith(b"HTTP/1.1 411"), first
    assert closed, "connection must be closed so the unread chunked bytes cannot be parsed as a request"
    assert first.count(b"HTTP/1.1") == 1, "a second (smuggled) request was answered on the same connection"


def test_idle_and_half_open_connections_are_dropped_after_the_request_timeout(services, users):
    mgr = SessionManager(services, tracker_factory=lambda cam, cfg: ScriptedTracker("stopped_car"))
    server = AppServer(services, mgr, host="127.0.0.1", port=0, request_timeout_s=0.6)
    server.start()
    try:
        host, port = "127.0.0.1", server.port
        s = socket.create_connection((host, port), timeout=5)
        s.sendall(b"GET /api/v1/hea")  # slow-loris: never completes the request line
        s.settimeout(4.0)
        t0 = time.time()
        try:
            data = s.recv(100)
        except socket.timeout:
            pytest.fail("a connection that never finished its request was not dropped (no per-connection timeout)")
        assert data == b"" and time.time() - t0 < 3.5
        s.close()
        # the server is still healthy
        c = Client(f"http://{host}:{port}")
        assert c.call("GET", "/health")[0] == 200
    finally:
        server.stop()


# -- file-source disclosure ------------------------------------------------------------------------------------


@pytest.fixture()
def svc_in_media_root(tmp_path, monkeypatch):
    """Services whose data directory lies INSIDE an allowed media root - the default layout, where the data
    directory is `<repo>/outputs/app` and the repository root is an allowed media root."""
    root = tmp_path / "media_root"
    root.mkdir()
    monkeypatch.setenv("ANVESH_MEDIA_ROOTS", str(root))
    cfg = AppConfig(data_dir=root / "data")
    svc = build_services(cfg, ocr=UnavailableOcr("test", "n/a"), plate_detector=UnavailablePlateDetector("test"))
    yield svc, root
    svc.close()


def _file_cam(uri, cid="c-file"):
    return {"camera_id": cid, "name": "f", "source_type": "file", "source_uri": str(uri), "data_class": "demo"}


def test_file_cameras_cannot_point_at_the_application_database_credentials_or_non_video_files(svc_in_media_root):
    svc, root = svc_in_media_root
    (root / "data" / "initial_credentials.txt").write_text("admin\tadministrator\tsecret\n")
    for bad in (svc.config.db_path, root / "data" / "initial_credentials.txt", root / "notes.txt", root / "clip.mp4.sqlite3"):
        with pytest.raises(ConfigError):
            svc.cameras.create("admin", _file_cam(bad, "c-" + bad.stem[:8]), "administrator")
    # video files in the media root and uploads are still fine
    svc.cameras.create("admin", _file_cam(root / "clip.mp4", "c-ok1"), "administrator")
    svc.cameras.create("admin", _file_cam(svc.config.uploads_dir / "abcd1234-up.mp4", "c-ok2"), "administrator")
    # ... but a video-named path inside the protected data tree (evidence/exports) is not
    with pytest.raises(ConfigError):
        svc.cameras.create("admin", _file_cam(svc.config.evidence_dir / "INC-1" / "AST-1.mp4", "c-bad2"), "administrator")


def test_update_cannot_repoint_a_file_camera_at_a_protected_file(svc_in_media_root):
    svc, root = svc_in_media_root
    svc.cameras.create("admin", _file_cam(root / "clip.mp4", "c-up"), "administrator")
    with pytest.raises(ConfigError):
        svc.cameras.update("admin", "c-up", {"source_uri": str(svc.config.db_path)}, "administrator")


def test_replay_endpoint_refuses_a_legacy_camera_that_points_at_a_non_video_or_protected_file(svc_in_media_root):
    """Defence in depth: a camera row written before the validation existed must not make /sessions/<id>/video a file reader."""
    svc, root = svc_in_media_root
    svc.auth.create_user("admin1", "Admin", "administrator", "pw-admin1-long")
    svc.auth.create_user("rev1", "Rev", "reviewer", "pw-rev1-long")
    now = "2026-10-20T00:00:00.000Z"
    svc.db.execute("INSERT INTO cameras(camera_id, name, source_type, source_uri, data_class, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                   ("c-legacy", "legacy", "file", str(svc.config.db_path), "demo", now, now))
    snap = json.dumps({"source_type": "file", "source": str(svc.config.db_path)})
    svc.db.execute("INSERT INTO processing_sessions(session_id, camera_id, run_id, state, data_class, started_at, time_basis, recording_start_utc, "
                   "source_snapshot_json) VALUES (?,?,?,?,?,?,?,?,?)", ("SES-LEGACY", "c-legacy", "r", "failed", "demo", now, "processing_clock", now, snap))
    server = AppServer(svc, SessionManager(svc, tracker_factory=lambda cam, cfg: ScriptedTracker("stopped_car")), host="127.0.0.1", port=0)
    server.start()
    try:
        c = client(f"http://127.0.0.1:{server.port}", "rev1")
        st, body, headers = c.call("GET", "/sessions/SES-LEGACY/video")
        assert st in (404, 422), (st, headers)
        assert not (isinstance(body, bytes) and body.startswith(b"SQLite format 3")), "the application database was served"
        # and the legacy camera cannot be started either
        adm = client(f"http://127.0.0.1:{server.port}", "admin1")
        st, body, _ = adm.call("POST", "/cameras/c-legacy/sessions", {})
        assert st == 422, (st, body)
    finally:
        server.stop()


# -- camera update validation ---------------------------------------------------------------------------------


def test_camera_update_validates_every_field_it_writes(api):
    base, svc = api
    adm = client(base, "admin1")
    st, cam, _ = adm.call("POST", "/cameras", {"camera_id": "c-upd", "name": "ok", "source_type": "rtsp", "source_uri": "rtsp://h/s", "nominal_fps": 25, "data_class": "demo"})
    assert st == 201, cam
    for body in ({"latitude": 500}, {"longitude": "east"}, {"utc_offset_minutes": 99999}, {"utc_offset_minutes": "330"}, {"name": ""}, {"name": 5},
                 {"nominal_fps": 0}, {"nominal_fps": 9999}, {"nominal_fps": "fast"}, {"enabled": "yes"}, {"location_label": 5}):
        st, resp, _ = adm.call("PATCH", "/cameras/c-upd", body)
        assert st == 422, (body, st, resp)
    st, resp, _ = adm.call("PATCH", "/cameras/c-upd", {"latitude": 12.9, "longitude": 77.6, "name": "renamed", "nominal_fps": 30, "enabled": False})
    assert st == 200 and resp["name"] == "renamed" and resp["enabled"] is False


def test_malformed_nested_camera_config_is_a_422_not_a_500(api):
    base, _ = api
    adm = client(base, "admin1")
    base_body = {"name": "x", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "simulated"}
    for cfg in ({"direction_zones": ["notadict"]}, {"lanes": [5]}, {"turn_rules": [[]]}, {"signal": "red"}, {"thresholds": [1]},
                {"calibration": "x"}, {"calibration": {"reference_points": [5]}}, {"direction_zones": {"a": 1}}, {"lanes": [{"lane_id": "a", "vertices": "x"}]}):
        st, resp, _ = adm.call("POST", "/cameras", dict(base_body, config=cfg))
        assert st == 422, (cfg, st, resp)


# -- accounts -------------------------------------------------------------------------------------------------


def test_user_update_input_types_and_self_lockout_cannot_be_bypassed(api):
    base, svc = api
    adm = client(base, "admin1")
    me = svc.auth.get_user(next(u["user_id"] for u in svc.auth.list_users() if u["username"] == "admin1"))
    other = next(u["user_id"] for u in svc.auth.list_users() if u["username"] == "inv1")
    for bad in ({"active": 0}, {"active": "false"}, {"active": "no"}):
        st, resp, _ = adm.call("PATCH", f"/users/{me['user_id']}", bad)
        assert st in (409, 422), (bad, st)
        assert svc.auth.get_user(me["user_id"])["active"] == 1, bad
        st, resp, _ = adm.call("PATCH", f"/users/{other}", bad)
        assert st == 422, (bad, st, resp)  # not coerced by truthiness
    assert svc.auth.get_user(other)["active"] == 1
    # the existing weak check accepted 200 here; the self-deactivation really must be refused
    assert adm.call("PATCH", f"/users/{me['user_id']}", {"active": False})[0] == 409
    assert adm.call("PATCH", f"/users/{me['user_id']}", {"role": "officer"})[0] == 409
    for bad in ({"password": 12345678901}, {"password": ["x"] * 12}, {"role": 5}, {"role": ["officer"]}):
        st, resp, _ = adm.call("PATCH", f"/users/{other}", bad)
        assert st == 422, (bad, st, resp)
    for bad in ({"username": 5, "display_name": "x", "role": "officer", "password": "longenough-pw"},
                {"username": "u2", "display_name": "x", "role": "officer", "password": 12345678901},
                {"username": "u2", "display_name": ["x"], "role": "officer", "password": "longenough-pw"},
                {"username": "u" * 500, "display_name": "x", "role": "officer", "password": "longenough-pw"},
                {"username": "u2", "display_name": "x", "role": "officer", "password": "p" * 5000}):
        st, resp, _ = adm.call("POST", "/users", bad)
        assert st == 422, (bad.get("username") if isinstance(bad.get("username"), int) else "...", st, resp)


def test_login_input_is_bounded_and_the_failure_table_cannot_grow_without_limit(api):
    base, svc = api
    c = Client(base)
    st, resp, _ = c.call("POST", "/auth/login", {"username": "x" * 100000, "password": "y"})
    assert st in (401, 422)
    last = svc.audit.query(action="auth.login_failed", limit=1)[0]
    assert all(len(r["actor"]) <= 128 for r in last), "a failed login must not write an unbounded actor name into the audit log"
    from anvesh.app.auth import LoginThrottle
    for i in range(LoginThrottle.MAX_ENTRIES + 50):
        svc.auth.throttle.failure(f"nobody{i}", f"10.0.{i // 250}.{i % 250}")
    assert all(n <= LoginThrottle.MAX_ENTRIES for n in svc.auth.throttle.sizes().values())


# -- case actions ---------------------------------------------------------------------------------------------


def test_case_action_cannot_be_fed_reserved_parameter_names(api):
    base, svc = api
    inc = seed_incident(svc)
    inv = client(base, "inv1")
    st, case, _ = inv.call("POST", f"/incidents/{inc['incident_id']}/cases", {"offence_code": "STOP-OBS"})
    assert st == 201, case
    for extra in ({"user": "x"}, {"case_id": "other"}, {"action": "submit_for_evidence_verification", "self": 1}, {"note": ["x"]}, {"note": 5}):
        st, resp, _ = inv.call("POST", f"/cases/{case['case_id']}/actions", dict({"action": "submit_for_evidence_verification"}, **extra))
        assert st < 500, (extra, st, resp)
    st, resp, _ = inv.call("POST", f"/cases/{case['case_id']}/actions", {"action": ["x"]})
    assert st == 422


# -- plate verification must follow the plate -----------------------------------------------------------------


def test_a_plate_changed_after_verification_cannot_be_approved_on_the_old_verification(services, users):
    """rev1 verified plate X; rev2 then corrects it to Y (a different vehicle). plate_verified_by still names rev1, who never saw Y."""
    from anvesh.app.cases import CaseError
    from .test_cases import _ready_case

    inc, cid = _ready_case(services, users, plate="MH12AB1234")
    head = services.plates.current(inc["incident_id"])[0]
    assert head["verified"]
    services.plates.correct(head["reading_id"], "KA01XY9999", "rev2", "reviewer", note="misread digit", can_verify=True)
    with pytest.raises(CaseError) as e:  # approval must be refused because the plate changed after verification
        services.cases.transition(cid, "approve", users["off1"])
    assert e.value.status == 422
    assert any(x.startswith("plate_verified") for x in e.value.errors), e.value.errors
    # the proper route back: request_more clears verifications, then the normal chain verifies the NEW plate
    services.cases.transition(cid, "request_more", users["off1"], note="plate changed after verification")
    services.cases.transition(cid, "resume_review", users["inv1"])
    services.cases.transition(cid, "verify_evidence", users["rev1"])
    services.cases.transition(cid, "verify_plate", users["rev1"], note="re-check")
    case = services.cases.get(cid, users["off1"])
    assert case["plate_verified_reading_id"] == services.cases.plate_for(case)["reading_id"]


# -- no unhandled exception for any malformed request body on any mutating route ------------------------------

_FIELDS = ("username password display_name role active name source_type source_uri nominal_fps config status assigned_to note cause basis text "
           "track_id offence_code reason limitations action adapter reading_id retention detection_defaults offences alert_rules state camera_id "
           "latitude longitude utc_offset_minutes enabled location_label data_class current_password new_password external_reference "
           "registry_override_reason allow_nonstandard evidence_days audit_days confirm dry_run").split()
_RAW_BODIES = (b"[]", b'"str"', b"null", b"123", b"true", b'{"a":[]}', b"{", b"\xff\xfe\x00", b"{" * 5000)
_WRONG = (None, [], 5, "x", {}, True)
_SKIP = {("POST", "/auth/logout"), ("POST", "/auth/login"), ("POST", "/cameras/{camera_id}/sessions"), ("POST", "/uploads")}


def _route_table(server, ids):
    out = []
    for method, regex, _handler, _public in server.routes:
        if method == "GET":
            continue
        pattern = regex.pattern[1:-1]
        names = re.findall(r"\(\?P<(\w+)>", pattern)
        template = re.sub(r"\(\?P<(\w+)>\[\^/\]\+\)", lambda m: "{" + m.group(1) + "}", pattern)
        if (method, template) in _SKIP:
            continue
        path = template
        for n in names:
            path = path.replace("{" + n + "}", ids.get(n, "x"))
        out.append((method, template, path))
    return out


def _fuzz(services, extra_routes=()):
    """POST/PATCH/PUT malformed and wrongly-typed JSON at every mutating route as each of four roles. Returns (failures, status counts)."""
    mgr = SessionManager(services, tracker_factory=lambda cam, cfg: ScriptedTracker("stopped_car"))
    server = AppServer(services, mgr, host="127.0.0.1", port=0)
    for method, pattern, handler in extra_routes:
        server.route(method, pattern, handler)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        inc = seed_incident(services)
        inv = client(base, "inv1")
        case = inv.call("POST", f"/incidents/{inc['incident_id']}/cases", {"offence_code": "STOP-OBS"})[1]
        reading = services.plates.manual_entry(inc["incident_id"], "MH12AB1234", "inv1", "investigator", track_id="run-s000-t00001")
        asset = services.evidence.list(inc["incident_id"])[0]
        ids = {"incident_id": inc["incident_id"], "case_id": case["case_id"], "camera_id": "cam-seed", "reading_id": reading["reading_id"],
               "asset_id": asset["asset_id"], "session_id": "SES-CAMEED1", "delivery_id": "x",
               "user_id": next(u["user_id"] for u in services.auth.list_users() if u["username"] == "rev2")}
        table = _route_table(server, ids)
        assert len(table) >= 20
        failures, counts = [], {}
        host, port = "127.0.0.1", server.port
        for user in ("admin1", "off1", "inv1", "rev1"):
            token = client(base, user).token
            hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
            conn = http.client.HTTPConnection(host, port, timeout=20)
            per_field = user in ("admin1", "off1")  # the two roles that reach the most handlers; the others get whole-body variants only
            bodies = [("raw", b) for b in _RAW_BODIES]
            bodies += [("all", json.dumps({f: w for f in _FIELDS}).encode()) for w in _WRONG]
            if per_field:
                bodies += [(f, json.dumps({f: w}).encode()) for f in _FIELDS for w in _WRONG]
            for method, template, path in table:
                for label, raw in bodies:
                    try:
                        conn.request(method, "/api/v1" + path, body=raw, headers=hdr)
                        r = conn.getresponse()
                        payload = r.read()
                        st = r.status
                    except (http.client.HTTPException, OSError):
                        conn.close()
                        conn = http.client.HTTPConnection(host, port, timeout=20)
                        continue
                    counts[st // 100] = counts.get(st // 100, 0) + 1
                    if st >= 500:
                        failures.append((user, method, template, label, raw[:60], st, payload[:120]))
                        break  # one report per route and user
            conn.close()
        return failures, counts
    finally:
        server.stop()


def test_fuzz_harness_detects_an_unhandled_exception(services, users):
    """Self-check: a handler that blows up is reported, so a green fuzz run below means something."""
    failures, counts = _fuzz(services, extra_routes=[("POST", "/_fuzz_probe", lambda req: 1 / 0)])
    assert any(f[2] == "/_fuzz_probe" and f[5] == 500 for f in failures), failures


def test_no_mutating_endpoint_returns_500_for_malformed_json_bodies(services, users):
    failures, counts = _fuzz(services)
    assert not failures, "unhandled errors:\n" + "\n".join(map(str, failures))
    # the run must actually have exercised the handlers, not just bounced off authentication
    assert counts.get(4, 0) > 2000 and counts.get(2, 0) > 0, counts
    assert services.audit.verify()["ok"]
