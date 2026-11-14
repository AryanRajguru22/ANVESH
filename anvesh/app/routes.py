"""All /api/v1 endpoints. Handlers return a dict (200), a (status, dict) tuple, or None after writing a raw
response themselves (files, streams, HTML). Permissions are checked first, inside each handler."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from anvesh.app import cameras as cams
from anvesh.app.auth import PRIVILEGED_ROLES, ROLES
from anvesh.app.capabilities import CAPABILITIES, all_category_status, category_status
from anvesh.app.cases import CASE_STATES
from anvesh.app.db import loads
from anvesh.app.incidents import STATUS_TRANSITIONS
from anvesh.app.reports import build_case_package, case_report_html
from anvesh.app.runner import TERMINAL_STATES
from anvesh.app.server import API, COOKIE, ApiError
from anvesh.app.timeutil import now_iso

_VIEW_AUDIT_THROTTLE: dict = {}
PRODUCT_STATUS = {
    "mode": "research_demo", "operational_use_authorised": False,
    "reasons": ["no authorised vehicle-registry or e-challan integration is connected", "no accuracy measurement on the intended cameras",
                "no real camera/RTSP validation", "model and dependency licences are not cleared for deployment",
                "multi-factor authentication is not implemented"],
}
# The only extra parameters POST /cases/{id}/actions forwards to the case service (name -> required type).
CASE_ACTION_PARAMS = {"reading_id": str, "allow_nonstandard": bool, "external_reference": str}


def _paged(items, total, limit, offset) -> dict:
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def register(app) -> None:  # noqa: C901 - a route table is naturally long
    s = app.s
    R = app.route

    def page(req):
        return req.qint("limit", 50, 1, 500), req.qint("offset", 0, 0, 10 ** 9)

    # DEMO accounts (run_app.py --demo-users) are confined to demo/simulated material: every WRITE that touches operational material is refused.
    # Reading is not restricted (a demo reviewer may look at what is on screen); see docs/ANVESH_ACCOUNT_SECURITY.md section 4.
    def refuse_demo_on_operational(req, data_class):
        if req.user.is_demo and data_class == "operational":
            raise ApiError(403, "demo_account", "demo accounts cannot change operational material")

    def incident_class(iid):
        inc = s.incidents.get(iid)
        if inc is None:
            raise LookupError("no such incident")
        return inc["data_class"]

    def reading_class(rid):
        r = s.plates.get(rid)
        if r is None:
            raise LookupError("no such plate reading")
        return incident_class(r["incident_id"])

    def camera_class(cid):
        c = s.cameras.get_raw(cid)
        if c is None:
            raise LookupError("no such camera")
        return c["data_class"]

    def audit_view(req, kind: str, entity: str, ident: str, throttle_s: float = 300.0) -> None:
        key = (req.user.username, kind, ident)
        now = time.time()
        if len(_VIEW_AUDIT_THROTTLE) > 5000:
            _VIEW_AUDIT_THROTTLE.clear()  # bounded: the throttle only suppresses duplicate audit rows
        if now - _VIEW_AUDIT_THROTTLE.get(key, 0) > throttle_s:
            _VIEW_AUDIT_THROTTLE[key] = now
            s.audit.append(req.user.username, kind, entity, ident, {}, actor_role=req.user.role, remote_addr=req.h.client_address[0])

    # ---------------------------------------------------------------- health / system
    def health(req):
        return {"status": "ok", "time": now_iso()}

    def ready(req):
        checks = {"database": False, "schema_version": None, "model_file": Path(s.config.model_path).is_file(),
                  "evidence_dir_writable": False}
        try:
            checks["schema_version"] = s.db.schema_version()
            checks["database"] = True
            probe = s.config.evidence_dir / ".probe"
            probe.write_text("x")
            probe.unlink()
            checks["evidence_dir_writable"] = True
        except Exception:  # noqa: BLE001
            pass
        ok = checks["database"] and checks["evidence_dir_writable"]
        notes = [] if checks["model_file"] else ["detector weights not found: live/recorded perception cannot start until they are present"]
        detector = {"source": s.config.model_source, "weights": Path(s.config.model_path).name}
        return (200 if ok else 503), {"ready": ok, "checks": checks, "notes": notes, "detector": detector}

    R("GET", "/health", health, public=True)
    R("GET", "/ready", ready, public=True)

    def system_status(req):
        req.require("view.overview")
        gpu = None
        try:
            import torch
            gpu = {"cuda": bool(torch.cuda.is_available()), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except Exception:  # noqa: BLE001
            gpu = {"cuda": False, "device": None, "note": "torch not importable"}
        return {
            "version": "0.2.0-mvp", "time": now_iso(), "gpu": gpu, "schema_version": s.db.schema_version(),
            "detector_weights_present": Path(s.config.model_path).is_file(),
            "integrations": {
                "ocr": s.plates.status()["ocr"], "plate_detector": s.plates.status()["detector"],
                "registry": s.registry.status(), "challan": s.handoff.status(), "alerts": s.alerts.adapter_status(),
            },
            "policy": {"require_distinct_approver": s.config.require_distinct_approver,
                       "evidence_retention_days": s.settings.get("retention", {}).get("evidence_days", s.config.evidence_retention_days),
                       "audit_retention_days": s.settings.get("retention", {}).get("audit_days", s.config.audit_retention_days)},
            "audit_head": s.audit.head(),
            "security": s.auth.posture(),
            "product": PRODUCT_STATUS,
            "models": model_status(),
        }

    def model_status():
        pd, ocr = s.plates.status()["detector"], s.plates.status()["ocr"]
        return {
            "vehicle_detector": {"available": Path(s.config.model_path).is_file(), "source": s.config.model_source, "weights": Path(s.config.model_path).name,
                                 "accuracy_on_your_cameras": "not measured (no labelled local footage)",
                                 "licence_status": "NOT CLEARED: Ultralytics AGPL-3.0 / enterprise question open; MIO-TCD (CC BY-NC-SA 4.0) governs fine-tuned weights"},
            "plate_detector": {"available": bool(pd.get("available")), "status": "available" if pd.get("available") else "not installed"},
            "ocr": {"available": bool(ocr.get("available")), "status": "available (accuracy unmeasured)" if ocr.get("available") else "not installed - plates are manual entry only"},
            "accident_model": {"available": False, "status": "not implemented; possible_collision is an experimental rule heuristic"},
        }

    R("GET", "/system/status", system_status)

    def capabilities(req):
        req.require("view.overview")
        cid = req.q("camera_id")
        cfg, geom = None, False
        if cid:
            cam = s.cameras.get_raw(cid)
            if cam is None:
                raise LookupError(f"no such camera {cid!r}")
            cfg = cams.parse_camera_config(loads(cam["config_json"], {}))
            if cfg.calibration:
                geom = cams.build_calibration_profile("probe", cid, cfg.calibration).supports_metric_speed
        return {"camera_id": cid, "events": all_category_status(cfg, geom), "product": PRODUCT_STATUS, "models": model_status()}

    R("GET", "/capabilities", capabilities)

    # ---------------------------------------------------------------- auth
    def login(req):
        token, user = s.auth.login(req.need("username"), req.need("password"), req.h.client_address[0])
        secure = "; Secure" if s.config.secure_cookies else ""
        cookie = f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={s.config.session_ttl_s}{secure}"
        return 200, {"token": token, "user": user.public()}, {"Set-Cookie": cookie}

    def logout(req):
        s.auth.logout(req.token, req.user)
        return 200, {"ok": True}, {"Set-Cookie": f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"}

    def me(req):
        return {"user": req.user.public()}

    def change_password(req):
        s.auth.change_password(req.user.user_id, req.need("current_password", maxlen=1024), req.need("new_password", maxlen=1024), req.h.client_address[0])
        return 200, {"ok": True, "note": "password changed; all sessions were signed out - sign in again"}, \
            {"Set-Cookie": f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"}

    def demo_login(req):
        """ONE-CLICK sign-in for a local demonstration. Only exists when the server was started with `demo_login=(user, password)`
        (scripts/run_demo.py --one-click) AND listens on a loopback address AND the caller is on loopback. Otherwise it does not exist."""
        cred = getattr(app, "demo_login", None)
        if not cred or app.host not in ("127.0.0.1", "localhost", "::1") or req.h.client_address[0] not in ("127.0.0.1", "::1"):
            raise ApiError(404, "not_found", "no such endpoint")
        token, user = s.auth.login(cred[0], cred[1], req.h.client_address[0])
        secure = "; Secure" if s.config.secure_cookies else ""
        cookie = f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={s.config.session_ttl_s}{secure}"
        return 200, {"token": token, "user": user.public()}, {"Set-Cookie": cookie}

    def demo_available(req):
        cred = getattr(app, "demo_login", None)
        ok = bool(cred) and app.host in ("127.0.0.1", "localhost", "::1") and req.h.client_address[0] in ("127.0.0.1", "::1")
        return {"one_click": ok}

    R("POST", "/auth/login", login, public=True)
    R("POST", "/auth/demo", demo_login, public=True)
    R("GET", "/auth/demo", demo_available, public=True)
    R("POST", "/auth/logout", logout)
    R("GET", "/auth/me", me)
    R("POST", "/auth/password", change_password)

    # ---------------------------------------------------------------- users / settings (admin)
    def users_list(req):
        req.require("admin.users")
        return {"items": s.auth.list_users(), "roles": list(ROLES), "privileged_roles": list(PRIVILEGED_ROLES),
                "pending_requests": len(s.accounts.list_requests(req.user, "pending"))}

    def users_create(req):
        req.require("admin.users")
        r = s.accounts.create_user(req.user, req.need("username", maxlen=128), req.need("display_name", maxlen=256), req.need("role", maxlen=32),
                                   req.need("password", maxlen=1024))
        return (201 if r["status"] == "created" else 202), r

    def users_update(req):
        req.require("admin.users")
        req.body  # parse (and reject a malformed body) before anything else
        active = req.opt("active", bool)  # a non-boolean is refused, never coerced by truthiness ("false", 0, ...)
        r = s.accounts.update_user(req.user, req.params["user_id"], req.opt("role", str, None, maxlen=32), active, req.opt("password", str, None, maxlen=1024))
        return (202 if r["status"] == "pending" else 200), r

    def users_unlock(req):
        req.require("admin.users")
        return s.accounts.unlock(req.user, req.params["user_id"])

    def requests_list(req):
        req.require("admin.users")
        return {"items": s.accounts.list_requests(req.user, req.q("status"))}

    def request_approve(req):
        req.require("admin.users")
        return s.accounts.approve(req.user, req.params["request_id"], req.opt("note", str, ""))

    def request_reject(req):
        req.require("admin.users")
        return s.accounts.reject(req.user, req.params["request_id"], req.opt("note", str, ""))

    R("GET", "/users", users_list)
    R("POST", "/users", users_create)
    R("GET", "/users/requests", requests_list)
    R("POST", "/users/requests/{request_id}/approve", request_approve)
    R("POST", "/users/requests/{request_id}/reject", request_reject)
    R("PATCH", "/users/{user_id}", users_update)
    R("POST", "/users/{user_id}/unlock", users_unlock)

    def settings_get(req):
        req.require("admin.settings")
        return {"retention": s.settings.get("retention", {"evidence_days": s.config.evidence_retention_days,
                                                          "audit_days": s.config.audit_retention_days}),
                "detection_defaults": s.settings.get("detection_defaults", {"window_s": 10.0, "dwell_s": 3.0, "long_stop_s": 30.0,
                                                                            "max_capacity_vehicles": 20}),
                "offences": s.cases.offences(), "alert_rules": s.alerts.rules(), "alert_adapters": s.alerts.adapter_status(),
                "policy": {"require_distinct_approver": s.config.require_distinct_approver,
                           "note": "set ANVESH_DISTINCT_APPROVER=0 to disable; changing it requires a restart"},
                "retention_note": "Retention is recorded and reported. Automatic purge is NOT implemented: evidence and audit rows are immutable "
                                  "by design, so expiry needs a reviewed offline procedure."}

    def settings_put(req):
        req.require("admin.settings")
        b = req.body
        if req.user.is_demo and {"retention", "offences", "alert_rules"} & set(b):  # these shape operational behaviour for everyone
            raise ApiError(403, "demo_account", "demo accounts cannot change retention, the offence catalogue or alert rules")
        for key, typ in (("retention", dict), ("detection_defaults", dict), ("offences", list), ("alert_rules", list)):
            if key in b and not isinstance(b[key], typ):
                raise ApiError(422, "validation", f"'{key}' must be a {'list' if typ is list else 'object'}")
        for key in ("offences", "alert_rules"):
            if key in b and not all(isinstance(i, dict) for i in b[key]):
                raise ApiError(422, "validation", f"every entry of '{key}' must be an object")
        if "retention" in b:
            r = b["retention"]
            for k in ("evidence_days", "audit_days"):
                if isinstance(r.get(k), bool) or not isinstance(r.get(k), int) or not 1 <= r[k] <= 36500:
                    raise ApiError(422, "validation", f"retention.{k} must be an integer in [1, 36500]")
            s.settings.set("retention", {"evidence_days": r["evidence_days"], "audit_days": r["audit_days"]}, req.user.username)
            s.audit.append(req.user.username, "settings.retention", "settings", "retention", r, actor_role=req.user.role)
        if "detection_defaults" in b:
            d = b["detection_defaults"]
            cams.parse_camera_config({"thresholds": d})  # validates names and ranges
            s.settings.set("detection_defaults", d, req.user.username)
            s.audit.append(req.user.username, "settings.detection_defaults", "settings", "detection_defaults", d, actor_role=req.user.role)
        if "offences" in b:
            s.cases.set_offences(b["offences"], req.user.username, req.user.role)
        if "alert_rules" in b:
            req.require("alert.configure")
            s.alerts.set_rules(b["alert_rules"], req.user.username, req.user.role)
        return settings_get(req)

    R("GET", "/settings", settings_get)
    R("PUT", "/settings", settings_put)

    def retention_report(req):
        req.require("admin.settings")
        plan = s.retention.plan()
        return {"evidence_retention_days": plan["evidence_retention_days"], "cutoff_utc": plan["cutoff_utc"],
                "past_retention": len(plan["eligible_incidents"]) + len(plan["protected_incidents"]),
                "blocked_by_open_cases": len(plan["protected_incidents"]), "assets_to_purge": plan["assets_to_purge"],
                "bytes_to_free": plan["bytes_to_free"], "purge_implemented": True, "purge_scope": "evidence files only",
                "audit_purge_implemented": False,
                "note": "Report only. Nothing is deleted until an administrator reviews GET /retention/plan and POSTs /retention/purge with its "
                        "plan_id. Evidence rows, hashes and the audit trail are never deleted."}

    def retention_plan(req):
        req.require("retention.manage")
        return s.retention.public_plan(s.retention.plan())

    def retention_purge(req):
        req.require("retention.manage")
        if req.user.is_demo:  # deleting evidence is never a demo account's job, whatever class the evidence is
            raise ApiError(403, "demo_account", "demo accounts cannot purge evidence")
        return s.retention.purge(req.need("plan_id", maxlen=64), req.need("reason", maxlen=1000), req.user.username, req.user.role)

    R("GET", "/retention/report", retention_report)
    R("GET", "/retention/plan", retention_plan)
    R("POST", "/retention/purge", retention_purge)

    # ---------------------------------------------------------------- cameras
    def cam_public(c):
        c = dict(c)
        cfg = c.get("config") or {}
        geom = {"validity": "image", "label": "Image space (pixels) - no calibration", "metric_speed": False, "synthetic": False}
        if cfg.get("calibration"):
            try:
                geom = cams.geometry_label(cams.build_calibration_profile("probe", c["camera_id"], cfg["calibration"]))
            except cams.ConfigError as e:
                geom = {"validity": "invalid", "label": f"calibration invalid: {e}", "metric_speed": False, "synthetic": False}
        c["geometry"] = geom
        active = [r for r in app.sessions.active_runtimes() if r["camera_id"] == c["camera_id"]]
        c["active_session"] = active[0] if active else None
        last = s.db.one("SELECT session_id, state, started_at, ended_at, error FROM processing_sessions WHERE camera_id = ? ORDER BY started_at DESC LIMIT 1",
                        (c["camera_id"],))
        c["last_session"] = last
        return c

    def cameras_list(req):
        req.require("view.cameras")
        items = [cam_public(c) for c in s.cameras.list()]
        return {"items": items, "adapters": {k: v.__dict__ for k, v in cams.ADAPTERS.items()}}

    def cameras_create(req):
        req.require("camera.manage")
        if req.user.is_demo and req.body.get("data_class") == "operational":
            raise ApiError(403, "demo_account", "demo accounts cannot register operational cameras")
        return 201, cam_public(s.cameras.create(req.user.username, req.body, req.user.role, actor_is_demo=req.user.is_demo))

    def camera_get(req):
        req.require("view.cameras")
        c = s.cameras.get(req.params["camera_id"])
        if c is None:
            raise LookupError("no such camera")
        return cam_public(c)

    def camera_patch(req):
        req.require("camera.manage")
        cid = req.params["camera_id"]
        refuse_demo_on_operational(req, camera_class(cid))
        refuse_demo_on_operational(req, req.body.get("data_class"))  # nor may a camera be turned operational
        if any(r["camera_id"] == cid for r in app.sessions.active_runtimes()) and set(req.body) - {"name", "location_label", "enabled"}:
            raise ApiError(409, "active_session", "stop the camera's session before changing its source or detection config")
        return cam_public(s.cameras.update(req.user.username, cid, req.body, req.user.role, actor_is_demo=req.user.is_demo))

    R("GET", "/cameras", cameras_list)
    R("POST", "/cameras", cameras_create)
    R("GET", "/cameras/{camera_id}", camera_get)
    R("PATCH", "/cameras/{camera_id}", camera_patch)

    def upload(req):
        req.require("camera.manage")
        return 201, app.save_upload(req)

    R("POST", "/uploads", upload)

    # ---------------------------------------------------------------- sessions
    def session_start(req):
        req.require("session.control")
        refuse_demo_on_operational(req, camera_class(req.params["camera_id"]))
        return 202, app.sessions.start(req.params["camera_id"], req.user.username, req.user.role)

    def sessions_list(req):
        req.require("view.cameras")
        active = {r["session_id"]: r for r in app.sessions.active_runtimes()}
        rows = s.db.query("SELECT * FROM processing_sessions ORDER BY started_at DESC LIMIT ?", (req.qint("limit", 30, 1, 200),))
        out = []
        for r in rows:
            r["stats"] = loads(r.pop("stats_json"), {})
            r["source_snapshot"] = loads(r.pop("source_snapshot_json"), {})
            r["runtime"] = active.get(r["session_id"])
            out.append(r)
        return {"items": out, "active": len(active)}

    def session_get(req):
        req.require("view.cameras")
        sid = req.params["session_id"]
        rt = app.sessions.runtime(sid)
        row = s.db.one("SELECT * FROM processing_sessions WHERE session_id = ?", (sid,))
        if row is None:
            raise LookupError("no such session")
        row["stats"] = loads(row.pop("stats_json"), {})
        row["source_snapshot"] = loads(row.pop("source_snapshot_json"), {})
        row["runtime"] = rt
        return row

    def session_control(action):
        def handler(req):
            req.require("session.control")
            getattr(app.sessions, action)(req.params["session_id"], req.user.username, req.user.role)
            return 200, {"ok": True, "state": (app.sessions.runtime(req.params["session_id"]) or {}).get("state")}
        return handler

    def session_signal(req):
        req.require("session.control")
        sess = app.sessions.get(req.params["session_id"])
        if sess is None or sess.signal is None:
            raise ApiError(409, "no_signal", "this session's camera has no signal configured")
        if sess.signal.mode == "schedule":
            raise ApiError(409, "schedule_mode", "the signal follows a configured schedule; set mode 'manual' or 'external' to push states")
        state = req.need("state", maxlen=32)
        sess.signal.set(state, sess.runtime()["stream_time_s"])
        s.audit.append(req.user.username, "session.signal", "session", req.params["session_id"], {"state": state, "mode": sess.signal.mode},
                       actor_role=req.user.role)
        return {"ok": True, "state": state, "source": sess.signal.source_label}

    def session_snapshot(req):
        req.require("view.cameras")
        sess = app.sessions.get(req.params["session_id"])
        if sess is None or sess.last_jpeg is None:
            raise LookupError("no frame available for this session")
        app.send_bytes(req.h, sess.last_jpeg, "image/jpeg")

    def session_stream(req):
        req.require("view.cameras")
        sess = app.sessions.get(req.params["session_id"])
        if sess is None:
            raise LookupError("session is not active")
        h = req.h
        h.send_response(200)
        h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        last = None
        idle_since = time.time()
        try:
            while True:
                jpeg = sess.last_jpeg
                if jpeg is not None and jpeg is not last:
                    h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n")
                    last, idle_since = jpeg, time.time()
                elif sess.runtime()["state"] in TERMINAL_STATES and time.time() - idle_since > 2:
                    break
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        h.close_connection = True

    def session_video(req):
        req.require("view.evidence")
        row = s.db.one("SELECT source_snapshot_json FROM processing_sessions WHERE session_id = ?", (req.params["session_id"],))
        if row is None:
            raise LookupError("no such session")
        snap = loads(row["source_snapshot_json"], {})
        if snap.get("source_type") != "file":
            raise ApiError(404, "no_source_file", "only recorded-file sessions have a replayable source video")
        path = Path(snap["source"])
        try:  # defence in depth: serve only what a file camera is allowed to be registered on today
            cams.check_file_source(str(path), s.media_roots, (s.config.data_dir,))
        except cams.ConfigError:
            raise ApiError(404, "no_source_file", "this session's source is not a replayable video") from None
        if not path.is_file():
            raise ApiError(404, "source_missing", "the source video file is no longer on disk")
        app.send_file(req.h, path)

    def session_officer(req):
        req.require("view.cameras")
        sess = app.sessions.get(req.params["session_id"])
        if sess is None:
            raise LookupError("session is not active")
        return sess.officer_state()

    def session_alert_thumb(req):
        req.require("view.cameras")
        sess = app.sessions.get(req.params["session_id"])
        if sess is None or sess.officer is None:
            raise LookupError("session is not active")
        jpeg = sess.officer.alerts.thumb(int(req.params["alert_id"]))
        if jpeg is None:
            raise LookupError("no picture for this alert")
        app.send_bytes(req.h, jpeg, "image/jpeg")

    R("GET", "/sessions/{session_id}/officer", session_officer)
    R("GET", "/sessions/{session_id}/alerts/{alert_id}/thumb.jpg", session_alert_thumb)
    R("POST", "/cameras/{camera_id}/sessions", session_start)
    R("GET", "/sessions", sessions_list)
    R("GET", "/sessions/{session_id}", session_get)
    R("POST", "/sessions/{session_id}/stop", session_control("stop"))
    R("POST", "/sessions/{session_id}/pause", session_control("pause"))
    R("POST", "/sessions/{session_id}/resume", session_control("resume"))
    R("POST", "/sessions/{session_id}/signal", session_signal)
    R("GET", "/sessions/{session_id}/snapshot.jpg", session_snapshot)
    R("GET", "/sessions/{session_id}/stream.mjpg", session_stream)
    R("GET", "/sessions/{session_id}/video", session_video)

    # ---------------------------------------------------------------- incidents
    def _thumbs(ids):
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = s.db.query(f"SELECT incident_id, asset_id FROM evidence_assets WHERE kind = 'thumbnail' AND incident_id IN ({marks}) ORDER BY created_at", ids)
        return {r["incident_id"]: r["asset_id"] for r in rows}

    def incidents_list(req):
        req.require("view.incidents")
        limit, offset = page(req)
        f = {k: req.q(k) for k in ("event_type", "status", "severity", "camera_id", "session_id", "data_class", "assigned_to", "q", "from_utc", "to_utc")}
        f["open_only"] = req.q("open_only") == "1"
        rows, total = s.incidents.list(f, limit, offset, order=req.q("order", "newest"))
        th = _thumbs([r["incident_id"] for r in rows])
        for r in rows:
            r["thumbnail_asset_id"] = th.get(r["incident_id"])
        return _paged(rows, total, limit, offset)

    def incident_get(req):
        req.require("view.incidents")
        iid = req.params["incident_id"]
        inc = s.incidents.get(iid, with_timeline=True)
        if inc is None:
            raise LookupError("no such incident")
        inc["evidence"] = s.evidence.list(iid)
        inc["plates"] = s.plates.history(iid)
        inc["current_plates"] = s.plates.current(iid)
        inc["alerts"] = s.alerts.list(incident_id=iid)
        inc["cases"] = [{"case_id": c["case_id"], "state": c["state"], "offence_code": c["offence_code"]} for c in
                        s.db.query("SELECT case_id, state, offence_code FROM cases WHERE incident_id = ?", (iid,))]
        inc["camera"] = (s.cameras.get(inc["camera_id"]) or {}).get("name")
        sess = s.db.one("SELECT state, source_snapshot_json FROM processing_sessions WHERE session_id = ?", (inc["session_id"],))
        snap = loads(sess["source_snapshot_json"], {}) if sess else {}
        inc["source_video"] = bool(snap.get("source_type") == "file" and Path(snap.get("source", "")).is_file())
        inc["capability"] = category_status(inc["event_type"]) if inc["event_type"] in CAPABILITIES else None
        inc["allowed_status"] = sorted(STATUS_TRANSITIONS[inc["status"]])
        return inc

    def incident_patch(req):
        req.require("incident.update")
        iid, b = req.params["incident_id"], req.body
        refuse_demo_on_operational(req, incident_class(iid))
        out = None
        status, note, assignee = req.opt("status"), req.opt("note", str, ""), req.opt("assigned_to", str, "", maxlen=128)
        if status is not None:
            out = s.incidents.change_status(iid, status, req.user.username, req.user.role, note)
        if "assigned_to" in b:
            out = s.incidents.assign(iid, assignee, req.user.username, req.user.role)
        if out is None:
            raise ApiError(422, "validation", "nothing to update (status or assigned_to)")
        return out

    def incident_note(req):
        req.require("incident.update")
        refuse_demo_on_operational(req, incident_class(req.params["incident_id"]))
        return s.incidents.add_note(req.params["incident_id"], req.need("note"), req.user.username, req.user.role)

    def incident_cause(req):
        req.require("incident.establish_cause")
        refuse_demo_on_operational(req, incident_class(req.params["incident_id"]))
        return s.incidents.establish_cause(req.params["incident_id"], req.need("cause"), req.need("basis"), req.user.username, req.user.role)

    def incident_report(req):
        req.require("view.incidents")
        iid = req.params["incident_id"]
        inc = s.incidents.get(iid, with_timeline=True)
        if inc is None:
            raise LookupError("no such incident")
        audit_rows, _ = s.audit.query(entity_type="incident", entity_id=iid, limit=500)
        return {"generated_at": now_iso(), "notice": "Candidate detection awaiting human review; confidence tier is a heuristic label, not a probability.",
                "incident": inc, "evidence": s.evidence.list(iid), "plates": s.plates.history(iid), "alerts": s.alerts.list(incident_id=iid),
                "audit": audit_rows}

    R("GET", "/incidents", incidents_list)
    R("GET", "/incidents/{incident_id}", incident_get)
    R("PATCH", "/incidents/{incident_id}", incident_patch)
    R("POST", "/incidents/{incident_id}/notes", incident_note)
    R("POST", "/incidents/{incident_id}/cause", incident_cause)
    R("GET", "/incidents/{incident_id}/report", incident_report)

    # ---------------------------------------------------------------- evidence
    def evidence_file(req):
        req.require("view.evidence")
        a = s.evidence.get(req.params["asset_id"])
        if a is None:
            raise LookupError("no such evidence asset")
        if a["kind"] in ("clip", "plate_crop"):
            audit_view(req, "evidence.view", "incident", a["incident_id"])
        purge = s.evidence.purge_record(a["asset_id"])
        if purge is not None:
            raise ApiError(410, "purged", "this evidence file was removed under the retention policy", {"purged_at": purge["purged_at"], "sha256": a["sha256"]})
        path = s.evidence.path_of(a)
        if not path.is_file():
            raise ApiError(410, "file_missing", "the evidence file is missing from disk")
        app.send_file(req.h, path, a["mime"])

    def evidence_verify(req):
        req.require("view.evidence")
        return s.evidence.verify(req.params["asset_id"])

    R("GET", "/evidence/{asset_id}/file", evidence_file)
    R("GET", "/evidence/{asset_id}/verify", evidence_verify)

    # ---------------------------------------------------------------- plates
    def plates_get(req):
        req.require("view.incidents")
        iid = req.params["incident_id"]
        return {"history": s.plates.history(iid), "current": s.plates.current(iid), "engine": s.plates.status()}

    def plate_enter(req):
        req.require("plate.enter")
        refuse_demo_on_operational(req, incident_class(req.params["incident_id"]))
        b = req.body
        r = s.plates.manual_entry(req.params["incident_id"], req.need("text", maxlen=32), req.user.username, req.user.role,
                                  track_id=req.opt("track_id", str, None, maxlen=128), note=req.opt("note", str, ""))
        return 201, r

    def plate_ocr(req):
        req.require("plate.enter")
        refuse_demo_on_operational(req, incident_class(req.params["incident_id"]))
        return s.plates.read_from_evidence(req.params["incident_id"], req.need("track_id", maxlen=128), req.user.username, req.user.role)

    def plate_verify(req):
        req.require("case.verify_plate")
        refuse_demo_on_operational(req, reading_class(req.params["reading_id"]))
        return s.plates.verify(req.params["reading_id"], req.user.username, req.user.role, req.opt("note", str, ""),
                               bool(req.opt("allow_nonstandard", bool, False)))

    def plate_correct(req):
        req.require("plate.enter")
        refuse_demo_on_operational(req, reading_class(req.params["reading_id"]))
        return 201, s.plates.correct(req.params["reading_id"], req.need("text", maxlen=32), req.user.username, req.user.role,
                                     req.opt("note", str, ""), can_verify=req.user.can("case.verify_plate"))

    R("GET", "/incidents/{incident_id}/plates", plates_get)
    R("POST", "/incidents/{incident_id}/plates", plate_enter)
    R("POST", "/incidents/{incident_id}/plates/ocr", plate_ocr)
    R("POST", "/plates/{reading_id}/verify", plate_verify)
    R("POST", "/plates/{reading_id}/correct", plate_correct)

    # ---------------------------------------------------------------- cases
    def case_create(req):
        return 201, s.cases.create(req.params["incident_id"], req.need("offence_code", maxlen=64), req.user,
                                   req.opt("track_id", str, None, maxlen=128), req.opt("reason", str, None))

    def cases_list(req):
        req.require("view.cases")
        limit, offset = page(req)
        f = {k: req.q(k) for k in ("state", "offence_code", "data_class", "incident_id", "created_by", "q")}
        rows, total = s.cases.list(f, limit, offset)
        return _paged(rows, total, limit, offset)

    def case_get(req):
        req.require("view.cases")
        case = s.cases.get(req.params["case_id"], req.user)
        if any(l["data"].get("owner_name") for l in case["registry_lookups"]):
            audit_view(req, "registry.view_owner", "case", case["case_id"])
        return case

    def case_patch(req):
        return s.cases.update_fields(req.params["case_id"], req.user, req.opt("offence_code", str, None, maxlen=64),
                                     req.opt("reason", str, None), req.opt("limitations", str, None))

    def case_action(req):
        action, note = req.need("action", maxlen=64), req.opt("note", str, "")
        params = {}
        for key, typ in CASE_ACTION_PARAMS.items():  # only these are forwarded; anything else is refused rather than splatted into the service
            value = req.opt(key, typ)
            if value is not None:
                params[key] = value
        extra = set(req.body) - set(CASE_ACTION_PARAMS) - {"action", "note"}
        if extra:
            hint = ("; there is no registry override - approval needs an authorised registry result for the verified plate" if "registry_override_reason" in extra else "")
            if action in ("approve", "retry_submission"):  # a refused attempt to supply approval-relevant fields is itself audit-worthy
                s.audit.append(req.user.username, f"case.{action}_blocked", "case", req.params["case_id"],
                               {"http_status": 422, "reasons": [f"request carried unsupported field(s): {sorted(extra)[:5]}"]}, actor_role=req.user.role)
            raise ApiError(422, "validation", f"unexpected field(s): {sorted(extra)[:5]}{hint}")
        return s.cases.transition(req.params["case_id"], action, req.user, note, **params)

    def case_lookup(req):
        req.require("registry.lookup")
        case = s.cases.get(req.params["case_id"], req.user)
        rid = req.opt("reading_id", str, None, maxlen=64)
        reading = s.plates.get(rid) if rid else case["current_plate"]
        if reading is not None and reading["incident_id"] != case["incident_id"]:
            raise ApiError(422, "validation", "reading does not belong to this case")
        if req.user.is_demo and case["data_class"] == "operational":
            raise ApiError(403, "demo_account", "demo accounts cannot work operational cases")
        look = s.registry.lookup(case, reading, req.user.username, req.user.role)
        return 201, s.registry.get(look["lookup_id"], req.user)

    def case_handoff(req):
        status = s.handoff.submit(req.params["case_id"], req.need("adapter", maxlen=32), req.user)
        return (200 if status.get("duplicate") else 201), status

    def case_report(req):
        req.require("export.case")
        case = s.cases.get(req.params["case_id"], req.user)
        inc = s.incidents.get(case["incident_id"])
        assets = s.evidence.list(case["incident_id"])
        images = {}
        for a in assets:
            if a["kind"] == "frame_annotated":
                data, state = s.evidence.read_verified(a)  # purged / missing / hash-mismatched files are not embedded
                if data is not None:
                    images[a["asset_id"]] = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        rows = []
        for et, eid in (("case", case["case_id"]), ("incident", inc["incident_id"])):
            rows += s.audit.query(entity_type=et, entity_id=eid, limit=500)[0]
        rows.sort(key=lambda r: r["seq"])
        cam = s.cameras.get(inc["camera_id"]) or {}
        clip = next((a for a in assets if a["kind"] == "clip"), None)
        html = case_report_html(case, inc, assets, images, rows, cam.get("utc_offset_minutes", 330),
                                clip_href=f"{API}/evidence/{clip['asset_id']}/file" if clip else None, generated_for=req.user.username)
        s.audit.append(req.user.username, "case.report_view", "case", case["case_id"], {}, actor_role=req.user.role)
        app.send_bytes(req.h, html.encode("utf-8"), "text/html; charset=utf-8")

    def case_package(req):
        req.require("export.case")
        path = build_case_package(s, req.params["case_id"], req.user, s.config.exports_dir)
        s.audit.append(req.user.username, "case.package_download", "case", req.params["case_id"], {"file": path.name}, actor_role=req.user.role)
        app.send_file(req.h, path, "application/zip", download_name=path.name)

    R("POST", "/incidents/{incident_id}/cases", case_create)
    R("GET", "/cases", cases_list)
    R("GET", "/cases/{case_id}", case_get)
    R("PATCH", "/cases/{case_id}", case_patch)
    R("POST", "/cases/{case_id}/actions", case_action)
    R("POST", "/cases/{case_id}/registry-lookup", case_lookup)
    R("POST", "/cases/{case_id}/handoff", case_handoff)
    R("GET", "/cases/{case_id}/report.html", case_report)
    R("GET", "/cases/{case_id}/package", case_package)

    def offences(req):
        req.require("view.cases")
        return {"items": s.cases.offences(), "states": list(CASE_STATES)}

    R("GET", "/offences", offences)

    # ---------------------------------------------------------------- alerts
    def alerts_list(req):
        req.require("view.incidents")
        return {"items": s.alerts.list(req.q("incident_id"), req.q("status"), req.qint("limit", 100, 1, 500)),
                "adapters": s.alerts.adapter_status(), "rules": s.alerts.rules()}

    def alert_retry(req):
        req.require("alert.retry")
        d = s.db.one("SELECT incident_id FROM alert_deliveries WHERE delivery_id = ?", (req.params["delivery_id"],))
        if d is not None:
            refuse_demo_on_operational(req, incident_class(d["incident_id"]))
        return s.alerts.retry(req.params["delivery_id"], req.user.username, req.user.role)

    R("GET", "/alerts", alerts_list)
    R("POST", "/alerts/{delivery_id}/retry", alert_retry)

    # ---------------------------------------------------------------- overview / congestion / stats / audit
    def overview(req):
        req.require("view.overview")
        runtimes = app.sessions.active_runtimes()
        cams_all = s.cameras.list()
        live_states = {}
        for c in cams_all:
            rt = next((r for r in runtimes if r["camera_id"] == c["camera_id"]), None)
            live_states[c["camera_id"]] = rt["state"] if rt else "idle"
        open_rows, open_total = s.incidents.list({"open_only": True}, 8, 0, order="priority")
        th = _thumbs([r["incident_id"] for r in open_rows])
        for r in open_rows:
            r["thumbnail_asset_id"] = th.get(r["incident_id"])
        awaiting = s.db.one("SELECT COUNT(*) AS n FROM cases WHERE state IN ('awaiting_evidence_verification','awaiting_plate_verification','ready_for_review','needs_further_review')")["n"]
        pending_approval = s.db.one("SELECT COUNT(*) AS n FROM cases WHERE state = 'ready_for_review'")["n"]
        cong_open = s.db.query("SELECT incident_id, camera_id, severity, indicators_json, start_utc FROM incidents WHERE event_type = 'congestion' AND end_offset_s IS NULL "
                               "AND status IN ('detected','under_investigation','escalated') ORDER BY detected_utc DESC LIMIT 10")
        for r in cong_open:
            ind = loads(r.pop("indicators_json"), {})
            r["current_level"], r["peak_level"], r["duration_s"] = ind.get("current_level"), ind.get("peak_level"), ind.get("duration_s")
        activity, _ = s.audit.query(limit=12)
        failed_alerts = s.db.one("SELECT COUNT(*) AS n FROM alert_deliveries WHERE status IN ('abandoned') AND last_error NOT LIKE 'blocked:%'")["n"]
        pending_alerts = s.db.one("SELECT COUNT(*) AS n FROM alert_deliveries WHERE status = 'pending'")["n"]
        errors = [{"session_id": r["session_id"], "camera_id": r["camera_id"], "errors": r["errors"], "state": r["state"]} for r in runtimes if r["errors"] or r["state"] in ("stalled", "reconnecting")]
        return {
            "cameras": {"total": len(cams_all), "by_state": {st: list(live_states.values()).count(st) for st in set(live_states.values())},
                        "states": live_states},
            "sessions": [{k: r[k] for k in ("session_id", "camera_id", "state", "frames_processed", "frames_dropped", "processing_fps", "concurrent_vehicles",
                                            "last_frame_age_s", "events_emitted", "data_class", "simulated", "stream_time_s")} for r in runtimes],
            "incidents": {"open_total": open_total, "top": open_rows, **s.incidents.counts()},
            "cases": {"by_state": s.cases.counts(), "awaiting_review": awaiting, "ready_for_officer": pending_approval},
            "congestion": cong_open, "alerts": {"pending": pending_alerts, "failed": failed_alerts},
            "recent_activity": activity, "errors": errors, "time": now_iso(),
        }

    def congestion(req):
        req.require("view.congestion")
        runtimes = {r["camera_id"]: r for r in app.sessions.active_runtimes()}
        per_camera = []
        for c in s.cameras.list():
            rt = runtimes.get(c["camera_id"])
            rows, _ = s.incidents.list({"event_type": "congestion", "camera_id": c["camera_id"]}, 5, 0)
            per_camera.append({"camera_id": c["camera_id"], "name": c["name"], "location": c["location_label"], "data_class": c["data_class"],
                               "live_trend": rt["congestion_trend"] if rt else [], "state": rt["state"] if rt else "idle",
                               "unique_tracks_by_class": rt["unique_tracks_by_class"] if rt else {},
                               "recent_episodes": [{k: r[k] for k in ("incident_id", "severity", "start_utc", "end_utc", "start_offset_s", "end_offset_s",
                                                                        "indicators", "candidate_causes", "uncertainty", "status")} for r in rows]})
        return {"cameras": per_camera}

    def stats(req):
        req.require("view.overview")
        by_day = s.db.query("SELECT substr(detected_utc, 1, 10) AS day, event_type, COUNT(*) AS n FROM incidents GROUP BY day, event_type ORDER BY day DESC LIMIT 200")
        outcomes = s.db.query("SELECT state, COUNT(*) AS n FROM cases GROUP BY state")
        alerts = s.db.query("SELECT status, is_test, delivery_confirmed, COUNT(*) AS n FROM alert_deliveries GROUP BY status, is_test, delivery_confirmed")
        sessions = s.db.query("SELECT state, COUNT(*) AS n FROM processing_sessions GROUP BY state")
        return {"incidents_by_day": by_day, "case_states": outcomes, "alert_deliveries": alerts, "sessions": sessions, **s.incidents.counts()}

    def audit_list(req):
        req.require("audit.view")
        limit, offset = page(req)
        rows, total = s.audit.query(req.q("entity_type"), req.q("entity_id"), req.q("actor"), req.q("action"), limit, offset)
        return _paged(rows, total, limit, offset)

    def audit_verify(req):
        req.require("audit.verify")
        return {**s.audit.verify(), "head": s.audit.head()}

    R("GET", "/overview", overview)
    R("GET", "/congestion", congestion)
    R("GET", "/stats", stats)
    R("GET", "/audit", audit_list)
    R("GET", "/audit/verify", audit_verify)
