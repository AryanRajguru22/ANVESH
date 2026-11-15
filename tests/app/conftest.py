import pytest

from anvesh.app.audit import AuditLog
from anvesh.app.auth import AuthService
from anvesh.app.db import AppDatabase


@pytest.fixture(autouse=True)
def _cheap_password_hashing(monkeypatch):
    """Production scrypt costs ~0.4 s and 128 MiB per hash; the suite creates hundreds of users. Tests lower the cost for speed ONLY - the
    production constants are asserted in test_account_security.py and are what the application uses."""
    from anvesh.app import auth as auth_module

    monkeypatch.setitem(auth_module.SCRYPT, "n", 2 ** 10)


@pytest.fixture()
def db(tmp_path):
    d = AppDatabase(tmp_path / "app.sqlite3")
    yield d
    d.close()


@pytest.fixture()
def audit(db):
    return AuditLog(db)


@pytest.fixture()
def auth(db, audit):
    return AuthService(db, audit)


@pytest.fixture()
def services(tmp_path):
    from anvesh.app.config import AppConfig
    from anvesh.app.services import build_services

    from anvesh.app.plates import UnavailableOcr, UnavailablePlateDetector

    svc = build_services(AppConfig(data_dir=tmp_path / "data"), ocr=UnavailableOcr("test", "n/a"), plate_detector=UnavailablePlateDetector("test"))
    yield svc
    svc.close()


# -- helpers for workflow tests (no perception needed) ----------------------------------------------

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from anvesh.app.detectors import DetectedEvent  # noqa: E402
from anvesh.app.incidents import SessionInfo  # noqa: E402
from anvesh.app.timeutil import now_iso  # noqa: E402


@pytest.fixture()
def users(services):
    out = {}
    for name, role in (("admin1", "administrator"), ("inv1", "investigator"), ("rev1", "reviewer"), ("rev2", "reviewer"),
                       ("off1", "officer"), ("off2", "officer")):
        services.auth.create_user(name, name.title(), role, "pw-" + name + "-long")
        out[name] = services.auth.login(name, "pw-" + name + "-long")[1]
    return out


def seed_incident(services, data_class="demo", event_type="stopped_vehicle", camera_id="cam-seed", with_evidence=True,
                  time_basis="declared_recording_start", severity="medium", key="ep1", tracks=("run-s000-t00001",)):
    """An incident + session + camera row, optionally with real (tiny) frame/clip evidence files."""
    if services.cameras.get(camera_id) is None:
        cam_class = "simulated" if data_class == "simulated" else "demo"
        services.cameras.create("admin", {"camera_id": camera_id, "name": camera_id, "source_type": "synthetic", "source_uri": "synthetic:queue",
                                          "data_class": "simulated"})
        services.db.execute("UPDATE cameras SET data_class = ? WHERE camera_id = ?", (data_class if data_class != cam_class else cam_class, camera_id))
    session_id = "SES-" + camera_id[-4:].upper() + key.upper()
    if services.db.one("SELECT 1 AS x FROM processing_sessions WHERE session_id = ?", (session_id,)) is None:
        services.db.execute(
            "INSERT INTO processing_sessions(session_id, camera_id, run_id, state, data_class, started_at, time_basis, recording_start_utc, "
            "source_snapshot_json) VALUES (?,?,?,?,?,?,?,?,'{}')",
            (session_id, camera_id, session_id.lower(), "completed", data_class, now_iso(), time_basis, "2026-10-01T10:00:00.000Z"))
    info = SessionInfo(session_id, camera_id, data_class, "2026-10-01T10:00:00.000Z", time_basis,
                       location={"label": "Test junction", "latitude": 12.9, "longitude": 77.6})
    ev = DetectedEvent(
        event_type=event_type, natural_key=key, start_offset_s=10.0, detected_offset_s=14.0, detector="test_detector", track_ids=list(tracks),
        involved=[{"track_id": t, "role": "observed", "vehicle_class": "car", "note": "test"} for t in tracks],
        indicators={"dwell_s": 4.0}, severity=severity, severity_basis="test", confidence_tier="medium", confidence_basis="test fixture",
        uncertainty=["test uncertainty"], summary="test event", motion_space="image", geometry_validity="image")
    inc, created, _ = services.incidents.upsert(info, ev)
    if with_evidence and created:
        iid = inc["incident_id"]
        img = np.full((90, 160, 3), 100, np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        for off in (10.0, 14.0):
            services.evidence.add_bytes(iid, "frame", buf.tobytes(), ".jpg", "image/jpeg", offset_s=off, frame_index=int(off * 10),
                                        track_id=tracks[0], meta={"label": "t", "track_bbox": [10, 10, 50, 50]})
        services.evidence.add_bytes(iid, "frame_annotated", buf.tobytes(), ".jpg", "image/jpeg", offset_s=10.0, track_id=tracks[0], meta={"label": "t"})
        services.evidence.add_clip(iid, [img.copy() for _ in range(5)], fps=10.0, offset_s=10.0)
        services.incidents.set_evidence_completeness(iid, services.evidence.list(iid), True)
        inc = services.incidents.get(iid)
    return inc


@pytest.fixture()
def stub():
    from .stubserver import StubServer

    made = []

    def make(respond):
        s = StubServer(respond)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


# -- registry test doubles (TEST ONLY) -------------------------------------------------------------------------------------
# ANVESH has no authorised registry. `AuthorisedRegistryDouble` is a stand-in that lets the tests exercise the code path that a real,
# authorised gateway would use. It lives in the tests, never in the application, and its results are fabricated for tests only.

from anvesh.app.registry import RegistryResult  # noqa: E402


class AuthorisedRegistryDouble:
    kind = "production"
    name = "authorised_registry_gateway"  # the name the service checks for: "the configured authorised gateway"
    label = "TEST DOUBLE standing in for an authorised registry gateway"

    def __init__(self, records=None, is_connected=True):
        self.records = {"MH12AB1234": {"registration_status": "active", "vehicle_class": "motor car", "owner_name": "TEST DOUBLE OWNER"}} \
            if records is None else records
        self._connected = is_connected
        self.calls = []

    def connected(self):
        return self._connected

    def lookup(self, plate, case_id, requested_by):
        self.calls.append((plate, case_id, requested_by))
        rec = self.records.get(plate)
        if rec is None:
            return RegistryResult("not_found", source_label=self.label)
        return RegistryResult("found", dict(rec, as_of=now_iso()), source_label="TEST authority (double)")


def use_authorised_registry(services, **kw):
    """Install the test double as the configured authorised gateway."""
    services.registry.production = AuthorisedRegistryDouble(**kw)
    return services.registry.production


def registry_lookup(services, cid, user, reading=None):
    """Look up the case's current plate through whichever adapter its data class routes to."""
    case = services.cases.get(cid, user)
    return services.registry.lookup(case, reading or case["current_plate"], user.username, user.role)


def lookup_and_approve(services, users, cid, who="off1", **kw):
    """The correct path: registry lookup bound to the verified reading, then approval. Operational cases get the authorised-gateway double."""
    case = services.cases.get(cid, users[who])
    if case["data_class"] == "operational" and not isinstance(services.registry.production, AuthorisedRegistryDouble):
        use_authorised_registry(services)
    registry_lookup(services, cid, users[who])
    return services.cases.transition(cid, "approve", users[who], **kw)


# -- schema downgrade helper (tests only) ----------------------------------------------------------------------------------------

_UNDO = {
    4: ["DROP TABLE account_requests", "ALTER TABLE users DROP COLUMN must_change_password", "ALTER TABLE users DROP COLUMN is_demo",
        "ALTER TABLE users DROP COLUMN password_changed_at", "ALTER TABLE auth_sessions DROP COLUMN last_seen_at"],
    3: ["DROP TRIGGER cases_operational_approval_needs_registry", "ALTER TABLE registry_lookups DROP COLUMN reading_id"],
    2: ["UPDATE cases SET plate_verified_reading_id = NULL", "DROP TABLE evidence_purges", "ALTER TABLE cases DROP COLUMN plate_verified_reading_id"],
}


def downgrade_to(database, target: int) -> None:
    """Turn a current-schema database into a schema-`target` one (undo v4, v3, v2 in that order) so a real upgrade can be exercised."""
    for version in (4, 3, 2):
        if version > target:
            for statement in _UNDO[version]:
                database.execute(statement)
            database.execute("DELETE FROM schema_migrations WHERE version = ?", (version,))
