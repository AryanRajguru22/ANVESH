"""Retention: plan / guarded purge of evidence files, tombstones, export integrity, schema-v2 migration.

All of this runs on temporary directories. Nothing here touches outputs/app."""

import sqlite3
import time
from pathlib import Path

import pytest

from anvesh.app import db as dbmod
from anvesh.app.cases import CaseError
from anvesh.app.db import AppDatabase
from anvesh.app.reports import build_case_package
from anvesh.app.retention import MIN_PURGE_DAYS, RetentionError
from anvesh.app.timeutil import add_seconds

from .conftest import downgrade_to, seed_incident

REPO_ROOT = Path(__file__).resolve().parents[2]
from .test_api import api, client  # noqa: F401  (api is a fixture)


def _later(inc, days=400):
    return add_seconds(inc["detected_utc"], days * 86400)


def _resolve(services, users, inc, case_action="reject"):
    """Make an incident purge-eligible: resolve the incident, and finish any case."""
    services.incidents.change_status(inc["incident_id"], "under_investigation", "inv1", "investigator")
    services.incidents.change_status(inc["incident_id"], "closed", "inv1", "investigator", "done")


def test_fresh_evidence_is_never_eligible_and_open_incidents_are_protected(services, users):
    inc = seed_incident(services)
    fresh = services.retention.plan()
    assert fresh["assets_to_purge"] == 0 and fresh["eligible_incidents"] == [] and fresh["protected_incidents"] == []
    old = services.retention.plan(now=_later(inc))
    assert old["assets_to_purge"] == 0
    assert [p["incident_id"] for p in old["protected_incidents"]] == [inc["incident_id"]]
    assert "still detected" in old["protected_incidents"][0]["reasons"][0]
    # the same incident is eligible once it is resolved
    _resolve(services, users, inc)
    plan = services.retention.plan(now=_later(inc))
    assert [e["incident_id"] for e in plan["eligible_incidents"]] == [inc["incident_id"]]
    assert plan["assets_to_purge"] == len(services.evidence.list(inc["incident_id"])) > 0 and plan["bytes_to_free"] > 0


def test_an_unfinished_case_protects_the_incident_until_it_is_closed_or_rejected(services, users):
    inc = seed_incident(services)
    _resolve(services, users, inc)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    plan = services.retention.plan(now=_later(inc))
    assert plan["assets_to_purge"] == 0 and f"case {cid} is detected" in plan["protected_incidents"][0]["reasons"]
    services.cases.transition(cid, "reject", users["off1"], note="not an offence")
    assert services.retention.plan(now=_later(inc))["assets_to_purge"] > 0


def test_purge_needs_a_reason_and_the_exact_reviewed_plan(services, users):
    inc = seed_incident(services)
    _resolve(services, users, inc)
    now = _later(inc)
    plan = services.retention.plan(now=now)
    with pytest.raises(RetentionError) as e:
        services.retention.purge(plan["plan_id"], "short", "admin1", "administrator", now=now)
    assert e.value.status == 422
    with pytest.raises(RetentionError) as e:
        services.retention.purge("0000000000000000", "annual retention run 2027", "admin1", "administrator", now=now)
    assert e.value.status == 409 and "plan changed" in str(e.value)
    assert all(services.evidence.path_of(a).is_file() for a in services.evidence.list(inc["incident_id"]))  # nothing was removed


def test_purge_removes_files_keeps_rows_hashes_and_the_audit_chain(services, users):
    inc = seed_incident(services)
    iid = inc["incident_id"]
    _resolve(services, users, inc)
    before = services.evidence.list(iid)
    paths = [services.evidence.path_of(a) for a in before]
    assert all(p.is_file() for p in paths)
    now = _later(inc)
    plan = services.retention.plan(now=now)
    out = services.retention.purge(plan["plan_id"], "annual retention run 2027", "admin1", "administrator", now=now)
    assert out["files_removed"] == len(before) and out["failed"] == [] and out["bytes_freed"] == sum(a["size_bytes"] for a in before)
    assert not any(p.exists() for p in paths) and not (services.evidence.root / iid).exists()  # files and the empty incident dir are gone
    after = services.evidence.list(iid)
    assert [(a["asset_id"], a["sha256"]) for a in after] == [(a["asset_id"], a["sha256"]) for a in before]  # rows and hashes kept
    assert all(a["purged_at"] for a in after)
    v = services.evidence.verify(before[0]["asset_id"])
    assert v["ok"] is False and v["purged"] is True and v["registered_sha256"] == before[0]["sha256"]
    tomb = services.db.query("SELECT * FROM evidence_purges WHERE incident_id = ?", (iid,))
    assert len(tomb) == len(before) and {t["purged_by"] for t in tomb} == {"admin1"} and all(t["reason"] == "annual retention run 2027" for t in tomb)
    audit, _ = services.audit.query(action="retention.purge", limit=50)
    assert {a["action"] for a in audit} == {"retention.purge", "retention.purge_run"}
    assert services.audit.verify()["ok"]
    # idempotent: a second plan has nothing to do
    again = services.retention.plan(now=now)
    assert again["assets_to_purge"] == 0 and again["files_pending_removal"] == 0
    # tombstones are append-only
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute("DELETE FROM evidence_purges")
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute("UPDATE evidence_assets SET sha256 = 'x'")


def test_a_short_retention_period_cannot_be_used_to_purge(services, users):
    inc = seed_incident(services)
    _resolve(services, users, inc)
    services.settings.set("retention", {"evidence_days": MIN_PURGE_DAYS - 1, "audit_days": 3650}, "admin1")
    now = add_seconds(inc["detected_utc"], 3 * 86400 * 10)
    plan = services.retention.plan(now=now)
    assert plan["purge_allowed"] is False and plan["assets_to_purge"] > 0  # it shows what WOULD match, but refuses to act
    with pytest.raises(RetentionError) as e:
        services.retention.purge(plan["plan_id"], "trying a very short period", "admin1", "administrator", now=now)
    assert "minimum" in str(e.value)
    assert all(services.evidence.path_of(a).is_file() for a in services.evidence.list(inc["incident_id"]))


def test_files_left_behind_by_an_interrupted_purge_are_finished_by_the_next_run(services, users):
    inc = seed_incident(services)
    _resolve(services, users, inc)
    a = services.evidence.list(inc["incident_id"])[0]
    services.db.execute("INSERT INTO evidence_purges(purge_id, asset_id, incident_id, kind, sha256, size_bytes, relpath, reason, purged_at, purged_by, file_existed) "
                        "VALUES ('PRG-X', ?, ?, ?, ?, ?, ?, 'interrupted run', '2027-01-01T00:00:00.000Z', 'admin1', 1)",
                        (a["asset_id"], a["incident_id"], a["kind"], a["sha256"], a["size_bytes"], a["relpath"]))
    # the tombstone alone already stops the asset being served ...
    assert services.evidence.verify(a["asset_id"])["purged"] is True
    now = _later(inc)
    plan = services.retention.plan(now=now)
    assert plan["files_pending_removal"] == 1
    out = services.retention.purge(plan["plan_id"], "finishing the interrupted run", "admin1", "administrator", now=now)
    assert not services.evidence.path_of(a).exists() and out["failed"] == []


def test_an_asset_path_that_escapes_the_evidence_root_is_never_deleted(services, users, tmp_path):
    inc = seed_incident(services, with_evidence=False)
    _resolve(services, users, inc)
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive")
    services.db.execute("INSERT INTO evidence_assets(asset_id, incident_id, kind, relpath, sha256, size_bytes, mime, created_at, created_by) "
                        "VALUES ('AST-ESC', ?, 'frame', '../../outside.txt', 'x', 12, 'text/plain', '2026-01-01T00:00:00.000Z', 'system')", (inc["incident_id"],))
    now = _later(inc)
    plan = services.retention.plan(now=now)
    out = services.retention.purge(plan["plan_id"], "path escape attempt check", "admin1", "administrator", now=now)
    assert outside.read_text() == "must survive"
    assert out["files_removed"] == 0 and out["failed"] and "escapes" in out["failed"][0]["error"]


def test_retention_endpoints_are_administrator_only_and_require_the_reviewed_plan(api):
    base, svc = api
    inc = seed_incident(svc)
    svc.incidents.change_status(inc["incident_id"], "under_investigation", "inv1", "investigator")
    svc.incidents.change_status(inc["incident_id"], "closed", "inv1", "investigator", "done")
    adm, off, inv = client(base, "admin1"), client(base, "off1"), client(base, "inv1")
    for who in (off, inv):
        assert who.call("GET", "/retention/plan")[0] == 403
        assert who.call("POST", "/retention/purge", {"plan_id": "x", "reason": "abcdefghijkl"})[0] == 403
    st, plan, _ = adm.call("GET", "/retention/plan")
    assert st == 200 and plan["assets_to_purge"] == 0 and "_assets" not in plan  # nothing is old enough yet, internals are not exposed
    assert adm.call("POST", "/retention/purge", {"plan_id": plan["plan_id"]})[0] == 422  # reason required
    st, body, _ = adm.call("POST", "/retention/purge", {"plan_id": "wrong", "reason": "a perfectly good reason"})
    assert st == 409 and body["error"]["code"] == "retention"
    # a purge run with the real (empty) plan is allowed and harmless
    st, body, _ = adm.call("POST", "/retention/purge", {"plan_id": plan["plan_id"], "reason": "scheduled check, nothing due"})
    assert st == 200 and body["files_removed"] == 0
    # purged evidence answers 410 'purged' (not a bare 'missing' or a 500) and the report still renders
    now = _later(inc)
    p2 = svc.retention.plan(now=now)
    svc.retention.purge(p2["plan_id"], "retention run for the endpoint test", "admin1", "administrator", now=now)
    asset = svc.evidence.list(inc["incident_id"])[0]
    st, body, _ = off.call("GET", f"/evidence/{asset['asset_id']}/file")
    assert st == 410 and body["error"]["code"] == "purged"
    assert off.call("GET", f"/evidence/{asset['asset_id']}/verify")[1]["purged"] is True


def test_export_refuses_tampered_evidence_and_skips_purged_files(services, users, tmp_path):
    from .test_cases import _ready_case

    inc, cid = _ready_case(services, users)
    ok = build_case_package(services, cid, users["off1"], tmp_path / "x1")
    assert ok.is_file()
    import json
    import zipfile
    with zipfile.ZipFile(ok) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["purged_under_retention"] == [] and set(manifest["registered_evidence_sha256"]) == {a["asset_id"] for a in services.evidence.list(inc["incident_id"])}
    frame = next(a for a in services.evidence.list(inc["incident_id"]) if a["kind"] == "frame")
    services.evidence.path_of(frame).write_bytes(b"tampered after capture")
    with pytest.raises(CaseError) as e:
        build_case_package(services, cid, users["off1"], tmp_path / "x2")
    assert e.value.status == 409 and any("mismatch" in p for p in e.value.errors)
    assert not (tmp_path / "x2").exists() or not list((tmp_path / "x2").glob("*.zip"))  # no package was written


def test_schema_v2_migration_backs_up_a_populated_database_and_backfills_plate_verification(tmp_path, services, users):
    from .test_cases import _ready_case

    inc, cid = _ready_case(services, users)
    case = services.cases.get(cid)
    assert case["plate_verified_reading_id"]  # recorded by the new code
    want = case["plate_verified_reading_id"]
    path = services.config.db_path
    # turn the database back into a schema-v1 one
    downgrade_to(services.db, 1)
    services.close()
    reopened = AppDatabase(path)
    try:
        assert reopened.schema_version() == dbmod.SCHEMA_VERSION  # v1 -> current in one start
        backup = path.with_name(f"{path.stem}.pre-v2-migration{path.suffix}.bak")
        assert backup.is_file()
        old = sqlite3.connect(backup)
        assert old.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 1  # the copy is the pre-migration state
        old.close()
        row = reopened.one("SELECT plate_verified_reading_id FROM cases WHERE case_id = ?", (cid,))
        assert row["plate_verified_reading_id"] == want  # backfilled: newest verified, unsuperseded reading
        assert reopened.one("SELECT COUNT(*) AS n FROM evidence_purges")["n"] == 0
    finally:
        reopened.close()


# -- the retention screen's API contract (added 2026-10-21) -------------------------------------------------------------------


def test_the_plan_shows_files_protected_records_and_already_purged_counts_and_hides_internals(api):
    base, svc = api
    done = seed_incident(svc, key="done1")
    svc.incidents.change_status(done["incident_id"], "under_investigation", "inv1", "investigator")
    svc.incidents.change_status(done["incident_id"], "closed", "inv1", "investigator", "done")
    open_inc = seed_incident(svc, key="open1")  # still 'detected': must be protected
    adm = client(base, "admin1")
    now = _later(done)
    plan = svc.retention.public_plan(svc.retention.plan(now=now))
    assert plan["assets_to_purge"] > 0 and plan["already_purged_assets"] == 0
    assert {f["incident_id"] for f in plan["files_to_purge"]} == {done["incident_id"]}
    assert all(set(f) == {"asset_id", "incident_id", "kind", "relpath", "size_bytes", "sha256"} for f in plan["files_to_purge"])
    assert [p["incident_id"] for p in plan["protected_incidents"]] == [open_inc["incident_id"]]
    assert "_assets" not in plan and "_leftovers" not in plan
    svc.retention.purge(plan["plan_id"], "retention run for the screen contract test", "admin1", "administrator", now=now)
    after = svc.retention.public_plan(svc.retention.plan(now=now))
    assert after["already_purged_assets"] == plan["assets_to_purge"] and after["assets_to_purge"] == 0 and after["files_to_purge"] == []
    st, over, _ = adm.call("GET", "/retention/report")
    assert st == 200 and "assets_to_purge" in over


def test_a_stale_plan_id_is_refused_when_state_changed_after_the_preview(api):
    base, svc = api
    inc = seed_incident(svc, key="stale1")
    svc.incidents.change_status(inc["incident_id"], "under_investigation", "inv1", "investigator")
    svc.incidents.change_status(inc["incident_id"], "closed", "inv1", "investigator", "done")
    now = _later(inc)
    shown = svc.retention.plan(now=now)["plan_id"]
    # between the preview and the click a second resolved incident becomes eligible -> the reviewed plan no longer describes reality
    other = seed_incident(svc, key="stale2")
    svc.incidents.change_status(other["incident_id"], "under_investigation", "inv1", "investigator")
    svc.incidents.change_status(other["incident_id"], "closed", "inv1", "investigator", "done")
    with pytest.raises(RetentionError, match="plan changed"):
        svc.retention.purge(shown, "purge with a stale preview", "admin1", "administrator", now=_later(other))
    assert all(svc.evidence.path_of(a).is_file() for a in svc.evidence.list(inc["incident_id"]) + svc.evidence.list(other["incident_id"]))


def test_the_dashboard_retention_screen_gates_the_destructive_button_on_reason_plan_id_and_acknowledgement():
    js = (REPO_ROOT / "anvesh" / "app" / "static" / "views_admin.js").read_text(encoding="utf-8")
    assert "go.disabled = !(reason.value.trim().length >= 10 && typed.value.trim() === plan.plan_id && ack.checked)" in js
    assert 'A.post("/retention/purge", { plan_id: plan.plan_id' in js and "A.can(\"retention.manage\")" in js
    assert "audit records are never deleted" in js or "audit log is never deleted" in js
