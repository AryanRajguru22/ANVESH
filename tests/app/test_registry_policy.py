"""Registry approval invariant (project-owner decision 2026-10-21).

    An operational case cannot be approved unless its CURRENT plate reading is verified and a registry result that came from the
    configured AUTHORISED (non-sandbox) gateway, is fresh, well-formed, confirmed by the audit log and bound to exactly that reading says `found`.

There is no override. Every test here drives the real service layer (and, where stated, the real HTTP API or raw SQL). The
`AuthorisedRegistryDouble` is a TEST double standing in for a gateway that does not exist in the product; nothing here shows that any
real registry is connected or that any real vehicle record was retrieved.
"""

import json
import sqlite3
import time

import pytest

from anvesh.app import db as dbmod
from anvesh.app.cases import CaseError, CaseService
from anvesh.app.timeutil import add_seconds, now_iso

from .conftest import AuthorisedRegistryDouble, downgrade_to, lookup_and_approve, registry_lookup, seed_incident, use_authorised_registry
from .test_api import Client, api, client  # noqa: F401  (api is a fixture)
from .test_cases import _ready_case

RULE_REF = [{"code": "STOP-OBS", "description": "d", "event_types": ["stopped_vehicle"], "rule_reference": "CONFIGURED-REF"}]


def op_case(services, users, key="op1", plate="MH12AB1234", authorised=True):
    """An OPERATIONAL case at ready_for_review (rule reference configured, declared time basis, plate verified by someone else)."""
    services.cases.set_offences(RULE_REF, "admin1")
    double = use_authorised_registry(services) if authorised else None
    inc, cid = _ready_case(services, users, data_class="operational", key=key, plate=plate)
    return inc, cid, double


def blocked_audit(services, cid, action="case.approve_blocked"):
    return [a for a in services.audit.query(entity_type="case", entity_id=cid, limit=200)[0] if a["action"] == action]


def approve_error(services, users, cid, who="off1"):
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users[who])
    return e.value


def insert_lookup(services, cid, reading_id, *, adapter="authorised_registry_gateway", kind="production", simulated=0, status="found",
                  data=None, plate="MH12AB1234", source="TEST authority", retrieved=None, lookup_id="REG-FORGED", audited=False):
    """Hand-write a registry_lookups row (what a compromised or buggy writer could do) - optionally with a matching audit entry."""
    services.db.execute(
        "INSERT INTO registry_lookups(lookup_id, case_id, plate, adapter, adapter_kind, status, is_simulated, data_json, error, source_label, "
        "requested_by, retrieved_at, reading_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (lookup_id, cid, plate, adapter, kind, status, simulated, json.dumps({"registration_status": "active"} if data is None else data), None,
         source, "off1", retrieved or now_iso(), reading_id))
    if audited:
        services.audit.append("off1", "registry.lookup", "case", cid, {"lookup_id": lookup_id, "status": status, "adapter": kind}, actor_role="officer")


# -- the invariant ---------------------------------------------------------------------------------------------------------


def test_operational_approval_is_blocked_without_any_registry_result_and_the_attempt_is_audited(services, users):
    inc, cid, _ = op_case(services, users)
    err = approve_error(services, users, cid)
    assert err.status == 422 and any(x.startswith("registry") and "no registry lookup" in x for x in err.errors)
    assert services.cases.get(cid)["state"] == "ready_for_review"
    rows = blocked_audit(services, cid)
    assert len(rows) == 1 and rows[0]["actor"] == "off1" and rows[0]["actor_role"] == "officer"
    assert any(r.startswith("registry") for r in rows[0]["details"]["reasons"]) and rows[0]["details"]["http_status"] == 422
    assert services.audit.verify()["ok"]


def test_a_manual_plate_entry_or_a_verified_plate_is_not_a_registry_result(services, users):
    inc, cid, _ = op_case(services, users)
    case = services.cases.get(cid, users["off1"])
    assert case["current_plate"]["verified"] and case["current_plate"]["source"] in ("officer_verification", "officer_correction")
    assert "evidence_verified" not in " ".join(approve_error(services, users, cid).errors)  # everything else is fine ...
    assert any(i["item"] == "registry" and not i["ok"] for i in services.cases.get(cid, users["off1"])["readiness"])  # ... only the registry is missing


def test_with_no_authorised_registry_configured_operational_approval_stays_blocked_and_says_why(services, users):
    inc, cid, _ = op_case(services, users, authorised=False)
    look = registry_lookup(services, cid, users["off1"])
    assert look["status"] == "not_connected" and not look["is_simulated"] and not look["registry_verified"]
    err = approve_error(services, users, cid)
    assert any("no authorised registry is configured" in x or "returned 'not_connected'" in x for x in err.errors)


def test_a_sandbox_or_simulated_result_never_satisfies_an_operational_case(services, users):
    inc, cid, _ = op_case(services, users, authorised=False)
    reading = services.cases.get(cid)["plate_verified_reading_id"]
    # (1) the schema refuses a sandbox lookup on an operational case outright
    with pytest.raises(sqlite3.DatabaseError):
        insert_lookup(services, cid, reading, adapter="sandbox_fixture", kind="sandbox", simulated=1, lookup_id="REG-SB")
    # (2) a 'production' row that is flagged simulated (the fixture's tell-tale) is rejected by the service ...
    insert_lookup(services, cid, reading, simulated=1, lookup_id="REG-SIM", audited=True)
    err = approve_error(services, users, cid)
    assert any("sandbox/synthetic" in x for x in err.errors)
    # (3) ... and, if the service were bypassed, by the database
    with pytest.raises(sqlite3.DatabaseError, match="authorised registry result"):
        services.db.execute("UPDATE cases SET state = 'approved_for_submission' WHERE case_id = ?", (cid,))
    # (4) matching plate text and a high-quality payload change nothing
    assert services.cases.get(cid)["state"] == "ready_for_review"


def test_an_unauthorised_provider_or_unconfigured_gateway_is_refused(services, users):
    inc, cid, _ = op_case(services, users, authorised=False)
    reading = services.cases.get(cid)["plate_verified_reading_id"]
    insert_lookup(services, cid, reading, adapter="rogue_gateway", lookup_id="REG-ROGUE", audited=True)
    assert any("did not come from the configured authorised registry" in x for x in approve_error(services, users, cid).errors)
    insert_lookup(services, cid, reading, lookup_id="REG-RIGHTNAME", audited=True)  # right name, but no gateway is configured on this server
    assert any("did not come from the configured authorised registry" in x for x in approve_error(services, users, cid).errors)
    use_authorised_registry(services, is_connected=False)
    assert any("authorised registry" in x for x in approve_error(services, users, cid).errors)
    assert services.cases.get(cid)["state"] == "ready_for_review"


def test_a_hand_inserted_registry_row_without_an_audit_trail_does_not_count(services, users):
    inc, cid, _ = op_case(services, users)  # a gateway IS configured
    reading = services.cases.get(cid)["plate_verified_reading_id"]
    insert_lookup(services, cid, reading, lookup_id="REG-NOAUDIT", audited=False)
    assert any("not confirmed by the audit log" in x for x in approve_error(services, users, cid).errors)
    insert_lookup(services, cid, reading, lookup_id="REG-AUDITED", audited=True)  # a row that the audit chain also vouches for is accepted
    assert services.cases.transition(cid, "approve", users["off1"])["state"] == "approved_for_submission"


@pytest.mark.parametrize("bad_data", [{}, {"registration_status": None}, {"owner_name": "only owner fields"}, "not-a-dict"])
def test_malformed_or_empty_registry_data_is_refused(services, users, bad_data):
    inc, cid, _ = op_case(services, users)
    reading = services.cases.get(cid)["plate_verified_reading_id"]
    insert_lookup(services, cid, reading, data=bad_data, audited=True)
    assert any("malformed or empty" in x for x in approve_error(services, users, cid).errors)


def test_a_result_with_no_source_or_a_future_timestamp_is_refused(services, users):
    inc, cid, _ = op_case(services, users)
    reading = services.cases.get(cid)["plate_verified_reading_id"]
    insert_lookup(services, cid, reading, source=" ", lookup_id="REG-NOSRC", audited=True)
    insert_lookup(services, cid, reading, retrieved=add_seconds(now_iso(), 3600), lookup_id="REG-FUTURE", audited=True)
    msg = " ".join(approve_error(services, users, cid).errors)
    assert "malformed or empty" in msg and "future" in msg


def test_a_result_for_a_different_plate_or_reading_is_refused(services, users):
    inc, cid, double = op_case(services, users)
    registry_lookup(services, cid, users["off1"])  # genuine result for MH12AB1234
    other = services.plates.manual_entry(inc["incident_id"], "MH12AB1234", "inv1", "investigator", track_id="run-s000-t00001")  # same TEXT, new reading
    case = services.cases.get(cid)
    # a lookup row whose reading belongs to some other reading id / another plate text
    insert_lookup(services, cid, other["reading_id"], lookup_id="REG-OTHERREADING", audited=True)
    insert_lookup(services, cid, case["plate_verified_reading_id"], plate="KA01AB9999", lookup_id="REG-OTHERPLATE", audited=True)
    # the genuine one still satisfies the verified reading
    assert services.cases.registry_support(case, case["current_plate"])["lookup_id"] not in ("REG-OTHERREADING", "REG-OTHERPLATE")
    assert services.cases.registry_support(case, case["current_plate"])["ok"]
    # but with only the foreign rows, nothing is accepted
    inc2, cid2, _ = op_case(services, users, key="op2")
    r2 = services.cases.get(cid2)["plate_verified_reading_id"]
    insert_lookup(services, cid2, other["reading_id"], lookup_id="REG-FOREIGN1", audited=True)
    insert_lookup(services, cid2, r2, plate="KA01AB9999", lookup_id="REG-FOREIGN2", audited=True)
    assert any("different plate reading" in x for x in approve_error(services, users, cid2).errors)


def test_a_plate_correction_after_verification_and_lookup_invalidates_both(services, users):
    inc, cid, double = op_case(services, users)
    first = registry_lookup(services, cid, users["off1"])
    assert first["status"] == "found" and first["reading_id"] == services.cases.get(cid)["plate_verified_reading_id"]
    head = services.plates.current(inc["incident_id"])[0]
    services.plates.correct(head["reading_id"], "KA01XY9999", "rev2", "reviewer", note="misread", can_verify=True)
    err = approve_error(services, users, cid)
    assert any(x.startswith("plate_verified") for x in err.errors) and any(x.startswith("registry") for x in err.errors)
    # the operational lookup for the NEW (corrected but not yet case-verified) reading is refused: verify first
    with pytest.raises(ValueError, match="verified plate reading"):
        registry_lookup(services, cid, users["off1"], reading=services.plates.current(inc["incident_id"])[0])
    # proper route back: request_more, re-verify the new reading, look it up again, approve
    services.cases.transition(cid, "request_more", users["off1"], note="plate changed")
    services.cases.transition(cid, "resume_review", users["inv1"])
    services.cases.transition(cid, "verify_evidence", users["rev1"])
    services.cases.transition(cid, "verify_plate", users["rev1"], note="re-check")
    double.records["KA01XY9999"] = {"registration_status": "active", "vehicle_class": "motor car"}
    second = registry_lookup(services, cid, users["off1"])
    assert second["plate"] == "KA01XY9999" and second["lookup_id"] != first["lookup_id"]
    assert services.cases.transition(cid, "approve", users["off2"])["state"] == "approved_for_submission"


def test_a_stale_registry_result_is_refused_and_a_fresh_one_accepted(services, users):
    inc, cid, _ = op_case(services, users)
    registry_lookup(services, cid, users["off1"])
    case = services.cases.get(cid)
    assert services.cases.registry_support(case, case["current_plate"])["ok"]
    later = add_seconds(now_iso(), services.config.registry_result_max_age_s + 60)
    stale = services.cases.registry_support(case, case["current_plate"], now=later)
    assert not stale["ok"] and "stale" in stale["detail"]
    # end to end with a short window
    services.cases._registry_max_age_s = 1
    time.sleep(1.2)
    assert any("stale" in x for x in approve_error(services, users, cid).errors)
    registry_lookup(services, cid, users["off1"])  # a new lookup is fresh again
    assert services.cases.transition(cid, "approve", users["off1"])["state"] == "approved_for_submission"


def test_valid_authorised_result_allows_approval_and_everything_is_audited(services, users):
    inc, cid, double = op_case(services, users)
    look = registry_lookup(services, cid, users["off1"])
    assert look["adapter_kind"] == "production" and not look["is_simulated"] and look["registry_verified"] and double.calls == [("MH12AB1234", cid, "off1")]
    out = services.cases.transition(cid, "approve", users["off1"], note="plate and registry agree")
    assert out["state"] == "approved_for_submission" and out["approved_by"] == "off1" and not out["registry_override_reason"]
    ready = {i["item"]: i for i in services.cases.get(cid, users["off1"])["readiness"]}
    assert ready["registry"]["ok"] and look["lookup_id"] in ready["registry"]["detail"]
    audit = services.audit.query(entity_type="case", entity_id=cid, limit=200)[0]
    assert {"registry.lookup", "case.approve"} <= {a["action"] for a in audit}
    blob = json.dumps(audit)
    assert "TEST DOUBLE OWNER" not in blob and "owner_name" not in blob  # owner details never reach the audit trail
    assert not blocked_audit(services, cid)
    assert services.audit.verify()["ok"]
    # and the supported manual route works on the approved operational case
    assert services.handoff.submit(cid, "export", users["off1"])["status"] == "exported"


def test_direct_sql_cannot_move_an_operational_case_into_approved(services, users):
    inc, cid, _ = op_case(services, users)
    with pytest.raises(sqlite3.DatabaseError, match="authorised registry result"):
        services.db.execute("UPDATE cases SET state = 'approved_for_submission' WHERE case_id = ?", (cid,))
    with pytest.raises(sqlite3.DatabaseError, match="authorised registry result"):
        services.cases.apply_handoff_result(cid, "approved_for_submission", "x", "officer", "internal call must not bypass the rule")
    assert services.cases.get(cid)["state"] == "ready_for_review"
    registry_lookup(services, cid, users["off1"])
    services.db.execute("UPDATE cases SET state = 'approved_for_submission' WHERE case_id = ?", (cid,))  # allowed once the registry result exists


def test_a_registry_result_that_expires_after_approval_blocks_submission_and_retry(services, users):
    inc, cid, _ = op_case(services, users)
    lookup_and_approve(services, users, cid)
    services.cases._registry_max_age_s = 1
    time.sleep(1.2)
    with pytest.raises(CaseError) as e:
        services.handoff.submit(cid, "export", users["off1"])
    assert e.value.status == 409 and any("registry" in x for x in e.value.errors)
    assert blocked_audit(services, cid, "handoff.blocked")
    assert services.cases.get(cid)["state"] == "approved_for_submission"
    # the other way into 'approved': retry_submission after a failed hand-off
    services.cases.apply_handoff_result(cid, "submission_failed", "off1", "officer", "gateway down")
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "retry_submission", users["off1"])
    assert e.value.status == 422 and any(x.startswith("registry") for x in e.value.errors)
    assert blocked_audit(services, cid, "case.retry_submission_blocked")
    assert services.cases.get(cid)["state"] == "submission_failed"
    services.cases._registry_max_age_s = 24 * 3600  # a fresh lookup makes the retry legitimate again
    registry_lookup(services, cid, users["off1"])
    assert services.cases.transition(cid, "retry_submission", users["off1"])["state"] == "approved_for_submission"


def test_demo_cases_follow_the_same_shape_but_accept_only_the_labelled_sandbox(services, users):
    inc, cid = _ready_case(services, users)
    assert any(x.startswith("registry") for x in approve_error(services, users, cid).errors)
    look = registry_lookup(services, cid, users["off1"])
    assert look["is_simulated"] and "SANDBOX" in look["source_label"] and look["reading_id"]
    ready = {i["item"]: i for i in services.cases.get(cid, users["off1"])["readiness"]}
    assert "SANDBOX (test data)" in ready["registry"]["detail"]
    out = services.cases.transition(cid, "approve", users["off1"])
    assert out["state"] == "approved_for_submission" and out["labels"]["non_production"]


# -- unexpected parameters / spoofing -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("registry_override_reason", "officer verified against the RC book"), ("registry_verified", True), ("registry_status", "found"),
    ("lookup_id", "REG-FORGED"), ("is_simulated", False), ("adapter_kind", "production"), ("data_class", "demo"), ("state", "approved_for_submission"),
    ("approved_by", "off2"), ("plate_verified_reading_id", "PLT-X")])
def test_spoofing_registry_status_through_action_parameters_is_refused_and_audited(services, users, field, value):
    inc, cid, _ = op_case(services, users)
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users["off1"], **{field: value})
    assert e.value.status == 422 and "unexpected action parameter" in str(e.value)
    assert services.cases.get(cid)["state"] == "ready_for_review" and services.cases.get(cid)["data_class"] == "operational"
    assert blocked_audit(services, cid)


def test_a_direct_edit_cannot_set_state_or_class(services, users):
    inc, cid, _ = op_case(services, users)
    out = services.cases.update_fields(cid, users["inv1"], reason="new reason text")
    assert out["state"] == "ready_for_review"
    with pytest.raises(TypeError):
        services.cases.update_fields(cid, users["inv1"], state="approved_for_submission")  # no such parameter exists


# -- over HTTP ------------------------------------------------------------------------------------------------------------


def test_http_approval_policy_spoofed_bodies_and_the_correct_path(api):
    base, svc = api
    users = {}
    for name, role in (("inv9", "investigator"), ("rev9", "reviewer"), ("off9", "officer"), ("off8", "officer")):
        svc.auth.create_user(name, name, role, "pw-" + name + "-long")
        users[name] = svc.auth.login(name, "pw-" + name + "-long")[1]
    svc.cases.set_offences(RULE_REF, "admin1")
    inc = seed_incident(svc, data_class="operational", key="http1", camera_id="cam-http1")
    inv, rev, off = client(base, "inv9"), client(base, "rev9"), client(base, "off9")
    st, case, _ = inv.call("POST", f"/incidents/{inc['incident_id']}/cases", {"offence_code": "STOP-OBS"})
    cid = case["case_id"]
    assert inv.call("POST", f"/cases/{cid}/actions", {"action": "submit_for_evidence_verification"})[0] == 200
    assert rev.call("POST", f"/cases/{cid}/actions", {"action": "verify_evidence"})[0] == 200
    svc.plates.manual_entry(inc["incident_id"], "MH12AB1234", "inv9", "investigator", track_id="run-s000-t00001")
    assert rev.call("POST", f"/cases/{cid}/actions", {"action": "verify_plate", "note": "ok"})[0] == 200
    # no registry result: 422 with the reasons, no sensitive data
    st, body, _ = off.call("POST", f"/cases/{cid}/actions", {"action": "approve"})
    assert st == 422 and any(x.startswith("registry") for x in body["error"]["details"])
    # every spoof attempt is a 422 and changes nothing
    for extra in ({"registry_override_reason": "trust me"}, {"registry_verified": True}, {"lookup_id": "REG-X"}, {"data_class": "demo"}):
        st, body, _ = off.call("POST", f"/cases/{cid}/actions", dict({"action": "approve"}, **extra))
        assert st == 422, extra
    assert svc.cases.get(cid)["state"] == "ready_for_review"
    # a lookup request cannot choose its adapter or outcome
    st, look, _ = off.call("POST", f"/cases/{cid}/registry-lookup", {"adapter": "sandbox", "status": "found", "is_simulated": False, "adapter_kind": "production"})
    assert st == 201 and look["status"] == "not_connected" and look["adapter_kind"] == "production" and not look["registry_verified"]
    assert off.call("POST", f"/cases/{cid}/actions", {"action": "approve"})[0] == 422
    # PATCH cannot change state or class
    st, body, _ = inv.call("PATCH", f"/cases/{cid}", {"state": "approved_for_submission", "data_class": "demo", "reason": "edited reason"})
    assert st in (200, 409) and svc.cases.get(cid)["state"] == "ready_for_review" and svc.cases.get(cid)["data_class"] == "operational"
    # once an authorised gateway (TEST DOUBLE) is configured and looked up, approval succeeds
    use_authorised_registry(svc)
    st, look, _ = off.call("POST", f"/cases/{cid}/registry-lookup", {})
    assert st == 201 and look["status"] == "found" and look["registry_verified"] and not look["is_simulated"]
    st, body, _ = off.call("POST", f"/cases/{cid}/actions", {"action": "approve"})
    assert st == 200 and body["state"] == "approved_for_submission"
    audit = svc.audit.query(entity_type="case", entity_id=cid, limit=300)[0]
    assert sum(1 for a in audit if a["action"] == "case.approve_blocked") >= 5 and any(a["action"] == "case.approve" for a in audit)
    assert svc.audit.verify()["ok"]


# -- schema v3 ----------------------------------------------------------------------------------------------------------------


def test_schema_v3_migration_adds_the_reading_binding_and_the_trigger_with_a_backup(tmp_path, services, users):
    inc, cid = _ready_case(services, users)
    lookup = registry_lookup(services, cid, users["off1"])
    assert lookup["reading_id"]
    path = services.config.db_path
    downgrade_to(services.db, 2)
    services.close()
    reopened = dbmod.AppDatabase(path)
    try:
        assert reopened.schema_version() == dbmod.SCHEMA_VERSION
        backup = path.with_name(f"{path.stem}.pre-v3-migration{path.suffix}.bak")
        assert backup.is_file()
        old = sqlite3.connect(backup)
        assert old.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 2
        old.close()
        assert reopened.one("SELECT reading_id FROM registry_lookups WHERE lookup_id = ?", (lookup["lookup_id"],))["reading_id"] is None  # old lookups are unbound
        assert reopened.one("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = 'cases_operational_approval_needs_registry'")
    finally:
        reopened.close()
