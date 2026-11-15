import sqlite3

import pytest

from anvesh.app.cases import CaseError

from .conftest import AuthorisedRegistryDouble, lookup_and_approve, registry_lookup, seed_incident, use_authorised_registry


def _ready_case(services, users, data_class="demo", event_type="stopped_vehicle", offence="STOP-OBS", key="ep1", plate="MH12AB1234"):
    """A case taken to `ready_for_review` by the proper actors."""
    inc = seed_incident(services, data_class=data_class, event_type=event_type, key=key)
    c = services.cases.create(inc["incident_id"], offence, users["inv1"])
    cid = c["case_id"]
    services.cases.transition(cid, "submit_for_evidence_verification", users["inv1"])
    services.cases.transition(cid, "verify_evidence", users["rev1"])
    services.plates.manual_entry(inc["incident_id"], plate, "inv1", "investigator", track_id="run-s000-t00001")
    services.cases.transition(cid, "verify_plate", users["rev1"], note="read from frame")
    return inc, cid


def test_happy_path_sandbox_end_to_end_and_audit_history(services, users):
    inc, cid = _ready_case(services, users)
    case = services.cases.get(cid, users["off1"])
    assert case["state"] == "ready_for_review" and "approve" in case["allowed_actions"]
    lookup = services.registry.lookup(case, case["current_plate"], "off1", "officer")
    assert lookup["status"] == "found" and lookup["is_simulated"] and "SANDBOX" in lookup["notice"]
    services.cases.transition(cid, "approve", users["off1"])
    out = services.handoff.submit(cid, "sandbox", users["off1"])
    assert out["status"] == "succeeded" and out["external_reference"].startswith("TEST-CHALLAN-") and out["is_simulated"]
    final = services.cases.get(cid, users["off1"])
    assert final["state"] == "challan_reference_received" and final["external_reference_kind"] == "sandbox_simulated"
    assert "TEST reference" in final["labels"]["reference_notice"]
    services.cases.transition(cid, "close", users["off1"], note="done")
    states = [t["to_state"] for t in services.cases.get(cid, users["off1"])["transitions"]]
    assert states == ["detected", "awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review",
                      "approved_for_submission", "submitted", "challan_reference_received", "closed"]
    actors = {t["action"]: t["actor"] for t in services.cases.get(cid, users["off1"])["transitions"]}
    assert actors["verify_evidence"] == "rev1" and actors["approve"] == "off1" and actors["create"] == "inv1"
    audit, _ = services.audit.query(entity_type="case", entity_id=cid, limit=200)
    assert {"case.create", "case.verify_evidence", "case.verify_plate", "case.approve", "registry.lookup", "handoff.result", "case.close"} <= {a["action"] for a in audit}
    assert services.audit.verify()["ok"]


def test_role_permissions_are_enforced(services, users):
    inc = seed_incident(services)
    with pytest.raises(CaseError) as e:
        services.cases.create(inc["incident_id"], "STOP-OBS", users["rev1"])  # reviewers cannot create
    assert e.value.status == 403
    c = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])
    cid = c["case_id"]
    services.cases.transition(cid, "submit_for_evidence_verification", users["inv1"])
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "verify_evidence", users["inv1"])  # investigators cannot verify
    assert e.value.status == 403
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users["admin1"])  # administrators cannot approve
    assert e.value.status == 403


def test_invalid_transitions_and_notes(services, users):
    inc = seed_incident(services)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users["off1"])  # still 'detected'
    assert e.value.status == 409
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "reject", users["off1"])  # note required
    assert e.value.status == 422
    out = services.cases.transition(cid, "reject", users["off1"], note="not an offence")
    assert out["state"] == "rejected"
    with pytest.raises(CaseError):
        services.cases.transition(cid, "verify_evidence", users["rev1"])


def test_offence_must_match_incident_type_and_duplicates_are_refused(services, users):
    inc = seed_incident(services)
    with pytest.raises(CaseError) as e:
        services.cases.create(inc["incident_id"], "RLV", users["inv1"])
    assert e.value.status == 422
    services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])
    with pytest.raises(CaseError) as e:
        services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])
    assert e.value.status == 409


def test_evidence_verification_blocked_without_evidence_and_on_tampering(services, users):
    inc = seed_incident(services, with_evidence=False)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    services.cases.transition(cid, "submit_for_evidence_verification", users["inv1"])
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "verify_evidence", users["rev1"])
    assert e.value.status == 422 and any("no key frame" in p for p in e.value.errors)
    # now real evidence, then corrupt a file on disk: the hash check must catch it
    inc2 = seed_incident(services, key="ep2")
    cid2 = services.cases.create(inc2["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    services.cases.transition(cid2, "submit_for_evidence_verification", users["inv1"])
    frame = next(a for a in services.evidence.list(inc2["incident_id"]) if a["kind"] == "frame")
    services.evidence.path_of(frame).write_bytes(b"tampered")
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid2, "verify_evidence", users["rev1"])
    assert any("integrity" in p for p in e.value.errors)


def test_plate_step_needs_a_reading_and_a_different_verifier(services, users):
    inc = seed_incident(services)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    services.cases.transition(cid, "submit_for_evidence_verification", users["inv1"])
    services.cases.transition(cid, "verify_evidence", users["rev1"])
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "verify_plate", users["rev1"])
    assert "no plate reading" in str(e.value)
    services.plates.manual_entry(inc["incident_id"], "MH12AB1234", "rev1", "reviewer", track_id="run-s000-t00001")
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "verify_plate", users["rev1"])  # rev1 entered it
    assert "four-eyes" in str(e.value)
    assert services.cases.get(cid)["state"] == "awaiting_plate_verification"  # rolled back entirely
    assert services.cases.transition(cid, "verify_plate", users["rev2"])["state"] == "ready_for_review"


def test_approval_requires_a_registry_result_and_there_is_no_override(services, users):
    """Policy change 2026-10-21 (project owner): the former 'registry override reason' is abolished. This replaces the test that asserted it."""
    inc, cid = _ready_case(services, users)
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users["off1"])
    assert e.value.status == 422 and any(x.startswith("registry") for x in e.value.errors)
    with pytest.raises(CaseError) as e:  # the old escape hatch is now an unexpected parameter, not a way through
        services.cases.transition(cid, "approve", users["off1"], registry_override_reason="registry unavailable; verified against physical RC")
    assert e.value.status == 422 and "no registry override" in str(e.value)
    assert services.cases.get(cid)["state"] == "ready_for_review"
    out = lookup_and_approve(services, users, cid)  # the only way: a (here: sandbox, demo case) registry result for the verified reading
    assert out["state"] == "approved_for_submission" and not out["registry_override_reason"]


def test_four_eyes_blocks_approver_who_verified_evidence(services, users, monkeypatch):
    # With the shipped roles the actors are always distinct; this proves the rule still holds if a
    # deployment customises roles so one user holds both permissions.
    from anvesh.app import auth
    monkeypatch.setitem(auth.PERMISSIONS, "officer", auth.PERMISSIONS["officer"] | {"case.verify_evidence", "case.create"})
    inc = seed_incident(services)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    services.cases.transition(cid, "submit_for_evidence_verification", users["inv1"])
    services.cases.transition(cid, "verify_evidence", users["off1"])  # the officer verifies the evidence ...
    services.plates.manual_entry(inc["incident_id"], "MH12AB1234", "inv1", "investigator", track_id="run-s000-t00001")
    services.cases.transition(cid, "verify_plate", users["rev1"])
    registry_lookup(services, cid, users["off1"])
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users["off1"])  # ... and may not approve it
    assert any("four-eyes" in x for x in e.value.errors)
    assert services.cases.transition(cid, "approve", users["off2"])["approved_by"] == "off2"


def test_cases_cannot_be_edited_after_approval_but_can_before(services, users):
    inc, cid = _ready_case(services, users)
    before = services.cases.get(cid)["version"]
    out = services.cases.update_fields(cid, users["inv1"], reason="Vehicle stopped for 4 s in a live lane.")
    assert out["version"] == before + 1 and out["reason"].startswith("Vehicle stopped")
    lookup_and_approve(services, users, cid)
    with pytest.raises(CaseError) as e:
        services.cases.update_fields(cid, users["inv1"], reason="changed after approval")
    assert e.value.status == 409


def test_request_more_clears_verifications_and_resume_restarts_review(services, users):
    inc, cid = _ready_case(services, users)
    out = services.cases.transition(cid, "request_more", users["rev1"], note="frame is blurry")
    assert out["state"] == "needs_further_review" and out["evidence_verified_by"] is None and out["plate_verified_by"] is None
    out = services.cases.transition(cid, "resume_review", users["inv1"])
    assert out["state"] == "awaiting_evidence_verification"
    services.cases.transition(cid, "verify_evidence", users["rev2"])
    # the plate reading is already verified (immutable history); re-confirmation by someone else is allowed
    assert services.cases.transition(cid, "verify_plate", users["rev2"])["state"] == "ready_for_review"


def test_operational_case_needs_rule_reference_event_time_and_never_uses_sandbox_registry(services, users):
    inc, cid = _ready_case(services, users, data_class="operational", key="op1")
    case = services.cases.get(cid, users["off1"])
    assert case["data_class"] == "operational" and not case["labels"]["non_production"]
    # sandbox registry cannot attach: an operational case routes to the production adapter, which is not connected
    look = services.registry.lookup(case, case["current_plate"], "off1", "officer")
    assert look["status"] == "not_connected" and not look["is_simulated"] and not look["registry_verified"]
    with pytest.raises(CaseError) as e:
        services.cases.transition(cid, "approve", users["off1"])
    items = " ".join(e.value.errors)
    assert "rule_reference" in items and "registry" in items
    # defence in depth: the SCHEMA itself refuses a sandbox lookup / handoff on an operational case
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute(
            "INSERT INTO registry_lookups(lookup_id, case_id, plate, adapter, adapter_kind, status, is_simulated, source_label, requested_by, retrieved_at) "
            "VALUES ('x', ?, 'MH12AB1234', 'sandbox_fixture', 'sandbox', 'found', 1, 's', 'u', 't')", (cid,))
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute(
            "INSERT INTO handoffs(handoff_id, case_id, adapter, adapter_kind, idempotency_key, case_version, status, is_simulated, submitted_by, created_at) "
            "VALUES ('h', ?, 'sandbox', 'sandbox', 'k', 1, 'pending', 1, 'u', 't')", (cid,))
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute("UPDATE cases SET data_class = 'demo' WHERE case_id = ?", (cid,))


def test_operational_case_with_processing_clock_time_cannot_be_approved(services, users):
    services.cases.set_offences([{"code": "STOP-OBS", "description": "d", "event_types": ["stopped_vehicle"], "rule_reference": "CONFIGURED-REF"}], "admin1")
    use_authorised_registry(services)  # TEST DOUBLE for an authorised gateway; none exists in the product
    inc, cid = _ready_case(services, users, data_class="operational", key="op2")
    ok = lookup_and_approve(services, users, cid)
    assert ok["state"] == "approved_for_submission"  # configured rule reference + declared event time + authorised registry result
    seed = seed_incident(services, data_class="operational", key="op3", time_basis="processing_clock", camera_id="cam-op3")
    c2 = services.cases.create(seed["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    services.cases.transition(c2, "submit_for_evidence_verification", users["inv1"])
    services.cases.transition(c2, "verify_evidence", users["rev1"])
    services.plates.manual_entry(seed["incident_id"], "MH12AB1234", "inv1", "investigator", track_id="run-s000-t00001")
    services.cases.transition(c2, "verify_plate", users["rev1"])
    registry_lookup(services, c2, users["off1"])  # the registry requirement is met here, so ONLY the time basis blocks
    with pytest.raises(CaseError) as e:
        services.cases.transition(c2, "approve", users["off1"])
    assert any("event_time" in x and "unknown" in x for x in e.value.errors) and not any(x.startswith("registry") for x in e.value.errors)


def test_list_and_search_cases_by_plate(services, users):
    inc, cid = _ready_case(services, users)
    rows, total = services.cases.list({"q": "mh12ab"})
    assert total == 1 and rows[0]["plate"]["text"] == "MH12AB1234" and rows[0]["plate"]["verified"]
    assert services.cases.list({"state": "closed"})[1] == 0
    assert services.cases.counts() == {"ready_for_review": 1}
