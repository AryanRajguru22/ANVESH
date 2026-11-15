import hashlib
import json
import zipfile

import pytest

from anvesh.app.cases import CaseError
from anvesh.app.handoff import OfficialChallanAdapter, SandboxChallanAdapter
from anvesh.app.registry import ProductionRegistryAdapter

from .conftest import lookup_and_approve, seed_incident
from .test_cases import _ready_case


# -- registry ---------------------------------------------------------------------

def test_lookup_requires_a_reviewed_plate_not_an_ocr_candidate(services, users):
    inc = seed_incident(services)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    case = services.cases.get(cid, users["off1"])
    with pytest.raises(ValueError, match="officer-reviewed"):
        services.registry.lookup(case, None, "off1", "officer")
    ocr = {"status": "ocr_candidate", "verified": False, "normalized_text": "MH12AB1234"}
    with pytest.raises(ValueError, match="officer-reviewed"):
        services.registry.lookup(case, ocr, "off1", "officer")


def test_sandbox_lookup_is_labelled_synthetic_and_owner_fields_are_role_filtered(services, users):
    inc, cid = _ready_case(services, users)
    case = services.cases.get(cid, users["off1"])
    look = services.registry.lookup(case, case["current_plate"], "off1", "officer")
    assert look["status"] == "found" and look["registry_verified"] and look["is_simulated"]
    assert look["adapter_kind"] == "sandbox" and "SANDBOX" in look["source_label"]
    officer_view = services.registry.get(look["lookup_id"], users["off1"])
    assert "SYNTHETIC" in officer_view["data"]["owner_name"] and not officer_view["owner_fields_hidden"]
    reviewer_view = services.registry.get(look["lookup_id"], users["rev1"])
    assert "owner_name" not in reviewer_view["data"] and reviewer_view["owner_fields_hidden"]
    assert reviewer_view["data"]["registration_status"] == "active"
    admin_view = services.registry.get(look["lookup_id"], users["admin1"])
    assert "owner_name" not in admin_view["data"]


def test_unknown_plate_is_not_found_never_invented(services, users):
    inc, cid = _ready_case(services, users, plate="MH12ZZ0001")
    case = services.cases.get(cid, users["off1"])
    look = services.registry.lookup(case, case["current_plate"], "off1", "officer")
    assert look["status"] == "not_found" and look["data"] == {} and not look["registry_verified"]


def test_invalid_plate_format_is_not_sent_to_any_adapter(services, users):
    inc = seed_incident(services)
    cid = services.cases.create(inc["incident_id"], "STOP-OBS", users["inv1"])["case_id"]
    r = services.plates.manual_entry(inc["incident_id"], "NOTAPLATE", "inv1", "investigator")
    look = services.registry.lookup(services.cases.get(cid, users["off1"]), r, "off1", "officer")
    assert look["status"] == "invalid_request"


@pytest.mark.parametrize("status,payload,expected", [
    (200, {"found": True, "registration_status": "active", "owner_name": "X", "source": "Test Authority"}, "found"),
    (200, {"found": False}, "not_found"),
    (403, {"error": "no"}, "denied"),
    (500, {"error": "boom"}, "error"),
    (200, "<html>not json</html>", "error"),
    (200, {"unexpected": 1}, "error"),
])
def test_production_adapter_failure_modes(stub, status, payload, expected):
    srv = stub(lambda p, h, b: (status, payload, 0))
    res = ProductionRegistryAdapter(srv.url, "k").lookup("MH12AB1234", "CASE-1", "off1")
    assert res.status == expected
    assert srv.requests[0]["headers"]["Authorization"] == "Bearer k"
    assert json.loads(srv.requests[0]["body"])["purpose"] == "traffic_enforcement_case"
    if expected != "found":
        assert res.data == {}


def test_production_adapter_timeout_and_unconfigured(stub):
    srv = stub(lambda p, h, b: (200, {"found": True}, 1.5))
    slow = ProductionRegistryAdapter(srv.url, "k", timeout_s=0.3).lookup("MH12AB1234", "C", "u")
    assert slow.status == "error"
    nc = ProductionRegistryAdapter(None, None).lookup("MH12AB1234", "C", "u")
    assert nc.status == "not_connected" and "no authorised registry" in nc.error


def test_registry_status_reports_not_connected_honestly(services):
    st = {s["key"]: s for s in services.registry.status()}
    assert st["production"]["state"] == "not_connected" and "no government-database access" in st["production"]["detail"].lower()
    assert st["sandbox"]["state"] == "sandbox"


# -- hand-off ----------------------------------------------------------------------

def _approved(services, users, **kw):
    inc, cid = _ready_case(services, users, **kw)
    lookup_and_approve(services, users, cid)
    return inc, cid


def test_duplicate_submission_is_prevented_and_idempotent(services, users):
    inc, cid = _approved(services, users)
    first = services.handoff.submit(cid, "export", users["off1"])
    assert first["status"] == "exported" and first["duplicate"] is False
    assert services.db.one("SELECT COUNT(*) AS n FROM handoffs WHERE case_id = ?", (cid,))["n"] == 1
    with pytest.raises(CaseError):  # state is now 'submitted': a repeat, or another adapter, cannot submit
        services.handoff.submit(cid, "export", users["off1"])
    with pytest.raises(CaseError):
        services.handoff.submit(cid, "sandbox", users["off1"])
    assert services.db.one("SELECT COUNT(*) AS n FROM handoffs WHERE case_id = ?", (cid,))["n"] == 1


def test_idempotency_key_returns_the_existing_handoff_for_a_concurrent_duplicate(services, users):
    inc, cid = _approved(services, users)
    case = services.cases.get(cid, users["off1"])
    key = services.handoff.idempotency_key(cid, case["version"], "sandbox")
    services.db.execute(
        "INSERT INTO handoffs(handoff_id, case_id, adapter, adapter_kind, idempotency_key, case_version, status, is_simulated, submitted_by, created_at) "
        "VALUES ('HND-EXISTING', ?, 'sandbox_challan_simulator', 'sandbox', ?, ?, 'pending', 1, 'off1', 't')", (cid, key, case["version"]))
    again = services.handoff.submit(cid, "sandbox", users["off1"])
    assert again["duplicate"] is True and again["handoff_id"] == "HND-EXISTING"
    assert services.db.one("SELECT COUNT(*) AS n FROM handoffs WHERE case_id = ?", (cid,))["n"] == 1


def test_failed_submission_is_recorded_and_retry_succeeds(services, users):
    services.handoff.adapters["sandbox"] = SandboxChallanAdapter(fail_next=1)
    inc, cid = _approved(services, users)
    out = services.handoff.submit(cid, "sandbox", users["off1"])
    assert out["status"] == "failed" and out["error"] and out["external_reference"] is None
    assert services.cases.get(cid)["state"] == "submission_failed"
    services.cases.transition(cid, "retry_submission", users["off1"])
    ok = services.handoff.submit(cid, "sandbox", users["off1"])
    assert ok["status"] == "succeeded" and ok["duplicate"] is False
    assert services.cases.get(cid)["state"] == "challan_reference_received"
    hs = services.db.query("SELECT status FROM handoffs WHERE case_id = ? ORDER BY created_at", (cid,))
    assert [h["status"] for h in hs] == ["failed", "succeeded"]


def test_only_an_approved_case_and_only_officers_can_submit(services, users):
    inc, cid = _ready_case(services, users)
    with pytest.raises(CaseError) as e:
        services.handoff.submit(cid, "sandbox", users["off1"])
    assert e.value.status == 409
    lookup_and_approve(services, users, cid)
    with pytest.raises(CaseError) as e:
        services.handoff.submit(cid, "sandbox", users["rev1"])
    assert e.value.status == 403


def test_official_gateway_is_not_connected_and_sandbox_refused_for_operational(services, users):
    services.cases.set_offences([{"code": "STOP-OBS", "description": "d", "event_types": ["stopped_vehicle"], "rule_reference": "REF"}], "admin1")
    inc, cid = _approved(services, users, data_class="operational", key="op9")
    with pytest.raises(CaseError, match="sandbox simulator cannot be used"):
        services.handoff.submit(cid, "sandbox", users["off1"])
    with pytest.raises(CaseError, match="not connected"):
        services.handoff.submit(cid, "official", users["off1"])
    assert services.cases.get(cid)["state"] == "approved_for_submission"  # nothing happened
    out = services.handoff.submit(cid, "export", users["off1"])  # the supported manual route
    assert out["status"] == "exported" and not out["is_simulated"]
    st = {s["key"]: s for s in services.handoff.status()}
    assert st["official"]["state"] == "not_connected" and "NOT CONNECTED" in st["official"]["detail"]


def test_export_package_manifest_hashes_and_contents(services, users):
    inc, cid = _approved(services, users)
    out = services.handoff.submit(cid, "export", users["off1"])
    path = services.config.exports_dir / out["artifact_relpath"]
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("manifest.json"))
        assert {"report.html", "case.json", "audit.json", "manifest.json"} <= names
        assert any(n.startswith("evidence/") and n.endswith((".webm", ".mp4")) for n in names)
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        html = z.read("report.html").decode()
        assert "NON-PRODUCTION DATA" in html and "Candidate detection" in html and "no challan is issued" in html
        assert json.loads(z.read("case.json"))["case_id"] == cid and manifest["non_production"] is True
    assert services.cases.get(cid)["state"] == "submitted"
    out = services.cases.transition(cid, "record_reference", users["off1"], note="from portal", external_reference="EXT-12345")
    assert out["state"] == "challan_reference_received" and out["external_reference_kind"] == "officer_entered"


def test_official_adapter_gateway_contract_against_local_stub(stub):
    seen = {}

    def respond(path, headers, body):
        seen.update(headers=headers, body=json.loads(body))
        return 200, {"reference": "GW-001"}, 0

    srv = stub(respond)
    case = {"case_id": "CASE-1", "version": 3, "offence_code": "RLV", "rule_reference": "R", "current_plate": {"normalized_text": "MH12AB1234"},
            "incident": {"start_utc": "2026-10-01T10:00:00.000Z", "camera_id": "c1"}}
    res = OfficialChallanAdapter(srv.url, "key").submit(case, None, "idem-1")
    assert res.status == "succeeded" and res.external_reference == "GW-001"
    assert seen["headers"]["Idempotency-Key"] == "idem-1" and seen["body"]["plate"] == "MH12AB1234"
    bad = stub(lambda p, h, b: (200, {"nope": 1}, 0))
    assert OfficialChallanAdapter(bad.url, "key").submit(case, None, "i").status == "failed"
    down = stub(lambda p, h, b: (503, {}, 0))
    assert OfficialChallanAdapter(down.url, "key").submit(case, None, "i").status == "failed"
    assert OfficialChallanAdapter(None, None).submit(case, None, "i").status == "failed"
