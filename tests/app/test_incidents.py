import pytest

from anvesh.app.detectors import DetectedEvent
from anvesh.app.incidents import SessionInfo

from .conftest import seed_incident


def _info(session_id, camera_id="cam-seed"):
    return SessionInfo(session_id, camera_id, "demo", "2026-10-01T10:00:00.000Z", "declared_recording_start")


def test_upsert_is_idempotent_per_natural_key(services):
    inc = seed_incident(services)
    again = seed_incident(services, with_evidence=False)  # same key, same session -> same incident
    assert again["incident_id"] == inc["incident_id"]
    assert services.db.one("SELECT COUNT(*) AS n FROM incidents")["n"] == 1


def test_update_escalates_but_never_downgrades_severity_and_records_end(services):
    inc = seed_incident(services, severity="medium")
    info = _info(inc["session_id"])
    ev = DetectedEvent(event_type="long_duration_stop", natural_key="ep1", start_offset_s=10, detected_offset_s=14, detector="d",
                       severity="high", severity_basis="long", end_offset_s=50.0, escalate_to="long_duration_stop", indicators={"dwell_s": 40})
    inc2, created, changes = services.incidents.upsert(info, ev)
    assert not created
    assert inc2["event_type"] == "long_duration_stop" and inc2["severity"] == "high" and inc2["end_offset_s"] == 50.0
    assert any("escalated" in c for c in changes) and "ended" in changes
    low = DetectedEvent(event_type="stopped_vehicle", natural_key="ep1", start_offset_s=10, detected_offset_s=14, detector="d",
                        severity="low", severity_basis="x", indicators={"dwell_s": 40})
    inc3, _, _ = services.incidents.upsert(info, low)
    assert inc3["severity"] == "high"  # not downgraded by a later, weaker window
    full = services.incidents.get(inc["incident_id"], with_timeline=True)
    assert full["timeline"][0]["kind"] == "detected"
    assert any(t["kind"] == "update" for t in full["timeline"])


def test_congestion_after_incident_is_linked_as_temporal_association_not_cause(services):
    stop = seed_incident(services, key="stop1")
    info = _info(stop["session_id"])
    cong = DetectedEvent(event_type="congestion", natural_key="cong1", start_offset_s=40.0, detected_offset_s=40.0, detector="c",
                         severity="medium", severity_basis="x", indicators={"current_level": "congested"})
    c_inc, created, _ = services.incidents.upsert(info, cong)
    assert created
    stop2 = services.incidents.get(stop["incident_id"], with_timeline=True)
    assert stop2["disruption"]["causal_claim"] is False
    assert stop2["disruption"]["followed_by_congestion"] == c_inc["incident_id"]
    assert c_inc["related"][0]["causal_claim"] is False and c_inc["related"][0]["relation"] == "temporally_associated_with"
    assert any(t["kind"] == "link" and "temporal association only" in t["summary"] for t in stop2["timeline"])


def test_officer_actions_status_notes_cause(services, users):
    inc = seed_incident(services)
    iid = inc["incident_id"]
    s = services.incidents
    with pytest.raises(ValueError):
        s.change_status(iid, "closed", "inv1", "investigator", "no")  # detected -> closed not allowed
    s.change_status(iid, "under_investigation", "inv1", "investigator")
    with pytest.raises(ValueError):
        s.change_status(iid, "dismissed", "inv1", "investigator")  # needs a note
    s.add_note(iid, "checked the clip", "inv1", "investigator")
    assert s.get(iid)["established_cause"] is None
    with pytest.raises(ValueError):
        s.establish_cause(iid, "mechanical failure", "", "inv1", "investigator")
    out = s.establish_cause(iid, "mechanical failure", "driver statement + clip", "inv1", "investigator")
    assert out["established_cause_by"] == "inv1" and "driver statement" in out["established_cause"]
    s.change_status(iid, "confirmed", "inv1", "investigator")
    s.change_status(iid, "closed", "inv1", "investigator", "resolved")
    with pytest.raises(ValueError):
        s.change_status(iid, "detected", "inv1", "investigator")
    kinds = [a["action"] for a in services.audit.query(entity_type="incident", entity_id=iid)[0]]
    assert "incident.establish_cause" in kinds and "incident.status" in kinds


def test_list_filters_pagination_and_priority(services):
    seed_incident(services, key="a", severity="low")
    seed_incident(services, key="b", severity="high", event_type="wrong_way")
    seed_incident(services, key="c", severity="medium")
    rows, total = services.incidents.list(order="priority")
    assert total == 3 and [r["severity"] for r in rows] == ["high", "medium", "low"]
    rows, total = services.incidents.list({"event_type": "wrong_way"})
    assert total == 1 and rows[0]["severity"] == "high"
    rows, total = services.incidents.list(limit=2, offset=2)
    assert total == 3 and len(rows) == 1
    assert services.incidents.list({"severity": "critical"})[1] == 0
