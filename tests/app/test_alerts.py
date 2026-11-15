import json

import pytest

from anvesh.app.alerts import BACKOFF_BASE_S, AlertRouter, WebhookAlertAdapter

from .conftest import seed_incident


def _rules(services, adapter="test", severity="high", recipient="control-room"):
    return services.alerts.set_rules([{"name": "r", "min_severity": severity, "adapter": adapter, "recipient": recipient}], "admin1")


def test_default_rule_routes_high_severity_to_test_adapter_and_never_confirms(services):
    inc = seed_incident(services, severity="high", event_type="possible_collision", key="c1")
    rows = services.alerts.list(incident_id=inc["incident_id"])
    assert len(rows) == 1 and rows[0]["status"] == "pending" and rows[0]["is_test"] == 1
    assert services.alerts.process_due() == 1
    row = services.alerts.list(incident_id=inc["incident_id"])[0]
    assert row["status"] == "sent" and row["delivery_confirmed"] == 0
    assert "TEST ADAPTER" in row["notice"] and "not delivered to any external party" in row["notice"]
    out = list(services.config.outbox_dir.glob("*.json"))
    body = json.loads(out[0].read_text())
    assert body["TEST_ADAPTER"] is True and body["delivered_to_external_party"] is False
    assert body["payload"]["candidate_notice"].startswith("Automated candidate detection")
    assert services.incidents.get(inc["incident_id"])["alert_status"] == "test_recorded_only"


def test_below_threshold_and_event_type_filter(services):
    _rules(services, severity="high")
    seed_incident(services, severity="medium", key="m1")
    assert services.alerts.list() == []
    services.alerts.set_rules([{"name": "only-wrong-way", "min_severity": "low", "adapter": "test", "recipient": "x", "event_types": ["wrong_way"]}], "admin1")
    seed_incident(services, severity="high", key="s1")  # stopped_vehicle: no match
    assert services.alerts.list() == []
    seed_incident(services, severity="low", event_type="wrong_way", key="w1")
    assert len(services.alerts.list()) == 1


def test_unconfigured_channel_cannot_be_used_in_a_rule(services):
    with pytest.raises(ValueError, match="not configured"):
        _rules(services, adapter="webhook")
    with pytest.raises(ValueError, match="not configured"):
        _rules(services, adapter="email")
    with pytest.raises(ValueError):
        services.alerts.set_rules([{"min_severity": "nope", "adapter": "test", "recipient": "x"}], "admin1")


def _with_webhook(services, url, operational=False):
    services.alerts.adapters["webhook"] = WebhookAlertAdapter(url, "secret")
    return _rules(services, adapter="webhook", severity="low")


def test_non_operational_data_can_never_use_a_real_channel(services, stub):
    srv = stub(lambda p, h, b: (200, {}, 0))
    _with_webhook(services, srv.url)
    inc = seed_incident(services, severity="high", key="d1")  # data class demo
    row = services.alerts.list(incident_id=inc["incident_id"])[0]
    assert row["status"] == "abandoned" and "blocked" in row["last_error"]
    assert services.alerts.process_due() == 0 and srv.requests == []  # nothing was sent
    with pytest.raises(ValueError, match="blocked"):
        services.alerts.retry(row["delivery_id"], "inv1")
    assert services.incidents.get(inc["incident_id"])["alert_status"] == "failed"


def test_operational_webhook_confirmed_only_on_2xx_and_signed(services, stub):
    srv = stub(lambda p, h, b: (200, {"ok": True}, 0))
    _with_webhook(services, srv.url)
    inc = seed_incident(services, data_class="operational", severity="high", key="o1", camera_id="cam-o1")
    assert services.alerts.process_due() == 1
    row = services.alerts.list(incident_id=inc["incident_id"])[0]
    assert row["status"] == "sent" and row["delivery_confirmed"] == 1 and "confirmed by the receiving endpoint" in row["notice"]
    req = srv.requests[0]
    headers = {k.lower(): v for k, v in req["headers"].items()}
    assert headers["x-anvesh-signature"].startswith("sha256=") and json.loads(req["body"])["payload"]["incident_id"] == inc["incident_id"]
    assert services.incidents.get(inc["incident_id"])["alert_status"] == "delivered_confirmed"


def test_delivery_failure_retries_with_backoff_then_abandons_and_is_audited(services, stub):
    srv = stub(lambda p, h, b: (503, {}, 0))
    _with_webhook(services, srv.url)
    inc = seed_incident(services, data_class="operational", severity="high", key="o2", camera_id="cam-o2")
    iid = inc["incident_id"]
    now = "2999-01-01T00:00:00.000Z"
    for attempt in range(1, 5):
        assert services.alerts.process_due(now) == 1
        row = services.alerts.list(incident_id=iid)[0]
        assert row["attempts"] == attempt and row["delivery_confirmed"] == 0
        if attempt < 4:
            assert row["status"] == "pending" and row["last_error"] == "HTTP 503" and row["next_attempt_at"] is not None
    assert row["status"] == "abandoned" and "not confirmed" in row["notice"]
    assert services.alerts.process_due(now) == 0  # no more attempts
    assert services.incidents.get(iid)["alert_status"] == "failed"
    attempts = [a for a in services.audit.query(entity_type="incident", entity_id=iid, action="alert.attempt")[0]]
    assert len(attempts) == 4 and all(a["details"]["ok"] is False for a in attempts)
    # a manual retry is possible, and a recovered endpoint then confirms
    srv2 = stub(lambda p, h, b: (200, {}, 0))
    services.alerts.adapters["webhook"] = WebhookAlertAdapter(srv2.url, "secret")
    services.alerts.retry(row["delivery_id"], "inv1", "investigator")
    assert services.alerts.process_due() == 1
    assert services.alerts.list(incident_id=iid)[0]["delivery_confirmed"] == 1


def test_backoff_grows_exponentially(services, stub):
    srv = stub(lambda p, h, b: (500, {}, 0))
    _with_webhook(services, srv.url)
    inc = seed_incident(services, data_class="operational", severity="high", key="o3", camera_id="cam-o3")
    services.alerts.process_due("2999-01-01T00:00:00.000Z")
    d1 = services.alerts.list(incident_id=inc["incident_id"])[0]["next_attempt_at"]
    services.alerts.process_due("2999-01-01T00:00:00.000Z")
    d2 = services.alerts.list(incident_id=inc["incident_id"])[0]["next_attempt_at"]
    from anvesh.app.timeutil import parse_iso
    gap = (parse_iso(d2) - parse_iso(d1)).total_seconds()
    assert gap > BACKOFF_BASE_S * 0.9  # second backoff (60 s) is longer than the first (30 s)


def test_crashing_adapter_is_a_failed_attempt_not_a_crashed_worker(services):
    class Boom:
        kind = "webhook"
        label = "boom"

        def connected(self):
            return True

        def send(self, d):
            raise RuntimeError("kaboom")

    services.alerts.adapters["webhook"] = Boom()
    _rules(services, adapter="webhook", severity="low")
    inc = seed_incident(services, data_class="operational", severity="high", key="o4", camera_id="cam-o4")
    assert services.alerts.process_due() == 1
    row = services.alerts.list(incident_id=inc["incident_id"])[0]
    assert row["status"] == "pending" and "kaboom" in row["last_error"]


def test_alert_failure_never_breaks_incident_creation(services):
    def explode(incident):
        raise RuntimeError("alert subsystem down")

    services.incidents.on_created.insert(0, explode)
    inc = seed_incident(services, severity="high", key="x1")
    assert inc["incident_id"] and services.incidents.get(inc["incident_id"]) is not None
