"""Alert routing to authorised recipients, with delivery records, retries and an audit trail.

CHANNELS (adapters) and what each can truthfully claim:
  test     -- writes the alert to a local outbox file. It reaches NOBODY. It is labelled as a test
              adapter everywhere and its deliveries are never `delivery_confirmed`.
  webhook  -- HTTP POST (HMAC-signed) to a configured URL; `delivery_confirmed` only on a 2xx
              response, i.e. "the receiving endpoint accepted it", not "a human read it".
  email    -- SMTP to a configured relay; confirmed only means the relay accepted the message.
Only `test` works without configuration. A webhook/email rule can be created only when that
channel is configured (`AppConfig.webhook_url` / `smtp_*`), and ANVESH never claims an authority
was notified unless a delivery row says `delivery_confirmed = 1`.

SAFETY RAILS
  - Incidents from non-operational data (`demo`, `simulated`) may ONLY use the test adapter: the
    delivery is recorded as `abandoned` with the reason instead of sending a real message.
  - Tests never use a real channel: webhook tests post to a local stub server.
  - A delivery failing never interrupts detection (incident callbacks swallow and log).

RETRIES. A failed attempt is retried with exponential backoff up to `max_attempts`, then
`abandoned`; an authorised user can retry manually. Every attempt is audited.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import logging
import smtplib
import threading
import urllib.error
import urllib.request
import uuid
from email.message import EmailMessage

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.incidents import SEVERITY_ORDER
from anvesh.app.timeutil import add_seconds, now_iso

log = logging.getLogger(__name__)

BACKOFF_BASE_S = 30.0


@dataclasses.dataclass(frozen=True)
class AdapterResult:
    ok: bool
    confirmed: bool = False
    confirmation: str = None
    error: str = None


class TestAlertAdapter:
    kind = "test"
    label = "TEST ADAPTER - writes a local file; delivers to no external party"

    def __init__(self, outbox_dir) -> None:
        self._dir = outbox_dir

    def connected(self) -> bool:
        return True

    def send(self, delivery: dict) -> AdapterResult:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{delivery['delivery_id']}.json"
        body = {"TEST_ADAPTER": True, "delivered_to_external_party": False, "recipient": delivery["recipient"],
                "subject": delivery["subject"], "payload": loads(delivery["payload_json"], {})}
        with path.open("w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=2)
        return AdapterResult(True, confirmed=False, confirmation=f"recorded in local test outbox: {path.name}")


class WebhookAlertAdapter:
    kind = "webhook"
    label = "Webhook (HTTP POST)"

    def __init__(self, url: str, secret: str = None, timeout_s: float = 10.0) -> None:
        self._url, self._secret, self._timeout = url, secret, timeout_s

    def connected(self) -> bool:
        return bool(self._url)

    def send(self, delivery: dict) -> AdapterResult:
        if not self._url:
            return AdapterResult(False, error="webhook not configured")
        body = json.dumps({"recipient": delivery["recipient"], "subject": delivery["subject"],
                           "payload": loads(delivery["payload_json"], {}), "delivery_id": delivery["delivery_id"]}).encode()
        headers = {"Content-Type": "application/json", "X-ANVESH-Delivery": delivery["delivery_id"]}
        if self._secret:
            headers["X-ANVESH-Signature"] = "sha256=" + hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 - operator-configured URL
                if 200 <= resp.status < 300:
                    return AdapterResult(True, confirmed=True, confirmation=f"HTTP {resp.status} from webhook endpoint")
                return AdapterResult(False, error=f"HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            return AdapterResult(False, error=f"HTTP {exc.code}")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return AdapterResult(False, error=f"{type(exc).__name__}: {exc}")


class EmailAlertAdapter:
    kind = "email"
    label = "Email (SMTP relay)"

    def __init__(self, host, port, user, password, sender) -> None:
        self._cfg = (host, port, user, password, sender)

    def connected(self) -> bool:
        return bool(self._cfg[0] and self._cfg[4])

    def send(self, delivery: dict) -> AdapterResult:
        host, port, user, password, sender = self._cfg
        if not (host and sender):
            return AdapterResult(False, error="SMTP not configured")
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = sender, delivery["recipient"], delivery["subject"]
        msg.set_content(json.dumps(loads(delivery["payload_json"], {}), indent=2))
        try:
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.starttls()
                if user:
                    smtp.login(user, password or "")
                smtp.send_message(msg)
            return AdapterResult(True, confirmed=True, confirmation="accepted by SMTP relay (not a read receipt)")
        except (smtplib.SMTPException, OSError) as exc:
            return AdapterResult(False, error=f"{type(exc).__name__}: {exc}")


DEFAULT_RULES = [{
    "rule_id": "default-control-room", "name": "Control room - high severity (TEST ADAPTER)", "enabled": True,
    "min_severity": "high", "event_types": [], "adapter": "test", "recipient": "control-room (test outbox)",
}]


class AlertRouter:
    def __init__(self, db: AppDatabase, audit: AuditLog, incidents, adapters: dict, settings) -> None:
        self._db, self._audit, self._incidents, self.adapters, self._settings = db, audit, incidents, adapters, settings
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        incidents.on_created.append(self.on_incident)

    # -- configuration -------------------------------------------------------

    def rules(self) -> list:
        return self._settings.get("alert_rules", DEFAULT_RULES)

    def adapter_status(self) -> list:
        return [{"adapter": k, "label": a.label, "connected": a.connected(), "is_test": k == "test"} for k, a in self.adapters.items()]

    def set_rules(self, rules: list, actor: str, role: str = None) -> list:
        clean = []
        for i, r in enumerate(rules):
            adapter = r.get("adapter")
            if adapter not in self.adapters:
                raise ValueError(f"rule {i + 1}: adapter must be one of {sorted(self.adapters)}")
            if not self.adapters[adapter].connected():
                raise ValueError(f"rule {i + 1}: the {adapter} channel is not configured on this server")
            if r.get("min_severity") not in SEVERITY_ORDER:
                raise ValueError(f"rule {i + 1}: min_severity must be one of {SEVERITY_ORDER}")
            if not (r.get("recipient") or "").strip():
                raise ValueError(f"rule {i + 1}: recipient must be non-empty")
            clean.append({"rule_id": r.get("rule_id") or "rule-" + uuid.uuid4().hex[:6], "name": (r.get("name") or r["recipient"]).strip(),
                          "enabled": bool(r.get("enabled", True)), "min_severity": r["min_severity"],
                          "event_types": list(r.get("event_types") or []), "adapter": adapter, "recipient": r["recipient"].strip()})
        self._settings.set("alert_rules", clean, actor)
        self._audit.append(actor, "alert.rules_update", "settings", "alert_rules", {"rules": len(clean)}, actor_role=role)
        return clean

    # -- routing -------------------------------------------------------------

    def on_incident(self, incident: dict) -> list:
        created = []
        for rule in self.rules():
            if not rule.get("enabled", True):
                continue
            if SEVERITY_ORDER.index(incident["severity"]) < SEVERITY_ORDER.index(rule["min_severity"]):
                continue
            if rule.get("event_types") and incident["event_type"] not in rule["event_types"]:
                continue
            created.append(self._enqueue(incident, rule))
        if created:
            self._refresh_incident_status(incident["incident_id"])
            self._wake.set()
        return created

    def _enqueue(self, incident: dict, rule: dict) -> str:
        adapter_kind = rule["adapter"]
        non_operational = incident["data_class"] != "operational"
        blocked = non_operational and adapter_kind != "test"
        marker = "[TEST DATA] " if non_operational else ""
        subject = f"{marker}ANVESH {incident['severity'].upper()}: {incident['event_type'].replace('_', ' ')} on {incident['camera_id']}"
        payload = {
            "incident_id": incident["incident_id"], "event_type": incident["event_type"], "severity": incident["severity"],
            "camera_id": incident["camera_id"], "detected_utc": incident["detected_utc"], "time_basis": incident["time_basis"],
            "location": incident["location"], "data_class": incident["data_class"],
            "candidate_notice": "Automated candidate detection awaiting human review; not an established finding.",
            "confidence_tier": incident["confidence_tier"], "uncertainty": incident["uncertainty"][:3],
        }
        delivery_id = "ALR-" + uuid.uuid4().hex[:10].upper()
        status = "abandoned" if blocked else "pending"
        error = "blocked: non-operational (demo/simulated) data may only use the test adapter" if blocked else None
        now = now_iso()
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO alert_deliveries(delivery_id, incident_id, rule_id, channel, adapter_kind, recipient, subject, payload_json, "
                "status, is_test, attempts, max_attempts, next_attempt_at, last_error, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,0,4,?,?,?,?)",
                (delivery_id, incident["incident_id"], rule["rule_id"], rule["name"], adapter_kind, rule["recipient"], subject, dumps(payload),
                 status, 1 if adapter_kind == "test" else 0, None if blocked else now, error, now, now))
        self._audit.append("system", "alert.queued" if not blocked else "alert.blocked", "incident", incident["incident_id"],
                           {"delivery_id": delivery_id, "adapter": adapter_kind, "rule": rule["rule_id"], "reason": error})
        return delivery_id

    # -- delivery ------------------------------------------------------------

    def process_due(self, now: str = None) -> int:
        """Attempt every pending delivery that is due. Returns the number of attempts made."""
        now = now or now_iso()
        due = self._db.query("SELECT * FROM alert_deliveries WHERE status = 'pending' AND next_attempt_at <= ? ORDER BY created_at", (now,))
        for d in due:
            self._attempt(d)
        return len(due)

    def _attempt(self, d: dict) -> None:
        adapter = self.adapters.get(d["adapter_kind"])
        try:
            result = adapter.send(d) if adapter is not None else AdapterResult(False, error="adapter missing")
        except Exception as exc:  # noqa: BLE001 - a crashing adapter is a failed attempt, not a crashed worker
            result = AdapterResult(False, error=f"adapter crashed: {type(exc).__name__}: {exc}")
        attempts = d["attempts"] + 1
        now = now_iso()
        if result.ok:
            status, nxt, err = "sent", None, None
        elif attempts >= d["max_attempts"]:
            status, nxt, err = "abandoned", None, result.error
        else:
            status, nxt, err = "pending", add_seconds(now, BACKOFF_BASE_S * (2 ** (attempts - 1))), result.error
        with self._db.write() as conn:
            conn.execute(
                "UPDATE alert_deliveries SET status = ?, attempts = ?, next_attempt_at = ?, last_error = ?, delivery_confirmed = ?, "
                "confirmation = ?, updated_at = ? WHERE delivery_id = ?",
                (status, attempts, nxt, err, 1 if (result.ok and result.confirmed) else 0, result.confirmation, now, d["delivery_id"]))
        self._audit.append("system", "alert.attempt", "incident", d["incident_id"],
                           {"delivery_id": d["delivery_id"], "attempt": attempts, "ok": result.ok, "confirmed": result.confirmed,
                            "error": result.error, "adapter": d["adapter_kind"]})
        self._refresh_incident_status(d["incident_id"])

    def retry(self, delivery_id: str, actor: str, role: str = None) -> dict:
        d = self._db.one("SELECT * FROM alert_deliveries WHERE delivery_id = ?", (delivery_id,))
        if d is None:
            raise LookupError(delivery_id)
        if d["status"] == "sent":
            raise ValueError("delivery already sent")
        if "blocked:" in (d["last_error"] or ""):
            raise ValueError("this delivery was blocked by the data-class rule and cannot be retried")
        with self._db.write() as conn:
            conn.execute("UPDATE alert_deliveries SET status = 'pending', attempts = 0, next_attempt_at = ?, updated_at = ? WHERE delivery_id = ?",
                         (now_iso(), now_iso(), delivery_id))
        self._audit.append(actor, "alert.retry", "incident", d["incident_id"], {"delivery_id": delivery_id}, actor_role=role)
        self._wake.set()
        return self._db.one("SELECT * FROM alert_deliveries WHERE delivery_id = ?", (delivery_id,))

    def _refresh_incident_status(self, incident_id: str) -> None:
        rows = self._db.query("SELECT status, delivery_confirmed, is_test FROM alert_deliveries WHERE incident_id = ?", (incident_id,))
        if not rows:
            return
        if any(r["delivery_confirmed"] and not r["is_test"] for r in rows):
            status = "delivered_confirmed"
        elif any(r["status"] == "pending" for r in rows):
            status = "pending"
        elif any(r["status"] == "sent" and r["is_test"] for r in rows):
            status = "test_recorded_only"
        elif any(r["status"] in ("abandoned",) for r in rows):
            status = "failed"
        else:
            status = "none"
        self._incidents.set_alert_status(incident_id, status)

    def list(self, incident_id: str = None, status: str = None, limit: int = 100) -> list:
        where, params = [], []
        if incident_id:
            where.append("incident_id = ?")
            params.append(incident_id)
        if status:
            where.append("status = ?")
            params.append(status)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self._db.query(f"SELECT * FROM alert_deliveries {clause} ORDER BY created_at DESC LIMIT ?", params + [limit])
        for r in rows:
            r["payload"] = loads(r.pop("payload_json"), {})
            r["notice"] = ("TEST ADAPTER: not delivered to any external party" if r["is_test"]
                           else ("delivery confirmed by the receiving endpoint" if r["delivery_confirmed"] else "not confirmed"))
        return rows

    # -- worker ----------------------------------------------------------------

    def start_worker(self, interval_s: float = 1.0) -> None:
        if self._thread is not None:
            return

        def loop():
            while not self._stop.is_set():
                try:
                    self.process_due()
                except Exception:  # noqa: BLE001
                    log.exception("alert worker error")
                self._wake.wait(interval_s)
                self._wake.clear()

        self._thread = threading.Thread(target=loop, name="anvesh-alerts", daemon=True)
        self._thread.start()

    def stop_worker(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(5)
        self._thread = None
