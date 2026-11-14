"""Case hand-off: how an APPROVED case leaves ANVESH. Three adapters, none of which issues a real challan
on its own:

  export    -- ExportPackageAdapter: builds an evidence package (zip + manifest) for MANUAL submission by the
               officer through whatever approved channel the department uses. Status `exported`; the case goes
               to `submitted` ("exported, awaiting external reference") and an officer later records the
               reference the official system gave them (`record_reference`, kind `officer_entered`).
  sandbox   -- SandboxChallanAdapter: SIMULATES a submission and returns a `TEST-CHALLAN-...` reference. Allowed
               only for demo/simulated cases; the reference is labelled non-production everywhere. The schema
               refuses to attach it to an operational case.
  official  -- OfficialChallanAdapter: NOT CONNECTED. The e-Challan system is operated under the Ministry of
               Road Transport & Highways and built by NIC, integrated with the Vahan and Sarathi databases; public
               API documentation was not found, and access is by departmental arrangement. ANVESH therefore
               contains NO endpoint for it. If ANVESH_CHALLAN_URL/KEY are set, this adapter posts ANVESH's own
               gateway contract (below) to that URL, UNTESTED against any real environment; a real integration
               means building a gateway to the department's approved interface.

GATEWAY CONTRACT (official adapter, ANVESH-defined)
  POST <ANVESH_CHALLAN_URL>  Authorization: Bearer <key>  Idempotency-Key: <key>
  body: {"case_id", "case_version", "offence_code", "rule_reference", "plate", "event_time_utc", "camera_id", "evidence_manifest": {...}}
  2xx: {"reference": "<external challan reference>"}      anything else = failure (never a guessed success)

IDEMPOTENCY. The key is sha256(case_id | case_version | adapter kind). A succeeded/exported/pending handoff with
the same key is returned instead of submitting again (a unique partial index enforces it in the database too), so
a double-click or a retried request cannot double-submit. A failed handoff does not block a retry.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.error
import urllib.request
import uuid

from anvesh.app.audit import AuditLog
from anvesh.app.cases import CaseError
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.reports import build_case_package
from anvesh.app.timeutil import now_iso


@dataclasses.dataclass(frozen=True)
class HandoffResult:
    status: str  # succeeded | failed | exported
    external_reference: str = None
    response: dict = dataclasses.field(default_factory=dict)
    error: str = None
    artifact_relpath: str = None


class ExportPackageAdapter:
    kind = "export"
    name = "evidence_package_export"
    label = "Evidence package export (manual submission by the officer)"

    def __init__(self, services) -> None:
        self._s = services

    def connected(self) -> bool:
        return True

    def submit(self, case: dict, user, idempotency_key: str) -> HandoffResult:
        path = build_case_package(self._s, case["case_id"], user, self._s.config.exports_dir)
        return HandoffResult("exported", response={"package": path.name, "note": "Submit this package through the department's approved channel, "
                                                                              "then record the reference the official system returns."},
                             artifact_relpath=path.name)


class SandboxChallanAdapter:
    kind = "sandbox"
    name = "sandbox_challan_simulator"
    label = "SANDBOX challan simulator - NOT a real submission"

    def __init__(self, fail_next: int = 0) -> None:
        self.fail_next = fail_next  # test hook: make the next N submissions fail

    def connected(self) -> bool:
        return True

    def submit(self, case: dict, user, idempotency_key: str) -> HandoffResult:
        if self.fail_next > 0:
            self.fail_next -= 1
            return HandoffResult("failed", error="simulated sandbox failure")
        ref = "TEST-CHALLAN-" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:10].upper()
        return HandoffResult("succeeded", external_reference=ref,
                             response={"simulated": True, "notice": "TEST reference from the sandbox simulator; no real challan exists"})


class OfficialChallanAdapter:
    kind = "production"
    name = "official_challan_gateway"
    label = "Official e-challan gateway"

    def __init__(self, base_url: str = None, api_key: str = None, timeout_s: float = 15.0) -> None:
        self._url, self._key, self._timeout = base_url, api_key, timeout_s

    def connected(self) -> bool:
        return bool(self._url and self._key)

    def submit(self, case: dict, user, idempotency_key: str) -> HandoffResult:
        if not self.connected():
            return HandoffResult("failed", error="no official e-challan integration is configured (not connected)")
        plate = (case.get("current_plate") or {}).get("normalized_text")
        body = json.dumps({"case_id": case["case_id"], "case_version": case["version"], "offence_code": case["offence_code"],
                           "rule_reference": case["rule_reference"], "plate": plate, "event_time_utc": case["incident"]["start_utc"],
                           "camera_id": case["incident"]["camera_id"]}).encode()
        req = urllib.request.Request(self._url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self._key}", "Idempotency-Key": idempotency_key})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 - operator-configured gateway
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return HandoffResult("failed", error=f"HTTP {exc.code}")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            return HandoffResult("failed", error=f"{type(exc).__name__}: {exc}")
        ref = payload.get("reference") if isinstance(payload, dict) else None
        if not ref:
            return HandoffResult("failed", error="gateway response had no reference")
        return HandoffResult("succeeded", external_reference=str(ref), response={"gateway": "official"})


class HandoffService:
    def __init__(self, db: AppDatabase, audit: AuditLog, cases, adapters: dict) -> None:
        self._db, self._audit, self._cases, self.adapters = db, audit, cases, adapters

    def status(self) -> list:
        notes = {
            "export": "available: builds an evidence package for manual submission",
            "sandbox": "simulator for demo/simulated cases only; returns TEST references",
            "official": "NOT CONNECTED: no official e-challan API is configured or tested. Access is by departmental arrangement (NIC/MoRTH).",
        }
        return [{"key": k, "label": a.label, "state": "connected" if (a.connected() and k == "official") else ("sandbox" if k == "sandbox" else ("available" if a.connected() else "not_connected")),
                 "detail": notes[k]} for k, a in self.adapters.items()]

    @staticmethod
    def idempotency_key(case_id: str, version: int, kind: str) -> str:
        return hashlib.sha256(f"{case_id}|{version}|{kind}".encode()).hexdigest()

    def submit(self, case_id: str, adapter_key: str, user) -> dict:
        if not user.can("case.submit"):
            raise CaseError("not permitted to submit cases", 403)
        adapter = self.adapters.get(adapter_key)
        if adapter is None:
            raise CaseError(f"unknown handoff adapter {adapter_key!r}", 422)
        case = self._cases.get(case_id, user)
        if case["state"] != "approved_for_submission":
            raise CaseError(f"only an approved case can be submitted (state: {case['state']})", 409)
        operational = case["data_class"] == "operational"
        if operational and getattr(user, "is_demo", False):
            raise CaseError("demo accounts cannot submit operational cases", 403)
        if adapter.kind == "sandbox" and operational:
            raise CaseError("the sandbox simulator cannot be used for an operational case", 409)
        if adapter.kind == "production" and not operational:
            raise CaseError("the official gateway is not used for demo/simulated cases", 409)
        if adapter.kind == "production" and not adapter.connected():
            raise CaseError("the official e-challan integration is not connected on this server", 409)
        if operational:  # re-check at the moment of submission: the state alone is not proof (a result can expire, a plate can change)
            blocked = [f"{c['item']}: {c['detail']}" for c in self._cases.readiness(case) if not c["ok"] and c["item"] in ("plate_verified", "registry")]
            if blocked:
                self._audit.append(user.username, "handoff.blocked", "case", case_id, {"reasons": blocked[:5]}, actor_role=user.role)
                raise CaseError("an operational case cannot be submitted without a current authorised registry result", 409, blocked)
        key = self.idempotency_key(case_id, case["version"], adapter.kind)
        existing = self._db.one("SELECT * FROM handoffs WHERE idempotency_key = ? AND status IN ('succeeded','exported','pending')", (key,))
        if existing is not None:
            return dict(self._present(existing), duplicate=True)
        handoff_id = "HND-" + uuid.uuid4().hex[:10].upper()
        try:
            with self._db.write() as conn:
                conn.execute(
                    "INSERT INTO handoffs(handoff_id, case_id, adapter, adapter_kind, idempotency_key, case_version, status, is_simulated, submitted_by, created_at) "
                    "VALUES (?,?,?,?,?,?,'pending',?,?,?)",
                    (handoff_id, case_id, adapter.name, adapter.kind, key, case["version"], 1 if adapter.kind == "sandbox" else 0, user.username, now_iso()))
        except Exception as exc:  # noqa: BLE001
            if "UNIQUE" in str(exc):  # lost a race to an identical concurrent submission
                existing = self._db.one("SELECT * FROM handoffs WHERE idempotency_key = ? AND status IN ('succeeded','exported','pending')", (key,))
                return dict(self._present(existing), duplicate=True)
            raise CaseError(str(exc), 409) from None
        self._audit.append(user.username, "handoff.start", "case", case_id, {"handoff_id": handoff_id, "adapter": adapter.kind}, actor_role=user.role)
        try:
            result = adapter.submit(case, user, key)
        except Exception as exc:  # noqa: BLE001 - an adapter crash is a failed submission, recorded as such
            result = HandoffResult("failed", error=f"adapter crashed: {type(exc).__name__}: {exc}")
        with self._db.write() as conn:
            conn.execute("UPDATE handoffs SET status = ?, external_reference = ?, response_json = ?, error = ?, artifact_relpath = ?, completed_at = ? "
                         "WHERE handoff_id = ?", (result.status, result.external_reference, dumps(result.response), result.error,
                                                  result.artifact_relpath, now_iso(), handoff_id))
        if result.status == "failed":
            self._cases.apply_handoff_result(case_id, "submission_failed", user.username, user.role, f"{adapter.kind}: {result.error}")
        elif result.status == "exported":
            self._cases.apply_handoff_result(case_id, "submitted", user.username, user.role, "evidence package exported for manual submission")
        else:
            kind = "sandbox_simulated" if adapter.kind == "sandbox" else "official_gateway"
            self._cases.apply_handoff_result(case_id, "submitted", user.username, user.role, f"submitted via {adapter.kind}")
            self._cases.apply_handoff_result(case_id, "challan_reference_received", user.username, user.role,
                                             f"{adapter.kind}: reference {result.external_reference}",
                                             {"external_reference": result.external_reference, "external_reference_kind": kind})
        self._audit.append(user.username, "handoff.result", "case", case_id,
                           {"handoff_id": handoff_id, "adapter": adapter.kind, "status": result.status, "reference": result.external_reference,
                            "error": result.error}, actor_role=user.role)
        return dict(self._present(self._db.one("SELECT * FROM handoffs WHERE handoff_id = ?", (handoff_id,))), duplicate=False)

    @staticmethod
    def _present(row: dict) -> dict:
        row = dict(row)
        row["response"] = loads(row.pop("response_json"), {})
        row["is_simulated"] = bool(row["is_simulated"])
        row["notice"] = "SANDBOX / TEST: not a real submission" if row["is_simulated"] else ""
        return row
