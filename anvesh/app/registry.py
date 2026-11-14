"""Vehicle-registry lookup behind an adapter, with safe failure behaviour.

ANVESH HAS NO GOVERNMENT-DATABASE ACCESS. Nothing here scrapes a public website or bypasses
access controls. The pieces are:

  SandboxRegistryAdapter    SYNTHETIC fixture records from `configs/sandbox_registry.json`, labelled as
                            test data in every response. Only demo/simulated cases may use it (the schema
                            also rejects a sandbox lookup on an operational case).
  ProductionRegistryAdapter a generic HTTPS connector for an AUTHORISED registry GATEWAY the department
                            provides. It is "not connected" unless ANVESH_REGISTRY_URL (+ key) is set, and
                            it is UNTESTED against any real registry: the request/response schema below is
                            ANVESH's own connector contract, not NIC/Vahan's API. A real integration means
                            writing a gateway (or adapter) to the department's actual, approved interface.

CONTRACT (production adapter)
  POST <ANVESH_REGISTRY_URL>   Authorization: Bearer <ANVESH_REGISTRY_API_KEY>
  request : {"plate": "MH12AB1234", "purpose": "traffic_enforcement_case", "case_id": "CASE-...", "requested_by": "<username>"}
  200     : {"found": true, "registration_status": "...", "vehicle_class": "...", "make_model": "...", "colour": "...",
             "owner_name": "...", "owner_address": "...", "owner_contact": "...", "source": "<authority name>", "as_of": "<ISO time>"}
  200     : {"found": false}                       -> status not_found
  401/403 : access denied                           -> status denied
  other / timeout / malformed -> status error (never a guessed result)

OCR confidence is NOT registry verification. A lookup is `found` only if the adapter says so, and the
row records the source, retrieval time and requester. Lookups require an officer-reviewed plate
reading (never a raw OCR candidate), and every lookup is audited. Owner fields (name/address/contact)
are shown only to roles with `registry.view_owner`.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.plates import validate_format
from anvesh.app.timeutil import now_iso

VEHICLE_FIELDS = ("registration_status", "vehicle_class", "make_model", "colour", "fuel", "insurance_valid_until")
OWNER_FIELDS = ("owner_name", "owner_address", "owner_contact")


@dataclasses.dataclass(frozen=True)
class RegistryResult:
    status: str  # found | not_found | error | not_connected | denied | invalid_request
    data: dict = dataclasses.field(default_factory=dict)
    error: str = None
    source_label: str = ""


class SandboxRegistryAdapter:
    kind = "sandbox"
    name = "sandbox_fixture"
    label = "SANDBOX - synthetic test records, not a real registry"

    def __init__(self, path) -> None:
        self._records = {}
        p = Path(path)
        if p.is_file():
            self._records = json.loads(p.read_text(encoding="utf-8")).get("records", {})

    def connected(self) -> bool:
        return True

    def lookup(self, plate: str, case_id: str, requested_by: str) -> RegistryResult:
        rec = self._records.get(plate)
        if rec is None:
            return RegistryResult("not_found", source_label=self.label)
        return RegistryResult("found", dict(rec, synthetic=True, notice="SYNTHETIC TEST RECORD"), source_label=self.label)


class ProductionRegistryAdapter:
    kind = "production"
    name = "authorised_registry_gateway"
    label = "Authorised registry gateway"

    def __init__(self, base_url: str = None, api_key: str = None, timeout_s: float = 10.0) -> None:
        self._url, self._key, self._timeout = base_url, api_key, timeout_s

    def connected(self) -> bool:
        return bool(self._url and self._key)

    def lookup(self, plate: str, case_id: str, requested_by: str) -> RegistryResult:
        if not self.connected():
            return RegistryResult("not_connected", error="no authorised registry is configured on this server", source_label=self.label)
        body = json.dumps({"plate": plate, "purpose": "traffic_enforcement_case", "case_id": case_id, "requested_by": requested_by}).encode()
        req = urllib.request.Request(self._url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 - operator-configured gateway
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return RegistryResult("denied", error=f"HTTP {exc.code}: access not permitted", source_label=self.label)
            return RegistryResult("error", error=f"HTTP {exc.code}", source_label=self.label)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            return RegistryResult("error", error=f"{type(exc).__name__}: {exc}", source_label=self.label)
        if not isinstance(payload, dict) or "found" not in payload:
            return RegistryResult("error", error="malformed response from registry gateway", source_label=self.label)
        if not payload["found"]:
            return RegistryResult("not_found", source_label=payload.get("source") or self.label)
        keep = {k: payload.get(k) for k in VEHICLE_FIELDS + OWNER_FIELDS + ("as_of",)}
        return RegistryResult("found", keep, source_label=payload.get("source") or self.label)


class RegistryService:
    def __init__(self, db: AppDatabase, audit: AuditLog, sandbox, production) -> None:
        self._db, self._audit = db, audit
        self.sandbox, self.production = sandbox, production

    def status(self) -> list:
        return [
            {"key": "production", "label": self.production.label, "state": "connected" if self.production.connected() else "not_connected",
             "detail": "authorised gateway configured (untested against a real registry)" if self.production.connected()
             else "No authorised registry access is configured. ANVESH holds no government-database access.",
             "setup": "Obtain departmental authorisation, then set ANVESH_REGISTRY_URL and ANVESH_REGISTRY_API_KEY for a gateway "
                      "implementing the connector contract in anvesh/app/registry.py."},
            {"key": "sandbox", "label": self.sandbox.label, "state": "sandbox",
             "detail": "synthetic fixture records; demo/simulated cases only", "setup": ""},
        ]

    def lookup(self, case: dict, reading: dict, user: str, role: str) -> dict:
        if reading is None or not (reading["verified"] or reading["status"] == "officer_entered"):
            raise ValueError("a registry lookup needs an officer-reviewed plate reading, not an unreviewed OCR candidate")
        operational = case["data_class"] == "operational"
        if operational and (not reading["verified"] or reading["reading_id"] != case.get("plate_verified_reading_id")):
            # Data minimisation and policy in one: an operational case queries the registry only for the plate that a second person
            # has verified and that is still the case's current verified reading. Anything else could never support an approval.
            raise ValueError("an operational registry lookup needs the case's verified plate reading (verify the plate first; a corrected plate must be re-verified)")
        plate = reading["normalized_text"]
        adapter = self.production if operational else self.sandbox
        fmt = validate_format(plate)
        if not fmt["valid"]:
            return self._record(case, plate, adapter, RegistryResult("invalid_request", error=f"plate format invalid: {fmt['reason']}"), user, role,
                                reading["reading_id"])
        try:
            result = adapter.lookup(plate, case["case_id"], user)
        except Exception as exc:  # noqa: BLE001 - never let an adapter bug break the workflow
            result = RegistryResult("error", error=f"adapter crashed: {type(exc).__name__}: {exc}", source_label=adapter.label)
        return self._record(case, plate, adapter, result, user, role, reading["reading_id"])

    def _record(self, case, plate, adapter, result, user, role=None, reading_id=None) -> dict:
        lookup_id = "REG-" + uuid.uuid4().hex[:10].upper()
        sandbox = adapter.kind == "sandbox"
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO registry_lookups(lookup_id, case_id, plate, adapter, adapter_kind, status, is_simulated, data_json, error, source_label, "
                "requested_by, retrieved_at, reading_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (lookup_id, case["case_id"], plate, adapter.name, adapter.kind, result.status, 1 if sandbox else 0, dumps(result.data),
                 result.error, result.source_label or adapter.label, user, now_iso(), reading_id))
        self._audit.append(user, "registry.lookup", "case", case["case_id"],
                           {"lookup_id": lookup_id, "plate": plate, "adapter": adapter.kind, "status": result.status, "error": result.error,
                            "reading_id": reading_id},
                           actor_role=role)
        return self.get(lookup_id)

    def get(self, lookup_id: str, user=None):
        row = self._db.one("SELECT * FROM registry_lookups WHERE lookup_id = ?", (lookup_id,))
        return self._present(row, user) if row else None

    def for_case(self, case_id: str, user=None) -> list:
        return [self._present(r, user) for r in
                self._db.query("SELECT * FROM registry_lookups WHERE case_id = ? ORDER BY retrieved_at DESC", (case_id,))]

    def _present(self, row: dict, user) -> dict:
        try:
            data = loads(row.pop("data_json"), {})
        except ValueError:
            data = None
        row["data_malformed"] = not isinstance(data, dict)  # a damaged row must fail closed, never crash the caller
        data = data if isinstance(data, dict) else {}
        permitted = user is not None and user.can("registry.view_owner")
        row["data"] = {k: v for k, v in data.items() if k not in OWNER_FIELDS or permitted}
        row["owner_fields_hidden"] = bool(data) and not permitted and any(k in data for k in OWNER_FIELDS)
        row["is_simulated"] = bool(row["is_simulated"])
        row["registry_verified"] = row["status"] == "found"
        row["notice"] = ("SANDBOX: synthetic test data, not a real registry result" if row["is_simulated"]
                         else ("Retrieved from an authorised registry gateway" if row["status"] == "found" else ""))
        return row
