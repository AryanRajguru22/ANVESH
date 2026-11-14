"""Officer-led violation cases and the review state machine.

ANVESH never decides enforcement. A case is a work item built on a candidate incident; it advances
only through explicit, permissioned, audited actions by named users, and only when its preconditions
hold. Nothing here issues a challan.

    detected
      -> awaiting_evidence_verification   (investigator submits)
      -> awaiting_plate_verification      (reviewer verifies the evidence: frames + clip present, hashes intact)
      -> ready_for_review                 (reviewer verifies the plate: an officer-reviewed reading)
      -> approved_for_submission          (OFFICER approves; four-eyes; every mandatory field present)
      -> submitted                        (handoff: sandbox / manual export / official gateway)
      -> challan_reference_received       (sandbox reference, or an officer-entered external reference)
      -> closed
    side exits: rejected, needs_further_review (back to awaiting_evidence_verification),
                submission_failed (retry -> approved_for_submission)

FOUR-EYES (`require_distinct_approver`, default on): the approving officer must differ from the user who
verified the evidence, the user who verified the plate, and the case creator.

OPERATIONAL vs DEMO. A case inherits the data class of its incident and can never change it. For an
`operational` case approval additionally needs a configured rule reference for the offence, an event
time that is not just the processing clock, and a registry result from the AUTHORISED registry (the
sandbox cannot attach to an operational case at all).

REGISTRY INVARIANT (project-owner decision 2026-10-21, no override exists): a case cannot be approved - and an approved case cannot be
retried or handed off - unless its CURRENT plate reading is verified by someone other than whoever entered it AND a registry lookup
bound to exactly that reading says `found`. For an operational case that lookup must come from the configured authorised (production)
adapter: not a sandbox/synthetic record, not a manual entry, not OCR confidence, not an officer's note. The lookup must be fresh
(`registry_result_max_age_s`), well-formed, for the same plate text, and confirmed by the append-only audit log (so a hand-inserted row
does not count). The same rule is enforced a third time by a database trigger. Demo/simulated cases follow the same shape but accept the
labelled sandbox result. There is NO registry override.
"""

from __future__ import annotations

import uuid

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.plates import VERIFIED_STATUSES
from anvesh.app.timeutil import now_iso

DEFAULT_OFFENCES = [
    {"code": "STOP-OBS", "description": "Vehicle stopped/standing causing obstruction (candidate)", "event_types": ["stopped_vehicle", "long_duration_stop"]},
    {"code": "RLV", "description": "Red-light / signal violation", "event_types": ["red_light_violation"]},
    {"code": "WRONG-WAY", "description": "Driving against the direction of traffic", "event_types": ["wrong_way"]},
    {"code": "WRONG-LANE", "description": "Lane discipline: vehicle class in a lane that excludes it", "event_types": ["wrong_lane"]},
    {"code": "SPEED", "description": "Exceeding the speed limit", "event_types": ["overspeed"]},
    {"code": "ILLEGAL-TURN", "description": "Prohibited turn / movement", "event_types": ["illegal_turn"]},
    {"code": "DANGEROUS-DRIVING", "description": "Dangerous / rash driving (candidate)", "event_types": ["rash_driving", "sudden_braking"]},
    {"code": "HELMET", "description": "Rider without protective headgear", "event_types": ["helmet_violation"]},
    {"code": "SEATBELT", "description": "Seatbelt not worn", "event_types": ["seatbelt_violation"]},
]
for _o in DEFAULT_OFFENCES:  # rule references are the DEPARTMENT's to configure; ANVESH ships none
    _o.setdefault("rule_reference", "")

CASE_STATES = ("detected", "awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review", "approved_for_submission",
               "rejected", "needs_further_review", "submitted", "submission_failed", "challan_reference_received", "closed")

_PRE_APPROVAL = {"detected", "awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review", "needs_further_review"}
# action -> (allowed source states, target state, permission)
TRANSITIONS = {
    "submit_for_evidence_verification": ({"detected"}, "awaiting_evidence_verification", "case.create"),
    "verify_evidence": ({"awaiting_evidence_verification"}, "awaiting_plate_verification", "case.verify_evidence"),
    "verify_plate": ({"awaiting_plate_verification"}, "ready_for_review", "case.verify_plate"),
    "approve": ({"ready_for_review"}, "approved_for_submission", "case.approve"),
    "reject": (_PRE_APPROVAL | {"approved_for_submission"}, "rejected", "case.review"),
    "request_more": ({"awaiting_evidence_verification", "awaiting_plate_verification", "ready_for_review", "approved_for_submission"},
                     "needs_further_review", "case.review"),
    "resume_review": ({"needs_further_review"}, "awaiting_evidence_verification", "case.create"),
    "retry_submission": ({"submission_failed"}, "approved_for_submission", "case.submit"),
    "record_reference": ({"submitted"}, "challan_reference_received", "case.submit"),
    "close": ({"rejected", "challan_reference_received"}, "closed", "case.close"),
}
NOTE_REQUIRED = {"reject", "request_more", "close", "record_reference"}


def _refuse_demo_on_operational(user, data_class: str) -> None:
    """Demo accounts (created by `run_app.py --demo-users`, credentials in a labelled file) are never operational credentials."""
    if data_class == "operational" and getattr(user, "is_demo", False):
        raise CaseError("demo accounts cannot create or act on operational cases", 403)


class CaseError(Exception):
    """A workflow rule was violated. `status` is the HTTP status the API should return."""

    def __init__(self, message: str, status: int = 409, errors: list = None) -> None:
        super().__init__(message)
        self.status, self.errors = status, errors or []


class CaseService:
    def __init__(self, db: AppDatabase, audit: AuditLog, incidents, evidence, plates, registry, settings, require_distinct: bool = True,
                 registry_max_age_s: int = 24 * 3600) -> None:
        self._registry_max_age_s = registry_max_age_s
        self._db, self._audit = db, audit
        self._incidents, self._evidence, self._plates, self._registry, self._settings = incidents, evidence, plates, registry, settings
        self._distinct = require_distinct

    # -- catalogue -----------------------------------------------------------------

    def offences(self) -> list:
        return self._settings.get("offence_catalog", DEFAULT_OFFENCES)

    def offence(self, code: str):
        return next((o for o in self.offences() if o["code"] == code), None)

    def set_offences(self, offences: list, actor: str, role: str = None) -> list:
        clean = []
        for o in offences:
            code = (o.get("code") or "").strip()
            if not code or not (o.get("description") or "").strip():
                raise ValueError("every offence needs a code and a description")
            clean.append({"code": code, "description": o["description"].strip(), "event_types": list(o.get("event_types") or []),
                          "rule_reference": (o.get("rule_reference") or "").strip()})
        if len({o["code"] for o in clean}) != len(clean):
            raise ValueError("duplicate offence code")
        self._settings.set("offence_catalog", clean, actor)
        self._audit.append(actor, "settings.offences_update", "settings", "offence_catalog", {"offences": len(clean)}, actor_role=role)
        return clean

    # -- creation / editing -----------------------------------------------------------

    def create(self, incident_id: str, offence_code: str, user, track_id: str = None, reason: str = None) -> dict:
        if not user.can("case.create"):
            raise CaseError("not permitted to create cases", 403)
        inc = self._incidents.get(incident_id, with_timeline=True)
        if inc is None:
            raise CaseError(f"no such incident {incident_id!r}", 404)
        _refuse_demo_on_operational(user, inc["data_class"])
        if not isinstance(offence_code, str) or (reason is not None and not isinstance(reason, str)) \
                or (track_id is not None and not isinstance(track_id, str)):
            raise CaseError("offence_code, reason and track_id must be strings", 422)
        offence = self.offence(offence_code)
        if offence is None:
            raise CaseError(f"unknown offence code {offence_code!r}", 422)
        if inc["event_type"] not in offence["event_types"]:
            raise CaseError(f"offence {offence_code} does not apply to {inc['event_type']} incidents (applies to: {offence['event_types']})", 422)
        if inc["status"] == "dismissed":
            raise CaseError("a dismissed incident cannot seed a case; reopen it first", 409)
        involved = [v["track_id"] for v in inc["involved"]]
        track_id = track_id or (involved[0] if involved else None)
        if track_id is not None and involved and track_id not in involved:
            raise CaseError("track_id is not one of the incident's involved vehicles", 422)
        case_id = f"CASE-{now_iso()[:10].replace('-', '')}-{uuid.uuid4().hex[:6].upper()}"
        summary = next((t["summary"] for t in inc["timeline"] if t["kind"] == "detected"), inc["event_type"])
        reason = (reason or "").strip() or f"Candidate {inc['event_type'].replace('_', ' ')}: {summary}. Detector: {inc['detector']}; confidence tier {inc['confidence_tier']} ({inc['confidence_basis']})."
        limitations = "\n".join(inc["uncertainty"])
        now = now_iso()
        try:
            with self._db.write() as conn:
                conn.execute(
                    "INSERT INTO cases(case_id, incident_id, data_class, state, offence_code, offence_description, rule_reference, reason, limitations, "
                    "track_id, created_by, created_at, updated_at) VALUES (?,?,?,'detected',?,?,?,?,?,?,?,?,?)",
                    (case_id, incident_id, inc["data_class"], offence["code"], offence["description"], offence.get("rule_reference") or None,
                     reason, limitations, track_id, user.username, now, now))
                conn.execute("INSERT INTO case_transitions(case_id, from_state, to_state, action, actor, actor_role, note, ts) VALUES (?,?,?,?,?,?,?,?)",
                             (case_id, None, "detected", "create", user.username, user.role, "", now))
        except Exception as exc:  # noqa: BLE001
            if "UNIQUE" in str(exc):
                raise CaseError(f"a case for this incident and offence ({offence_code}) already exists", 409) from None
            raise
        self._audit.append(user.username, "case.create", "case", case_id, {"incident_id": incident_id, "offence": offence["code"],
                                                                         "data_class": inc["data_class"]}, actor_role=user.role)
        self._incidents.add_timeline(incident_id, "case", f"case {case_id} opened ({offence['code']})", user.username)
        return self.get(case_id, user)

    def update_fields(self, case_id: str, user, offence_code: str = None, reason: str = None, limitations: str = None) -> dict:
        case = self._row(case_id)
        if not (user.can("case.create") or user.can("case.review")):
            raise CaseError("not permitted to edit cases", 403)
        if case["state"] not in _PRE_APPROVAL:
            raise CaseError(f"a case in state {case['state']} can no longer be edited", 409)
        sets, params, changed = [], [], {}
        if offence_code is not None and offence_code != case["offence_code"]:
            off = self.offence(offence_code)
            inc = self._incidents.get(case["incident_id"])
            if off is None or inc["event_type"] not in off["event_types"]:
                raise CaseError("offence does not apply to this incident", 422)
            sets += ["offence_code = ?", "offence_description = ?", "rule_reference = ?"]
            params += [off["code"], off["description"], off.get("rule_reference") or None]
            changed["offence_code"] = {"from": case["offence_code"], "to": off["code"]}
        if reason is not None:
            if not reason.strip():
                raise CaseError("reason must not be empty", 422)
            sets.append("reason = ?")
            params.append(reason.strip())
            changed["reason"] = "edited"
        if limitations is not None:
            sets.append("limitations = ?")
            params.append(limitations.strip())
            changed["limitations"] = "edited"
        if not sets:
            return self.get(case_id, user)
        sets += ["version = version + 1", "updated_at = ?"]
        params.append(now_iso())
        with self._db.write() as conn:
            conn.execute(f"UPDATE cases SET {', '.join(sets)} WHERE case_id = ?", params + [case_id])
        self._audit.append(user.username, "case.edit", "case", case_id, changed, actor_role=user.role)
        return self.get(case_id, user)

    # -- readiness ------------------------------------------------------------------

    def _row(self, case_id: str) -> dict:
        row = self._db.one("SELECT * FROM cases WHERE case_id = ?", (case_id,))
        if row is None:
            raise CaseError(f"no such case {case_id!r}", 404)
        return row

    def evidence_check(self, incident_id: str) -> dict:
        assets = self._evidence.list(incident_id)
        frames = [a for a in assets if a["kind"] == "frame"]
        clips = [a for a in assets if a["kind"] == "clip"]
        problems = []
        if not frames:
            problems.append("no key frame was captured")
        if not clips:
            problems.append("no clip was captured")
        for a in frames + clips:
            r = self._evidence.verify(a["asset_id"])
            if not r["ok"]:
                problems.append(f"{a['kind']} {a['asset_id']} failed integrity check: {r['reason']}")
        return {"ok": not problems, "problems": problems, "frames": len(frames), "clips": len(clips)}

    def plate_for(self, case: dict):
        heads = self._plates.current(case["incident_id"], case["track_id"]) or self._plates.current(case["incident_id"])
        verified = [h for h in heads if h["verified"]]
        return (verified or heads or [None])[0]

    def registry_support(self, case: dict, plate: dict, now: str = None) -> dict:
        """{ok, detail, lookup_id}: does an acceptable registry result back THIS case's CURRENT verified plate reading?
        Fails closed; `detail` says why in terms that carry no personal data."""
        from anvesh.app.registry import VEHICLE_FIELDS
        from anvesh.app.timeutil import add_seconds, parse_iso

        def no(detail):
            return {"ok": False, "detail": detail, "lookup_id": None}

        operational = case["data_class"] == "operational"
        verified_id = case.get("plate_verified_reading_id")
        if plate is None or not plate["verified"] or not verified_id or plate["reading_id"] != verified_id:
            return no("the plate is not verified, or was changed after verification; a registry result can only support the current verified reading")
        production = self._registry.production
        reasons, now = [], now or now_iso()
        confirmed = {a["details"].get("lookup_id") for a in self._audit.query(entity_type="case", entity_id=case["case_id"], action="registry.lookup", limit=500)[0]
                     if a["details"].get("status") == "found"}
        for l in self._registry.for_case(case["case_id"]):
            if l["status"] != "found":
                reasons.append(f"a lookup returned '{l['status']}'")
                continue
            if l.get("reading_id") != plate["reading_id"] or l["plate"] != plate["normalized_text"]:
                reasons.append("a registry result belongs to a different plate reading")
                continue
            if l["adapter_kind"] == "sandbox" or l["is_simulated"]:
                if operational:
                    reasons.append("a sandbox/synthetic registry result cannot support an operational case")
                    continue
                if l["adapter_kind"] != "sandbox" or not l["is_simulated"]:
                    reasons.append("a registry result is internally inconsistent")
                    continue
            elif l["adapter_kind"] != "production" or l["adapter"] != production.name or not production.connected():
                reasons.append("a registry result did not come from the configured authorised registry")
                continue
            data = l.get("data")
            if (l.get("data_malformed") or not isinstance(data, dict) or not any(data.get(k) not in (None, "") for k in VEHICLE_FIELDS)
                    or not (l.get("source_label") or "").strip()):
                reasons.append("a registry result is malformed or empty")
                continue
            try:
                retrieved = parse_iso(l["retrieved_at"])
                too_old = retrieved < parse_iso(add_seconds(now, -self._registry_max_age_s))
                from_future = retrieved > parse_iso(add_seconds(now, 120))
            except (ValueError, TypeError):
                reasons.append("a registry result has no valid retrieval time")
                continue
            if too_old or from_future:
                reasons.append("a registry result is stale (older than the allowed age)" if too_old else "a registry result has a retrieval time in the future")
                continue
            if l["lookup_id"] not in confirmed:
                reasons.append("a registry result is not confirmed by the audit log")
                continue
            return {"ok": True, "detail": f"{'authorised registry' if l['adapter_kind'] == 'production' else 'SANDBOX (test data)'} result {l['lookup_id']} "
                                          f"found a record for the verified reading", "lookup_id": l["lookup_id"]}
        if not reasons:
            if operational and not production.connected():
                return no("no authorised registry is configured on this server, so an operational case cannot be approved")
            return no("no registry lookup has been made for the verified plate reading")
        return no(reasons[0] if len(set(reasons)) == 1 else "; ".join(sorted(set(reasons))[:3]))

    def readiness(self, case: dict) -> list:
        """The approval checklist: [{item, ok, detail}]. Used for the UI and enforced by `approve`."""
        inc = self._incidents.get(case["incident_id"])
        plate = self.plate_for(case)
        support = self.registry_support(case, plate)
        operational = case["data_class"] == "operational"
        offence = self.offence(case["offence_code"]) if case["offence_code"] else None
        ev = self.evidence_check(case["incident_id"])
        # The verification must cover the reading that is current NOW. If the plate was corrected after plate_verified_by
        # signed off, that person never saw the new text. (Cases verified before schema v2 were backfilled by the migration.)
        plate_ok = bool(plate and plate["verified"] and case["plate_verified_by"]
                        and case.get("plate_verified_reading_id") == plate["reading_id"])
        items = [
            ("offence", bool(offence), "offence selected" if offence else "no offence selected"),
            ("reason", bool((case["reason"] or "").strip()), "reason for classification recorded" if case["reason"] else "reason missing"),
            ("evidence_captured", ev["ok"], f"{ev['frames']} frame(s), {ev['clips']} clip(s), hashes intact" if ev["ok"] else "; ".join(ev["problems"])),
            ("evidence_verified", bool(case["evidence_verified_by"]),
             f"verified by {case['evidence_verified_by']}" if case["evidence_verified_by"] else "evidence not yet verified by a reviewer"),
            ("plate_verified", plate_ok,
             (f"{plate['normalized_text']} verified by {case['plate_verified_by']}" if plate_ok
              else ("the plate was changed after it was verified (now " + plate["normalized_text"] + "); request more review and re-verify it"
                    if plate and plate["verified"] and case["plate_verified_by"] else
                    ("plate not yet verified" if plate else "no plate reading")))),
            ("registry", support["ok"], support["detail"]),
        ]
        if operational:
            items.append(("rule_reference", bool(case["rule_reference"]), "rule reference configured" if case["rule_reference"]
                          else "no rule reference configured for this offence in the catalogue (department to configure)"))
            items.append(("event_time", inc["time_basis"] != "processing_clock", "event time from a declared/live clock" if inc["time_basis"] != "processing_clock"
                          else "event wall-clock time is unknown (only the processing clock); set the camera's recording_start_utc"))
        return [{"item": i, "ok": ok, "detail": d} for i, ok, d in items]

    # -- transitions -----------------------------------------------------------------

    def transition(self, case_id: str, action: str, user, note: str = "", **params) -> dict:
        """Run one workflow action. A BLOCKED approval (or retry-into-approved) is recorded in the audit log with the reasons, so refused
        attempts are as visible as granted ones; the case itself is left exactly as it was (the transaction rolls back)."""
        try:
            return self._transition(case_id, action, user, note, **params)
        except CaseError as exc:
            if action in ("approve", "retry_submission") and exc.status in (403, 409, 422):
                self._audit.append(user.username, f"case.{action}_blocked", "case", case_id,
                                   {"http_status": exc.status, "reasons": ([str(e)[:300] for e in exc.errors[:10]] or [str(exc)[:300]])},
                                   actor_role=user.role)
            raise

    def _transition(self, case_id: str, action: str, user, note: str = "", **params) -> dict:
        unexpected = set(params) - {"reading_id", "allow_nonstandard", "external_reference"}
        if unexpected:  # e.g. the removed registry_override_reason, or any attempt to hand the service a registry verdict
            raise CaseError(f"unexpected action parameter(s): {sorted(unexpected)[:5]}"
                            + ("; there is no registry override - an authorised registry result is required" if "registry_override_reason" in unexpected else ""), 422)
        if not isinstance(action, str) or action not in TRANSITIONS:
            raise CaseError(f"unknown action {action!r}", 422)
        sources, target, permission = TRANSITIONS[action]
        if not user.can(permission):
            raise CaseError(f"role {user.role} may not perform {action}", 403)
        if not isinstance(action, str) or not isinstance(note or "", str):
            raise CaseError("action and note must be strings", 422)
        note = (note or "").strip()
        if action in NOTE_REQUIRED and not note:
            raise CaseError(f"a note is required for {action}", 422)
        extra = {}
        with self._db.write() as conn:
            case = self._row(case_id)
            _refuse_demo_on_operational(user, case["data_class"])
            if case["state"] not in sources:
                raise CaseError(f"cannot {action} a case in state {case['state']}", 409)
            now = now_iso()
            if action == "verify_evidence":
                ev = self.evidence_check(case["incident_id"])
                if not ev["ok"]:
                    raise CaseError("evidence is incomplete or damaged", 422, ev["problems"])
                extra = {"evidence_verified_by": user.username, "evidence_verified_at": now}
            elif action == "verify_plate":
                extra = self._verify_plate(case, user, params, note)
            elif action == "retry_submission":
                # A failed submission goes back to approved only if the case STILL satisfies every approval requirement (the registry
                # result may have expired, the plate may have been corrected): this is a second entry into `approved_for_submission`.
                errors = [f"{c['item']}: {c['detail']}" for c in self.readiness(case) if not c["ok"]]
                if errors:
                    raise CaseError("case no longer meets the approval requirements", 422, errors)
            elif action == "approve":
                errors = [f"{c['item']}: {c['detail']}" for c in self.readiness(case) if not c["ok"]]
                if self._distinct:
                    for field, label in (("evidence_verified_by", "evidence"), ("plate_verified_by", "plate"), ("created_by", "the case")):
                        if case[field] == user.username:
                            errors.append(f"four-eyes: the approving officer cannot be the user who verified {label}" if field != "created_by"
                                          else "four-eyes: the approving officer cannot be the user who created the case")
                if errors:
                    raise CaseError("case is not ready for approval", 422, errors)
                extra.update({"approved_by": user.username, "approved_at": now})
            elif action in ("request_more", "resume_review"):
                extra = {"evidence_verified_by": None, "evidence_verified_at": None, "plate_verified_by": None, "plate_verified_at": None,
                         "plate_verified_reading_id": None, "approved_by": None, "approved_at": None}
            elif action == "record_reference":
                ref = (params.get("external_reference") or "").strip()
                if not ref:
                    raise CaseError("external_reference is required", 422)
                handoff = self._db.one("SELECT * FROM handoffs WHERE case_id = ? AND status = 'exported' ORDER BY created_at DESC LIMIT 1", (case_id,))
                if handoff is None:
                    raise CaseError("there is no exported handoff to attach a reference to", 409)
                extra = {"external_reference": ref, "external_reference_kind": "officer_entered"}
            sets = ["state = ?", "updated_at = ?", "version = version + 1"] + [f"{k} = ?" for k in extra]
            conn.execute(f"UPDATE cases SET {', '.join(sets)} WHERE case_id = ?", [target, now] + list(extra.values()) + [case_id])
            conn.execute("INSERT INTO case_transitions(case_id, from_state, to_state, action, actor, actor_role, note, ts) VALUES (?,?,?,?,?,?,?,?)",
                         (case_id, case["state"], target, action, user.username, user.role, note, now))
        self._audit.append(user.username, f"case.{action}", "case", case_id, {"from": case["state"], "to": target, "note": note,
                                                                           **{k: v for k, v in extra.items() if k.endswith("_by") or k == "external_reference"}},
                           actor_role=user.role)
        return self.get(case_id, user)

    def _verify_plate(self, case: dict, user, params: dict, note: str) -> dict:
        heads = self._plates.current(case["incident_id"], case["track_id"]) or self._plates.current(case["incident_id"])
        if not heads:
            raise CaseError("no plate reading exists for this case; enter the plate first", 422)
        reading_id = params.get("reading_id") or heads[0]["reading_id"]
        reading = self._plates.get(reading_id)
        if reading is None or reading["incident_id"] != case["incident_id"]:
            raise CaseError("reading does not belong to this case's incident", 422)
        verified_id = reading_id
        try:
            if not reading["verified"]:
                verified_id = self._plates.verify(reading_id, user.username, user.role, note=note,
                                                  allow_nonstandard=bool(params.get("allow_nonstandard")))["reading_id"]
            elif self._distinct and reading["created_by"] == user.username:
                raise CaseError("four-eyes: you cannot re-confirm a reading you made yourself", 409)
        except ValueError as exc:
            raise CaseError(str(exc), 422) from None
        return {"plate_verified_by": user.username, "plate_verified_at": now_iso(), "plate_verified_reading_id": verified_id}

    def apply_handoff_result(self, case_id: str, new_state: str, actor: str, role: str, note: str, extra: dict = None) -> None:
        """Called only by the handoff service (which holds the submit permission and idempotency checks)."""
        extra = extra or {}
        with self._db.write() as conn:
            case = self._row(case_id)
            sets = ["state = ?", "updated_at = ?", "version = version + 1"] + [f"{k} = ?" for k in extra]
            conn.execute(f"UPDATE cases SET {', '.join(sets)} WHERE case_id = ?", [new_state, now_iso()] + list(extra.values()) + [case_id])
            conn.execute("INSERT INTO case_transitions(case_id, from_state, to_state, action, actor, actor_role, note, ts) VALUES (?,?,?,?,?,?,?,?)",
                         (case_id, case["state"], new_state, "handoff", actor, role, note, now_iso()))

    # -- reading ----------------------------------------------------------------------

    def allowed_actions(self, case: dict, user) -> list:
        return sorted(a for a, (src, _t, perm) in TRANSITIONS.items() if case["state"] in src and user.can(perm))

    def get(self, case_id: str, user=None, detail: bool = True) -> dict:
        case = self._row(case_id)
        out = dict(case)
        out["transitions"] = self._db.query("SELECT * FROM case_transitions WHERE case_id = ? ORDER BY transition_id", (case_id,))
        if detail:
            inc = self._incidents.get(case["incident_id"])
            out["incident"] = {k: inc[k] for k in ("incident_id", "event_type", "severity", "status", "camera_id", "start_utc", "detected_utc",
                                                    "start_offset_s", "detected_offset_s", "time_basis", "location", "confidence_tier",
                                                    "confidence_basis", "uncertainty", "indicators", "involved", "detector", "data_class")}
            out["plates"] = self._plates.history(case["incident_id"])
            out["current_plate"] = self.plate_for(case)
            out["registry_lookups"] = self._registry.for_case(case_id, user)
            out["evidence"] = self._evidence.list(case["incident_id"])
            out["handoffs"] = self._db.query("SELECT * FROM handoffs WHERE case_id = ? ORDER BY created_at DESC", (case_id,))
            for h in out["handoffs"]:
                h["response"] = loads(h.pop("response_json"), {})
                h["is_simulated"] = bool(h["is_simulated"])
            out["readiness"] = self.readiness(case)
        out["allowed_actions"] = self.allowed_actions(case, user) if user is not None else []
        out["labels"] = {"data_class": case["data_class"],
                         "non_production": case["data_class"] != "operational",
                         "reference_notice": ("TEST reference - not a real challan" if (case["external_reference_kind"] or "") == "sandbox_simulated" else None)}
        return out

    def list(self, filters: dict = None, limit: int = 50, offset: int = 0) -> tuple:
        f = filters or {}
        where, params = [], []
        for col in ("state", "offence_code", "data_class", "incident_id", "created_by"):
            if f.get(col):
                where.append(f"c.{col} = ?")
                params.append(f[col])
        if f.get("q"):
            where.append("(c.case_id LIKE ? OR c.incident_id LIKE ? OR EXISTS (SELECT 1 FROM plate_readings p WHERE p.incident_id = c.incident_id AND p.normalized_text LIKE ?))")
            like = f"%{f['q'].upper()}%"
            params += [like, like, like]
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = self._db.one(f"SELECT COUNT(*) AS n FROM cases c {clause}", params)["n"]
        rows = self._db.query(f"SELECT c.* FROM cases c {clause} ORDER BY c.updated_at DESC LIMIT ? OFFSET ?", params + [limit, offset])
        out = []
        for r in rows:
            plate = self.plate_for(r)
            r["plate"] = {"text": plate["normalized_text"], "status": plate["status"], "verified": plate["verified"]} if plate else None
            out.append(r)
        return out, total

    def counts(self) -> dict:
        return {r["state"]: r["n"] for r in self._db.query("SELECT state, COUNT(*) AS n FROM cases GROUP BY state")}
