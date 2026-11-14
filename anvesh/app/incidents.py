"""Incident records: persistence, de-duplication, updates, timeline, officer actions.

An incident is a CANDIDATE event with provenance, not a finding. Three things are kept
deliberately apart (and the data model has a place for each):

  observed event               -- what the detector saw (`event_type`, `indicators`, times);
  potentially involved vehicle -- `involved[].role` is `observed` or `potentially_involved`;
  evidence-supported contributing vehicle / established cause
                               -- NOT set by any detector. `established_cause` is empty unless an
                                  authorised officer records it, with their name and a basis.

IDEMPOTENCY. `natural_key` (session id + the detector's own key, e.g. a stop episode id) is
UNIQUE: re-delivering the same episode from later windows updates the one incident, never
creates a second. Session ids are random per processing run, so reprocessing a file produces
new incidents rather than colliding with the old ones.
"""

from __future__ import annotations

import logging
import uuid

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.timeutil import add_seconds, now_iso

log = logging.getLogger(__name__)

SEVERITY_ORDER = ("info", "low", "medium", "high", "critical")
STATUSES = ("detected", "under_investigation", "confirmed", "dismissed", "escalated", "closed")
STATUS_TRANSITIONS = {
    "detected": {"under_investigation", "confirmed", "dismissed", "escalated"},
    "under_investigation": {"confirmed", "dismissed", "escalated", "closed"},
    "escalated": {"under_investigation", "confirmed", "dismissed"},
    "confirmed": {"closed", "escalated"},
    "dismissed": {"under_investigation", "closed"},
    "closed": set(),
}
_JSON_COLS = {"location_json": "location", "track_ids_json": "track_ids", "involved_json": "involved", "indicators_json": "indicators",
              "disruption_json": "disruption", "candidate_causes_json": "candidate_causes", "uncertainty_json": "uncertainty",
              "related_json": "related", "detector_config_json": "detector_config"}
DISRUPTION_LINK_WINDOW_S = 120.0


class SessionInfo:
    """What the incident service needs to know about the session an event came from."""

    def __init__(self, session_id, camera_id, data_class, recording_start_utc, time_basis, location=None, utc_offset_minutes=330):
        self.session_id, self.camera_id, self.data_class = session_id, camera_id, data_class
        self.recording_start_utc, self.time_basis = recording_start_utc, time_basis
        self.location = location or {}
        self.utc_offset_minutes = utc_offset_minutes


def _present(row: dict) -> dict:
    row = dict(row)
    for col, name in _JSON_COLS.items():
        row[name] = loads(row.pop(col), [] if name in ("track_ids", "involved", "candidate_causes", "uncertainty", "related") else {})
    return row


class IncidentService:
    def __init__(self, db: AppDatabase, audit: AuditLog) -> None:
        self._db, self._audit = db, audit
        self.on_created: list = []  # callbacks(incident_dict); failures are logged, never raised
        self.on_updated: list = []

    # -- detector-driven ---------------------------------------------------

    def upsert(self, session: SessionInfo, ev) -> tuple:
        """Create or update the incident for `ev`. Returns (incident, created: bool, changes: list)."""
        key = f"{session.session_id}:{ev.natural_key}"
        existing = self._db.one("SELECT * FROM incidents WHERE natural_key = ?", (key,))
        now = now_iso()
        if existing is None:
            incident_id = f"INC-{now[:10].replace('-', '')}-{uuid.uuid4().hex[:6].upper()}"
            start_utc = add_seconds(session.recording_start_utc, ev.start_offset_s)
            detected_utc = add_seconds(session.recording_start_utc, ev.detected_offset_s)
            end_utc = add_seconds(session.recording_start_utc, ev.end_offset_s) if ev.end_offset_s is not None else None
            with self._db.write() as conn:
                conn.execute(
                    "INSERT INTO incidents(incident_id, natural_key, session_id, camera_id, event_type, status, severity, severity_basis, "
                    "confidence_tier, confidence_basis, evidence_completeness, start_offset_s, detected_offset_s, end_offset_s, start_utc, "
                    "detected_utc, end_utc, time_basis, location_json, track_ids_json, involved_json, indicators_json, disruption_json, "
                    "candidate_causes_json, uncertainty_json, related_json, detector, detector_version, detector_config_json, motion_space, "
                    "geometry_validity, speed_basis, data_class, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,'detected',?,?,?,?,'pending',?,?,?,?,?,?,?,?,?,?,?,?,?,?,'[]',?,?,?,?,?,?,?,?,?)",
                    (incident_id, key, session.session_id, session.camera_id, ev.event_type, ev.severity, ev.severity_basis,
                     ev.confidence_tier, ev.confidence_basis, ev.start_offset_s, ev.detected_offset_s, ev.end_offset_s,
                     start_utc, detected_utc, end_utc, session.time_basis, dumps(session.location), dumps(ev.track_ids),
                     dumps(ev.involved), dumps(ev.indicators), "{}", dumps(ev.candidate_causes), dumps(ev.uncertainty),
                     ev.detector, "1.0.0", dumps(ev.detector_config), ev.motion_space, ev.geometry_validity, ev.speed_basis,
                     session.data_class, now, now),
                )
                self._timeline(conn, incident_id, detected_utc, ev.detected_offset_s, "detected", ev.summary or ev.event_type,
                               {"detector": ev.detector, "severity": ev.severity}, "system")
            self._audit.append("system", "incident.create", "incident", incident_id,
                               {"event_type": ev.event_type, "session": session.session_id, "camera": session.camera_id,
                                "data_class": session.data_class})
            inc = self.get(incident_id)
            if ev.event_type == "congestion":
                self._link_disruption(session, inc)
                inc = self.get(incident_id)
            for cb in self.on_created:
                self._safe(cb, inc)
            return inc, True, ["created"]
        return self._update_existing(session, existing, ev, now)

    def _update_existing(self, session, row, ev, now) -> tuple:
        incident_id = row["incident_id"]
        changes, sets, params = [], [], []
        indicators = loads(row["indicators_json"], {})
        if ev.event_type != row["event_type"] and ev.escalate_to == ev.event_type and row["event_type"] == "stopped_vehicle":
            changes.append(f"escalated to {ev.event_type}")
            sets.append("event_type = ?")
            params.append(ev.event_type)
        if SEVERITY_ORDER.index(ev.severity) > SEVERITY_ORDER.index(row["severity"]):
            changes.append(f"severity {row['severity']} -> {ev.severity}")
            sets += ["severity = ?", "severity_basis = ?"]
            params += [ev.severity, ev.severity_basis]
        if ev.end_offset_s is not None and row["end_offset_s"] is None:
            changes.append("ended")
            sets += ["end_offset_s = ?", "end_utc = ?"]
            params += [ev.end_offset_s, add_seconds(session.recording_start_utc, ev.end_offset_s)]
        if ev.event_type == "congestion":
            prev_level, new_level = indicators.get("current_level"), ev.indicators.get("current_level")
            if prev_level != new_level:
                changes.append(f"congestion level {prev_level} -> {new_level}")
        if ev.indicators != indicators:
            sets.append("indicators_json = ?")
            params.append(dumps({**indicators, **ev.indicators}))
        if ev.candidate_causes and dumps(ev.candidate_causes) != row["candidate_causes_json"]:
            sets.append("candidate_causes_json = ?")
            params.append(dumps(ev.candidate_causes))
        if ev.uncertainty and dumps(ev.uncertainty) != row["uncertainty_json"]:
            sets.append("uncertainty_json = ?")
            params.append(dumps(ev.uncertainty))
        if not sets:
            return self.get(incident_id), False, []
        sets += ["updated_at = ?", "update_count = update_count + 1"]
        params.append(now)
        with self._db.write() as conn:
            conn.execute(f"UPDATE incidents SET {', '.join(sets)} WHERE incident_id = ?", params + [incident_id])
            for c in changes:
                off = ev.end_offset_s if c == "ended" and ev.end_offset_s is not None else ev.detected_offset_s
                self._timeline(conn, incident_id, add_seconds(session.recording_start_utc, off), off, "update", c,
                               {"detector": ev.detector}, "system")
        inc = self.get(incident_id)
        if changes:
            self._audit.append("system", "incident.update", "incident", incident_id, {"changes": changes})
        for cb in self.on_updated:
            self._safe(cb, inc, changes)
        return inc, False, changes

    @staticmethod
    def _safe(cb, *args) -> None:
        try:
            cb(*args)
        except Exception:  # noqa: BLE001 - a failing alert/webhook must never break detection
            log.exception("incident callback failed")

    @staticmethod
    def _timeline(conn, incident_id, ts_utc, offset_s, kind, summary, data, actor) -> None:
        conn.execute(
            "INSERT INTO incident_timeline(incident_id, ts_utc, offset_s, kind, summary, data_json, actor) VALUES (?,?,?,?,?,?,?)",
            (incident_id, ts_utc, offset_s, kind, summary, dumps(data or {}), actor),
        )

    def add_timeline(self, incident_id: str, kind: str, summary: str, actor: str, offset_s=None, data=None) -> None:
        with self._db.write() as conn:
            self._timeline(conn, incident_id, now_iso(), offset_s, kind, summary, data, actor)

    def _link_disruption(self, session, congestion_inc: dict) -> None:
        """A congestion episode that starts within DISRUPTION_LINK_WINDOW_S after another incident on
        the same camera and session is LINKED to it as temporally associated. Never causal."""
        start = congestion_inc["start_offset_s"]
        rows = self._db.query(
            "SELECT incident_id, event_type, start_offset_s FROM incidents WHERE session_id = ? AND incident_id != ? AND event_type != 'congestion' "
            "AND start_offset_s <= ? AND start_offset_s >= ?", (session.session_id, congestion_inc["incident_id"], start, start - DISRUPTION_LINK_WINDOW_S))
        if not rows:
            return
        with self._db.write() as conn:
            links = []
            for r in rows:
                gap = start - r["start_offset_s"]
                links.append({"incident_id": r["incident_id"], "relation": "temporally_associated_with", "event_type": r["event_type"],
                              "gap_s": round(gap, 1), "causal_claim": False})
                conn.execute("UPDATE incidents SET disruption_json = ?, related_json = ?, updated_at = ? WHERE incident_id = ?",
                             (dumps({"followed_by_congestion": congestion_inc["incident_id"], "gap_s": round(gap, 1), "causal_claim": False,
                                     "note": "Congestion began after this incident; association is temporal only"}),
                              dumps([{"incident_id": congestion_inc["incident_id"], "relation": "followed_by_congestion", "gap_s": round(gap, 1),
                                      "causal_claim": False}]), now_iso(), r["incident_id"]))
                self._timeline(conn, r["incident_id"], now_iso(), None, "link",
                               f"Congestion {congestion_inc['incident_id']} began {gap:.0f}s later (temporal association only)",
                               {"causal_claim": False}, "system")
            conn.execute("UPDATE incidents SET related_json = ? WHERE incident_id = ?", (dumps(links), congestion_inc["incident_id"]))

    def set_evidence_completeness(self, incident_id: str, assets: list, has_track: bool) -> str:
        kinds = [a["kind"] for a in assets]
        frames, clips = kinds.count("frame"), kinds.count("clip")
        if frames >= 2 and clips >= 1 and has_track:
            level = "full"
        elif frames >= 1:
            level = "partial"
        else:
            level = "minimal"
        with self._db.write() as conn:
            conn.execute("UPDATE incidents SET evidence_completeness = ? WHERE incident_id = ?", (level, incident_id))
        return level

    # -- reading -----------------------------------------------------------

    def get(self, incident_id: str, with_timeline: bool = False):
        row = self._db.one("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
        if row is None:
            return None
        inc = _present(row)
        if with_timeline:
            inc["timeline"] = [dict(t, data=loads(t.pop("data_json"), {})) for t in
                               self._db.query("SELECT * FROM incident_timeline WHERE incident_id = ? ORDER BY entry_id", (incident_id,))]
        return inc

    def list(self, filters: dict = None, limit: int = 50, offset: int = 0, order: str = "newest") -> tuple:
        f = filters or {}
        where, params = [], []
        for col in ("event_type", "status", "severity", "camera_id", "session_id", "data_class", "assigned_to"):
            if f.get(col):
                where.append(f"{col} = ?")
                params.append(f[col])
        if f.get("from_utc"):
            where.append("detected_utc >= ?")
            params.append(f["from_utc"])
        if f.get("to_utc"):
            where.append("detected_utc <= ?")
            params.append(f["to_utc"])
        if f.get("open_only"):
            where.append("status IN ('detected','under_investigation','escalated')")
        if f.get("q"):
            where.append("(incident_id LIKE ? OR event_type LIKE ? OR involved_json LIKE ? OR investigation_notes LIKE ?)")
            like = f"%{f['q']}%"
            params += [like] * 4
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = self._db.one(f"SELECT COUNT(*) AS n FROM incidents {clause}", params)["n"]
        sev = "CASE severity " + " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(SEVERITY_ORDER)) + " END"
        orderby = f"{sev} DESC, detected_utc DESC" if order == "priority" else "detected_utc DESC, incident_id"
        rows = self._db.query(f"SELECT * FROM incidents {clause} ORDER BY {orderby} LIMIT ? OFFSET ?", params + [limit, offset])
        return [_present(r) for r in rows], total

    def counts(self) -> dict:
        by_status = {r["status"]: r["n"] for r in self._db.query("SELECT status, COUNT(*) AS n FROM incidents GROUP BY status")}
        by_type = {r["event_type"]: r["n"] for r in self._db.query("SELECT event_type, COUNT(*) AS n FROM incidents GROUP BY event_type")}
        by_sev = {r["severity"]: r["n"] for r in self._db.query(
            "SELECT severity, COUNT(*) AS n FROM incidents WHERE status IN ('detected','under_investigation','escalated') GROUP BY severity")}
        return {"by_status": by_status, "by_type": by_type, "open_by_severity": by_sev}

    # -- officer actions -----------------------------------------------------

    def change_status(self, incident_id: str, new_status: str, actor: str, role: str, note: str = "") -> dict:
        inc = self.get(incident_id)
        if inc is None:
            raise LookupError(incident_id)
        if new_status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if new_status not in STATUS_TRANSITIONS[inc["status"]]:
            raise ValueError(f"cannot move an incident from {inc['status']} to {new_status}")
        if new_status in ("dismissed", "closed") and not (note or "").strip():
            raise ValueError("a note is required to dismiss or close an incident")
        with self._db.write() as conn:
            conn.execute("UPDATE incidents SET status = ?, updated_at = ? WHERE incident_id = ?", (new_status, now_iso(), incident_id))
            self._timeline(conn, incident_id, now_iso(), None, "status", f"{inc['status']} -> {new_status}" + (f": {note}" if note else ""),
                           {"from": inc["status"], "to": new_status}, actor)
        self._audit.append(actor, "incident.status", "incident", incident_id, {"from": inc["status"], "to": new_status, "note": note}, actor_role=role)
        return self.get(incident_id)

    def assign(self, incident_id: str, assignee: str, actor: str, role: str) -> dict:
        if self.get(incident_id) is None:
            raise LookupError(incident_id)
        with self._db.write() as conn:
            conn.execute("UPDATE incidents SET assigned_to = ?, updated_at = ? WHERE incident_id = ?", (assignee or None, now_iso(), incident_id))
            self._timeline(conn, incident_id, now_iso(), None, "assign", f"assigned to {assignee or 'nobody'}", {}, actor)
        self._audit.append(actor, "incident.assign", "incident", incident_id, {"assignee": assignee}, actor_role=role)
        return self.get(incident_id)

    def add_note(self, incident_id: str, note: str, actor: str, role: str) -> dict:
        note = (note or "").strip()
        if not note:
            raise ValueError("note must be non-empty")
        inc = self.get(incident_id)
        if inc is None:
            raise LookupError(incident_id)
        body = (inc["investigation_notes"] + "\n" if inc["investigation_notes"] else "") + f"[{now_iso()} {actor}] {note}"
        with self._db.write() as conn:
            conn.execute("UPDATE incidents SET investigation_notes = ?, updated_at = ? WHERE incident_id = ?", (body, now_iso(), incident_id))
            self._timeline(conn, incident_id, now_iso(), None, "note", note, {}, actor)
        self._audit.append(actor, "incident.note", "incident", incident_id, {"length": len(note)}, actor_role=role)
        return self.get(incident_id)

    def establish_cause(self, incident_id: str, cause: str, basis: str, actor: str, role: str) -> dict:
        """Record an officer's finding as the established cause. Never done by a detector."""
        cause, basis = (cause or "").strip(), (basis or "").strip()
        if not cause or not basis:
            raise ValueError("an established cause needs both the cause and the evidence basis")
        if self.get(incident_id) is None:
            raise LookupError(incident_id)
        with self._db.write() as conn:
            conn.execute("UPDATE incidents SET established_cause = ?, established_cause_by = ?, updated_at = ? WHERE incident_id = ?",
                         (f"{cause}\nBasis: {basis}", actor, now_iso(), incident_id))
            self._timeline(conn, incident_id, now_iso(), None, "cause", f"cause recorded by {actor}: {cause}", {"basis": basis}, actor)
        self._audit.append(actor, "incident.establish_cause", "incident", incident_id, {"cause": cause, "basis": basis}, actor_role=role)
        return self.get(incident_id)

    def set_alert_status(self, incident_id: str, status: str) -> None:
        with self._db.write() as conn:
            conn.execute("UPDATE incidents SET alert_status = ? WHERE incident_id = ?", (status, incident_id))
