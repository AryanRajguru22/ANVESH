"""Evidence retention: a reviewable plan and a guarded purge of evidence FILES.

WHAT IS AND IS NOT DELETED
  - Evidence FILES (frames, clips, thumbnails, plate crops) of incidents older than the evidence-retention period are
    removed from disk.
  - The `evidence_assets` row (with its SHA-256) is KEPT, and an append-only `evidence_purges` row records what was removed,
    when, by whom and why. A purged asset therefore stays identifiable and its hash stays on record; it just cannot be
    served. The dashboard/API report it as "purged under the retention policy", never as silently missing.
  - Incident, case, plate, registry, hand-off and AUDIT rows are never deleted by this module. The audit log is a hash chain
    (deleting a row breaks every later hash) and is protected by SQL triggers; expiring it needs a reviewed offline
    procedure (export + anchor the head hash + start a new segment) that is NOT implemented here.

SAFETY RULES
  - Nothing runs automatically. A purge is an explicit administrator action.
  - Two steps: `plan()` (read-only, lists exactly what would go and what is protected and why) then `purge(plan_id, reason)`.
    `plan_id` is a digest of the asset list; a purge whose plan no longer matches the current state is refused, so an
    administrator can only delete what they were shown.
  - An incident is protected while it is still open (detected / under_investigation / escalated) or has ANY case that is not
    closed or rejected (including submitted / awaiting a challan reference).
  - The retention period may not be shorter than MIN_PURGE_DAYS for a purge, whatever the setting allows.
  - Paths are resolved through `EvidenceStore.path_of`, which refuses anything outside the evidence root.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase
from anvesh.app.timeutil import add_seconds, now_iso

MIN_PURGE_DAYS = 30
OPEN_INCIDENT_STATUSES = ("detected", "under_investigation", "escalated")
FINISHED_CASE_STATES = ("closed", "rejected")


class RetentionError(Exception):
    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


class RetentionService:
    def __init__(self, db: AppDatabase, audit: AuditLog, evidence, settings, default_days: int) -> None:
        self._db, self._audit, self._evidence, self._settings, self._default_days = db, audit, evidence, settings, default_days

    def evidence_days(self) -> int:
        return int(self._settings.get("retention", {}).get("evidence_days", self._default_days))

    # -- planning ----------------------------------------------------------------

    def _blockers(self, incident: dict) -> list:
        reasons = []
        if incident["status"] in OPEN_INCIDENT_STATUSES:
            reasons.append(f"incident is still {incident['status']}")
        marks = ",".join("?" * len(FINISHED_CASE_STATES))
        for c in self._db.query(f"SELECT case_id, state FROM cases WHERE incident_id = ? AND state NOT IN ({marks})",
                                [incident["incident_id"], *FINISHED_CASE_STATES]):
            reasons.append(f"case {c['case_id']} is {c['state']}")
        return reasons

    def plan(self, now: str = None) -> dict:
        """What a purge would do right now. Read-only."""
        now = now or now_iso()
        days = self.evidence_days()
        cutoff = add_seconds(now, -days * 86400)
        eligible, protected, assets = [], [], []
        for inc in self._db.query("SELECT incident_id, detected_utc, status FROM incidents WHERE detected_utc < ? ORDER BY detected_utc, incident_id", (cutoff,)):
            reasons = self._blockers(inc)
            if reasons:
                protected.append({"incident_id": inc["incident_id"], "detected_utc": inc["detected_utc"], "reasons": reasons})
                continue
            rows = self._db.query(
                "SELECT a.asset_id, a.incident_id, a.kind, a.sha256, a.size_bytes, a.relpath FROM evidence_assets a "
                "LEFT JOIN evidence_purges p ON p.asset_id = a.asset_id WHERE a.incident_id = ? AND p.asset_id IS NULL ORDER BY a.asset_id",
                (inc["incident_id"],))
            if rows:
                eligible.append({"incident_id": inc["incident_id"], "detected_utc": inc["detected_utc"], "assets": len(rows),
                                 "bytes": sum(r["size_bytes"] for r in rows)})
                assets.extend(rows)
        # tombstoned assets whose file is still on disk (a purge that was interrupted or whose unlink failed)
        leftovers = []
        for p in self._db.query("SELECT asset_id, incident_id, relpath FROM evidence_purges ORDER BY asset_id"):
            try:
                if self._evidence.path_of({"relpath": p["relpath"]}).is_file():
                    leftovers.append(p)
            except PermissionError:
                continue
        already_purged = self._db.one("SELECT COUNT(*) AS n FROM evidence_purges")["n"]
        plan_id = hashlib.sha256(json.dumps([[a["asset_id"] for a in assets], [l["asset_id"] for l in leftovers]]).encode()).hexdigest()[:16]
        return {"evidence_retention_days": days, "cutoff_utc": cutoff, "evaluated_at": now, "plan_id": plan_id,
                "purge_allowed": days >= MIN_PURGE_DAYS,
                "purge_blocked_reason": None if days >= MIN_PURGE_DAYS else f"retention period {days} d is below the {MIN_PURGE_DAYS} d minimum for a purge",
                "eligible_incidents": eligible, "protected_incidents": protected,
                "assets_to_purge": len(assets), "bytes_to_free": sum(a["size_bytes"] for a in assets),
                "files_pending_removal": len(leftovers), "already_purged_assets": already_purged,
                "_assets": assets, "_leftovers": leftovers,
                "note": "Evidence files only. Rows, hashes and the audit trail are kept. Nothing happens until a purge is requested with this plan_id."}

    def public_plan(self, plan: dict, max_items: int = 200) -> dict:
        out = {k: v for k, v in plan.items() if not k.startswith("_")}
        out["eligible_incidents"] = plan["eligible_incidents"][:max_items]
        out["protected_incidents"] = plan["protected_incidents"][:max_items]
        out["truncated"] = len(plan["eligible_incidents"]) > max_items or len(plan["protected_incidents"]) > max_items
        # exactly which files a purge with this plan_id would remove (paths are inside the evidence root and carry no personal data)
        out["files_to_purge"] = [{k: a[k] for k in ("asset_id", "incident_id", "kind", "relpath", "size_bytes", "sha256")} for a in plan["_assets"][:max_items]]
        out["files_truncated"] = len(plan["_assets"]) > max_items
        return out

    # -- execution ---------------------------------------------------------------

    def purge(self, plan_id: str, reason: str, actor: str, role: str = None, now: str = None) -> dict:
        if not isinstance(reason, str) or len(reason.strip()) < 10:
            raise RetentionError("a reason of at least 10 characters is required", 422)
        plan = self.plan(now)
        if not plan["purge_allowed"]:
            raise RetentionError(plan["purge_blocked_reason"], 409)
        if plan_id != plan["plan_id"]:
            raise RetentionError("the plan changed since it was reviewed (or plan_id is wrong); run a new plan and review it", 409)
        removed, failed, per_incident = 0, [], {}
        freed = 0
        for a in plan["_assets"]:
            # Re-check at the moment of deletion: a case may have been opened since the plan was made.
            inc = self._db.one("SELECT incident_id, status FROM incidents WHERE incident_id = ?", (a["incident_id"],))
            if inc is None or self._blockers(inc):
                failed.append({"asset_id": a["asset_id"], "error": "incident became protected after the plan was made"})
                continue
            try:
                path = self._evidence.path_of(a)
            except PermissionError as exc:
                failed.append({"asset_id": a["asset_id"], "error": str(exc)})
                continue
            existed = path.is_file()
            with self._db.write() as conn:  # tombstone first: from here on the asset is never served, whatever happens to the file
                conn.execute(
                    "INSERT INTO evidence_purges(purge_id, asset_id, incident_id, kind, sha256, size_bytes, relpath, reason, purged_at, purged_by, file_existed) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("PRG-" + uuid.uuid4().hex[:12], a["asset_id"], a["incident_id"], a["kind"], a["sha256"], a["size_bytes"], a["relpath"],
                     reason.strip(), now_iso(), actor, 1 if existed else 0))
            if self._remove(path, failed, a["asset_id"]) and existed:
                removed += 1
                freed += a["size_bytes"]
            per_incident.setdefault(a["incident_id"], []).append({"asset_id": a["asset_id"], "kind": a["kind"], "sha256": a["sha256"], "file_existed": existed})
        for leftover in plan["_leftovers"]:
            try:
                path = self._evidence.path_of(leftover)
            except PermissionError:
                continue
            if self._remove(path, failed, leftover["asset_id"]):
                removed += 1  # (only listed as a leftover because the file was still on disk)
        for incident_id, items in per_incident.items():
            self._audit.append(actor, "retention.purge", "incident", incident_id,
                               {"plan_id": plan_id, "reason": reason.strip(), "cutoff_utc": plan["cutoff_utc"], "assets": items}, actor_role=role)
        self._audit.append(actor, "retention.purge_run", "retention", plan_id,
                           {"reason": reason.strip(), "files_removed": removed, "bytes_freed": freed, "failed": len(failed), "incidents": len(per_incident)},
                           actor_role=role)
        return {"plan_id": plan_id, "files_removed": removed, "bytes_freed": freed, "incidents": len(per_incident), "failed": failed}

    def _remove(self, path, failed: list, asset_id: str) -> bool:
        try:
            if path.is_file():
                path.unlink()
            parent = path.parent
            if parent != self._evidence.root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
            return True
        except OSError as exc:
            failed.append({"asset_id": asset_id, "error": f"{type(exc).__name__}: {exc}"})
            return False
