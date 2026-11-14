"""Append-only, hash-chained audit log.

Each row stores `prev_hash` and `hash = sha256(prev_hash || canonical(row))`, so
removing, reordering or editing a past row breaks every later hash. The table also
rejects UPDATE/DELETE at the SQL level (see `anvesh.app.db`). `verify()` re-derives
the chain.

This is tamper-EVIDENT, not tamper-PROOF: someone with write access to the database
file and the code can rewrite the whole chain. Anchoring the head hash outside the
database (e.g. printing it to an external log) is the production mitigation; the head
is exposed through `head()` for that purpose.

Never put secrets, passwords, tokens or full registry/owner payloads in `details`.
"""

from __future__ import annotations

import hashlib
import json

from anvesh.app.db import AppDatabase, dumps
from anvesh.app.timeutil import now_iso

GENESIS = "0" * 64


def _row_hash(prev_hash: str, ts, actor, actor_role, action, entity_type, entity_id, details_json, remote_addr) -> str:
    body = json.dumps(
        [ts, actor, actor_role, action, entity_type, entity_id, details_json, remote_addr],
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, db: AppDatabase) -> None:
        self._db = db

    def append(self, actor: str, action: str, entity_type: str = None, entity_id: str = None,
               details: dict = None, actor_role: str = None, remote_addr: str = None) -> int:
        details_json = dumps(details or {})
        with self._db.write() as conn:
            last = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last["hash"] if last else GENESIS
            ts = now_iso()
            h = _row_hash(prev, ts, actor, actor_role, action, entity_type, entity_id, details_json, remote_addr)
            cur = conn.execute(
                "INSERT INTO audit_log(ts, actor, actor_role, action, entity_type, entity_id, details_json, remote_addr, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts, actor, actor_role, action, entity_type, entity_id, details_json, remote_addr, prev, h),
            )
            return cur.lastrowid

    def head(self) -> dict:
        row = self._db.one("SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1")
        return row or {"seq": 0, "hash": GENESIS}

    def verify(self) -> dict:
        """Re-derive the whole chain. Returns {ok, rows, first_bad_seq}."""
        prev, rows = GENESIS, 0
        for r in self._db.conn.execute("SELECT * FROM audit_log ORDER BY seq"):
            expected = _row_hash(prev, r["ts"], r["actor"], r["actor_role"], r["action"], r["entity_type"],
                                 r["entity_id"], r["details_json"], r["remote_addr"])
            if r["prev_hash"] != prev or r["hash"] != expected:
                return {"ok": False, "rows": rows, "first_bad_seq": r["seq"]}
            prev, rows = r["hash"], rows + 1
        return {"ok": True, "rows": rows, "first_bad_seq": None}

    def query(self, entity_type: str = None, entity_id: str = None, actor: str = None, action: str = None,
              limit: int = 100, offset: int = 0) -> tuple:
        where, params = [], []
        for col, val in (("entity_type", entity_type), ("entity_id", entity_id), ("actor", actor)):
            if val:
                where.append(f"{col} = ?")
                params.append(val)
        if action:
            where.append("action LIKE ?")
            params.append(action + "%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = self._db.one(f"SELECT COUNT(*) AS n FROM audit_log {clause}", params)["n"]
        rows = self._db.query(f"SELECT * FROM audit_log {clause} ORDER BY seq DESC LIMIT ? OFFSET ?", params + [limit, offset])
        for r in rows:
            r["details"] = json.loads(r.pop("details_json"))
        return rows, total
