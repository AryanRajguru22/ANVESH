"""Account provisioning policy (who may create or change which account).

PRINCIPLES
  * Least privilege: only an administrator can provision accounts; nobody can change their OWN role or deactivate themselves.
  * Two-person control for privileged roles (`administrator`, `officer`): creating such an account, moving anyone into or out of such a
    role, resetting such an account's password and re-activating it are REQUESTS that a DIFFERENT active administrator must approve.
    Without this a single administrator could mint an officer and approve their own enforcement case. The requester's password
    choice is stored only as a scrypt hash inside the request.
  * Everything else (investigator / reviewer accounts, deactivating anyone but yourself or the last administrator) is immediate.
  * Passwords set by somebody else are TEMPORARY: the user must change them at first login (the API refuses everything else until then).
  * Pending requests expire after 72 h; decided requests are immutable (database triggers) and every step is audited.
  * Recovery / break-glass: a locked-out or sole administrator uses the OFFLINE `scripts/manage_users.py` on the host (the root of
    trust is access to the data directory). It is audited with actor `cli:<os user>`.
Who can create administrators: a second administrator approving a request, or the offline CLI. Who can create officers: same.
"""

from __future__ import annotations

import json
import uuid

from anvesh.app.audit import AuditLog
from anvesh.app.auth import MAX_DISPLAY_NAME_LENGTH, MAX_USERNAME_LENGTH, PRIVILEGED_ROLES, ROLES, AuthService, check_password_policy, hash_password
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.timeutil import add_seconds, now_iso

REQUEST_TTL_S = 72 * 3600


class AccountError(Exception):
    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


class AccountService:
    def __init__(self, db: AppDatabase, audit: AuditLog, auth: AuthService) -> None:
        self._db, self._audit, self._auth = db, audit, auth

    # -- direct operations by an acting administrator ------------------------------------------------------------------

    @staticmethod
    def _require_admin(actor) -> None:
        if actor is None or not actor.can("admin.users"):
            raise AccountError("only an administrator can manage accounts", 403)

    @staticmethod
    def _scope(actor, target_is_demo: bool) -> None:
        """A DEMO administrator manages only demo accounts: it must not be able to reset, re-role or approve anything real."""
        if getattr(actor, "is_demo", False) and not target_is_demo:
            raise AccountError("demo accounts can only manage demo accounts", 403)

    def create_user(self, actor, username, display_name, role, password) -> dict:
        """-> {"status": "created", "user": {...}} for investigator/reviewer, {"status": "pending", "request": {...}} for privileged roles."""
        self._require_admin(actor)
        for name, value in (("username", username), ("display_name", display_name), ("role", role), ("password", password)):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        username = username.strip().lower()
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if not username or len(username) > MAX_USERNAME_LENGTH or not username.replace("_", "").replace(".", "").replace("-", "").isalnum():
            raise ValueError(f"username must be 1-{MAX_USERNAME_LENGTH} letters, digits, '.', '_' or '-'")
        if not display_name.strip() or len(display_name.strip()) > MAX_DISPLAY_NAME_LENGTH:
            raise ValueError(f"display_name must be 1-{MAX_DISPLAY_NAME_LENGTH} characters")
        check_password_policy(password, username)
        if role not in PRIVILEGED_ROLES:
            # whatever a demo administrator creates is a demo account too: demo status cannot be laundered into real credentials
            user = self._auth.create_user(username, display_name, role, password, actor.username, must_change_password=True,
                                          is_demo=getattr(actor, "is_demo", False))
            return {"status": "created", "user": user}
        if self._auth.get_by_username(username) is not None:
            raise ValueError(f"username {username!r} already exists")
        req = self._open_request(actor, "create_user", None, username, {"display_name": display_name.strip(), "role": role, "password_hash": hash_password(password),
                                                                          "is_demo": bool(getattr(actor, "is_demo", False))})
        return {"status": "pending", "request": req, "note": f"creating a {role} account needs approval by a different administrator"}

    def update_user(self, actor, user_id: str, role=None, active=None, password=None) -> dict:
        self._require_admin(actor)
        given = [n for n, v in (("role", role), ("active", active), ("password", password)) if v is not None]
        if len(given) != 1:
            raise ValueError("change exactly one of role, active or password per request")
        row = self._auth.get_user(user_id)
        if row is None:
            raise AccountError("no such user", 404)
        self._scope(actor, bool(row["is_demo"]))
        is_self = row["user_id"] == actor.user_id
        target_priv = row["role"] in PRIVILEGED_ROLES
        if role is not None:
            if not isinstance(role, str) or role not in ROLES:
                raise ValueError(f"role must be one of {ROLES}")
            if is_self:
                raise AccountError("you cannot change the role of your own account", 409)
            if role == row["role"]:
                return {"status": "unchanged", "user": row}
            if target_priv or role in PRIVILEGED_ROLES:
                return {"status": "pending", "request": self._open_request(actor, "change_role", row, row["username"], {"role": role}),
                        "note": "a role change into or out of administrator/officer needs approval by a different administrator"}
            return {"status": "updated", "user": self._auth.update_user(user_id, actor.username, role=role)}
        if active is not None:
            if not isinstance(active, bool):
                raise ValueError("active must be true or false")
            if is_self and active is False:
                raise AccountError("you cannot deactivate your own account", 409)
            if bool(row["active"]) == active:
                return {"status": "unchanged", "user": row}
            if active and target_priv:
                return {"status": "pending", "request": self._open_request(actor, "activate_user", row, row["username"], {}),
                        "note": "re-activating an administrator/officer account needs approval by a different administrator"}
            return {"status": "updated", "user": self._auth.update_user(user_id, actor.username, active=active)}
        # password
        if is_self:
            raise AccountError("use the change-password function for your own account", 409)
        check_password_policy(password, row["username"])
        if target_priv:
            return {"status": "pending", "request": self._open_request(actor, "reset_password", row, row["username"], {"password_hash": hash_password(password)}),
                    "note": "resetting an administrator/officer password needs approval by a different administrator"}
        return {"status": "updated", "user": self._auth.update_user(user_id, actor.username, password=password, must_change_password=True)}

    def unlock(self, actor, user_id: str) -> dict:
        self._require_admin(actor)
        row = self._auth.get_user(user_id)
        if row is None:
            raise AccountError("no such user", 404)
        self._scope(actor, bool(row["is_demo"]))
        return {"cleared": self._auth.unlock(row["username"], actor.username),
                "scope": "this server process only; counters are in memory and also expire by themselves after 5 minutes"}

    # -- requests ------------------------------------------------------------------------------------------------------

    def _open_request(self, actor, kind: str, target, username: str, payload: dict) -> dict:
        self._expire()
        if self._db.one("SELECT 1 AS x FROM account_requests WHERE status = 'pending' AND kind = ? AND target_username = ?", (kind, username)):
            raise AccountError(f"a pending {kind} request for {username!r} already exists", 409)
        request_id = "ACR-" + uuid.uuid4().hex[:10].upper()
        now = now_iso()
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO account_requests(request_id, kind, target_user_id, target_username, payload_json, status, requested_by, requested_by_user_id, "
                "requested_at, expires_at) VALUES (?,?,?,?,?,'pending',?,?,?,?)",
                (request_id, kind, target["user_id"] if target else None, username, dumps(payload), actor.username, actor.user_id, now,
                 add_seconds(now, REQUEST_TTL_S)))
        # the payload (which may contain a password hash) is never copied to the audit trail
        self._audit.append(actor.username, "account.request", "account_request", request_id,
                           {"kind": kind, "target": username, "role": payload.get("role")}, actor_role=actor.role)
        return self.get_request(request_id)

    def _expire(self) -> None:
        with self._db.write() as conn:
            conn.execute("UPDATE account_requests SET status = 'expired', decided_at = ?, decision_note = 'expired unapproved' "
                         "WHERE status = 'pending' AND expires_at < ?", (now_iso(), now_iso()))

    @staticmethod
    def _present(row: dict) -> dict:
        row = dict(row)
        payload = loads(row.pop("payload_json"), {})
        row["role"] = payload.get("role")
        row["display_name"] = payload.get("display_name")
        row["needs_approval_by"] = "a different active administrator" if row["status"] == "pending" else None
        return row  # the password hash in the payload is deliberately never returned

    def get_request(self, request_id: str):
        row = self._db.one("SELECT * FROM account_requests WHERE request_id = ?", (request_id,))
        return self._present(row) if row else None

    def list_requests(self, actor, status: str = None) -> list:
        self._require_admin(actor)
        self._expire()
        rows = self._db.query("SELECT * FROM account_requests " + ("WHERE status = ? " if status else "") + "ORDER BY requested_at DESC LIMIT 200",
                              (status,) if status else ())
        if getattr(actor, "is_demo", False):  # a demo administrator sees only requests raised by demo administrators
            demo_ids = {u["user_id"] for u in self._auth.list_users() if u["is_demo"]}
            rows = [r for r in rows if r["requested_by_user_id"] in demo_ids]
        return [self._present(r) for r in rows]

    def approve(self, actor, request_id: str, note: str = "") -> dict:
        self._require_admin(actor)
        self._expire()
        row = self._db.one("SELECT * FROM account_requests WHERE request_id = ?", (request_id,))
        if row is None:
            raise AccountError("no such request", 404)
        if row["status"] != "pending":
            raise AccountError(f"this request is already {row['status']}", 409)
        if row["requested_by_user_id"] == actor.user_id:
            self._audit.append(actor.username, "account.request_self_approval_refused", "account_request", request_id, {}, actor_role=actor.role)
            raise AccountError("you cannot approve your own request; a different administrator must", 403)
        requester = self._auth.get_user(row["requested_by_user_id"])
        if requester is not None:
            self._scope(actor, bool(requester["is_demo"]))
        if requester is None or not requester["active"] or requester["role"] != "administrator":
            return self._finish(actor, row, "failed", note, "the requesting administrator is no longer an active administrator")
        payload = loads(row["payload_json"], {})
        try:
            if row["kind"] == "create_user":
                self._auth.create_user(row["target_username"], payload["display_name"], payload["role"], None, actor.username,
                                       must_change_password=True, password_hash=payload["password_hash"],
                                       is_demo=bool(payload.get("is_demo")) or bool(requester and requester["is_demo"]) or getattr(actor, "is_demo", False))
            else:
                target = self._auth.get_user(row["target_user_id"])
                if target is None:
                    raise ValueError("the target account no longer exists")
                if row["kind"] == "change_role":
                    self._auth.update_user(target["user_id"], actor.username, role=payload["role"])
                elif row["kind"] == "reset_password":
                    self._auth.update_user(target["user_id"], actor.username, password_hash=payload["password_hash"], must_change_password=True)
                elif row["kind"] == "activate_user":
                    self._auth.update_user(target["user_id"], actor.username, active=True)
        except (ValueError, KeyError) as exc:
            return self._finish(actor, row, "failed", note, f"could not be applied: {exc}")
        return self._finish(actor, row, "approved", note, "applied")

    def reject(self, actor, request_id: str, note: str = "") -> dict:
        """Any administrator may reject; the requester rejecting their own request is a withdrawal."""
        self._require_admin(actor)
        self._expire()
        row = self._db.one("SELECT * FROM account_requests WHERE request_id = ?", (request_id,))
        if row is None:
            raise AccountError("no such request", 404)
        if row["status"] != "pending":
            raise AccountError(f"this request is already {row['status']}", 409)
        requester = self._auth.get_user(row["requested_by_user_id"])
        if requester is not None:
            self._scope(actor, bool(requester["is_demo"]))
        return self._finish(actor, row, "rejected", note, "withdrawn" if row["requested_by_user_id"] == actor.user_id else "rejected")

    def _finish(self, actor, row: dict, status: str, note: str, result: str) -> dict:
        with self._db.write() as conn:
            conn.execute("UPDATE account_requests SET status = ?, decided_by = ?, decided_at = ?, decision_note = ?, result = ? WHERE request_id = ?",
                         (status, actor.username, now_iso(), (note or "")[:500], result, row["request_id"]))
        self._audit.append(actor.username, f"account.request_{status}", "account_request", row["request_id"],
                           {"kind": row["kind"], "target": row["target_username"], "requested_by": row["requested_by"], "result": result}, actor_role=actor.role)
        return self.get_request(row["request_id"])
