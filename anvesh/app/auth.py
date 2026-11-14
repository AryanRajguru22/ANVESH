"""Users, roles, permissions and authenticated sessions.

ROLES (separation of duties is deliberate):
  administrator -- configures cameras, users, thresholds, alert destinations, retention;
                   reads the audit trail. Can NOT approve or submit an enforcement case and
                   can NOT see registry owner details.
  investigator  -- runs sessions, works incidents, creates cases, enters plates, looks up the
                   registry. Cannot verify evidence/plates or approve.
  reviewer      -- verifies evidence and plates, rejects or sends cases back. Cannot approve.
  officer       -- the authorised enforcement officer: approves, submits/exports, closes.
                   The only role that sees registry owner details besides investigators.

PASSWORDS are hashed with scrypt (stdlib; N=2^17, r=8, p=1 = about 128 MiB and 0.4 s per hash, the OWASP-recommended minimum
for scrypt) and a per-user salt; the parameters are stored in the hash string so older, weaker hashes still verify and are
upgraded transparently at the next successful login. At most two hashes run at once, so a burst of logins cannot exhaust memory.
Session tokens are 256-bit random values; only their SHA-256 is stored, so a database leak does not yield usable tokens.

THROTTLING (in-process memory only - see LoginThrottle): failures are counted per (username, source) pair, per source and per
username. A pair locks after 5 failures, a source after 30, a username after 25 spread over any sources, each for 5 minutes. Pairing
username with source is deliberate: an attacker elsewhere cannot lock a known user out of their own desk with five bad guesses.
The counters are NOT persisted and NOT shared between server processes; a restart clears them, and an administrator can clear a
username with `unlock`. Unknown usernames are tracked identically, so the response never reveals whether an account exists.

NOT IMPLEMENTED: multi-factor authentication. There is no cryptography-grade library available in the pinned environment and none is
hand-rolled here; see docs/ANVESH_ACCOUNT_SECURITY.md for the design and what remains.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import secrets
import threading
import time
import uuid

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase
from anvesh.app.timeutil import add_seconds, iso, now_iso, now_utc

ROLES = ("administrator", "investigator", "reviewer", "officer")
PRIVILEGED_ROLES = ("administrator", "officer")  # roles whose provisioning needs a second administrator

_VIEW = {"view.overview", "view.cameras", "view.incidents", "view.cases", "view.evidence", "view.congestion"}

PERMISSIONS = {
    "administrator": _VIEW | {"camera.manage", "session.control", "admin.users", "admin.settings",
                              "audit.view", "audit.verify", "alert.configure", "alert.retry", "export.audit",
                              "retention.manage"},
    "investigator": _VIEW | {"session.control", "incident.update", "incident.establish_cause", "case.create",
                             "plate.enter", "registry.lookup", "registry.view_owner", "export.case", "alert.retry"},
    "reviewer": _VIEW | {"case.verify_evidence", "case.verify_plate", "case.review", "plate.enter",
                         "registry.lookup", "export.case", "incident.update"},
    "officer": _VIEW | {"case.review", "case.approve", "case.submit", "case.close", "plate.enter", "registry.lookup",
                        "registry.view_owner", "export.case", "incident.update", "incident.establish_cause",
                        "audit.view"},
}

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256
MAX_USERNAME_LENGTH = 64
MAX_DISPLAY_NAME_LENGTH = 120
_COMMON_PASSWORDS = frozenset(p.lower() for p in (
    "password123", "password1234", "passw0rd123", "1234567890", "12345678910", "qwertyuiop", "qwerty12345", "letmein1234",
    "welcome1234", "administrator", "admin123456", "changeme123", "iloveyou123", "abc1234567", "0123456789", "1q2w3e4r5t",
    "anvesh12345", "anvesh@1234", "officer1234", "reviewer1234", "investigator"))

# scrypt cost. PRODUCTION is the OWASP-recommended minimum for scrypt; the live dict can be lowered by tests only.
SCRYPT_PRODUCTION = {"n": 2 ** 17, "r": 8, "p": 1, "dklen": 32}
SCRYPT = dict(SCRYPT_PRODUCTION)
_SCRYPT_MAXMEM = 256 * 1024 * 1024
_SCRYPT_MAX_N = 2 ** 20  # never trust a stored cost beyond this (a tampered row must not be able to demand gigabytes)
_HASH_SLOTS = threading.BoundedSemaphore(2)


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    with _HASH_SLOTS:
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, maxmem=_SCRYPT_MAXMEM, **SCRYPT)
    return "scrypt${n}${r}${p}${s}${h}".format(
        n=SCRYPT["n"], r=SCRYPT["r"], p=SCRYPT["p"], s=base64.b64encode(salt).decode(), h=base64.b64encode(dk).decode())


def _parse(stored: str):
    scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
    if scheme != "scrypt":
        raise ValueError("unknown scheme")
    n, r, p = int(n), int(r), int(p)
    if not (2 <= n <= _SCRYPT_MAX_N and n & (n - 1) == 0 and 1 <= r <= 32 and 1 <= p <= 16):
        raise ValueError("implausible scrypt parameters")
    return n, r, p, base64.b64decode(salt_b64), base64.b64decode(hash_b64)


def _safe_parse(stored: str):
    try:
        return _parse(stored)
    except (ValueError, TypeError, AttributeError):
        return None


def verify_password(password: str, stored: str) -> bool:
    try:
        n, r, p, salt, want = _parse(stored)
        with _HASH_SLOTS:
            got = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_SCRYPT_MAXMEM, dklen=len(want))
    except (ValueError, TypeError, AttributeError):
        return False
    return hmac.compare_digest(got, want)


def needs_rehash(stored: str) -> bool:
    """True if `stored` was made with weaker scrypt cost than the current setting (upgrade it at the next successful login)."""
    parsed = _safe_parse(stored)
    if parsed is None:
        return False
    n, r, p = parsed[:3]
    return n < SCRYPT["n"] or r < SCRYPT["r"] or p < SCRYPT["p"]


def check_password_policy(password: str, username: str = None) -> None:
    if not isinstance(password, str):
        raise ValueError("password must be a string")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"password must be at most {MAX_PASSWORD_LENGTH} characters")
    low = password.lower()
    if low in _COMMON_PASSWORDS or (username and username.lower() in low and len(low) - len(username) < 6):
        raise ValueError("password is too easy to guess (a common password, or the username with little added)")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class User:
    __slots__ = ("user_id", "username", "display_name", "role", "is_demo", "must_change_password")

    def __init__(self, user_id, username, display_name, role, is_demo=False, must_change_password=False) -> None:
        self.user_id, self.username, self.display_name, self.role = user_id, username, display_name, role
        self.is_demo, self.must_change_password = bool(is_demo), bool(must_change_password)

    def can(self, permission: str) -> bool:
        return permission in PERMISSIONS.get(self.role, ())

    def public(self) -> dict:
        return {"user_id": self.user_id, "username": self.username, "display_name": self.display_name,
                "role": self.role, "permissions": sorted(PERMISSIONS.get(self.role, ())), "is_demo": self.is_demo,
                "must_change_password": self.must_change_password}


class LoginThrottle:
    """Bounded, in-process failure counters. NOT persisted, NOT shared across processes (documented limitation)."""

    LIMITS = {"user_source": 5, "source": 30, "user": 25}
    WINDOW_S = 300  # failures older than this (without a lock) are forgotten
    LOCKOUT_S = 300
    MAX_ENTRIES = 2000  # per table

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._tables = {scope: {} for scope in self.LIMITS}
        self._lock = threading.Lock()

    @staticmethod
    def _key(scope: str, username: str, source: str):
        return {"user_source": (username, source), "source": source, "user": username}[scope]

    def _live(self, table: dict, key, now: float):
        e = table.get(key)
        if e is None:
            return None
        if e["locked_until"] is not None:
            if now >= e["locked_until"]:
                del table[key]
                return None
        elif now - e["first"] >= self.WINDOW_S:
            del table[key]
            return None
        return e

    def check(self, username: str, source: str) -> None:
        now = self._clock()
        with self._lock:
            for scope in self.LIMITS:
                e = self._live(self._tables[scope], self._key(scope, username, source), now)
                if e is not None and e["locked_until"] is not None:
                    raise AuthError("too many failed attempts; try again later", 429)

    def failure(self, username: str, source: str) -> list:
        """Count one failure; returns the scopes that just became locked."""
        now, tripped = self._clock(), []
        with self._lock:
            for scope, limit in self.LIMITS.items():
                table, key = self._tables[scope], self._key(scope, username, source)
                e = self._live(table, key, now)
                if e is None:
                    if len(table) >= self.MAX_ENTRIES:
                        for k in list(table):  # drop expired entries first (the lookup deletes them)
                            self._live(table, k, now)
                        while len(table) >= self.MAX_ENTRIES:
                            table.pop(next(iter(table)))
                    e = table[key] = {"count": 0, "first": now, "locked_until": None}
                e["count"] += 1
                if e["count"] >= limit and e["locked_until"] is None:
                    e["locked_until"] = now + self.LOCKOUT_S
                    tripped.append(scope)
        return tripped

    def success(self, username: str, source: str) -> None:
        with self._lock:
            self._tables["user_source"].pop((username, source), None)
            self._tables["user"].pop(username, None)

    def clear_user(self, username: str) -> int:
        n = 0
        with self._lock:
            n += self._tables["user"].pop(username, None) is not None
            for k in [k for k in self._tables["user_source"] if k[0] == username]:
                del self._tables["user_source"][k]
                n += 1
        return n

    def sizes(self) -> dict:
        with self._lock:
            return {scope: len(t) for scope, t in self._tables.items()}


class AuthService:
    """Trusted account/session primitives. Policy about WHO may provision WHAT (two-person control for privileged roles) lives in
    `anvesh.app.accounts.AccountService`, which is what the HTTP API calls; this class is also used by the bootstrap and the offline
    `scripts/manage_users.py`, whose operator already holds the host."""

    MAX_FAILURES = LoginThrottle.LIMITS["user_source"]
    LOCKOUT_S = LoginThrottle.LOCKOUT_S

    def __init__(self, db: AppDatabase, audit: AuditLog, session_ttl_s: int = 8 * 3600, clock=time.monotonic, idle_timeout_s: int = 30 * 60) -> None:
        self._db, self._audit, self._ttl, self._clock, self._idle = db, audit, session_ttl_s, clock, idle_timeout_s
        self.throttle = LoginThrottle(clock)
        self._dummy_hash = hash_password("dummy-password-for-timing")  # same cost as real hashes, so unknown users cost the same time

    # -- users -------------------------------------------------------------

    def create_user(self, username: str, display_name: str, role: str, password: str, actor: str = "system", *,
                    must_change_password: bool = False, is_demo: bool = False, password_hash: str = None) -> dict:
        for field, value in (("username", username), ("display_name", display_name), ("role", role)):
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
        username = username.strip().lower()
        if not username or len(username) > MAX_USERNAME_LENGTH or not username.replace("_", "").replace(".", "").replace("-", "").isalnum():
            raise ValueError(f"username must be 1-{MAX_USERNAME_LENGTH} letters, digits, '.', '_' or '-'")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if password_hash is None:
            check_password_policy(password, username)
        elif _safe_parse(password_hash) is None:
            raise ValueError("password_hash is not a valid scrypt hash")
        if not display_name.strip() or len(display_name.strip()) > MAX_DISPLAY_NAME_LENGTH:
            raise ValueError(f"display_name must be 1-{MAX_DISPLAY_NAME_LENGTH} characters")
        user_id = "USR-" + uuid.uuid4().hex[:10]
        pw_hash = password_hash or hash_password(password)
        with self._db.write() as conn:
            if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                raise ValueError(f"username {username!r} already exists")
            conn.execute(
                "INSERT INTO users(user_id, username, display_name, role, password_hash, active, created_at, must_change_password, is_demo, password_changed_at) "
                "VALUES (?,?,?,?,?,1,?,?,?,?)",
                (user_id, username, display_name.strip(), role, pw_hash, now_iso(), 1 if must_change_password else 0, 1 if is_demo else 0, now_iso()))
        self._audit.append(actor, "user.create", "user", user_id,
                           {"username": username, "role": role, "demo": bool(is_demo), "temporary_password": bool(must_change_password)})
        return self.get_user(user_id)

    _USER_COLS = "user_id, username, display_name, role, active, created_at, last_login_at, must_change_password, is_demo, password_changed_at"

    def get_user(self, user_id: str):
        return self._db.one(f"SELECT {self._USER_COLS} FROM users WHERE user_id = ?", (user_id,))

    def get_by_username(self, username: str):
        return self._db.one(f"SELECT {self._USER_COLS} FROM users WHERE username = ?", ((username or "").strip().lower(),))

    def list_users(self) -> list:
        return self._db.query(f"SELECT {self._USER_COLS} FROM users ORDER BY username")

    def active_administrators(self, excluding: str = None) -> int:
        return self._db.one("SELECT COUNT(*) AS n FROM users WHERE role = 'administrator' AND active = 1 AND user_id <> ?", (excluding or "",))["n"]

    def update_user(self, user_id: str, actor: str, role: str = None, active: bool = None, password: str = None, *,
                    must_change_password: bool = None, password_hash: str = None) -> dict:
        row = self.get_user(user_id)
        if row is None:
            raise ValueError("no such user")
        if role is not None and not isinstance(role, str):
            raise ValueError("role must be a string")
        if active is not None and not isinstance(active, bool):
            raise ValueError("active must be true or false")
        if password is not None and not isinstance(password, str):
            raise ValueError("password must be a string")
        if role is not None and role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if password is not None:
            check_password_policy(password, row["username"])
        if password_hash is not None and _safe_parse(password_hash) is None:
            raise ValueError("password_hash is not a valid scrypt hash")
        # An installation must always keep an active administrator: deactivating or demoting the last one would lock everybody out of
        # user management (recovery then needs the offline scripts/manage_users.py).
        removing_admin = row["role"] == "administrator" and row["active"] and ((active is False) or (role is not None and role != "administrator"))
        if removing_admin and self.active_administrators(excluding=user_id) == 0:
            raise ValueError("this is the last active administrator; create or activate another one first (or use scripts/manage_users.py)")
        changes, revoke = {}, False
        with self._db.write() as conn:
            if role is not None and role != row["role"]:
                conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))
                changes["role"] = {"from": row["role"], "to": role}
                revoke = True  # a role change takes effect at once and forces a fresh login
            if active is not None and bool(row["active"]) != active:
                conn.execute("UPDATE users SET active = ? WHERE user_id = ?", (1 if active else 0, user_id))
                changes["active"] = active
                revoke = revoke or not active
            if password is not None or password_hash is not None:
                pw_hash = password_hash if password_hash is not None else hash_password(password)
                conn.execute("UPDATE users SET password_hash = ?, password_changed_at = ? WHERE user_id = ?", (pw_hash, now_iso(), user_id))
                if must_change_password is None:
                    must_change_password = True  # a password somebody else set is temporary by definition
                changes["password"] = "reset"
                revoke = True
            if must_change_password is not None:
                conn.execute("UPDATE users SET must_change_password = ? WHERE user_id = ?", (1 if must_change_password else 0, user_id))
                changes["must_change_password"] = bool(must_change_password)
            if revoke:
                conn.execute("UPDATE auth_sessions SET revoked = 1 WHERE user_id = ?", (user_id,))
        self._audit.append(actor, "user.update", "user", user_id, dict(changes, username=row["username"]))
        return self.get_user(user_id)

    def set_demo(self, user_id: str, is_demo: bool, actor: str) -> dict:
        """Flag or unflag an account as a DEMO account (used by the offline CLI; accounts that predate schema v4 are not flagged automatically)."""
        row = self.get_user(user_id)
        if row is None:
            raise ValueError("no such user")
        with self._db.write() as conn:
            conn.execute("UPDATE users SET is_demo = ? WHERE user_id = ?", (1 if is_demo else 0, user_id))
            conn.execute("UPDATE auth_sessions SET revoked = 1 WHERE user_id = ?", (user_id,))
        self._audit.append(actor, "user.update", "user", user_id, {"username": row["username"], "is_demo": bool(is_demo)})
        return self.get_user(user_id)

    def unlock(self, username: str, actor: str) -> int:
        """Clear the in-process lockout counters for `username` (this server process only)."""
        n = self.throttle.clear_user((username or "").strip().lower()[:MAX_USERNAME_LENGTH * 2])
        self._audit.append(actor, "auth.unlock", "user", None, {"username": (username or "")[:MAX_USERNAME_LENGTH], "cleared": n})
        return n

    # -- login / sessions ----------------------------------------------------

    def _purge_sessions(self) -> None:
        cutoff = add_seconds(now_iso(), -86400)
        with self._db.write() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE expires_at < ? OR (revoked = 1 AND created_at < ?)", (cutoff, cutoff))

    def login(self, username: str, password: str, remote_addr: str = None) -> tuple:
        """Returns (token, User). The token is shown once; only its hash is stored."""
        if not isinstance(username, str) or not isinstance(password, str):
            raise AuthError("invalid credentials")
        # No real username can exceed MAX_USERNAME_LENGTH, so truncating cannot match another account; it keeps the audit row small.
        username = username.strip().lower()[:MAX_USERNAME_LENGTH * 2]
        source = (remote_addr or "-")[:64]
        self.throttle.check(username, source)
        row = self._db.one("SELECT * FROM users WHERE username = ?", (username,))
        # Verify against a dummy hash for unknown users so timing does not reveal which usernames exist.
        stored = row["password_hash"] if row else self._dummy_hash
        ok = verify_password(password[:MAX_PASSWORD_LENGTH * 4], stored) and row is not None and row["active"]
        if not ok:
            tripped = self.throttle.failure(username, source)
            self._audit.append(username or "?", "auth.login_failed", "user", row["user_id"] if row else None, {}, remote_addr=remote_addr)
            for scope in tripped:
                # Only a known username is written to the audit row; an attacker-chosen string is not.
                self._audit.append("system", "auth.lockout", "user", row["user_id"] if row else None,
                                   {"scope": scope, "username": row["username"] if row else None, "locked_for_s": self.throttle.LOCKOUT_S},
                                   remote_addr=remote_addr)
            raise AuthError("invalid credentials")
        self.throttle.success(username, source)
        if needs_rehash(row["password_hash"]):
            with self._db.write() as conn:
                conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (hash_password(password), row["user_id"]))
        token = secrets.token_urlsafe(32)
        now = now_utc()
        self._purge_sessions()
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at, remote_addr, last_seen_at) VALUES (?,?,?,?,?,?)",
                (_token_hash(token), row["user_id"], iso(now), iso(now + _dt.timedelta(seconds=self._ttl)), remote_addr, iso(now)))
            conn.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?", (iso(now), row["user_id"]))
        self._audit.append(row["username"], "auth.login", "user", row["user_id"], {}, actor_role=row["role"], remote_addr=remote_addr)
        return token, User(row["user_id"], row["username"], row["display_name"], row["role"], row["is_demo"], row["must_change_password"])

    def authenticate(self, token: str):
        """The User behind `token`, or None (unknown, expired, idle, revoked or deactivated)."""
        if not token or not isinstance(token, str) or len(token) > 256:
            return None
        row = self._db.one(
            "SELECT s.token_hash, s.created_at, s.expires_at, s.revoked, s.last_seen_at, u.user_id, u.username, u.display_name, u.role, u.active, "
            "u.is_demo, u.must_change_password FROM auth_sessions s JOIN users u ON u.user_id = s.user_id WHERE s.token_hash = ?",
            (_token_hash(token),))
        if row is None or row["revoked"] or not row["active"]:
            return None
        now = now_iso()
        if row["expires_at"] <= now:
            return None
        last = row["last_seen_at"] or row["created_at"]
        if self._idle and last < add_seconds(now, -self._idle):
            self._revoke(row["token_hash"])  # idle too long: the session is over, not merely paused
            return None
        if last < add_seconds(now, -60):  # refresh at most once a minute, so a busy dashboard does not write on every request
            with self._db.write() as conn:
                conn.execute("UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?", (now, row["token_hash"]))
        return User(row["user_id"], row["username"], row["display_name"], row["role"], row["is_demo"], row["must_change_password"])

    def _revoke(self, token_hash: str) -> None:
        with self._db.write() as conn:
            conn.execute("UPDATE auth_sessions SET revoked = 1 WHERE token_hash = ?", (token_hash,))

    def logout(self, token: str, user: User = None) -> None:
        self._revoke(_token_hash(token))
        if user is not None:
            self._audit.append(user.username, "auth.logout", "user", user.user_id, {}, actor_role=user.role)

    def change_password(self, user_id: str, current_password: str, new_password: str, remote_addr: str = None) -> None:
        """Self-service change. Re-verifies the current password (throttled like a login), refuses reuse, signs every session out."""
        row = self._db.one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if row is None or not row["active"]:
            raise AuthError("invalid credentials")
        source = (remote_addr or "-")[:64]
        self.throttle.check(row["username"], source)
        if not isinstance(current_password, str) or not verify_password(current_password[:MAX_PASSWORD_LENGTH * 4], row["password_hash"]):
            tripped = self.throttle.failure(row["username"], source)
            self._audit.append(row["username"], "auth.password_change_failed", "user", user_id, {}, actor_role=row["role"], remote_addr=remote_addr)
            for scope in tripped:
                self._audit.append("system", "auth.lockout", "user", user_id, {"scope": scope, "username": row["username"], "locked_for_s": self.throttle.LOCKOUT_S},
                                   remote_addr=remote_addr)
            raise AuthError("invalid credentials")
        check_password_policy(new_password, row["username"])
        if new_password == current_password:
            raise ValueError("the new password must differ from the current one")
        self.throttle.success(row["username"], source)
        with self._db.write() as conn:
            conn.execute("UPDATE users SET password_hash = ?, must_change_password = 0, password_changed_at = ? WHERE user_id = ?",
                         (hash_password(new_password), now_iso(), user_id))
            conn.execute("UPDATE auth_sessions SET revoked = 1 WHERE user_id = ?", (user_id,))
        self._audit.append(row["username"], "auth.password_changed", "user", user_id, {}, actor_role=row["role"], remote_addr=remote_addr)

    def posture(self) -> dict:
        """What this server's authentication actually provides - shown on the status page so nobody has to guess."""
        return {
            "password_hash": f"scrypt N=2^{SCRYPT['n'].bit_length() - 1}, r={SCRYPT['r']}, p={SCRYPT['p']}",
            "session_ttl_s": self._ttl, "session_idle_timeout_s": self._idle,
            "login_throttle": {"limits": dict(LoginThrottle.LIMITS), "lockout_s": LoginThrottle.LOCKOUT_S,
                               "scope": "in-process memory; lost on restart; not shared between server processes"},
            "mfa": "not implemented", "privileged_account_provisioning": "two-person (a second administrator must approve) or offline CLI",
        }
