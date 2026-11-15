import sqlite3

import pytest

from anvesh.app.audit import GENESIS
from anvesh.app.auth import PERMISSIONS, AuthError, hash_password, verify_password
from anvesh.app.db import SCHEMA_VERSION, AppDatabase, SchemaTooNew
from anvesh.app.timeutil import iso, parse_iso, to_local_label
import datetime as dt


def test_migrations_apply_once_and_reopen(tmp_path):
    p = tmp_path / "a.sqlite3"
    d = AppDatabase(p)
    assert d.schema_version() == SCHEMA_VERSION
    d.close()
    d2 = AppDatabase(p)  # reopening must not re-run the migration
    assert d2.schema_version() == SCHEMA_VERSION
    d2.close()


def test_database_from_newer_build_is_refused(tmp_path):
    p = tmp_path / "a.sqlite3"
    AppDatabase(p).close()
    conn = sqlite3.connect(p)
    conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (99, 'x')")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaTooNew):
        AppDatabase(p)


def test_audit_chain_verifies_and_detects_tampering(db, audit, tmp_path):
    for i in range(5):
        audit.append("alice", "thing.do", "thing", str(i), {"i": i}, actor_role="officer")
    assert audit.verify() == {"ok": True, "rows": 5, "first_bad_seq": None}
    assert audit.head()["seq"] == 5
    # SQL-level append-only guard
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE audit_log SET actor = 'mallory' WHERE seq = 2")
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("DELETE FROM audit_log WHERE seq = 2")
    # an out-of-band edit (bypassing the trigger by dropping it) is still caught by the chain
    db.execute("DROP TRIGGER audit_no_update")
    db.execute("UPDATE audit_log SET details_json = '{\"i\":99}' WHERE seq = 3")
    res = audit.verify()
    assert res["ok"] is False and res["first_bad_seq"] == 3


def test_audit_first_row_chains_to_genesis(db, audit):
    audit.append("a", "x")
    assert db.one("SELECT prev_hash FROM audit_log WHERE seq = 1")["prev_hash"] == GENESIS


def test_password_hash_roundtrip_and_salt():
    h1, h2 = hash_password("correct horse battery"), hash_password("correct horse battery")
    assert h1 != h2
    assert verify_password("correct horse battery", h1)
    assert not verify_password("wrong", h1)
    assert not verify_password("x", "garbage")


def test_create_user_validation(auth):
    with pytest.raises(ValueError):
        auth.create_user("bob", "Bob", "emperor", "longenough-pw")
    with pytest.raises(ValueError):
        auth.create_user("bob", "Bob", "officer", "short")
    u = auth.create_user("Bob", "Bob", "officer", "longenough-pw")
    assert u["username"] == "bob" and "password_hash" not in u
    with pytest.raises(ValueError):
        auth.create_user("bob", "Bob2", "officer", "longenough-pw")


def test_login_token_lifecycle(auth, db):
    auth.create_user("carol", "Carol", "reviewer", "longenough-pw")
    token, user = auth.login("carol", "longenough-pw")
    assert user.role == "reviewer"
    # only the hash is stored
    assert db.one("SELECT COUNT(*) AS n FROM auth_sessions WHERE token_hash = ?", (token,))["n"] == 0
    assert auth.authenticate(token).username == "carol"
    auth.logout(token, user)
    assert auth.authenticate(token) is None
    assert auth.authenticate("nonsense") is None


def test_wrong_password_and_unknown_user_are_indistinguishable_and_audited(auth, db):
    auth.create_user("dave", "Dave", "officer", "longenough-pw")
    for name in ("dave", "ghost"):
        with pytest.raises(AuthError) as e:
            auth.login(name, "nope-nope-nope")
        assert str(e.value) == "invalid credentials"
    assert db.one("SELECT COUNT(*) AS n FROM audit_log WHERE action = 'auth.login_failed'")["n"] == 2


def test_lockout_after_repeated_failures(db, audit):
    from anvesh.app.auth import AuthService
    clock = [0.0]
    a = AuthService(db, audit, clock=lambda: clock[0])
    a.create_user("erin", "Erin", "officer", "longenough-pw")
    for _ in range(AuthService.MAX_FAILURES):
        with pytest.raises(AuthError):
            a.login("erin", "bad-bad-bad-bad")
    with pytest.raises(AuthError) as e:
        a.login("erin", "longenough-pw")  # even the right password is refused while locked
    assert e.value.status == 429
    clock[0] += AuthService.LOCKOUT_S + 1
    assert a.login("erin", "longenough-pw")[1].username == "erin"


def test_deactivated_user_session_is_revoked(auth):
    u = auth.create_user("frank", "Frank", "officer", "longenough-pw")
    token, _ = auth.login("frank", "longenough-pw")
    auth.update_user(u["user_id"], "admin", active=False)
    assert auth.authenticate(token) is None
    with pytest.raises(AuthError):
        auth.login("frank", "longenough-pw")


def test_separation_of_duties_in_permission_matrix():
    assert "case.approve" not in PERMISSIONS["administrator"]
    assert "case.approve" not in PERMISSIONS["investigator"]
    assert "case.approve" not in PERMISSIONS["reviewer"]
    assert "case.approve" in PERMISSIONS["officer"]
    assert "registry.view_owner" not in PERMISSIONS["administrator"]
    assert "registry.view_owner" not in PERMISSIONS["reviewer"]
    assert "case.verify_evidence" not in PERMISSIONS["officer"]


def test_time_helpers_are_utc_and_reject_naive():
    t = dt.datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=dt.timezone.utc)
    assert iso(t) == "2026-01-02T03:04:05.678Z"
    assert parse_iso(iso(t)) == t
    assert to_local_label("2026-01-02T03:04:05.000Z", 330) == "2026-01-02 08:34:05 UTC+05:30"
    with pytest.raises(ValueError):
        iso(dt.datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        parse_iso("2026-01-01T00:00:00")
