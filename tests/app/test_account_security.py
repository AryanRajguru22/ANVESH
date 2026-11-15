"""Account security: password hashing, throttling, two-person provisioning, temporary passwords, sessions, demo accounts, the offline CLI.

What this file does NOT show: MFA (not implemented), persistent or multi-process lockout (the counters are in memory), or that any of
this has been through an independent security review. It shows the implemented behaviour is what the documentation says.
"""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from anvesh.app import auth as auth_module
from anvesh.app.audit import AuditLog
from anvesh.app.auth import SCRYPT_PRODUCTION, AuthError, AuthService, LoginThrottle, check_password_policy, hash_password, needs_rehash, verify_password
from anvesh.app.bootstrap import ensure_users
from anvesh.app.cases import CaseError
from anvesh.app.runner import SessionManager
from anvesh.app.server import AppServer
from anvesh.app.synthetic import ScriptedTracker

from .conftest import seed_incident
from .test_api import Client, api, client  # noqa: F401  (api is a fixture)

REPO = Path(__file__).resolve().parents[2]


# -- password hashing ---------------------------------------------------------------------------------------------------------


def test_production_scrypt_parameters_are_the_owasp_minimum_not_the_cheap_test_setting():
    assert SCRYPT_PRODUCTION == {"n": 2 ** 17, "r": 8, "p": 1, "dklen": 32}
    assert auth_module.SCRYPT["n"] == 2 ** 10  # the autouse test fixture lowers the LIVE dict only
    assert auth_module._SCRYPT_MAXMEM >= 128 * 8 * 2 ** 17  # the memory ceiling really allows the production cost


def test_old_weaker_hashes_still_verify_and_are_upgraded_at_the_next_successful_login(auth, db, monkeypatch):
    auth.create_user("olga", "Olga", "officer", "long-enough-pw")
    old = db.one("SELECT password_hash FROM users WHERE username = 'olga'")["password_hash"]
    assert old.startswith("scrypt$1024$") and not needs_rehash(old)
    monkeypatch.setitem(auth_module.SCRYPT, "n", 2 ** 11)  # "production got stronger"
    assert needs_rehash(old) and verify_password("long-enough-pw", old)
    auth.login("olga", "long-enough-pw")
    new = db.one("SELECT password_hash FROM users WHERE username = 'olga'")["password_hash"]
    assert new.startswith("scrypt$2048$") and new != old and verify_password("long-enough-pw", new)
    assert not auth.get_user(db.one("SELECT user_id FROM users WHERE username = 'olga'")["user_id"]).get("password_hash")


@pytest.mark.parametrize("stored", ["", "garbage", "scrypt$1$8$1$AA==$AA==", f"scrypt${2 ** 30}$8$1$AA==$AA==", "scrypt$1024$8$1$not-base64$x", "bcrypt$1024$8$1$AA==$AA=="])
def test_a_damaged_or_hostile_stored_hash_fails_closed_without_demanding_memory(stored):
    assert verify_password("anything-at-all", stored) is False
    assert needs_rehash(stored) is False


@pytest.mark.parametrize("pw,user,ok", [
    ("short", "bob", False), ("x" * 257, "bob", False), ("password123", "bob", False), ("BobBobBobBob", "bob", True), ("administrator1", "administrator", False), ("bob123456789", "bob", True),
    ("correct horse battery staple", "bob", True), (12345678901, "bob", False), (None, "bob", False)])
def test_password_policy(pw, user, ok):
    if ok:
        check_password_policy(pw, user)
    else:
        with pytest.raises(ValueError):
            check_password_policy(pw, user)


def test_changing_a_password_re_verifies_refuses_reuse_signs_every_session_out_and_is_audited(auth, db, audit):
    u = auth.create_user("pam", "Pam", "reviewer", "first-password-1")
    t1, _ = auth.login("pam", "first-password-1")
    t2, _ = auth.login("pam", "first-password-1")
    with pytest.raises(AuthError):
        auth.change_password(u["user_id"], "WRONG-current-pw", "second-password-2")
    with pytest.raises(ValueError, match="differ"):
        auth.change_password(u["user_id"], "first-password-1", "first-password-1")
    with pytest.raises(ValueError):
        auth.change_password(u["user_id"], "first-password-1", "password123")
    auth.change_password(u["user_id"], "first-password-1", "second-password-2")
    assert auth.authenticate(t1) is None and auth.authenticate(t2) is None
    with pytest.raises(AuthError):
        auth.login("pam", "first-password-1")
    assert auth.login("pam", "second-password-2")[1].username == "pam"
    actions = [r["action"] for r in audit.query(entity_type="user", entity_id=u["user_id"], limit=50)[0]]
    assert "auth.password_changed" in actions and "auth.password_change_failed" in actions
    assert "first-password-1" not in json.dumps(audit.query(limit=500)[0]) and "second-password-2" not in json.dumps(audit.query(limit=500)[0])


# -- throttling -----------------------------------------------------------------------------------------------------------


@pytest.fixture()
def throttled(db, audit):
    clock = [0.0]
    a = AuthService(db, audit, clock=lambda: clock[0])
    a.create_user("vic", "Vic", "officer", "victim-password-1")
    a.create_user("wes", "Wes", "officer", "wesley-password-1")
    return a, clock


def fail(a, user, src, n=1):
    for _ in range(n):
        with pytest.raises(AuthError):
            a.login(user, "definitely-wrong-pw", src)


def test_five_bad_guesses_from_one_source_do_not_lock_the_real_user_out_at_their_own_desk(throttled):
    a, _ = throttled
    fail(a, "vic", "203.0.113.9", 5)  # an attacker elsewhere
    with pytest.raises(AuthError) as e:
        a.login("vic", "victim-password-1", "203.0.113.9")  # that pair is locked, even with the right password
    assert e.value.status == 429
    assert a.login("vic", "victim-password-1", "10.0.0.5")[1].username == "vic"  # the legitimate user from their own address is fine


def test_a_source_that_sprays_many_usernames_is_locked_as_a_whole(throttled):
    a, _ = throttled
    for i in range(LoginThrottle.LIMITS["source"]):
        fail(a, f"guess{i}", "198.51.100.7")
    with pytest.raises(AuthError) as e:
        a.login("vic", "victim-password-1", "198.51.100.7")
    assert e.value.status == 429
    assert a.login("wes", "wesley-password-1", "10.0.0.6")[1].username == "wes"


def test_a_distributed_attack_on_one_username_is_stopped_but_clearable_by_an_administrator(throttled):
    a, _ = throttled
    for i in range(LoginThrottle.LIMITS["user"]):
        fail(a, "vic", f"192.0.2.{i}")
    with pytest.raises(AuthError) as e:
        a.login("vic", "victim-password-1", "10.9.9.9")
    assert e.value.status == 429
    assert a.unlock("vic", "admin1") >= 1
    assert a.login("vic", "victim-password-1", "10.9.9.9")[1].username == "vic"


def test_unknown_and_known_usernames_are_indistinguishable_to_the_caller(throttled):
    a, _ = throttled
    seen = []
    for name in ("vic", "no_such_user"):
        msgs = set()
        for i in range(7):
            try:
                a.login(name, "definitely-wrong-pw", "10.1.1.1")
            except AuthError as e:
                msgs.add((e.status, str(e)))
                seen.append(e.status)
        assert msgs == {(401, "invalid credentials"), (429, "too many failed attempts; try again later")}
    assert seen[:5] == [401] * 5 and seen[5:7] == [429, 429]


def test_a_lockout_expires_on_the_monotonic_clock_and_success_resets_the_counters(throttled):
    a, clock = throttled
    fail(a, "vic", "10.2.2.2", 5)
    with pytest.raises(AuthError):
        a.login("vic", "victim-password-1", "10.2.2.2")
    clock[0] += LoginThrottle.LOCKOUT_S - 1
    with pytest.raises(AuthError):
        a.login("vic", "victim-password-1", "10.2.2.2")
    clock[0] += 2
    assert a.login("vic", "victim-password-1", "10.2.2.2")[1].username == "vic"
    fail(a, "vic", "10.2.2.2", 4)  # a success wiped the earlier count, so four more is still below the limit
    assert a.login("vic", "victim-password-1", "10.2.2.2")[1].username == "vic"
    fail(a, "wes", "10.3.3.3", 3)
    clock[0] += LoginThrottle.WINDOW_S + 1  # old, unlocked failures are forgotten
    fail(a, "wes", "10.3.3.3", 4)
    assert a.login("wes", "wesley-password-1", "10.3.3.3")[1].username == "wes"


def test_lockouts_are_audited_once_per_scope_without_attacker_controlled_strings_and_without_secrets(throttled, audit):
    a, _ = throttled
    fail(a, "vic", "10.4.4.4", 6)
    fail(a, "ghost-" + "x" * 50, "10.4.4.5", 6)
    rows = [r for r in audit.query(action="auth.lockout", limit=50)[0]]
    scopes = [(r["details"]["scope"], r["details"]["username"]) for r in rows]
    assert ("user_source", "vic") in scopes and scopes.count(("user_source", "vic")) == 1
    assert ("user_source", None) in scopes  # the unknown name is recorded as unknown, not as the string the attacker chose
    blob = json.dumps(audit.query(limit=500)[0])
    assert "definitely-wrong-pw" not in blob and "ghost-xxxx" not in json.dumps(rows)


def test_throttle_memory_is_bounded_under_a_flood_of_distinct_names_and_sources():
    t = LoginThrottle(clock=lambda: 0.0)
    for i in range(LoginThrottle.MAX_ENTRIES * 2):
        t.failure(f"u{i}", f"s{i}")
    assert all(n <= LoginThrottle.MAX_ENTRIES for n in t.sizes().values())


def test_the_posture_statement_is_honest_about_scope_and_mfa(auth):
    p = auth.posture()
    assert "in-process memory" in p["login_throttle"]["scope"] and "lost on restart" in p["login_throttle"]["scope"]
    assert p["mfa"] == "not implemented"


# -- sessions -------------------------------------------------------------------------------------------------------------------


def test_malformed_tokens_never_authenticate(auth):
    for t in (None, "", " ", "x" * 300, 12345, ["a"], {"a": 1}, "a b", "\x00"):
        assert auth.authenticate(t) is None


def test_sessions_end_on_idle_timeout_role_change_deactivation_and_logout(db, audit):
    a = AuthService(db, audit, idle_timeout_s=60)
    u = a.create_user("sam", "Sam", "reviewer", "samuel-password-1")
    t, _ = a.login("sam", "samuel-password-1")
    assert a.authenticate(t).role == "reviewer"
    db.execute("UPDATE auth_sessions SET last_seen_at = '2000-01-01T00:00:00.000Z'")  # unused for decades
    assert a.authenticate(t) is None
    assert db.one("SELECT revoked FROM auth_sessions")["revoked"] == 1  # idle expiry revokes, it does not merely hide
    t, _ = a.login("sam", "samuel-password-1")
    a.update_user(u["user_id"], "admin", role="investigator")
    assert a.authenticate(t) is None  # a role change forces a fresh login
    t, _ = a.login("sam", "samuel-password-1")
    assert a.authenticate(t).role == "investigator"
    a.update_user(u["user_id"], "admin", active=False)
    assert a.authenticate(t) is None
    a.update_user(u["user_id"], "admin", active=True)
    t, user = a.login("sam", "samuel-password-1")
    a.logout(t, user)
    assert a.authenticate(t) is None


def test_last_seen_is_refreshed_at_most_once_a_minute(db, audit):
    a = AuthService(db, audit)
    a.create_user("tia", "Tia", "reviewer", "tiana-password-1")
    t, _ = a.login("tia", "tiana-password-1")
    before = db.one("SELECT last_seen_at FROM auth_sessions")["last_seen_at"]
    for _ in range(5):
        a.authenticate(t)
    assert db.one("SELECT last_seen_at FROM auth_sessions")["last_seen_at"] == before
    db.execute("UPDATE auth_sessions SET last_seen_at = '2000-01-01T00:00:00.000Z'")
    a.idle = 0
    a._idle = 0  # no idle limit for this check
    assert a.authenticate(t) is not None
    assert db.one("SELECT last_seen_at FROM auth_sessions")["last_seen_at"] > "2020"


def test_old_sessions_are_purged_so_the_table_does_not_grow_forever(db, audit):
    a = AuthService(db, audit)
    a.create_user("uma", "Uma", "reviewer", "umaima-password-1")
    for i in range(3):
        db.execute("INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at, revoked) SELECT ?, user_id, '2000-01-01T00:00:00.000Z', "
                   "'2000-01-02T00:00:00.000Z', 0 FROM users", (f"old{i}",))
    a.login("uma", "umaima-password-1")
    assert db.one("SELECT COUNT(*) AS n FROM auth_sessions")["n"] == 1


# -- two-person provisioning over HTTP ----------------------------------------------------------------------------------------


def make_admin2(svc):
    svc.auth.create_user("admin2", "Admin Two", "administrator", "pw-admin2-long")


def test_only_an_authenticated_administrator_can_create_accounts(api):
    base, svc = api
    body = {"username": "mallory", "display_name": "M", "role": "investigator", "password": "mallory-password-1"}
    assert Client(base).call("POST", "/users", body)[0] == 401
    for who in ("inv1", "rev1", "off1"):
        assert client(base, who).call("POST", "/users", body)[0] == 403
    assert svc.auth.get_by_username("mallory") is None


def test_investigators_and_reviewers_are_created_at_once_with_a_temporary_password(api):
    base, svc = api
    adm = client(base, "admin1")
    for role in ("investigator", "reviewer"):
        st, r, _ = adm.call("POST", "/users", {"username": f"new_{role}", "display_name": "N", "role": role, "password": "temp-password-123"})
        assert st == 201 and r["status"] == "created" and r["user"]["must_change_password"] == 1 and "password_hash" not in json.dumps(r)
    assert any(a["action"] == "user.create" and a["details"]["temporary_password"] for a in svc.audit.query(action="user.create", limit=20)[0])


def test_officer_and_administrator_accounts_need_a_second_administrator(api):
    base, svc = api
    adm = client(base, "admin1")
    st, r, _ = adm.call("POST", "/users", {"username": "newofficer", "display_name": "New Officer", "role": "officer", "password": "temp-password-123"})
    assert st == 202 and r["status"] == "pending" and svc.auth.get_by_username("newofficer") is None
    rid = r["request"]["request_id"]
    assert "password_hash" not in json.dumps(r) and "temp-password-123" not in json.dumps(r)
    # the requester cannot approve their own request, and the refusal is audited
    st, body, _ = adm.call("POST", f"/users/requests/{rid}/approve", {})
    assert st == 403 and svc.auth.get_by_username("newofficer") is None
    assert svc.audit.query(action="account.request_self_approval_refused", limit=5)[0]
    # a duplicate pending request is refused
    assert adm.call("POST", "/users", {"username": "newofficer", "display_name": "x", "role": "officer", "password": "temp-password-123"})[0] == 409
    # non-administrators cannot see or decide requests
    assert client(base, "off1").call("POST", f"/users/requests/{rid}/approve", {})[0] == 403
    assert client(base, "off1").call("GET", "/users/requests")[0] == 403
    make_admin2(svc)
    adm2 = client(base, "admin2")
    st, done, _ = adm2.call("POST", f"/users/requests/{rid}/approve", {"note": "verified identity in person"})
    assert st == 200 and done["status"] == "approved"
    created = svc.auth.get_by_username("newofficer")
    assert created["role"] == "officer" and created["must_change_password"] == 1 and created["active"] == 1
    # decided requests are final
    assert adm2.call("POST", f"/users/requests/{rid}/approve", {})[0] == 409
    with pytest.raises(sqlite3.DatabaseError):
        svc.db.execute("UPDATE account_requests SET status = 'pending' WHERE request_id = ?", (rid,))
    with pytest.raises(sqlite3.DatabaseError):
        svc.db.execute("DELETE FROM account_requests")
    # audited end to end, and neither the password nor its hash is in the audit trail
    actions = [a["action"] for a in svc.audit.query(limit=500)[0]]
    assert {"account.request", "account.request_approved", "user.create"} <= set(actions)
    blob = json.dumps(svc.audit.query(limit=500)[0])
    assert "temp-password-123" not in blob and "scrypt$" not in blob


def test_the_temporary_password_must_be_changed_before_anything_else_works(api):
    base, svc = api
    adm = client(base, "admin1")
    adm.call("POST", "/users", {"username": "temp_user", "display_name": "T", "role": "investigator", "password": "temp-password-123"})
    c = Client(base)
    st, body, h = c.call("POST", "/auth/login", {"username": "temp_user", "password": "temp-password-123"})
    assert st == 200 and body["user"]["must_change_password"] is True
    c.token = body["token"]
    assert c.call("GET", "/auth/me")[0] == 200
    for path in ("/overview", "/incidents", "/cases", "/cameras"):
        st, resp, _ = c.call("GET", path)
        assert st == 403 and resp["error"]["code"] == "password_change_required", path
    assert c.call("POST", "/auth/password", {"current_password": "temp-password-123", "new_password": "temp-password-123"})[0] == 422  # reuse refused
    assert c.call("POST", "/auth/password", {"current_password": "WRONG-password-1", "new_password": "a-brand-new-pass-9"})[0] == 401
    st, resp, h = c.call("POST", "/auth/password", {"current_password": "temp-password-123", "new_password": "a-brand-new-pass-9"})
    assert st == 200 and "Max-Age=0" in h["Set-Cookie"]
    assert c.call("GET", "/overview")[0] == 401  # every old session is gone
    st, body, _ = c.call("POST", "/auth/login", {"username": "temp_user", "password": "a-brand-new-pass-9"})
    assert st == 200 and body["user"]["must_change_password"] is False
    c.token = body["token"]
    assert c.call("GET", "/overview")[0] == 200


def test_role_changes_reactivation_and_resets_for_privileged_accounts_are_requests_too(api):
    base, svc = api
    make_admin2(svc)
    adm, adm2 = client(base, "admin1"), client(base, "admin2")
    ids = {u["username"]: u["user_id"] for u in svc.auth.list_users()}
    # investigator -> reviewer: immediate. anything into/out of officer or administrator: a request
    assert adm.call("PATCH", f"/users/{ids['inv1']}", {"role": "reviewer"})[1]["status"] == "updated"
    st, r, _ = adm.call("PATCH", f"/users/{ids['rev2']}", {"role": "officer"})
    assert st == 202 and svc.auth.get_user(ids["rev2"])["role"] == "reviewer"
    assert adm2.call("POST", f"/users/requests/{r['request']['request_id']}/approve", {})[0] == 200
    assert svc.auth.get_user(ids["rev2"])["role"] == "officer"
    st, r, _ = adm.call("PATCH", f"/users/{ids['off2']}", {"role": "investigator"})  # demoting an officer is also a request
    assert st == 202
    adm2.call("POST", f"/users/requests/{r['request']['request_id']}/reject", {"note": "not now"})
    assert svc.auth.get_user(ids["off2"])["role"] == "officer"
    # reset of a privileged password: a request; of an ordinary one: immediate and temporary
    st, r, _ = adm.call("PATCH", f"/users/{ids['off1']}", {"password": "reset-password-123"})
    assert st == 202
    st, r2, _ = adm.call("PATCH", f"/users/{ids['rev1']}", {"password": "reset-password-123"})
    assert st == 200 and r2["user"]["must_change_password"] == 1
    # deactivation is immediate (it only removes access) but re-activating a privileged account is a request
    assert adm.call("PATCH", f"/users/{ids['off2']}", {"active": False})[1]["status"] == "updated"
    st, r3, _ = adm.call("PATCH", f"/users/{ids['off2']}", {"active": True})
    assert st == 202 and svc.auth.get_user(ids["off2"])["active"] == 0
    # exactly one change per call
    assert adm.call("PATCH", f"/users/{ids['inv1']}", {"role": "reviewer", "active": False})[0] == 422


def test_the_last_active_administrator_can_never_be_removed(api):
    base, svc = api
    ids = {u["username"]: u["user_id"] for u in svc.auth.list_users()}
    adm = client(base, "admin1")
    assert adm.call("PATCH", f"/users/{ids['admin1']}", {"active": False})[0] == 409  # self
    make_admin2(svc)
    ids["admin2"] = svc.auth.get_by_username("admin2")["user_id"]
    assert svc.auth.update_user(ids["admin2"], "x", active=False)["active"] == 0  # one of two may go
    with pytest.raises(ValueError, match="last active administrator"):
        svc.auth.update_user(ids["admin1"], "x", active=False)  # direct service call
    with pytest.raises(ValueError, match="last active administrator"):
        svc.auth.update_user(ids["admin1"], "x", role="officer")


def test_pending_requests_expire_and_a_request_from_a_since_removed_administrator_is_not_applied(api):
    base, svc = api
    make_admin2(svc)
    adm, adm2 = client(base, "admin1"), client(base, "admin2")
    rid = adm.call("POST", "/users", {"username": "late_officer", "display_name": "L", "role": "officer", "password": "temp-password-123"})[1]["request"]["request_id"]
    svc.db.execute("UPDATE account_requests SET expires_at = '2000-01-01T00:00:00.000Z' WHERE request_id = ?", (rid,))
    st, body, _ = adm2.call("POST", f"/users/requests/{rid}/approve", {})
    assert st == 409 and "expired" in body["error"]["message"] and svc.auth.get_by_username("late_officer") is None
    rid2 = adm.call("POST", "/users", {"username": "orphan_officer", "display_name": "O", "role": "officer", "password": "temp-password-123"})[1]["request"]["request_id"]
    svc.auth.update_user(svc.auth.get_by_username("admin1")["user_id"], "x", active=False)  # the requester loses administrator access
    st, done, _ = adm2.call("POST", f"/users/requests/{rid2}/approve", {})
    assert done["status"] == "failed" and "no longer an active administrator" in done["result"] and svc.auth.get_by_username("orphan_officer") is None


def test_unlock_is_administrator_only_and_audited(api):
    base, svc = api
    ids = {u["username"]: u["user_id"] for u in svc.auth.list_users()}
    for _ in range(5):
        Client(base).call("POST", "/auth/login", {"username": "inv1", "password": "wrong-wrong-wrong"})
    assert Client(base).call("POST", "/auth/login", {"username": "inv1", "password": "pw-inv1-long"})[0] == 429
    assert client(base, "off1").call("POST", f"/users/{ids['inv1']}/unlock", {})[0] == 403
    st, r, _ = client(base, "admin1").call("POST", f"/users/{ids['inv1']}/unlock", {})
    assert st == 200 and r["cleared"] >= 1 and "this server process only" in r["scope"]
    assert Client(base).call("POST", "/auth/login", {"username": "inv1", "password": "pw-inv1-long"})[0] == 200
    assert svc.audit.query(action="auth.unlock", limit=5)[0]


# -- demo accounts ---------------------------------------------------------------------------------------------------------------


def test_demo_accounts_are_flagged_labelled_and_can_never_act_on_operational_cases(services, tmp_path):
    created = ensure_users(services, demo_users=True)
    assert "admin" in created and "officer" in created
    users = {u["username"]: u for u in services.auth.list_users()}
    assert all(u["is_demo"] == 1 and "(DEMO ACCOUNT)" in u["display_name"] for u in users.values())
    creds = (services.config.data_dir / "initial_credentials.txt").read_text()
    assert creds.startswith("# DEMO ACCOUNTS - FOR REVIEW AND TESTING ONLY")
    pw = {l.split("\t")[0]: l.split("\t")[2] for l in creds.splitlines() if "\t" in l and not l.startswith("#")}
    investigator = services.auth.login("investigator", pw["investigator"])[1]
    assert investigator.is_demo
    op = seed_incident(services, data_class="operational", key="demo-op", camera_id="cam-demo-op")
    with pytest.raises(CaseError) as e:
        services.cases.create(op["incident_id"], "STOP-OBS", investigator)
    assert e.value.status == 403 and "demo accounts" in str(e.value)
    dm = seed_incident(services, data_class="demo", key="demo-dm", camera_id="cam-demo-dm")
    assert services.cases.create(dm["incident_id"], "STOP-OBS", investigator)["state"] == "detected"  # demo work is fine


def test_demo_accounts_cannot_register_operational_cameras_or_look_up_operational_registries_over_http(services):
    ensure_users(services, demo_users=True)
    pw = {l.split("\t")[0]: l.split("\t")[2] for l in (services.config.data_dir / "initial_credentials.txt").read_text().splitlines() if "\t" in l and not l.startswith("#")}
    server = AppServer(services, SessionManager(services, tracker_factory=lambda cam, cfg: ScriptedTracker("stopped_car")), host="127.0.0.1", port=0)
    server.start()
    try:
        c = Client(f"http://127.0.0.1:{server.port}")
        st, body, _ = c.call("POST", "/auth/login", {"username": "admin", "password": pw["admin"]})
        c.token = body["token"]
        st, resp, _ = c.call("POST", "/cameras", {"name": "x", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "operational"})
        assert st == 403 and resp["error"]["code"] == "demo_account"
    finally:
        server.stop()


def test_without_demo_users_only_one_real_administrator_is_created_and_nothing_is_flagged(services):
    assert ensure_users(services) == ["admin"]
    u = services.auth.get_by_username("admin")
    assert u["is_demo"] == 0 and "DEMO" not in u["display_name"]
    assert ensure_users(services) == []  # never re-created or reset on a later start
    assert not (services.config.data_dir / "initial_credentials.txt").read_text().startswith("# DEMO")


# -- status endpoint ------------------------------------------------------------------------------------------------------------


def test_system_status_states_the_product_mode_models_and_security_posture(api):
    base, svc = api
    st, body, _ = client(base, "admin1").call("GET", "/system/status")
    assert st == 200
    assert body["product"]["mode"] == "research_demo" and body["product"]["operational_use_authorised"] is False and body["product"]["reasons"]
    assert body["security"]["mfa"] == "not implemented" and "in-process memory" in body["security"]["login_throttle"]["scope"]
    m = body["models"]
    assert m["accident_model"]["available"] is False and "not implemented" in m["accident_model"]["status"]
    assert m["ocr"]["available"] is False and "manual entry" in m["ocr"]["status"]
    assert "NOT CLEARED" in m["vehicle_detector"]["licence_status"] and "not measured" in m["vehicle_detector"]["accuracy_on_your_cameras"]


# -- malformed authentication requests ------------------------------------------------------------------------------------------


def test_malformed_authentication_requests_are_clean_client_errors(api):
    base, _ = api
    c = Client(base)
    for body in ({}, {"username": "inv1"}, {"password": "x"}, {"username": 5, "password": "x"}, {"username": "inv1", "password": ["x"]},
                 {"username": "x" * 5000, "password": "y" * 5000}, {"username": None, "password": None}):
        assert c.call("POST", "/auth/login", body)[0] in (401, 422), body
    for h in ({"Authorization": "Bearer"}, {"Authorization": "Bearer " + "A" * 10000}, {"Authorization": "Basic abc"}, {"Authorization": "bearer   "},
              {"Cookie": "anvesh_session=" + "z" * 5000}, {"Cookie": "anvesh_session"}, {"Cookie": ";;;;"}):
        c2 = Client(base)
        st, _, _ = c2.call("GET", "/overview", headers=h)
        assert st == 401, h


# -- offline recovery CLI --------------------------------------------------------------------------------------------------------


def _cli():
    spec = importlib.util.spec_from_file_location("manage_users_cli", REPO / "scripts" / "manage_users.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_offline_cli_is_the_audited_recovery_path(services, capsys):
    cli = _cli()
    d = str(services.config.data_dir)
    services.auth.create_user("solo_admin", "Solo", "administrator", "solo-admin-pass-1")
    with pytest.raises(SystemExit):
        cli.main(["list"])  # --data-dir is mandatory: there is no default location to hit by accident
    assert cli.main(["--data-dir", str(services.config.data_dir / "nowhere"), "list"]) == 2
    assert cli.main(["--data-dir", d, "create", "--username", "breakglass", "--display-name", "Break Glass", "--role", "administrator"]) == 0
    out = capsys.readouterr().out
    temp = out.split("Temporary password (shown once): ")[1].split()[0]
    u = services.auth.get_by_username("breakglass")
    assert u["role"] == "administrator" and u["must_change_password"] == 1
    assert services.auth.login("breakglass", temp)[1].must_change_password
    # recovery of a locked-out administrator: reset the password offline
    assert cli.main(["--data-dir", d, "reset-password", "--username", "solo_admin"]) == 0
    temp2 = capsys.readouterr().out.split("Temporary password (shown once): ")[1].split()[0]
    with pytest.raises(AuthError):
        services.auth.login("solo_admin", "solo-admin-pass-1")
    assert services.auth.login("solo_admin", temp2)[1].username == "solo_admin"
    # the last-administrator rule holds for the CLI too (two admins exist now, so removing one is fine, the second is not)
    assert cli.main(["--data-dir", d, "set-active", "--username", "breakglass", "--active", "false"]) == 0
    assert cli.main(["--data-dir", d, "set-active", "--username", "solo_admin", "--active", "false"]) == 1
    assert "last active administrator" in capsys.readouterr().err
    assert cli.main(["--data-dir", d, "set-role", "--username", "nobody", "--role", "reviewer"]) == 1
    actors = {a["actor"] for a in services.audit.query(action="account.offline_cli", limit=20)[0]}
    assert actors and all(x.startswith("cli:") for x in actors)
    assert services.audit.verify()["ok"]
    assert temp not in json.dumps(services.audit.query(limit=500)[0]) and temp2 not in json.dumps(services.audit.query(limit=500)[0])
    assert cli.main(["--data-dir", d, "list"]) == 0 and "solo_admin" in capsys.readouterr().out


# -- schema v4 ------------------------------------------------------------------------------------------------------------------


def test_schema_v4_migration_with_backup_keeps_existing_users_and_marks_them_non_demo(tmp_path):
    from anvesh.app import db as dbmod

    path = tmp_path / "old.sqlite3"
    d = dbmod.AppDatabase(path)
    d.execute("INSERT INTO users(user_id, username, display_name, role, password_hash, active, created_at) VALUES ('U1','legacy','Legacy','officer','scrypt$1024$8$1$AA==$AA==',1,'t')")
    d.execute("DROP TABLE account_requests")
    d.execute("ALTER TABLE users DROP COLUMN must_change_password")
    d.execute("ALTER TABLE users DROP COLUMN is_demo")
    d.execute("ALTER TABLE users DROP COLUMN password_changed_at")
    d.execute("ALTER TABLE auth_sessions DROP COLUMN last_seen_at")
    d.execute("DELETE FROM schema_migrations WHERE version = 4")
    d.close()
    d2 = dbmod.AppDatabase(path)
    try:
        assert d2.schema_version() == 4
        assert path.with_name("old.pre-v4-migration.sqlite3.bak").is_file()
        row = d2.one("SELECT username, role, must_change_password, is_demo FROM users WHERE user_id = 'U1'")
        assert row == {"username": "legacy", "role": "officer", "must_change_password": 0, "is_demo": 0}  # existing accounts are NOT retro-flagged as demo
    finally:
        d2.close()


# -- demo isolation: no laundering, no reclassification, no writes to operational material (added after review, 2026-10-21) ------------


def make_demo(svc, name, role):
    svc.auth.create_user(name, name, role, "pw-" + name + "-long", is_demo=True)


def test_a_demo_administrator_cannot_turn_a_camera_operational_or_touch_an_operational_one(api):
    base, svc = api
    make_demo(svc, "demoadm", "administrator")
    adm = client(base, "demoadm")
    st, cam, _ = adm.call("POST", "/cameras", {"camera_id": "demo-cam", "name": "d", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "simulated"})
    assert st == 201
    for patch in ({"data_class": "operational"}, {"data_class": "operational", "source_type": "rtsp", "source_uri": "rtsp://h/s", "nominal_fps": 25}):
        st, resp, _ = adm.call("PATCH", "/cameras/demo-cam", patch)
        assert st == 403 and resp["error"]["code"] == "demo_account", patch
    assert svc.cameras.get("demo-cam")["data_class"] == "simulated"
    # the service layer refuses on its own as well (not only the route)
    with pytest.raises(PermissionError):
        svc.cameras.update("demoadm", "demo-cam", {"data_class": "operational"}, "administrator", actor_is_demo=True)
    with pytest.raises(PermissionError):
        svc.cameras.create("demoadm", {"name": "x", "source_type": "synthetic", "source_uri": "synthetic:queue", "data_class": "operational"}, "administrator", actor_is_demo=True)
    # an existing operational camera cannot even be renamed or have a session started by a demo account
    svc.db.execute("UPDATE cameras SET data_class = 'operational' WHERE camera_id = 'demo-cam'")
    assert adm.call("PATCH", "/cameras/demo-cam", {"name": "renamed"})[0] == 403
    assert client(base, "admin1").call("PATCH", "/cameras/demo-cam", {"name": "renamed"})[0] == 200  # a real administrator can
    make_demo(svc, "demoinv", "investigator")
    assert client(base, "demoinv").call("POST", "/cameras/demo-cam/sessions", {})[0] == 403


def test_demo_status_cannot_be_laundered_into_real_credentials(api):
    base, svc = api
    make_demo(svc, "demoadm", "administrator")
    make_demo(svc, "demoadm2", "administrator")
    adm = client(base, "demoadm")
    # direct creation: the new account is a demo account too
    st, r, _ = adm.call("POST", "/users", {"username": "kid_inv", "display_name": "K", "role": "investigator", "password": "temp-password-123"})
    assert st == 201 and r["user"]["is_demo"] == 1
    # a request raised by a demo administrator and approved by ANOTHER demo administrator still yields a demo account
    st, r, _ = adm.call("POST", "/users", {"username": "kid_off", "display_name": "K", "role": "officer", "password": "temp-password-123"})
    assert st == 202
    assert client(base, "demoadm2").call("POST", f"/users/requests/{r['request']['request_id']}/approve", {})[0] == 200
    assert svc.auth.get_by_username("kid_off")["is_demo"] == 1
    # a REAL administrator approving a demo administrator's request also yields a demo account (the request's origin decides)
    st, r, _ = adm.call("POST", "/users", {"username": "kid_adm", "display_name": "K", "role": "administrator", "password": "temp-password-123"})
    assert client(base, "admin1").call("POST", f"/users/requests/{r['request']['request_id']}/approve", {})[0] == 200
    assert svc.auth.get_by_username("kid_adm")["is_demo"] == 1
    # a demo administrator can neither manage real accounts nor decide real requests, nor see them
    real_id = svc.auth.get_by_username("inv1")["user_id"]
    assert adm.call("PATCH", f"/users/{real_id}", {"password": "hijack-password-12"})[0] == 403
    assert adm.call("PATCH", f"/users/{real_id}", {"active": False})[0] == 403
    assert adm.call("POST", f"/users/{real_id}/unlock", {})[0] == 403
    st, r, _ = client(base, "admin1").call("POST", "/users", {"username": "real_off", "display_name": "R", "role": "officer", "password": "temp-password-123"})
    assert adm.call("POST", f"/users/requests/{r['request']['request_id']}/approve", {})[0] == 403
    assert adm.call("POST", f"/users/requests/{r['request']['request_id']}/reject", {})[0] == 403
    assert all(q["requested_by"] != "admin1" for q in adm.call("GET", "/users/requests")[1]["items"])
    assert svc.auth.get_by_username("real_off") is None and svc.auth.get_user(real_id)["active"] == 1
    # a real administrator may manage demo accounts
    kid = svc.auth.get_by_username("kid_inv")["user_id"]
    assert client(base, "admin1").call("PATCH", f"/users/{kid}", {"active": False})[0] == 200


def test_demo_accounts_cannot_write_to_operational_material_but_can_to_demo_material(api):
    base, svc = api
    make_demo(svc, "demoinv", "investigator")
    make_demo(svc, "demorev", "reviewer")
    op = seed_incident(svc, data_class="operational", key="iso-op", camera_id="cam-iso-op")
    dm = seed_incident(svc, data_class="demo", key="iso-dm", camera_id="cam-iso-dm")
    inv, rev = client(base, "demoinv"), client(base, "demorev")
    reading = svc.plates.manual_entry(op["incident_id"], "MH12AB1234", "inv1", "investigator", track_id="run-s000-t00001")
    refused = [
        inv.call("POST", f"/incidents/{op['incident_id']}/plates", {"text": "MH12AB1234"}),
        inv.call("POST", f"/incidents/{op['incident_id']}/plates/ocr", {"track_id": "run-s000-t00001"}),
        inv.call("POST", f"/plates/{reading['reading_id']}/correct", {"text": "MH12AB9999", "note": "x"}),
        rev.call("POST", f"/plates/{reading['reading_id']}/verify", {}),
        inv.call("PATCH", f"/incidents/{op['incident_id']}", {"status": "under_investigation"}),
        inv.call("POST", f"/incidents/{op['incident_id']}/notes", {"note": "demo note"}),
        inv.call("POST", f"/incidents/{op['incident_id']}/cause", {"cause": "c", "basis": "b"}),
        inv.call("POST", f"/incidents/{op['incident_id']}/cases", {"offence_code": "STOP-OBS"}),
    ]
    assert [r[0] for r in refused] == [403] * len(refused), [(r[0], r[1]) for r in refused]
    assert svc.incidents.get(op["incident_id"])["status"] == "detected" and len(svc.plates.history(op["incident_id"])) == 1
    # the same actions on DEMO material are fine
    assert inv.call("POST", f"/incidents/{dm['incident_id']}/notes", {"note": "demo note"})[0] == 200
    assert inv.call("POST", f"/incidents/{dm['incident_id']}/plates", {"text": "MH12AB1234"})[0] == 201
    assert inv.call("PATCH", f"/incidents/{dm['incident_id']}", {"status": "under_investigation"})[0] == 200
    # system-wide settings that shape operational behaviour, and evidence deletion, are closed to demo administrators too
    make_demo(svc, "demoadm", "administrator")
    da = client(base, "demoadm")
    assert da.call("PUT", "/settings", {"offences": [{"code": "X", "description": "d", "event_types": []}]})[0] == 403
    assert da.call("PUT", "/settings", {"retention": {"evidence_days": 90, "audit_days": 3650}})[0] == 403
    assert da.call("POST", "/retention/purge", {"plan_id": "x", "reason": "a perfectly good reason"})[0] == 403
    assert da.call("PUT", "/settings", {"detection_defaults": {"window_s": 12}})[0] == 200


def test_the_offline_cli_can_flag_an_account_that_predates_demo_flagging(services, capsys):
    cli = _cli()
    d = str(services.config.data_dir)
    services.auth.create_user("old_demo_officer", "Old", "officer", "old-demo-password-1")
    assert services.auth.get_by_username("old_demo_officer")["is_demo"] == 0
    assert cli.main(["--data-dir", d, "set-demo", "--username", "old_demo_officer", "--demo", "true"]) == 0
    assert services.auth.get_by_username("old_demo_officer")["is_demo"] == 1
    assert cli.main(["--data-dir", d, "set-demo", "--username", "old_demo_officer", "--demo", "false"]) == 0
    assert services.auth.get_by_username("old_demo_officer")["is_demo"] == 0
    assert cli.main(["--data-dir", d, "set-demo", "--username", "nobody", "--demo", "true"]) == 1
    assert any(a["details"].get("command") == "set-demo" for a in services.audit.query(action="account.offline_cli", limit=10)[0])
