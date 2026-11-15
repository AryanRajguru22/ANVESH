"""Offline account management: the recovery / break-glass path (needs access to the server's data directory).

    .venv\\Scripts\\python.exe -I scripts\\manage_users.py --data-dir <dir> list
    .venv\\Scripts\\python.exe -I scripts\\manage_users.py --data-dir <dir> create --username alice --display-name "Alice" --role officer
    .venv\\Scripts\\python.exe -I scripts\\manage_users.py --data-dir <dir> reset-password --username admin
    .venv\\Scripts\\python.exe -I scripts\\manage_users.py --data-dir <dir> set-active --username alice --active false
    .venv\\Scripts\\python.exe -I scripts\\manage_users.py --data-dir <dir> set-role --username alice --role reviewer

WHY IT EXISTS. The HTTP API needs a second administrator to approve privileged accounts, and an administrator who is locked out or the
only one cannot approve anything. Whoever can run this on the host already controls the database and the evidence files, so host access
IS the root of trust here. Every change is written to the hash-chained audit log with actor `cli:<operating-system user>`.

Rules: `--data-dir` is mandatory (there is no default, so it can never touch `outputs/app` by accident); passwords are generated (or read
from a hidden prompt) and shown once; created or reset accounts must change the password at first login; the last active administrator
cannot be deactivated or demoted; nothing is deleted. The login throttle lives in the running server's memory: after a reset, an old
lockout expires on its own within 5 minutes, or restart the server / use the administrator `unlock` API function.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvesh.app.audit import AuditLog  # noqa: E402
from anvesh.app.auth import ROLES, AuthService, PRIVILEGED_ROLES  # noqa: E402
from anvesh.app.db import AppDatabase  # noqa: E402


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1"):
        return True
    if text.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _password(args) -> tuple:
    """(password, generated?)"""
    if args.prompt_password:
        p1, p2 = getpass.getpass("new password: "), getpass.getpass("repeat: ")
        if p1 != p2:
            raise SystemExit("passwords differ")
        return p1, False
    return secrets.token_urlsafe(14), True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="the application data directory that holds anvesh_app.sqlite3 (required; no default)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create")
    c.add_argument("--username", required=True)
    c.add_argument("--display-name", required=True)
    c.add_argument("--role", required=True, choices=ROLES)
    c.add_argument("--prompt-password", action="store_true", help="type the password instead of generating one")
    r = sub.add_parser("reset-password")
    r.add_argument("--username", required=True)
    r.add_argument("--prompt-password", action="store_true")
    a = sub.add_parser("set-active")
    a.add_argument("--username", required=True)
    a.add_argument("--active", required=True, type=_bool)
    dm = sub.add_parser("set-demo", help="flag or unflag an account as a DEMO account (demo accounts cannot act on operational material)")
    dm.add_argument("--username", required=True)
    dm.add_argument("--demo", required=True, type=_bool)
    ro = sub.add_parser("set-role")
    ro.add_argument("--username", required=True)
    ro.add_argument("--role", required=True, choices=ROLES)
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    db_path = data_dir / "anvesh_app.sqlite3"
    if not db_path.is_file():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    actor = f"cli:{getpass.getuser()}"
    db = AppDatabase(db_path)
    audit = AuditLog(db)
    auth = AuthService(db, audit)
    try:
        if args.cmd == "list":
            for u in auth.list_users():
                print(f"{u['username']:20s} {u['role']:14s} {'active' if u['active'] else 'DISABLED':9s} {'DEMO ' if u['is_demo'] else '     '}"
                      f"{'must-change-password' if u['must_change_password'] else ''}")
            return 0
        try:
            if args.cmd == "create":
                pw, generated = _password(args)
                user = auth.create_user(args.username, args.display_name, args.role, pw, actor, must_change_password=True)
                audit.append(actor, "account.offline_cli", "user", user["user_id"], {"command": "create", "role": args.role, "privileged": args.role in PRIVILEGED_ROLES})
                print(f"created {user['username']} ({user['role']}).", f"Temporary password (shown once): {pw}" if generated else "")
            else:
                user = auth.get_by_username(args.username)
                if user is None:
                    print(f"no such user {args.username!r}", file=sys.stderr)
                    return 1
                if args.cmd == "reset-password":
                    pw, generated = _password(args)
                    auth.update_user(user["user_id"], actor, password=pw, must_change_password=True)
                    audit.append(actor, "account.offline_cli", "user", user["user_id"], {"command": "reset-password"})
                    print("password reset; all sessions revoked.", f"Temporary password (shown once): {pw}" if generated else "")
                elif args.cmd == "set-active":
                    auth.update_user(user["user_id"], actor, active=args.active)
                    audit.append(actor, "account.offline_cli", "user", user["user_id"], {"command": "set-active", "active": args.active})
                    print(f"{user['username']} is now {'active' if args.active else 'disabled'}.")
                elif args.cmd == "set-demo":
                    auth.set_demo(user["user_id"], args.demo, actor)
                    audit.append(actor, "account.offline_cli", "user", user["user_id"], {"command": "set-demo", "demo": args.demo})
                    print(f"{user['username']} is now {'a DEMO' if args.demo else 'a NON-demo'} account; sessions revoked.")
                elif args.cmd == "set-role":
                    auth.update_user(user["user_id"], actor, role=args.role)
                    audit.append(actor, "account.offline_cli", "user", user["user_id"], {"command": "set-role", "role": args.role})
                    print(f"{user['username']} is now {args.role}; sessions revoked.")
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
