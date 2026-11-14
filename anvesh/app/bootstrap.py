"""First-run provisioning. No credential is ever stored in source: users are created with random
passwords that are written ONCE to `<data_dir>/initial_credentials.txt` (inside the gitignored outputs/
directory by default) and must be changed by the operator."""

from __future__ import annotations

import os
import secrets
import stat

DEMO_USERS = (("admin", "Administrator", "administrator"), ("investigator", "Investigator", "investigator"),
              ("reviewer", "Reviewer", "reviewer"), ("reviewer2", "Second Reviewer", "reviewer"),
              ("officer", "Enforcement Officer", "officer"), ("officer2", "Second Officer", "officer"))


def ensure_users(services, demo_users: bool = False) -> list:
    """Create the initial administrator (and, with `demo_users`, one account per role) if there are no users.
    Returns the created usernames; writes their passwords to the credentials file."""
    if services.auth.list_users():
        return []
    created, lines = [], []
    admin_pw = os.environ.get("ANVESH_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    plan = DEMO_USERS if demo_users else DEMO_USERS[:1]
    for username, display, role in plan:
        pw = admin_pw if username == "admin" else secrets.token_urlsafe(12)
        services.auth.create_user(username, display + (" (DEMO ACCOUNT)" if demo_users else ""), role, pw, actor="bootstrap", is_demo=demo_users)
        created.append(username)
        lines.append(f"{username}\t{role}\t{pw}")
    path = services.config.data_dir / "initial_credentials.txt"
    banner = ("# DEMO ACCOUNTS - FOR REVIEW AND TESTING ONLY. They are flagged in the database and cannot act on operational cases.\n"
              if demo_users else "")
    path.write_text(banner + "# ANVESH initial credentials (shown once). Change them, then delete this file.\n# username\trole\tpassword\n"
                    + "\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return created
