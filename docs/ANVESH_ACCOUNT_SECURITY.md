# ANVESH account security (state as of 2026-10-21)

Scope: what the application actually does today for accounts, passwords, throttling, sessions and recovery, what is **not** done (MFA), and the design for finishing it.
Nothing here is a security certification; it has had no independent review. ANVESH remains **research / demo only** (`docs/ANVESH_RELEASE_READINESS.md`).

## 1. Who can create what

| Action | Who | How |
|---|---|---|
| Anything unauthenticated | nobody | `POST /users` needs a valid session (401 otherwise); non-administrators get 403 |
| Create an **investigator** or **reviewer** | any active administrator | immediate; the password is **temporary** (the user must change it at first sign-in) |
| Create an **officer** or **administrator** | a **request** by one administrator, approved by a **different** active administrator | `POST /users` -> 202 pending; `POST /users/requests/{id}/approve`. The requester cannot approve (403, audited); requests expire after 72 h; a request from an administrator who has since been deactivated or demoted is not applied; decided requests are immutable (database triggers) |
| Move an account into or out of officer/administrator | same two-person request | `PATCH /users/{id}` `{"role": ...}` |
| Re-enable an officer/administrator account; reset their password | same two-person request | |
| Reset an investigator/reviewer password | any administrator, immediate, temporary password | |
| Deactivate any account | any administrator, immediate (it only removes access) | not your own account; never the last active administrator |
| Change your own role / deactivate yourself | nobody | 409 |
| Change your own password | the user | `POST /auth/password` re-verifies the current password, refuses reuse, signs every session out |
| Offline / break-glass | whoever holds the server host | `scripts/manage_users.py --data-dir <dir> ...` (below) |

One change per `PATCH` (role **or** active **or** password) so an approval always describes exactly one thing. All of it is audited (`user.create`, `user.update`, `account.request*`, `auth.*`); neither passwords nor password hashes are ever written to the audit trail or returned by the API (a pending request stores only a scrypt hash).

**Who can create administrators:** a second administrator approving a request, or the offline CLI. **Who can create officers:** the same. A single-administrator installation therefore needs the CLI to create its second administrator or its first officer - deliberately.

## 2. Recovery of a locked-out or sole administrator

1. A *throttle* lockout clears by itself after 5 minutes, on a server restart, or by another administrator: `POST /users/{id}/unlock` (clears the counters of **this** server process only).
2. A forgotten password or no administrator able to approve: on the server host run `scripts/manage_users.py --data-dir <data dir> reset-password --username <admin>` (a random temporary password is printed once; sessions are revoked; the user must change it at first sign-in) or `create --role administrator`. `--data-dir` is mandatory (there is no default, so it cannot touch another installation by accident). Every action is audited as actor `cli:<operating-system user>`.
3. The CLI refuses to deactivate or demote the last active administrator and never deletes anything. Host access is the root of trust: whoever can run it can already read the database and the evidence files; protect the host and its backups accordingly.

## 3. Password storage, throttling and sessions

- **Hashing**: scrypt N=2^17, r=8, p=1 (about 128 MiB and 0.4 s per hash on the development laptop; OWASP's recommended minimum for scrypt), 16-byte random salt, parameters stored in the hash string. Hashes made with weaker cost still verify and are **upgraded at the next successful login**. A stored cost beyond a sanity ceiling is treated as invalid (a tampered row cannot demand gigabytes). At most two hashes run concurrently, so a login burst cannot exhaust memory. (Before this change the cost was N=2^14.)
- **Password policy**: 10-256 characters; a short deny-list of very common passwords; not the username with little added; the new password must differ from the current one.
- **Throttling** (`LoginThrottle`): failures counted per (username, source address) - 5 locks that pair; per source - 30 locks that address; per username - 25 (any sources) locks that name. Each lock lasts 5 minutes on a monotonic clock; unlocked failures are forgotten after 5 minutes; every table is bounded (2,000 entries, oldest evicted). Unknown usernames are tracked identically and return the same generic message, so the response never reveals whether an account exists; the lock itself is audited (`auth.lockout`) with the scope and, only for an existing account, its name. Pairing username with source means five bad guesses from elsewhere cannot lock someone out of their own desk.
  **Limits, stated plainly**: the counters live in the server process's memory - they are lost on restart, they are not shared between server processes, and behind a reverse proxy every client appears with the proxy's address (so the per-pair and per-source limits degrade to per-proxy until a trusted-proxy setting exists). A distributed attack on one username can still lock that name for 5 minutes (an administrator can unlock it).
- **Sessions**: 256-bit random tokens, only their SHA-256 stored; 8 h lifetime; **30 minutes idle timeout** (last use refreshed at most once a minute); HttpOnly + `SameSite=Strict` cookie (the `Secure` flag only with `ANVESH_SECURE_COOKIES=1`, which is required behind HTTPS; `run_app.py` warns when bound to a non-loopback address without it); cookie-authenticated writes also need `X-Requested-With` and a matching `Origin`. A role change, deactivation or password change **revokes the user's sessions**; expired or old revoked sessions are purged at login. Tokens of any malformed shape simply do not authenticate.
- **Temporary passwords**: while `must_change_password` is set the API answers 403 `password_change_required` to everything except `/auth/me`, `/auth/password`, `/auth/logout`, `/health`, `/ready`, and the dashboard shows only the change-password screen.
- **Secrets in logs**: passwords, hashes, tokens and (when they exist) MFA codes are not logged or audited; failed-login audit rows hold the (length-capped) attempted username only.

## 4. Demo accounts

`run_app.py --demo-users` creates one account per role with random passwords in `initial_credentials.txt`, whose first line says **DEMO ACCOUNTS - FOR REVIEW AND TESTING ONLY**. They are flagged `is_demo` in the database, labelled "(DEMO ACCOUNT)", shown with a banner in the dashboard, and are confined to demo/simulated material. **Enforced for WRITES** (403, by the route and, for cameras and cases, also by the service layer): cases (create and every action), registry look-up, hand-off, plate entry / OCR / correction / verification, incident status / assignment / notes / cause, starting a session, registering, changing or reclassifying a camera to or from `operational`, alert retry, evidence purge, and `PUT /settings` for retention, the offence catalogue and alert rules. **Not enforced for READS**: a demo account can still view operational incidents, evidence and cases it has the role permission to see. **Demo status cannot be laundered**: any account a demo administrator creates (directly, or through a request - the origin of the request decides, whoever approves it) is itself a demo account, and a demo administrator can only manage, unlock, approve or reject things belonging to demo accounts. Without `--demo-users` only one real `admin` is created. No default or production credential exists anywhere in the source. Accounts that existed before schema v4 are not retro-flagged (the migration cannot know which ones were demo): flag them with `scripts/manage_users.py --data-dir <dir> set-demo --username <name> --demo true` (sessions are revoked) or disable them. `run_app.py` warns at start-up when `initial_credentials.txt` is still present. **The existing `outputs/app` database (2026-10-18) holds six `--demo-users` accounts that migrate as NON-demo, and their plaintext first-run passwords are in `outputs/app/initial_credentials.txt`; they were not changed by this work - flag or disable them and delete that file before any non-demo use.**

## 5. MFA - status: NOT IMPLEMENTED (no fake MFA)

Reason: a TOTP secret has to be stored in recoverable form, so protecting it needs authenticated encryption with a key held outside the database. The pinned environment contains no vetted cryptography library (`cryptography`, `pyotp`, ... are not installed) and the Python standard library has no AEAD primitive; hand-rolling one is exactly what should not be done, and storing the secrets in plaintext beside the password hashes would make a database leak defeat the second factor. I did not install packages into the pinned environment. The TOTP algorithm itself (RFC 6238, HMAC-SHA1 from the standard library) is the easy part; secure storage and recovery are what remain.

**Proposed design (to implement once a dependency is approved):**
1. Dependency: `cryptography` (Apache-2.0/BSD) for AES-GCM/Fernet of the secret at rest; `pyotp` (MIT) or ~40 lines of stdlib RFC 6238 verified against the RFC test vectors. Check both against `requirements-lock.txt` and record versions/licences.
2. Key: 32 random bytes in `<data dir>/mfa.key` (mode 0600), created on first use, or `ANVESH_MFA_KEY`; never in the database or repository; backed up separately from the database; losing it forces re-enrolment.
3. Enrolment (authenticated user, re-enters the password): server generates a secret, shows an `otpauth://` URI/QR **once**, stores it encrypted but **inactive**; the user submits a valid code to activate; 10 single-use recovery codes are shown once and stored only as scrypt hashes.
4. Login: password first, then a code; a code is accepted once (store the last accepted time-step per user to stop replay), with +-1 step tolerance; failed codes use the same throttle and audit; the session is issued only after both factors.
5. Reset / disable: only by a **different** administrator through the existing two-person request kind (`reset_mfa`), or the offline CLI; both audited; the user must re-enrol; disabling your own MFA requires re-authentication (password + code).
6. Migration: `users.mfa_secret_enc`, `mfa_enabled`, `mfa_last_step`, `mfa_recovery_hashes` (schema v5, with the existing backup-before-migrate); MFA optional per account first, then **mandatory for administrator and officer** once enrolment and recovery have been exercised for every such account.
7. Tests to write: RFC 6238 vectors, replay refused, skew window, enrolment not active until a code verifies, recovery code single use, reset needs a second administrator, secrets never in API/audit/logs, wrong-key and corrupted-ciphertext fail closed.

## 6. Residual risks

Two colluding administrators (or anyone with host access) can still create privileged accounts - audited, not preventable in code. No MFA. Throttling is per process and per source address. HTTP only on localhost; any exposure needs a TLS proxy, `ANVESH_SECURE_COOKIES=1`, and a connection cap at the proxy. Uploaded video is parsed by FFmpeg/OpenCV. No independent review or penetration test has been done.
