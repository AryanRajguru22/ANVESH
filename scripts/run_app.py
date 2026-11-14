"""Launch the ANVESH application: API + dashboard on http://127.0.0.1:8750 by default.

    .venv\\Scripts\\python.exe scripts\\run_app.py                 # first run creates an 'admin' user
    .venv\\Scripts\\python.exe scripts\\run_app.py --demo-users    # also one account per role (demo / review use)

Credentials are generated randomly and written to <data-dir>/initial_credentials.txt (default data dir:
outputs/app, which is gitignored). Nothing secret lives in the repository.

Configuration is by environment variable (see anvesh/app/config.py): ANVESH_DATA_DIR, ANVESH_HOST, ANVESH_PORT,
ANVESH_MODEL, ANVESH_MEDIA_ROOTS, and -- only if you have them -- ANVESH_ALERT_WEBHOOK_URL / SMTP_*,
ANVESH_REGISTRY_URL, ANVESH_CHALLAN_URL. Unset integrations are shown as NOT CONNECTED in the dashboard.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvesh.app.bootstrap import ensure_users  # noqa: E402
from anvesh.app.config import load_config  # noqa: E402
from anvesh.app.server import serve  # noqa: E402
from anvesh.app.services import build_services  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="database / evidence / uploads directory (default: outputs/app)")
    ap.add_argument("--host", help="bind address (default 127.0.0.1; binding elsewhere exposes the API on the network)")
    ap.add_argument("--port", type=int, help="port (default 8750)")
    ap.add_argument("--demo-users", action="store_true", help="on first run create one account per role")
    ap.add_argument("--simple-home", action="store_true",
                    help="show the simple one-screen officer view at '/' (the full dashboard stays at /advanced). Normal sign-in applies")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(data_dir=args.data_dir)
    services = build_services(config)
    created = ensure_users(services, demo_users=args.demo_users)
    server = serve(services, host=args.host, port=args.port)
    from anvesh.app import warmup
    warmup.start(config.model_path)
    server.simple_home = bool(args.simple_home)
    print(f"\nANVESH is running:  http://{server.host}:{server.port}/")
    print(f"data directory:     {config.data_dir}")
    if created:
        print(f"created users:      {', '.join(created)}  (passwords in {config.data_dir / 'initial_credentials.txt'})")
    if not Path(config.model_path).is_file():
        print(f"WARNING: detector weights not found at {config.model_path}; sessions on real video cannot start.")
    print("MODE: RESEARCH / DEMO ONLY - not cleared for operational enforcement (see docs/ANVESH_RELEASE_READINESS.md).")
    creds_file = config.data_dir / "initial_credentials.txt"
    if creds_file.is_file() and not created:
        print(f"WARNING: {creds_file} still holds first-run passwords in plain text. Change those passwords, then delete the file. Accounts created before "
              "schema v4 are NOT flagged as demo accounts even if they came from --demo-users: flag them with scripts/manage_users.py set-demo, or disable them.")
    if args.demo_users:
        print("WARNING: --demo-users creates DEMO accounts (flagged; they cannot act on operational cases). Do not use them for real work.")
    if server.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: bound to a non-loopback address over plain HTTP." + ("" if config.secure_cookies else
              " Session cookies are NOT marked Secure (set ANVESH_SECURE_COOKIES=1 behind an HTTPS proxy). Login throttling is per source address, "
              "so behind a proxy every client appears as the proxy until a trusted-proxy setting exists."))
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        server.stop()
        services.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
