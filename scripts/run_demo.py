"""Start the ANVESH presentation demo on a DISPOSABLE data directory (never outputs/app).

    .venv\Scripts\python.exe scripts\run_demo.py --reset            # fresh data dir, camera registered, then browse to the URL
    .venv\Scripts\python.exe scripts\run_demo.py --reset --autostart # also start the video session immediately (acceptance runs)
    .venv\Scripts\python.exe scripts\run_demo.py                     # keep the previous demo data (cases, audit) and restart the server

What it does: builds the services against demo/data, creates the flagged DEMO accounts on first run (passwords in
demo/data/initial_credentials.txt), registers the camera `demo-motorway` (demo.mp4 + demo/camera_config.json) if it is not there,
and serves the dashboard. Nothing is processed until a session is started (Cameras -> Start, or --autostart) and
detection/tracking/events then happen WHILE the video plays (realtime_factor 1.0, no whole-video preprocessing).

MODE: RESEARCH / DEMONSTRATION ONLY - see demo/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anvesh.app.bootstrap import ensure_users  # noqa: E402
from anvesh.app.config import load_config  # noqa: E402
from anvesh.app.server import serve  # noqa: E402
from anvesh.app.services import build_services  # noqa: E402

DEMO_DIR = REPO / "demo"
MARKER = ".anvesh-demo-data"
CAMERA_ID = "demo-motorway"


def reset_data_dir(path: Path) -> None:
    """Delete a previous DEMO data dir. Refuses anything that is not inside <repo>/demo or lacks our marker file."""
    path = path.resolve()
    if DEMO_DIR.resolve() not in path.parents:
        raise SystemExit(f"refusing to reset {path}: only directories inside {DEMO_DIR} may be reset")
    if path.exists():
        if not (path / MARKER).is_file():
            raise SystemExit(f"refusing to reset {path}: it has no {MARKER} marker (was it created by run_demo.py?)")
        shutil.rmtree(path)


# SIMULATED scenario cameras: scripted boxes, not footage. They exist so detectors that real footage never triggers (collision, wrong-way) can be shown
# working - and so the "vehicles merely queued together is NOT a collision" case can be shown. Every record they produce is labelled SIMULATED.
def _sim_lanes() -> list:
    return [{"lane_id": f"LANE-{i + 1}", "vertices": [[0, 172 + 52 * i], [640, 172 + 52 * i], [640, 172 + 52 * (i + 1)], [0, 172 + 52 * (i + 1)]]} for i in range(3)]


_EAST = [{"zone_id": "east", "polygon": [[0, 100], [640, 100], [640, 340], [0, 340]], "expected": [1, 0]}]
SIMULATED = (
    ("sim-collision", "SIMULATED: two vehicles meet head-on and stop (scripted)", "synthetic:collision", {}),
    ("sim-wrong-way", "Test: wrong-way driver (simulated scene)", "synthetic:show_wrong_way", {"direction_zones": _EAST, "lanes": _sim_lanes()}),
    ("sim-accident", "Test: accident (simulated scene)", "synthetic:show_accident", {"lanes": _sim_lanes()}),
    ("sim-jam", "Test: jam from a breakdown (simulated scene)", "synthetic:show_jam",
     {"lanes": _sim_lanes(), "direction_zones": _EAST, "thresholds": {"max_capacity_vehicles": 12}}),
    ("sim-queue", "SIMULATED: eight vehicles queued bumper to bumper - must NOT be a collision (scripted)", "synthetic:queue",
     {"thresholds": {"max_capacity_vehicles": 6}}),
)


def register_simulated(services) -> None:
    have = {c["camera_id"] for c in services.cameras.list()}
    for cid, name, uri, cfg in SIMULATED:
        cfg = dict(cfg, realtime_factor=1.0)  # play at video speed so the audience can follow it
        if cid not in have:
            services.cameras.create("demo-setup", {"camera_id": cid, "name": name, "source_type": "synthetic", "source_uri": uri,
                                                   "data_class": "simulated", "config": cfg}, actor_role="administrator")


def _investigator_credentials(data_dir: Path):
    """(username, password) of the DEMO investigator from the credentials file the bootstrap wrote, or None."""
    path = Path(data_dir) / "initial_credentials.txt"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] == "investigator":
            return parts[0], parts[2]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEMO_DIR / "data"), help="disposable demo data directory (must be inside demo/)")
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--video", default=str(REPO / "demo.mp4"))
    ap.add_argument("--seconds", type=float, default=None, help="override max_stream_s of the camera config (default 1200 = 20 min)")
    ap.add_argument("--fast", action="store_true", help="process as fast as possible instead of at video speed (TESTING ONLY, not the presentation mode)")
    ap.add_argument("--reset", action="store_true", help="delete the previous demo data dir first (cases, audit, evidence are LOST)")
    ap.add_argument("--detector", choices=("pretrained", "finetuned"), default="pretrained",
                    help="pretrained = the validated default (COCO YOLOv8n). finetuned = the MIO-TCD checkpoint via ANVESH_FINETUNED_MODEL: NOT promoted "
                         "(no labelled evidence on this footage); on an unlabelled 3-minute segment it showed fewer track fragments - see demo/runs/detector_compare.json")
    ap.add_argument("--no-simulated", action="store_true", help="do not register the SIMULATED scenario cameras (collision, wrong-way, queue)")
    ap.add_argument("--autostart", action="store_true", help="start the processing session right away")
    ap.add_argument("--one-click", action="store_true",
                    help="presentation mode: open the simple officer screen at '/' and sign in automatically as the DEMO investigator. Loopback only "
                         "(127.0.0.1); the sign-in shortcut does not exist on any other address. The full dashboard stays at /advanced")
    ap.add_argument("--open", action="store_true", help="open the browser when the server is ready")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.reset:
        reset_data_dir(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MARKER).write_text("created by scripts/run_demo.py; safe to delete\n", encoding="utf-8")

    if args.detector == "finetuned":
        import os
        os.environ["ANVESH_FINETUNED_MODEL"] = str(REPO / "models" / "finetuned" / "anvesh_mio5_yolov8n_best.pt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(data_dir=str(data_dir))
    services = build_services(config)
    created = ensure_users(services, demo_users=True)

    cam_cfg = json.loads((DEMO_DIR / "camera_config.json").read_text(encoding="utf-8"))
    if args.seconds:
        cam_cfg["max_stream_s"] = args.seconds
    if args.fast:
        cam_cfg.pop("realtime_factor", None)
    if not any(c["camera_id"] == CAMERA_ID for c in services.cameras.list()):
        services.cameras.create("demo-setup", {
            "camera_id": CAMERA_ID, "name": "Demo motorway (fixed overpass camera, recorded)", "source_type": "file",
            "source_uri": str(Path(args.video).resolve()), "data_class": "demo",
            "location_label": "Recorded footage - road identity not verified (see demo/SOURCE.md)",
            "config": cam_cfg}, actor_role="administrator")
        print(f"registered camera {CAMERA_ID} ({Path(args.video).name}, first {cam_cfg.get('max_stream_s')} s)")

    plates_clip = DEMO_DIR / "plates_demo.mp4"
    if plates_clip.is_file() and not any(c["camera_id"] == "demo-plates" for c in services.cameras.list()):
        services.cameras.create("demo-setup", {
            "camera_id": "demo-plates", "name": "Test: number plates (real photographs, not CCTV)", "source_type": "file",
            "source_uri": str(plates_clip.resolve()), "data_class": "demo",
            "location_label": "Real vehicle photographs shown as a video to demonstrate the plate reader (see scripts/build_plate_demo.py)",
            "config": {"realtime_factor": 1.0, "thresholds": {"plate_min_vehicle_px": 110, "plates_only": 1}}}, actor_role="administrator")
        print("registered camera demo-plates (demo/plates_demo.mp4)")

    if not args.no_simulated:
        register_simulated(services)
    server = serve(services, port=args.port)
    from anvesh.app import warmup
    warmup.start(config.model_path)
    if args.one_click:
        cred = _investigator_credentials(config.data_dir)
        if cred is None:
            print("--one-click needs the investigator password from initial_credentials.txt; start once with --reset to create fresh demo accounts.")
        else:
            server.demo_login = cred
            server.simple_home = True
    print(f"\nANVESH DEMO is running:  http://{server.host}:{server.port}/")
    print(f"data directory:          {config.data_dir}   (disposable)")
    if created:
        print(f"created DEMO accounts:   {', '.join(created)}  -> passwords in {config.data_dir / 'initial_credentials.txt'}")
    else:
        print(f"existing accounts kept; passwords were in {config.data_dir / 'initial_credentials.txt'} (if not already deleted)")
    print("MODE: RESEARCH / DEMONSTRATION ONLY - not cleared for operational enforcement, real challans or legal decisions.")
    print("Sandbox registry/challan results are SIMULATED. No real registry or e-challan is contacted.")
    if args.autostart:
        sess = server.sessions.start(CAMERA_ID, "investigator", "investigator")
        print(f"started session {sess['session_id']}  ->  http://{server.host}:{server.port}/#/live/{sess['session_id']}")
    if args.one_click and server.demo_login:
        print("ONE-CLICK mode: the simple officer screen is at the address above; the full dashboard is at /advanced (sign in there with the credentials file).")
    if args.open:
        import webbrowser
        webbrowser.open(f"http://{server.host}:{server.port}/")
    print("Press Ctrl+C to stop.\n", flush=True)
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
