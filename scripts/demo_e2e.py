"""Reproducible end-to-end demonstration, driven through the SAME HTTP API the dashboard uses, as the
separate roles a real review would involve.

  PART A  REAL INFERENCE ON REAL FOOTAGE (recorded file -> YOLOv8 + ByteTrack -> events -> evidence)
          then the officer workflow on one resulting incident, ending in a SANDBOX hand-off.
  PART B  SIMULATED fixture (scripted collision) to exercise the accident-candidate alert path. It is labelled
          SIMULATED everywhere: the available footage contains no collision, so none is claimed.

Prerequisites: the server is running with demo users (`scripts/run_app.py --demo-users`), and the model weights
are present at models/yolov8n.pt.

    .venv\\Scripts\\python.exe scripts\\demo_e2e.py --base http://127.0.0.1:8750 --data-dir outputs\\app
    .venv\\Scripts\\python.exe scripts\\demo_e2e.py --verify-only      # after restarting the server: check persistence

WHAT IS AND IS NOT REAL HERE (also written into the report):
  - Detection, tracking, windows, stop episodes, congestion state: real, on the real video.
  - The recording's wall-clock start is unknown, so times are processing-clock times (flagged on every record).
  - The plate "MH12AB1234" is a DEMO PLACEHOLDER typed by the investigator to exercise the workflow; it was NOT read
    from the footage (no plate is legible in it, and no OCR engine is installed).
  - The registry lookup and the challan reference come from the SANDBOX adapters (synthetic data, TEST- references).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class Api:
    def __init__(self, base: str, username: str, password: str) -> None:
        self.base, self.username = base.rstrip("/"), username
        _, body = self.call("POST", "/auth/login", {"username": username, "password": password}, auth=False)
        self.token = body["token"]

    def call(self, method, path, body=None, raw=None, auth=True, binary=False):
        h = {"Content-Type": "application/json"} if raw is None else {"Content-Type": "application/octet-stream"}
        if auth:
            h["Authorization"] = "Bearer " + self.token
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + "/api/v1" + path, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = r.read()
                return r.status, (payload if binary else json.loads(payload))
        except urllib.error.HTTPError as e:
            payload = e.read()
            raise SystemExit(f"{method} {path} -> HTTP {e.code}: {payload.decode(errors='replace')[:400]}")

    def get(self, path, **kw):
        return self.call("GET", path, **kw)[1]

    def post(self, path, body=None):
        return self.call("POST", path, {} if body is None else body)[1]


def credentials(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            user, _role, pw = line.split("\t")
            out[user] = pw
    return out


def step(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def wait_session(api: Api, sid: str, timeout_s: float) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        s = api.get(f"/sessions/{sid}")
        rt = s.get("runtime") or {}
        line = f"   state={rt.get('state', s['state'])} frames={rt.get('frames_processed', (s.get('stats') or {}).get('frames_processed'))} fps={rt.get('processing_fps', 0):.1f} events={rt.get('events_emitted')}"
        if line != last:
            print(line, flush=True)
            last = line
        if (rt.get("state") or s["state"]) in ("completed", "failed", "stopped"):
            return s
        time.sleep(3)
    raise SystemExit("session did not finish in time")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8750")
    ap.add_argument("--data-dir", default=str(REPO / "outputs" / "app"))
    ap.add_argument("--video", default=str(REPO / "demo.mp4"))
    ap.add_argument("--seconds", type=float, default=120.0, help="seconds of the recording to process")
    ap.add_argument("--skip-simulated", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="only check persistence/audit of the previous run")
    ap.add_argument("--report", default=str(REPO / "outputs" / "demo" / "demo_report.json"))
    args = ap.parse_args()

    creds = credentials(Path(args.data_dir) / "initial_credentials.txt")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        prev = json.loads(report_path.read_text(encoding="utf-8"))
        admin = Api(args.base, "admin", creds["admin"])
        officer = Api(args.base, "officer", creds["officer"])
        case = officer.get(f"/cases/{prev['case_id']}")
        audit = admin.get("/audit/verify")
        inc = officer.get(f"/incidents/{prev['incident_id']}")
        ev = [officer.get(f"/evidence/{a['asset_id']}/verify") for a in inc["evidence"]]
        print(f"case {case['case_id']}: state={case['state']} ref={case['external_reference']} transitions={len(case['transitions'])}")
        print(f"incident {inc['incident_id']}: {len(inc['evidence'])} evidence assets, all hashes ok: {all(e['ok'] for e in ev)}")
        print(f"audit chain ok: {audit['ok']} ({audit['rows']} entries)")
        ok = case["state"] == prev["final_case_state"] and audit["ok"] and all(e["ok"] for e in ev)
        print("PERSISTENCE AFTER RESTART:", "VERIFIED" if ok else "FAILED")
        return 0 if ok else 1

    admin = Api(args.base, "admin", creds["admin"])
    inv = Api(args.base, "investigator", creds["investigator"])
    rev = Api(args.base, "reviewer", creds["reviewer"])
    off = Api(args.base, "officer", creds["officer"])
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "video": args.video, "seconds_processed": args.seconds, "notes": []}

    step("System status (what is and is not connected)")
    st = admin.get("/system/status")
    print("   GPU:", st["gpu"], "| detector weights present:", st["detector_weights_present"])
    print("   OCR available:", st["integrations"]["ocr"]["available"], "| production registry:", st["integrations"]["registry"][0]["state"],
          "| official e-challan:", [c for c in st["integrations"]["challan"] if c["key"] == "official"][0]["state"])
    report["integrations"] = st["integrations"]

    step("PART A: register the recorded video as a DEMO camera and process it with the real detector/tracker")
    cams = {c["camera_id"]: c for c in admin.get("/cameras")["items"]}
    cam_id = "demo-motorway"
    if cam_id not in cams:
        admin.post("/cameras", {"camera_id": cam_id, "name": "Demo motorway (recorded stock footage)", "source_type": "file", "source_uri": args.video,
                                "data_class": "demo", "location_label": "Stock footage - location not known",
                                "config": {"max_stream_s": args.seconds, "stop_criterion": "displacement"}})
    sess = inv.post(f"/cameras/{cam_id}/sessions")
    print("   session", sess["session_id"])
    final = wait_session(inv, sess["session_id"], 1800)
    stats = final["stats"]
    print(f"   finished: {stats['frames_processed']} frames in {stats['wall_elapsed_s']:.0f}s ({stats['processing_fps']:.1f} fps), "
          f"{stats['frames_dropped']} dropped, {stats['events_emitted']} events, tracker={stats['tracker']}")
    report["session"] = {"session_id": sess["session_id"], "stats": stats}

    step("Incidents produced by real inference")
    inc_list = inv.get(f"/incidents?session_id={sess['session_id']}&limit=200")
    by_type = {}
    for i in inc_list["items"]:
        by_type[i["event_type"]] = by_type.get(i["event_type"], 0) + 1
    print("   by type:", by_type)
    report["incidents_by_type"] = by_type
    stops = [i for i in inc_list["items"] if i["event_type"] in ("stopped_vehicle", "long_duration_stop") and i["evidence_completeness"] in ("full", "partial")]
    if not stops:
        raise SystemExit("no stopped-vehicle incident with evidence was produced; try --seconds larger")
    stops.sort(key=lambda i: (i["severity"] != "medium", i["detected_offset_s"]))
    inc = inv.get(f"/incidents/{stops[0]['incident_id']}")
    print(f"   using {inc['incident_id']} ({inc['event_type']}, severity {inc['severity']}, video t={inc['start_offset_s']:.1f}s, {len(inc['evidence'])} evidence assets)")
    print("   uncertainty:", inc["uncertainty"][0])
    ev_ok = [inv.get(f"/evidence/{a['asset_id']}/verify")["ok"] for a in inc["evidence"]]
    print("   evidence integrity (sha256):", all(ev_ok))
    out_dir = report_path.parent
    for a in inc["evidence"]:
        if a["kind"] in ("frame_annotated", "clip"):
            data = inv.call("GET", f"/evidence/{a['asset_id']}/file", binary=True)[1]
            ext = Path(a["relpath"]).suffix
            (out_dir / f"demo_{a['kind']}_{a['asset_id']}{ext}").write_bytes(data)
    report["incident_id"] = inc["incident_id"]

    step("Investigator opens a violation case; reviewer verifies evidence")
    case = inv.post(f"/incidents/{inc['incident_id']}/cases", {"offence_code": "STOP-OBS"})
    cid = case["case_id"]
    inv.post(f"/cases/{cid}/actions", {"action": "submit_for_evidence_verification"})
    rev.post(f"/cases/{cid}/actions", {"action": "verify_evidence", "note": "key frames and clip reviewed"})
    print("   case", cid, "-> awaiting_plate_verification")

    step("Plate: OCR is unavailable here, so the investigator enters a DEMO PLACEHOLDER; the reviewer verifies it")
    ocr = inv.post(f"/incidents/{inc['incident_id']}/plates/ocr", {"track_id": inc["track_ids"][0]})
    print("   OCR attempt:", ocr["status"], "-", "; ".join(ocr.get("reasons", [])))
    pl = inv.post(f"/incidents/{inc['incident_id']}/plates", {"text": "MH12AB1234", "track_id": inc["track_ids"][0],
                                                              "note": "DEMO PLACEHOLDER - not read from the footage; exercises the workflow only"})
    rev.post(f"/cases/{cid}/actions", {"action": "verify_plate", "note": "demo placeholder verified for workflow test"})
    print("   plate reading", pl["reading_id"], "source:", pl["source"])

    step("Officer: sandbox registry lookup, approval, SANDBOX hand-off, evidence package export")
    look = off.post(f"/cases/{cid}/registry-lookup")
    print("   registry:", look["status"], "|", look["notice"])
    approved = off.post(f"/cases/{cid}/actions", {"action": "approve", "note": "approved for the sandbox hand-off"})
    print("   state:", approved["state"], "| approved_by:", approved["approved_by"])
    ho = off.post(f"/cases/{cid}/handoff", {"adapter": "sandbox"})
    print("   hand-off:", ho["status"], ho["external_reference"], "|", ho["notice"])
    pkg = off.call("GET", f"/cases/{cid}/package", binary=True)[1]
    (out_dir / f"{cid}.zip").write_bytes(pkg)
    html = off.call("GET", f"/cases/{cid}/report.html", binary=True)[1]
    (out_dir / f"{cid}_report.html").write_bytes(html)
    final_case = off.get(f"/cases/{cid}")
    print("   final state:", final_case["state"], "| reference kind:", final_case["external_reference_kind"])
    report.update({"case_id": cid, "final_case_state": final_case["state"], "external_reference": final_case["external_reference"],
                   "package": f"{cid}.zip", "report_html": f"{cid}_report.html"})
    report["notes"] += ["plate MH12AB1234 is a DEMO PLACEHOLDER, not read from the footage", "registry and challan outputs are SANDBOX/TEST",
                        "event wall-clock times are processing-clock times (recording start unknown)"]

    if not args.skip_simulated:
        step("PART B (SIMULATED): scripted collision fixture -> possible_collision candidate -> control-room TEST alert")
        try:
            admin.post("/cameras", {"camera_id": "sim-collision", "name": "SIMULATED collision fixture", "source_type": "synthetic",
                                    "source_uri": "synthetic:collision", "data_class": "simulated"})
        except SystemExit:
            pass
        s2 = inv.post("/cameras/sim-collision/sessions")
        wait_session(inv, s2["session_id"], 300)
        items = inv.get(f"/incidents?session_id={s2['session_id']}&event_type=possible_collision")["items"]
        if items:
            i2 = inv.get(f"/incidents/{items[0]['incident_id']}")
            print(f"   {i2['incident_id']}: severity {i2['severity']}, confidence {i2['confidence_tier']}, data_class {i2['data_class']}")
            print("   roles:", {v["role"] for v in i2["involved"]}, "| established cause:", i2["established_cause"])
            for a in i2["alerts"]:
                print("   alert:", a["status"], "|", a["notice"])
            report["simulated_incident_id"] = i2["incident_id"]
        else:
            print("   no collision candidate produced")

    report["audit"] = admin.get("/audit/verify")
    print("\naudit chain:", report["audit"])
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
