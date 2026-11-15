"""Officer workflow acceptance check against a RUNNING demo server, through the same HTTP API the dashboard uses. Picks the best incident of the
current session and walks it: investigator opens a case -> reviewer verifies evidence -> investigator enters a plate (a DEMO PLACEHOLDER: nothing
in the footage is legible) -> a DIFFERENT reviewer verifies it -> officer runs the SANDBOX registry lookup -> approves -> SANDBOX hand-off ->
evidence package. Then it asserts the safety rules: no registry override, the official e-challan path is blocked, demo accounts cannot create
operational cameras, the audit chain verifies, every evidence file re-hashes. Writes a JSON report.

   python demo/tools/officer_workflow.py --base http://127.0.0.1:8760 --data-dir demo/data-run1 --out demo/runs/workflow_report.json
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _api import apis  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from demo_e2e import Api, credentials  # noqa: E402


def raw(api, method, path, body=None):
    """Like Api.call but returns (status, json) instead of exiting on an HTTP error."""
    req = urllib.request.Request(api.base + "/api/v1" + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + api.token}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760")
    ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--out", default="demo/runs/workflow_report.json")
    ap.add_argument("--event-type", default=None)
    a = ap.parse_args()
    A = apis(a.base, a.data_dir)
    cr = credentials(Path(a.data_dir) / "initial_credentials.txt")
    rev2 = Api(a.base, "reviewer2", cr["reviewer2"])
    inv, rev, off, adm = A["investigator"], A["reviewer"], A["officer"], A["admin"]
    rep = {"checks": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def check(name, ok, detail=""):
        rep["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name, detail)

    st = adm.get("/system/status")
    check("system status reports RESEARCH/DEMO mode and no operational authorisation",
          st["product"]["mode"] == "research_demo" and not st["product"]["operational_use_authorised"])
    check("no authorised (production) registry is connected", not any(r.get("state") == "connected" and r.get("kind") == "production"
                                                                       for r in st["integrations"]["registry"]), json.dumps(st["integrations"]["registry"])[:240])
    sid = inv.get("/sessions?limit=1")["items"][0]["session_id"]
    offs = inv.get("/offences")
    items_o = offs.get("items", offs) if isinstance(offs, dict) else offs
    by_type = {t: o["code"] for o in items_o for t in o["event_types"]}  # congestion is not an offence: it never gets a case
    items = inv.get(f"/incidents?session_id={sid}&limit=200")["items"]
    pick = [i for i in items if i["event_type"] in by_type and (a.event_type is None or i["event_type"] == a.event_type)
            and i["evidence_completeness"] in ("full", "partial")]
    pick.sort(key=lambda i: (i["severity"] not in ("high", "medium"), -i["detected_offset_s"]))
    if not pick:
        raise SystemExit("no offence-type incident with evidence yet")
    inc = inv.get(f"/incidents/{pick[0]['incident_id']}")
    check("incident has hashed evidence", len(inc["evidence"]) > 0, f"{inc['incident_id']} {inc['event_type']} {len(inc['evidence'])} assets")
    check("every evidence asset re-hashes", all(inv.get(f"/evidence/{e['asset_id']}/verify")["ok"] for e in inc["evidence"]))
    check("incident carries reasons, uncertainty and basis", bool(inc["uncertainty"]) and bool(inc["severity_basis"]) and bool(inc["confidence_basis"]))
    offence = by_type[inc["event_type"]]
    case = inv.post(f"/incidents/{inc['incident_id']}/cases", {"offence_code": offence} if offence else {})
    cid = case["case_id"]
    inv.post(f"/cases/{cid}/actions", {"action": "submit_for_evidence_verification"})
    rev.post(f"/cases/{cid}/actions", {"action": "verify_evidence", "note": "key frames and clip reviewed in the demo"})
    ocr = inv.post(f"/incidents/{inc['incident_id']}/plates/ocr", {"track_id": inc["track_ids"][0]})
    check("automatic plate reading is NOT claimed (OCR/detector unavailable)", ocr["status"] not in ("read", "ok"),
          f"status={ocr['status']}: {'; '.join(ocr.get('reasons', []))[:160]}")
    pl = inv.post(f"/incidents/{inc['incident_id']}/plates", {"text": "MH12AB1234", "track_id": inc["track_ids"][0],
                                                              "note": "DEMO PLACEHOLDER - not read from the footage; exercises the workflow only"})
    check("manual plate entry is recorded as manual_entry, not OCR", pl["source"] == "manual_entry", pl["source"])
    rev2.post(f"/cases/{cid}/actions", {"action": "verify_plate", "note": "placeholder verified for the workflow demo"})
    s, _ = raw(off, "POST", f"/cases/{cid}/actions", {"action": "approve", "registry_override_reason": "trust me"})
    check("registry override is refused (422)", s == 422, f"HTTP {s}")
    look = off.post(f"/cases/{cid}/registry-lookup")
    check("registry result is labelled SANDBOX/synthetic", "sandbox" in json.dumps(look).lower() or "synthetic" in json.dumps(look).lower(), str(look.get("notice", ""))[:100])
    off.post(f"/cases/{cid}/actions", {"action": "approve", "note": "approved for the SANDBOX hand-off only"})
    ho = off.post(f"/cases/{cid}/handoff", {"adapter": "sandbox"})
    check("hand-off is a SANDBOX reference", str(ho["external_reference"]).startswith("TEST-"), f"{ho['external_reference']} {str(ho.get('notice', ''))[:80]}")
    fin = off.get(f"/cases/{cid}")
    check("case reaches challan_reference_received (sandbox)", fin["state"] == "challan_reference_received", fin["state"])
    pkg = off.call("GET", f"/cases/{cid}/package", binary=True)[1]
    check("evidence package exports", len(pkg) > 1000, f"{len(pkg)} bytes")
    s, _ = raw(off, "POST", f"/cases/{cid}/handoff", {"adapter": "official"})
    check("official e-challan hand-off is blocked (no integration)", s in (409, 422, 503), f"HTTP {s}")
    s, _ = raw(inv, "POST", "/cameras", {"camera_id": "op-try", "name": "x", "source_type": "file", "source_uri": str(Path("demo.mp4").resolve()), "data_class": "operational"})
    check("demo account cannot register an operational camera", s in (403, 422), f"HTTP {s}")
    au = adm.get("/audit/verify")
    check("audit chain verifies", au["ok"], f"{au['rows']} rows")
    rep.update({"incident_id": inc["incident_id"], "case_id": cid, "final_case_state": fin["state"], "audit": au, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    fails = [c for c in rep["checks"] if not c["ok"]]
    print(f"\n{len(rep['checks']) - len(fails)}/{len(rep['checks'])} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
