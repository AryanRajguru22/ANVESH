"""Records / compares what a demo data dir holds, to prove persistence across a server restart.

  python demo/tools/restart_check.py snapshot demo/runs/restart_before.json   # server running
  (stop the server, start it again WITHOUT --reset)
  python demo/tools/restart_check.py compare  demo/runs/restart_before.json   # exits 1 on any difference
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _api import apis  # noqa: E402


def snap(base="http://127.0.0.1:8760", data_dir="demo/data"):
    a = apis(base, data_dir)
    admin, inv = a["admin"], a["investigator"]
    incidents = inv.get("/incidents?limit=200")
    cases = inv.get("/cases?limit=200")
    verify = admin.get("/audit/verify")
    sessions = inv.get("/sessions?limit=50")["items"]
    return {"incidents": incidents["total"], "incident_ids": sorted(i["incident_id"] for i in incidents["items"]),
            "cases": cases["total"], "case_states": sorted((c["case_id"], c["state"]) for c in cases["items"]),
            "audit_rows": verify["rows"], "audit_ok": verify["ok"], "audit_head": verify["head"]["hash"],
            "session_states": sorted((s["session_id"], s["state"]) for s in sessions)}


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    now = json.loads(json.dumps(snap()))  # same JSON types as the stored snapshot
    if mode == "snapshot":
        path.write_text(json.dumps(now, indent=1))
        print({k: (v if not isinstance(v, list) else len(v)) for k, v in now.items()})
    else:
        before = json.loads(path.read_text())
        bad = []
        for k in ("incidents", "incident_ids", "cases", "case_states"):
            if before[k] != now[k]:
                bad.append(k)
        if not now["audit_ok"]:
            bad.append("audit chain broken")
        if now["audit_rows"] < before["audit_rows"]:
            bad.append("audit rows lost")
        print("audit rows", before["audit_rows"], "->", now["audit_rows"], "chain ok:", now["audit_ok"])
        print("sessions before:", before["session_states"]); print("sessions after: ", now["session_states"])
        print("RESTART PERSISTENCE VERIFIED" if not bad else f"DIFFERENCES: {bad}")
        sys.exit(1 if bad else 0)
