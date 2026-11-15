"""Read-only summary of one demo data directory: sessions, incident counts per camera/type, queue membership, congestion episodes.

  .venv\\Scripts\\python.exe demo/tools/summarize_run.py demo/data-fast [--camera demo-motorway] [--wait]

`--wait` blocks until no session is starting/running/paused. Opens the SQLite file read-only; it never writes.
"""
import argparse
import collections
import json
import sqlite3
import sys
import time
from pathlib import Path


def connect(data_dir):
    return sqlite3.connect(f"file:{Path(data_dir) / 'anvesh_app.sqlite3'}?mode=ro", uri=True, timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--camera", default="demo-motorway")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--json", default=None, help="also write the summary to this file")
    a = ap.parse_args()
    if a.wait:
        while True:
            try:
                c = connect(a.data_dir)
                n = c.execute("select count(*) from processing_sessions where state in ('starting','running','paused')").fetchone()[0]
                c.close()
                if n == 0:
                    break
            except sqlite3.Error:
                pass
            time.sleep(3)
    c = connect(a.data_dir)
    c.row_factory = sqlite3.Row
    out = {"sessions": [dict(r) for r in c.execute("select session_id, camera_id, state, started_at, ended_at from processing_sessions")]}
    rows = [dict(r) for r in c.execute("select * from incidents where camera_id = ? order by start_offset_s", (a.camera,))]
    by_type = collections.Counter(r["event_type"] for r in rows)
    out["incidents_by_type"] = dict(by_type)
    stops, congestion = [], []
    for r in rows:
        ind = json.loads(r["indicators_json"] or "{}")
        if r["event_type"] in ("stopped_vehicle", "long_duration_stop"):
            stops.append({"start": r["start_offset_s"], "end": r["end_offset_s"], "severity": r["severity"], "queue_member": ind.get("queue_member"),
                          "queue_signal": ind.get("queue_signal"), "group": ind.get("queue_group_id"), "dwell_s": ind.get("dwell_s"),
                          "closed_reason": ind.get("closed_reason")})
        elif r["event_type"] == "congestion":
            congestion.append({"start": r["start_offset_s"], "end": r["end_offset_s"], "peak": ind.get("peak_level"), "severity": r["severity"],
                               "span_s": ind.get("duration_s"), "non_free_s": ind.get("non_free_s"), "interruptions": ind.get("interruptions"),
                               "windows": ind.get("windows"), "closed_reason": ind.get("closed_reason")})
    out["stopped"] = stops
    out["stopped_summary"] = {"total": len(stops), "queue_member": sum(1 for s in stops if s["queue_member"]),
                              "isolated_or_unlatched": sum(1 for s in stops if not s["queue_member"]),
                              "by_severity": dict(collections.Counter(s["severity"] for s in stops)),
                              "queue_groups": len({s["group"] for s in stops if s["group"]})}
    out["congestion"] = congestion
    out["other_events"] = {k: v for k, v in by_type.items() if k not in ("stopped_vehicle", "long_duration_stop", "congestion")}
    print(json.dumps(out["incidents_by_type"]), json.dumps(out["stopped_summary"]), flush=True)
    for e in congestion:
        print("congestion", e)
    for s in stops:
        print("stop", s)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
