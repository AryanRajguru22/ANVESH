"""Summarises a real-time benchmark: monitor_run.py CSV + sample_cpu.py CSV + the final session record -> JSON (and a markdown table).

    python demo/tools/summarize_realtime.py --csv demo/runs/s9_realtime_perf.csv --cpu demo/runs/s9_realtime_cpu.csv \
        --base http://127.0.0.1:8764 --data-dir demo/data-s9 --out docs/benchmarks/s9_realtime_summary.json

Every figure is read from the files; nothing is typed in. Run it while the server is still up (it fetches the final session record from the API
and stores the trimmed record next to the summary)."""
import argparse, csv, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def col(rows, k):
    return [float(r[k]) for r in rows if r.get(k) not in (None, "", "None")]


def stats(v):
    return {"min": min(v), "max": max(v), "mean": round(statistics.fmean(v), 2), "first_quarter_mean": round(statistics.fmean(v[:max(1, len(v) // 4)]), 2),
            "last_quarter_mean": round(statistics.fmean(v[-max(1, len(v) // 4):]), 2)} if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--cpu"); ap.add_argument("--base"); ap.add_argument("--data-dir")
    ap.add_argument("--camera", default="demo-motorway"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv, newline="")) if r.get("state") and not r["state"].startswith("error")]
    errors = sum(1 for r in csv.DictReader(open(a.csv, newline="")) if (r.get("state") or "").startswith("error"))
    out = {"samples": len(rows), "sample_errors": errors, "sample_interval_s": 10, "states_seen": sorted({r["state"] for r in rows})}
    out["sampled_wall_s"] = float(rows[-1]["wall_s"]) if rows else None
    out["rss_mb"] = stats(col(rows, "server_rss_mb"))
    out["gpu_mem_mb_nvidia_smi_total_used"] = stats(col(rows, "gpu_mem_mb"))
    out["gpu_util_pct"] = stats(col(rows, "gpu_util_pct"))
    out["frame_queue_depth_max_sampled"] = max(col(rows, "frames_q_depth") or [0])
    out["frame_queue_high_water_max"] = max(col(rows, "frames_q_high") or [0])
    out["ring_mb_max"] = max(col(rows, "ring_mb") or [0])
    # stall heuristic: the processed-frame counter did not advance between two samples while the state was live
    stalls = 0
    for x, y in zip(rows, rows[1:]):
        if x["state"] == "live" and y["state"] == "live" and float(y["frames"]) <= float(x["frames"]):
            stalls += 1
    out["live_samples_with_no_frame_progress"] = stalls
    if a.cpu and Path(a.cpu).is_file():
        cr = list(csv.DictReader(open(a.cpu, newline="")))
        out["cpu"] = {"server_cpu_pct_of_one_core": stats(col(cr, "server_cpu_pct")), "system_cpu_pct_all_cores": stats(col(cr, "system_cpu_pct")),
                      "logical_cores": int(float(cr[0]["logical_cores"])) if cr else None}
    if a.base:
        from _api import apis
        inv = apis(a.base, a.data_dir)["investigator"]
        sid = [x for x in inv.get("/sessions?limit=50")["items"] if x["camera_id"] == a.camera][0]["session_id"]
        s = inv.get(f"/sessions/{sid}")
        rt = s.get("runtime") or {}
        keep = {k: rt.get(k) for k in ("state", "frames_processed", "frames_dropped", "processing_fps", "wall_elapsed_s", "stream_time_s", "fps_nominal", "latency", "queues",
                                       "unique_tracks_total", "events_emitted", "source_health", "engine_health", "windows_emitted", "errors", "tracker")}
        if not rt:
            keep = {"stats": s.get("stats"), "state": s.get("state")}
        out["session_id"] = sid
        out["final_session_record"] = keep
        out["incidents_total"] = inv.get(f"/incidents?session_id={sid}&limit=1")["total"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "final_session_record"}, indent=1))


if __name__ == "__main__":
    main()
