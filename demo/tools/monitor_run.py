"""Samples a running demo session every N seconds and writes a CSV: stream time, frames, processing fps, drops, queue depth,
latency percentiles, incident count, server RSS (MB) and GPU memory (MB). Used for the full-length performance report.

    python demo/tools/monitor_run.py --camera demo-motorway --base http://127.0.0.1:8760 --data-dir demo/data --pid <server pid> --out demo/runs/perf.csv
Stops when the session reaches a terminal state."""
import argparse, csv, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _api import apis  # noqa: E402


_PROC = {}


def rss_mb(pid):
    try:
        import psutil
        p = _PROC.setdefault(pid, psutil.Process(pid))  # one Process object per pid: cpu_percent needs the previous call
        return round((p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 2**20, 1), round(p.cpu_percent(interval=None), 1)
    except Exception:
        return None, None


def gpu_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        a, b = out.strip().split(",")
        return int(a), int(b)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760"); ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--pid", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--every", type=float, default=10.0)
    ap.add_argument("--camera", default="demo-motorway", help="sample the session of THIS camera (other, e.g. SIMULATED, sessions may run beside it)")
    a = ap.parse_args()
    inv = apis(a.base, a.data_dir)["investigator"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["wall_s", "state", "stream_s", "progress", "frames", "dropped", "proc_fps_avg", "frames_q_depth", "frames_q_high", "p50_ms", "p95_ms", "p99_ms",
            "incidents", "server_rss_mb", "server_cpu_pct", "gpu_mem_mb", "gpu_util_pct", "ring_mb"]
    t0 = time.time()
    sid = None
    while sid is None:  # wait for the presenter to start the session
        try:
            mine = [x for x in inv.get("/sessions?limit=50")["items"] if x["camera_id"] == a.camera]
            sid = mine[0]["session_id"] if mine else None
        except Exception:  # noqa: BLE001
            pass
        if sid is None:
            time.sleep(2)
    t0 = time.time()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); rss_mb(a.pid)
        while True:
            try:
                s = {"session_id": sid}
                rt = inv.get(f"/sessions/{sid}")["runtime"]
                if rt is None:
                    break
                lat = (rt.get("latency") or {}).get("frame_processing") or (rt.get("latency") or {}).get("detect_track") or {}
                q = (rt.get("queues") or {}).get("frames") or {}
                n_inc = inv.get(f"/incidents?session_id={sid}&limit=1")["total"]
                rss, cpu = rss_mb(a.pid); gm, gu = gpu_mb()
                w.writerow([round(time.time() - t0), rt["state"], round(rt["stream_time_s"], 1), rt.get("progress"), rt["frames_processed"], rt["frames_dropped"],
                            round(rt["processing_fps"], 2), q.get("depth"), q.get("high_water"), lat.get("p50_ms"), lat.get("p95_ms"), lat.get("p99_ms"),
                            n_inc, rss, cpu, gm, gu, round((rt.get("ring") or {}).get("bytes", 0) / 2**20, 1)]); f.flush()
                if rt["state"] in ("completed", "failed", "stopped", "interrupted"):
                    break
            except Exception as exc:  # keep sampling through transient errors
                w.writerow([round(time.time() - t0), f"error: {exc}"]); f.flush()
            time.sleep(a.every)


if __name__ == "__main__":
    main()
