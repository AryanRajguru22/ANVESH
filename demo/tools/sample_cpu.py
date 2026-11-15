"""CPU sampler for a running server process (monitor_run.py's cpu column was always 0 because it built a new psutil.Process per sample).

    python demo/tools/sample_cpu.py --pid <server pid> --out demo/runs/cpu.csv [--every 10] [--seconds 1300]
Columns: wall_s, server_cpu_pct (percent of ONE logical core, summed over the process and its children; can exceed 100), system_cpu_pct (all cores)."""
import argparse, csv, time
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--pid", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--every", type=float, default=10.0); ap.add_argument("--seconds", type=float, default=1300)
a = ap.parse_args()
proc = psutil.Process(a.pid)
procs = {a.pid: proc}
psutil.cpu_percent(None)
t0 = time.time()
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["wall_s", "server_cpu_pct", "system_cpu_pct", "logical_cores"])
    while time.time() - t0 < a.seconds:
        time.sleep(a.every)
        try:
            for c in proc.children(recursive=True):
                procs.setdefault(c.pid, c)
            tot = sum(p.cpu_percent(None) for p in list(procs.values()) if p.is_running())
        except psutil.NoSuchProcess:
            break
        w.writerow([round(time.time() - t0, 1), round(tot, 1), psutil.cpu_percent(None), psutil.cpu_count()]); f.flush()
