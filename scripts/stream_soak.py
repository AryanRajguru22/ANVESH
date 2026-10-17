"""Memory soak: the ~34-minute demo video (demo.mp4: 640x360 @ 25 fps, 51,201
frames) through the streaming pipeline with the REAL detector+tracker, sampling
process RSS and GPU memory every few seconds.

PASS CRITERION (fixed before measurement): after a warm-up period, memory must
be approximately flat -- the mean RSS of the LAST quarter of the post-warm-up
run may exceed the mean of the SECOND quarter by no more than 5%, and the
fitted growth rate is reported (MiB/hour). An unbounded store, frame list or
track list would grow with every one of the 51k frames.

The run is UNPACED (as fast as the pipeline can go): it exercises exactly the
same per-frame code and retention logic as a 34-minute live run would, in
~10-15 minutes of wall time; it does not test wall-clock-dependent behaviour
(e.g. RTSP reconnects), and it is NOT a 34-minute real-time soak. `--realtime-factor
1` runs the real thing.

Also checks the identity invariant on the full run: every identity created is
unique (a set of all ids announced via the store's update hook has exactly
`tracks_created` members).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stream_common import MODEL, OUT, REPO_ROOT  # noqa: E402

import psutil  # noqa: E402
import torch  # noqa: E402

from anvesh.perception.ingestion import VideoFileFrameSource  # noqa: E402
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402
from anvesh.streaming.engine import StreamingPerceptionEngine  # noqa: E402
from anvesh.streaming.pipeline import StreamingPipeline  # noqa: E402
from anvesh.streaming.queues import DropPolicy  # noqa: E402
from anvesh.streaming.reasoning import WindowReasoner  # noqa: E402
from anvesh.streaming.windows import WindowSpec  # noqa: E402


def fit_slope(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(REPO_ROOT / "demo.mp4"))
    ap.add_argument("--realtime-factor", type=float, default=None)
    ap.add_argument("--sample-every", type=float, default=5.0)
    ap.add_argument("--warmup-s", type=float, default=120.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--tag", default="soak_demo")
    ap.add_argument("--out-dir", type=Path, default=None, help="default outputs/m_a; the M-A.1 run uses outputs/m_a1")
    ap.add_argument("--tracemalloc", action="store_true", help="diagnosis only: top allocation growth")
    args = ap.parse_args()

    src = VideoFileFrameSource(args.video, realtime_factor=args.realtime_factor)
    fps = src.fps()
    tracker = ByteTrackYoloTracker(model_name=MODEL, confidence_threshold=0.25)
    ids_seen: set = set()
    created = {"n": 0}

    def on_update(u):
        if u.is_new:
            created["n"] += 1
            ids_seen.add(u.track_id)

    eng = StreamingPerceptionEngine("cam-demo", tracker, fps, windows=WindowSpec(10.0), on_update=on_update)
    reasoner = WindowReasoner("cam-demo")
    outcomes: dict = {}
    n_windows = {"n": 0}

    def on_window(snap):
        a = reasoner.process(snap)
        o = "unavailable" if a.unavailable else a.result.candidate_ranking.outcome.value
        outcomes[o] = outcomes.get(o, 0) + 1
        n_windows["n"] += 1

    pipe = StreamingPipeline(src, eng, on_window=on_window, frame_queue_size=8, frame_policy=DropPolicy.BLOCK,
                             max_frames=args.max_frames or None)

    proc = psutil.Process()
    rows: list = []
    stop = threading.Event()
    t_start = time.perf_counter()

    if args.tracemalloc:
        import tracemalloc

        tracemalloc.start(15)
        snap0 = {}

    def sampler():
        while not stop.is_set():
            mem = proc.memory_info()
            cuda_alloc = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
            cuda_res = torch.cuda.memory_reserved() / 2**20 if torch.cuda.is_available() else 0.0
            rows.append({
                "t_s": round(time.perf_counter() - t_start, 2),
                "frames": eng.frames_processed,
                "rss_mib": round(mem.rss / 2**20, 2),
                "vms_mib": round(mem.vms / 2**20, 2),
                "cuda_alloc_mib": round(cuda_alloc, 2),
                "cuda_reserved_mib": round(cuda_res, 2),
                "tracks_created": eng.store.manager.total_created,
                "tracks_retained": eng.store.retained_track_count(),
                "frame_queue": len(pipe.frame_queue),
                "window_queue": len(pipe.window_queue),
                "frame_times_len": len(eng._frame_times),
                "windows": n_windows["n"],
            })
            stop.wait(args.sample_every)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    stats = pipe.run()
    stop.set()
    th.join(timeout=10)
    wall = time.perf_counter() - t_start

    out_dir = args.out_dir if args.out_dir is not None else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.tag}_memory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    post = [r for r in rows if r["t_s"] >= args.warmup_s]
    summary = {
        "video": Path(args.video).name, "source_fps": fps, "frames_processed": stats["engine"]["frames_processed"],
        "source_duration_s": stats["engine"]["frames_processed"] / fps, "wall_s": wall,
        "processing_fps": stats["processing_fps"], "frames_dropped": stats["engine"]["frames_dropped"],
        "windows": n_windows["n"], "window_outcomes": outcomes,
        "tracks_created": stats["engine"]["tracks_created"], "unique_identity_ids": len(ids_seen),
        "duplicate_identities": stats["engine"]["tracks_created"] - len(ids_seen),
        "stale_raw_id_restarts": stats["engine"]["stale_raw_id_restarts"],
        "samples": len(rows), "warmup_s": args.warmup_s,
    }
    if len(post) >= 8:
        q = len(post) // 4
        mean = lambda seg, k: sum(r[k] for r in seg) / len(seg)  # noqa: E731
        q2, q4 = post[q:2 * q], post[3 * q:]
        for key in ("rss_mib", "cuda_alloc_mib", "cuda_reserved_mib", "tracks_retained"):
            m2, m4 = mean(q2, key), mean(q4, key)
            slope_per_s = fit_slope([r["t_s"] for r in post], [r[key] for r in post])
            summary[key] = {
                "start": rows[0][key], "peak": max(r[key] for r in rows), "final": rows[-1][key],
                "post_warmup_mean_q2": m2, "post_warmup_mean_q4": m4,
                "q4_vs_q2_pct": 100.0 * (m4 - m2) / m2 if m2 else 0.0,
                "fitted_growth_per_hour": slope_per_s * 3600.0,
            }
        summary["verdict_rss_flat"] = abs(summary["rss_mib"]["q4_vs_q2_pct"]) <= 5.0
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
