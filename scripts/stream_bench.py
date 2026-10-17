"""Streaming latency / throughput benchmark on REAL video with the REAL
detector+tracker (YOLOv8n + ByteTrack, GPU), one or several concurrent streams.

Per stream it reports (p50/p95/p99, ms):
  decode              OpenCV `read()` time per frame
  detect_track        `tracker.update` (YOLO inference + ByteTrack)
  store_update        identity assignment + track store update
  windowing           building the closed window snapshots (per emission)
  reasoning           aggregate_traffic_state + run_corridor_window (per window)
  frame_processing    worker time per frame (everything above except decode/queue)
  frame_end_to_end    frame decoded -> its track update committed (includes queue wait)
  window_end_to_end   capture of the frame that CLOSED a window -> reasoning done
                      (processing latency of an event once its window closes; it
                       does NOT include the window's own duration or lateness --
                       a window of W seconds can only be reported W seconds after
                       it starts, by construction)
plus: nominal SOURCE fps, achieved PROCESSING fps (frames processed / wall), their
ratio (real-time multiple), dropped frames, queue high-water marks, GPU
utilisation (nvidia-smi poll) and peak RSS.

SOURCE fps and PROCESSING fps are different quantities: the former is the
camera's rate (10 fps for CityFlow, 25 for demo.mp4); the latter is how fast
this machine consumed frames. Unpaced runs measure the latter; paced runs
(`--realtime-factor 1`) show whether the pipeline keeps up with the former.

Usage:
    python scripts/stream_bench.py --streams c006 --tag single_c006_unpaced
    python scripts/stream_bench.py --streams c006,c009 --tag dual_unpaced
    python scripts/stream_bench.py --streams c006,c009 --realtime-factor 1 --seconds 60 --policy drop_oldest --tag dual_paced1x
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stream_common import MODEL, OUT, REPO_ROOT, GpuPoller  # noqa: E402

import psutil  # noqa: E402

from anvesh.evaluation import cityflow_gt as gt_io  # noqa: E402
from anvesh.perception.ingestion import VideoFileFrameSource  # noqa: E402
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402
from anvesh.streaming.engine import StreamingPerceptionEngine  # noqa: E402
from anvesh.streaming.latency import LatencyRecorder  # noqa: E402
from anvesh.streaming.pipeline import StreamingPipeline  # noqa: E402
from anvesh.streaming.queues import DropPolicy  # noqa: E402
from anvesh.streaming.reasoning import WindowReasoner  # noqa: E402
from anvesh.streaming.windows import WindowSpec  # noqa: E402
from anvesh.world.calibration import load_cityflow_calibration  # noqa: E402


def resolve(stream: str) -> tuple:
    """-> (camera_id, video_path, calibration_profile_or_None)"""
    if stream == "demo":
        return "cam-demo", REPO_ROOT / "demo.mp4", None
    cam_dir = gt_io.camera_dir("S02", stream)
    profile = load_cityflow_calibration(f"calib-s02-{stream}", f"s02_{stream}", cam_dir / "calibration.txt")
    return f"s02_{stream}", cam_dir / "vdo.avi", profile


def run_stream(stream: str, args, results: dict, barrier: threading.Barrier) -> None:
    camera_id, video, profile = resolve(stream)
    src = VideoFileFrameSource(video, realtime_factor=args.realtime_factor)
    fps = src.fps()
    recorder = LatencyRecorder()
    tracker = ByteTrackYoloTracker(model_name=MODEL, confidence_threshold=0.25)  # own model per stream
    eng = StreamingPerceptionEngine(camera_id, tracker, fps, windows=WindowSpec(10.0), calibration_profile=profile,
                                    recorder=recorder)
    reasoner = WindowReasoner(camera_id, calibration_profile=profile)
    outcomes = []

    def on_window(snap):
        t0 = time.perf_counter()
        analysis = reasoner.process(snap)
        done = time.perf_counter()
        recorder.record("reasoning", done - t0)
        if snap.trigger_capture_wall_s is not None:
            recorder.record("window_end_to_end", done - snap.trigger_capture_wall_s)
        outcomes.append("unavailable" if analysis.unavailable else analysis.result.candidate_ranking.outcome.value)

    max_frames = int(args.seconds * fps) if args.seconds else None
    policy = DropPolicy(args.policy)
    pipe = StreamingPipeline(src, eng, on_window=on_window, frame_queue_size=args.queue, frame_policy=policy,
                             max_frames=max_frames)
    # warm-up: one throwaway inference so CUDA init/cudnn autotune is not in the numbers
    import numpy as np
    tracker.update(np.zeros((360, 640, 3), dtype=np.uint8), 0, 0.0)
    tracker.reset()
    barrier.wait()
    stats = pipe.run()
    eng_stats = stats["engine"]
    wall = stats["wall_seconds"]
    results[stream] = {
        "camera_id": camera_id, "video": video.name,
        "source_nominal_fps": fps,
        "frames_processed": eng_stats["frames_processed"],
        "frames_dropped_total": eng_stats["frames_dropped"] + 0,
        "queue_drops": stats["frame_queue"]["dropped"],
        "wall_s": wall,
        "processing_fps": stats["processing_fps"],
        "realtime_multiple": stats["processing_fps"] / fps,
        "frame_queue": stats["frame_queue"], "window_queue": stats["window_queue"],
        "windows": eng_stats["windows_emitted"],
        "window_outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
        "latency_ms": recorder.summary_ms(),
        "identity": {"tracks_created": eng_stats["tracks_created"], "tracks_retained_end": eng_stats["tracks_retained"],
                     "stale_raw_id_restarts": eng_stats["stale_raw_id_restarts"],
                     "same_frame_raw_id_collisions": eng_stats["same_frame_raw_id_collisions"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", default="c006", help="comma list of: c006, c009, demo")
    ap.add_argument("--realtime-factor", type=float, default=None, help="None = unpaced")
    ap.add_argument("--seconds", type=float, default=0.0, help="limit each stream to N source seconds (0 = whole video)")
    ap.add_argument("--policy", default="block", choices=[p.value for p in DropPolicy])
    ap.add_argument("--queue", type=int, default=8)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    results: dict = {}
    barrier = threading.Barrier(len(streams))
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    gpu = GpuPoller(1.0)
    gpu.start()
    t0 = time.perf_counter()
    threads = [threading.Thread(target=run_stream, args=(s, args, results, barrier), name=f"bench-{s}") for s in streams]
    for t in threads:
        t.start()
    peak_rss = rss0
    while any(t.is_alive() for t in threads):
        peak_rss = max(peak_rss, proc.memory_info().rss)
        time.sleep(0.5)
    for t in threads:
        t.join()
    gpu_stats = gpu.stop()
    total_wall = time.perf_counter() - t0

    report = {
        "tag": args.tag, "streams": streams, "concurrent": len(streams), "realtime_factor": args.realtime_factor,
        "policy": args.policy, "queue": args.queue, "seconds_limit": args.seconds,
        "total_wall_s_including_model_load": total_wall,
        "aggregate_processing_fps": sum(r["processing_fps"] for r in results.values()),
        "gpu": gpu_stats, "peak_rss_mib": peak_rss / 2**20, "per_stream": results,
    }
    out = OUT / f"latency_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"== {args.tag}: {len(streams)} stream(s), realtime_factor={args.realtime_factor}, policy={args.policy}")
    print(f"   GPU: {gpu_stats}   peak RSS {peak_rss / 2**20:.0f} MiB")
    for name, r in results.items():
        print(f"   [{name}] source {r['source_nominal_fps']:.0f} fps -> processing {r['processing_fps']:.1f} fps "
              f"({r['realtime_multiple']:.2f}x real time); frames={r['frames_processed']} dropped={r['frames_dropped_total']} "
              f"queue_drops={r['queue_drops']} windows={r['windows']}")
        for stage in ("decode", "detect_track", "store_update", "windowing", "reasoning", "frame_processing",
                      "frame_end_to_end", "window_end_to_end"):
            s = r["latency_ms"].get(stage)
            if s:
                print(f"      {stage:<20} n={s['count']:<6} p50={s['p50_ms']:8.2f}  p95={s['p95_ms']:8.2f}  p99={s['p99_ms']:8.2f}  max={s['max_ms']:9.2f} ms")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
