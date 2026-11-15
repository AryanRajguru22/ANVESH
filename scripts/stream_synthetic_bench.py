"""Reproducible SYNTHETIC-stream benchmark of the ANVESH streaming pipeline (plumbing only).

WHAT THIS MEASURES: the threaded pipeline (reader -> bounded frame queue -> worker -> bounded window queue -> consumer), the
streaming engine, windowing, stop-episode bookkeeping and the RTSP source's reconnect machinery, driven by (a) a scripted
tracker that stands in for the detector and (b) in-memory frames or an injected fake capture. A configurable sleep can stand
in for detector latency so that queue/back-pressure behaviour can be exercised.

WHAT IT DOES NOT MEASURE: YOLO/GPU inference, video decode, JPEG evidence encoding, the HTTP/dashboard layer, a real webcam, a real
RTSP/RTMP server, or any camera. Numbers from here must never be quoted as application latency or as live-camera acceptance.

PROCEDURE (so a number can be reproduced and judged): every scenario is run `--repeats` times (default 5) in a fresh pipeline,
sequentially, after one discarded warm-up run. For each run we record wall time, throughput, frames offered/processed/dropped,
queue high-water marks and the engine's own per-stage latency percentiles. We then report min / median / max across the repeats
for the per-run p50 and p95 and flag a scenario UNSTABLE when max/min of the per-run p95 exceeds `--stability-ratio` (default 1.5):
an unstable scenario's numbers are noise on this machine and must not be quoted. One run is never a result.

    .venv\\Scripts\\python.exe scripts\\stream_synthetic_bench.py --repeats 5 --out outputs\\exec4\\stream_synthetic_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvesh.perception.ingestion import SourceHealth  # noqa: E402
from anvesh.perception.types import Frame, RawDetection, TrackedDetection  # noqa: E402
from anvesh.storage.schemas import VehicleClass  # noqa: E402
from anvesh.streaming.engine import StreamingPerceptionEngine  # noqa: E402
from anvesh.streaming.pipeline import StreamingPipeline  # noqa: E402
from anvesh.streaming.queues import DropPolicy  # noqa: E402
from anvesh.streaming.rtsp import RtspFrameSource  # noqa: E402
from anvesh.streaming.windows import WindowSpec  # noqa: E402

FPS = 10.0
IMG = np.zeros((12, 16, 3), dtype=np.uint8)


class StubTracker:
    """Stands in for the detector+tracker: a few moving vehicles and one parked one, optional fixed latency."""

    lost_buffer_frames = 30

    def __init__(self, latency_s: float = 0.0) -> None:
        self._latency = latency_s

    def reset(self) -> None:
        pass

    def update(self, image, frame_index: int, timestamp: float):
        if self._latency:
            time.sleep(self._latency)
        raw, tracked = [], []
        objs = [("p", (30.0, 30.0))] + [(f"v{j}", (60.0 + (frame_index - j * 20) * 2, 40.0 + 5 * (j % 5)))
                                         for j in range(max(0, frame_index // 20 - 1), frame_index // 20 + 1) if 0 <= frame_index - j * 20 < 45]
        for rid, (x, y) in objs:
            d = RawDetection(frame_index=frame_index, timestamp=timestamp, class_name="car", vehicle_class=VehicleClass.CAR, confidence=0.9,
                             bbox_xyxy=(x - 5, y - 5, x + 5, y + 5))
            raw.append(d)
            tracked.append(TrackedDetection(detection=d, track_id=rid))
        return raw, tracked


class StampedSource:
    """In-memory source that stamps each frame with the wall time at which it is handed to the pipeline (so end-to-end latency
    includes queueing). Not paced: it offers frames as fast as the pipeline will take them unless `pace_fps` is given."""

    def __init__(self, n: int, pace_fps: float = None) -> None:
        self._n, self._pace = n, pace_fps
        self.closed = False

    def fps(self) -> float:
        return FPS

    def health(self) -> SourceHealth:
        return SourceHealth("connected", 0, self._n, 0, 0, FPS, 0.0)

    def close(self) -> None:
        self.closed = True

    def __iter__(self):
        t0 = time.perf_counter()
        for i in range(self._n):
            if self.closed:
                return
            if self._pace:
                wait = t0 + i / self._pace - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            yield Frame(index=i, timestamp=i / FPS, image=IMG, capture_wall_s=time.perf_counter(), decode_s=0.0)


class FlappingCapture:
    """Fake cv2.VideoCapture that delivers `n_ok` frames then dies (a dropped connection). No real I/O."""

    def __init__(self, clock, n_ok: int) -> None:
        self._clock, self._n = clock, n_ok

    def isOpened(self):
        return True

    def get(self, prop):
        return FPS

    def read(self):
        if self._n <= 0:
            return False, None
        self._n -= 1
        self._clock.t += 1.0 / FPS
        return True, IMG

    def release(self):
        pass


class VirtualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _engine(camera: str, tracker) -> StreamingPerceptionEngine:
    return StreamingPerceptionEngine(camera, tracker, FPS, windows=WindowSpec(10.0), run_id=camera)


def _summarise_run(stats: dict, eng, wall: float) -> dict:
    lat = eng.recorder.summary_ms()
    e2e = lat.get("frame_end_to_end", {})
    proc = lat.get("detect_track", {})
    e = stats["engine"]
    fq, wq = stats["frame_queue"], stats["window_queue"]
    offered = e["frames_processed"] + fq["dropped"]
    return {"wall_s": wall, "offered": offered, "processed": e["frames_processed"], "queue_dropped": fq["dropped"],
            "engine_gap_frames": e["frames_dropped"], "windows_dropped": wq["dropped"],
            "throughput_fps": e["frames_processed"] / wall if wall else 0.0,
            "frame_queue_high_water": fq["high_water"], "window_queue_high_water": wq["high_water"],
            "e2e_p50_ms": e2e.get("p50_ms"), "e2e_p95_ms": e2e.get("p95_ms"),
            "detect_track_p50_ms": proc.get("p50_ms"), "detect_track_p95_ms": proc.get("p95_ms")}


def run_single(frames: int, policy: DropPolicy, detector_ms: float, qsize: int, pace_fps: float = None) -> dict:
    eng = _engine("bench", StubTracker(detector_ms / 1000.0))
    pipe = StreamingPipeline(StampedSource(frames, pace_fps), eng, on_window=lambda s: None, frame_queue_size=qsize, frame_policy=policy)
    t0 = time.perf_counter()
    stats = pipe.run()
    return _summarise_run(stats, eng, time.perf_counter() - t0)


def run_concurrent(n_sources: int, frames: int, detector_ms: float, qsize: int, pace_fps: float = None) -> dict:
    out, errs = [None] * n_sources, []

    def one(k):
        try:
            eng = _engine(f"bench{k}", StubTracker(detector_ms / 1000.0))
            pipe = StreamingPipeline(StampedSource(frames, pace_fps), eng, on_window=lambda s: None, frame_queue_size=qsize,
                                     frame_policy=DropPolicy.DROP_OLDEST)
            t0 = time.perf_counter()
            stats = pipe.run()
            out[k] = _summarise_run(stats, eng, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            errs.append(repr(exc))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(k,)) for k in range(n_sources)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    done = [o for o in out if o]
    agg = {"wall_s": wall, "sources": n_sources, "errors": errs,
           "offered": sum(o["offered"] for o in done), "processed": sum(o["processed"] for o in done),
           "queue_dropped": sum(o["queue_dropped"] for o in done),
           "throughput_fps": sum(o["processed"] for o in done) / wall if wall else 0.0,
           "frame_queue_high_water": max((o["frame_queue_high_water"] for o in done), default=0),
           "window_queue_high_water": max((o["window_queue_high_water"] for o in done), default=0),
           "e2e_p50_ms": statistics.median([o["e2e_p50_ms"] for o in done if o["e2e_p50_ms"] is not None] or [None]) if done else None,
           "e2e_p95_ms": max([o["e2e_p95_ms"] for o in done if o["e2e_p95_ms"] is not None] or [None]) if done else None}
    return agg


def run_flapping(frames: int, outage_every: int) -> dict:
    """RtspFrameSource over a fake capture that drops every `outage_every` frames; virtual clock, so it is fast and deterministic."""
    clock = VirtualClock()
    caps = [FlappingCapture(clock, outage_every) for _ in range(frames // outage_every + 2)]
    it = iter(caps)
    src = RtspFrameSource("rtsp://synthetic/fake", nominal_fps=FPS, capture_factory=lambda u: next(it), clock=clock, sleep=clock.sleep,
                          backoff_s=(0.5,), max_frames=frames)
    eng = _engine("flap", StubTracker())
    pipe = StreamingPipeline(src, eng, on_window=lambda s: None, frame_queue_size=8, frame_policy=DropPolicy.BLOCK)
    t0 = time.perf_counter()
    stats = pipe.run()
    r = _summarise_run(stats, eng, time.perf_counter() - t0)
    h = src.health()
    # The source runs on a virtual clock, so frame timestamps and the engine's wall clock are unrelated: latency is not measured here.
    for k in ("e2e_p50_ms", "e2e_p95_ms", "detect_track_p50_ms", "detect_track_p95_ms"):
        r[k] = None
    r.update({"reconnects": h.reconnects, "source_gap_frames": h.source_gap_frames, "final_epoch": h.epoch,
              "note": "plumbing correctness under repeated disconnects; latency not measured (virtual clock)"})
    return r


# Sources are PACED (frames offered at a fixed rate) wherever drops are being measured: an unpaced in-memory source offers frames
# infinitely fast, which would make any bounded queue "drop 99%" and say nothing about a camera.
SCENARIOS = {
    "steady_lossless_1src_unpaced": lambda a: run_single(a.frames, DropPolicy.BLOCK, 0.0, 8),
    "live_keeps_up_1src_100fps_5ms": lambda a: run_single(500, DropPolicy.DROP_OLDEST, 5.0, 4, pace_fps=100.0),
    "live_overload_1src_400fps_5ms": lambda a: run_single(1200, DropPolicy.DROP_OLDEST, 5.0, 4, pace_fps=400.0),
    "concurrent_4src_50fps_2ms": lambda a: run_concurrent(4, 250, 2.0, 4, pace_fps=50.0),
    "concurrent_8src_50fps_2ms": lambda a: run_concurrent(8, 250, 2.0, 4, pace_fps=50.0),
    "flapping_rtsp_fake_every_200": lambda a: run_flapping(a.frames, 200),
}


def _spread(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "median": None, "max": None, "max_over_min": None}
    return {"min": min(vals), "median": statistics.median(vals), "max": max(vals), "max_over_min": (max(vals) / min(vals)) if min(vals) > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--stability-ratio", type=float, default=1.5)
    ap.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), help="default: all")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.repeats < 3:
        print("refusing: fewer than 3 repeats cannot say anything about stability", file=sys.stderr)
        return 2
    report = {"label": "SYNTHETIC pipeline benchmark - stub tracker, in-memory/fake sources; NOT detector, camera, webcam, RTSP or RTMP performance",
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              "settings": {"frames": args.frames, "repeats": args.repeats, "stability_ratio": args.stability_ratio}, "scenarios": {}}
    for name in args.scenario or list(SCENARIOS):
        SCENARIOS[name](args)  # discarded warm-up
        runs = [SCENARIOS[name](args) for _ in range(args.repeats)]
        summary = {k: _spread([r.get(k) for r in runs]) for k in
                   ("throughput_fps", "e2e_p50_ms", "e2e_p95_ms", "queue_dropped", "frame_queue_high_water", "window_queue_high_water")}
        ratio = summary["e2e_p95_ms"]["max_over_min"]
        thr_ratio = summary["throughput_fps"]["max_over_min"]
        unstable = (ratio is not None and ratio > args.stability_ratio) or (thr_ratio is not None and thr_ratio > args.stability_ratio)
        drops = sum(r.get("queue_dropped", 0) for r in runs)
        report["scenarios"][name] = {"runs": runs, "summary": summary, "stable": not unstable,
                                     "verdict": ("UNSTABLE on this machine: do not quote these numbers" if unstable else "spread within the stability ratio"),
                                     "total_queue_drops_over_runs": drops}
        def fmt(d):
            return "n/a" if d["min"] is None else f"{d['min']:.1f}/{d['median']:.1f}/{d['max']:.1f}"

        print(f"{name:34s} thr fps min/med/max {fmt(summary['throughput_fps'])}  e2e p95 ms {fmt(summary['e2e_p95_ms'])}"
              f"  queue drops (all runs) {drops}  {'STABLE' if not unstable else 'UNSTABLE'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
