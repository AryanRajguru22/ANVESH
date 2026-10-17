"""Batch / stream parity on REAL data (CityFlow S02/c006, 1920x1080 @ 10 fps).

"Same source, same configuration" run through four execution modes, then the
resulting WORLD-MODEL content (window ids/bounds, track identities, first/last
seen, speed, class, occlusion state, geometry validity, position counts) and
the reasoner's per-window outcome are compared:

  SYNC      the engine driven frame-by-frame in one thread, as fast as possible
            ("batch-style")
  THREADED  reader -> bounded queue (BLOCK, lossless) -> worker -> window queue
            -> consumer, as fast as possible
  PACED     the same threaded pipeline fed by a file source paced at
            `--paced-factor` x real time (file-as-live-stream)
  POSTHOC   legacy-shaped batch: `run_m1_pipeline` over the whole file, THEN
            windows cut from the FINAL track states (what a recorded-file
            consumer that waits for EOF sees)

Part A replays the cached REAL tracker output (`CachedTracker`), so the only
variable is execution mode: identical frames in -> the claim is EXACT equality.
Part B runs the REAL detector+tracker (YOLOv8n+ByteTrack on the GPU) twice in
SYNC and once THREADED over the first N seconds, quantifying how much
GPU/runtime nondeterminism (not logic) moves the output.

TOLERANCE STATEMENT (see docs/M-A report): with lossless queues and no source
gaps the engine is a deterministic function of the frame sequence, so
SYNC == THREADED == PACED exactly (Part A). POSTHOC differs from the streaming
windows ONLY in `occlusion_state` for tracks whose fate (PARTIAL -> LOST) was
decided after the window closed, because a post-hoc consumer sees the future;
that difference is counted, not hidden. Part B's tolerance is whatever the
measured run-to-run difference of the real detector is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stream_common import MODEL, OUT, REPO_ROOT, CachedTracker, diff_windows, opened_episodes, window_signature  # noqa: E402
from anvesh.safety.stop_episodes import episode_views, stop_episodes_from_track_states  # noqa: E402

from anvesh.evaluation import cityflow_gt as gt_io  # noqa: E402
from anvesh.perception.ingestion import VideoFileFrameSource  # noqa: E402
from anvesh.perception.pipeline import run_m1_pipeline  # noqa: E402
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402
from anvesh.perception.windowing import generate_windows  # noqa: E402
from anvesh.streaming.engine import StreamingPerceptionEngine  # noqa: E402
from anvesh.streaming.pipeline import StreamingPipeline  # noqa: E402
from anvesh.streaming.queues import DropPolicy  # noqa: E402
from anvesh.streaming.reasoning import WindowReasoner  # noqa: E402
from anvesh.streaming.store import build_window_tracks  # noqa: E402
from anvesh.streaming.windows import WindowSnapshot, WindowSpec  # noqa: E402
from anvesh.world.calibration import load_cityflow_calibration  # noqa: E402

CAMERA = "s02_c006"
WINDOW_S = 10.0


def _profile():
    return load_cityflow_calibration("calib-s02-c006", CAMERA, gt_io.camera_dir("S02", "c006") / "calibration.txt")


def _outcome_sig(analysis) -> tuple:
    if analysis.unavailable:  # M-A.1: an unavailable window has no reasoning result
        return ("unavailable", None, 0, None, 0.0)
    r = analysis.result
    return (
        r.candidate_ranking.outcome.value,
        analysis.aggregation.traffic_state.congestion_level.value,
        analysis.aggregation.traffic_state.vehicle_count,
        r.candidate_vehicle.status.value,
        round(r.reliability_by_camera[CAMERA].score, 9),
    )


def _make_engine(tracker, fps, profile):
    return StreamingPerceptionEngine(CAMERA, tracker, fps, windows=WindowSpec(WINDOW_S), calibration_profile=profile,
                                     run_id="parity")  # the comparison needs both modes to share one run identity


def run_sync(video, tracker_factory, profile, max_frames=None):
    src = VideoFileFrameSource(video)
    fps = src.fps()
    tracker = tracker_factory()
    eng = _make_engine(tracker, fps, profile)
    reasoner = WindowReasoner(CAMERA, calibration_profile=profile)
    sigs, outs = [], []
    frames = itertools.islice(src, max_frames) if max_frames else src
    t0 = time.perf_counter()
    for frame in frames:
        for snap in eng.process_frame(frame):
            sigs.append(window_signature(snap))
            outs.append(_outcome_sig(reasoner.process(snap)))
    for snap in eng.finish():
        sigs.append(window_signature(snap))
        outs.append(_outcome_sig(reasoner.process(snap)))
    return sigs, outs, {"wall_s": time.perf_counter() - t0, **eng.stats()}


def run_threaded(video, tracker_factory, profile, max_frames=None, realtime_factor=None):
    src = VideoFileFrameSource(video, realtime_factor=realtime_factor)
    fps = src.fps()
    eng = _make_engine(tracker_factory(), fps, profile)
    reasoner = WindowReasoner(CAMERA, calibration_profile=profile)
    sigs, outs = [], []

    def on_window(snap):
        sigs.append(window_signature(snap))
        outs.append(_outcome_sig(reasoner.process(snap)))

    pipe = StreamingPipeline(src, eng, on_window=on_window, frame_queue_size=8, frame_policy=DropPolicy.BLOCK,
                             max_frames=max_frames)
    stats = pipe.run()
    return sigs, outs, stats


def run_posthoc(video, tracker_factory, profile, max_frames=None):
    """Legacy-shaped batch: whole file first, windows from FINAL states, then
    the SAME reasoner (so the downstream effect of post-hoc occlusion states is
    measured, not assumed)."""
    if max_frames:
        raise ValueError("posthoc runs the whole file")
    src = VideoFileFrameSource(video)
    fps = src.fps()
    result = run_m1_pipeline(video, CAMERA, tracker_factory(), "cached", "0", calibration_profile=profile, run_id="parity")
    duration = result.frame_count / fps
    sigs, outs = [], []
    posthoc_episodes = stop_episodes_from_track_states(CAMERA, result.track_states, profile)
    reasoner = WindowReasoner(CAMERA, calibration_profile=profile)
    for w in generate_windows(duration, WINDOW_S):
        tracks = build_window_tracks(result.track_states, w.start, w.end, CAMERA, profile)
        snap = WindowSnapshot(
            window_index=w.index, window_id=f"{CAMERA}:w{w.index:06d}", camera_id=CAMERA, start=w.start, end=w.end,
            tracks=tuple(tracks), frames_observed=0, dropped_frames=0, sessions=("s000",),
            closed_at_stream_time=w.end, final_partial=False,
        )
        outs.append(_outcome_sig(reasoner.process(snap)))
        sigs.append((
            f"{CAMERA}:w{w.index:06d}", w.start, w.end, None, None, w.end < w.start + WINDOW_S - 1e-9,
            tuple(
                (t.track_id, t.first_seen, t.last_seen, t.vehicle_class.value, t.occlusion_state.value,
                 t.motion_space.value, t.geometry_validity.value, t.speed_available,
                 round(t.speed_estimate.value, 9), len(t.position_history))
                for t in tracks
            ),
            # post-hoc windows carry no observability data; their stop episodes come from the FINAL states
            ("observed", 1.0, 0.0, None,
             tuple((v.phase.value, v.episode.episode_id, v.episode.start, v.episode.confirmed_at)
                   for v in episode_views(posthoc_episodes, w.start, w.end))),
        ))
    return sigs, outs, {"frames": result.frame_count, "posthoc_episodes": len(posthoc_episodes), **result.engine_stats}


def _strip_counts(sigs):
    """Drop frames_observed/dropped and the observability element (not available post-hoc) for comparison."""
    return [(s[0], s[1], s[2], None, None, s[5], s[6]) for s in sigs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paced-factor", type=float, default=8.0)
    ap.add_argument("--real-seconds", type=float, default=60.0, help="Part B duration")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write stream_parity.json (default: outputs/m_a; the M-A.1 run uses outputs/m_a1)")
    args = ap.parse_args()

    video = gt_io.camera_dir("S02", "c006") / "vdo.avi"
    cache = REPO_ROOT / "outputs" / "cityflow" / "s02_c006_perception_cache.jsonl.gz"
    profile = _profile()
    cached = lambda: CachedTracker(cache)  # noqa: E731
    report = {"camera": CAMERA, "geometry_validity": profile.geometry_validity.value, "window_s": WINDOW_S}

    print("PART A: cached real tracker output, identical frames in")
    sync_sigs, sync_out, sync_stats = run_sync(video, cached, profile)
    print(f"  SYNC      windows={len(sync_sigs)} wall={sync_stats['wall_s']:.1f}s frames={sync_stats['frames_processed']}")
    thr_sigs, thr_out, thr_stats = run_threaded(video, cached, profile)
    print(f"  THREADED  windows={len(thr_sigs)} wall={thr_stats['wall_seconds']:.1f}s dropped={thr_stats['engine']['frames_dropped']}")
    paced_sigs, paced_out, paced_stats = run_threaded(video, cached, profile, realtime_factor=args.paced_factor)
    print(f"  PACED x{args.paced_factor:g} windows={len(paced_sigs)} wall={paced_stats['wall_seconds']:.1f}s dropped={paced_stats['engine']['frames_dropped']}")
    post_sigs, post_out, post_stats = run_posthoc(video, cached, profile)
    print(f"  POSTHOC   windows={len(post_sigs)}")

    report["part_a_cached"] = {
        "sync_vs_threaded": {**diff_windows(sync_sigs, thr_sigs), "reasoner_outcomes_identical": sync_out == thr_out},
        "sync_vs_paced": {**diff_windows(sync_sigs, paced_sigs), "reasoner_outcomes_identical": sync_out == paced_out,
                          "paced_factor": args.paced_factor},
        "sync_vs_posthoc_batch": {
            **diff_windows(_strip_counts(sync_sigs), _strip_counts(post_sigs)),
            "reasoner_outcome_flips": sum(1 for a, b in zip(sync_out, post_out) if a[:4] != b[:4]),
            "reliability_score_deltas": [round(b[4] - a[4], 6) for a, b in zip(sync_out, post_out)],
            "max_abs_reliability_delta": max((abs(b[4] - a[4]) for a, b in zip(sync_out, post_out)), default=0.0),
            "windows_with_reliability_delta": sum(1 for a, b in zip(sync_out, post_out) if a[4] != b[4]),
        },
        # M-A.1: the new window fields are part of the signatures compared above (sync/threaded/paced must
        # agree EXACTLY on availability, coverage and stop-episode phases). Against the post-hoc batch only the
        # episodes that OPENED are comparable (phase CONTINUING vs CLOSED depends on look-ahead).
        "stop_episodes": {
            "availability_per_window_sync": sorted({s[7][0] for s in sync_sigs}),
            "opened_sync": opened_episodes(sync_sigs),
            "opened_posthoc": opened_episodes(post_sigs),
            "opened_identical_sync_vs_posthoc": opened_episodes(sync_sigs) == opened_episodes(post_sigs),
            "opened_identical_sync_vs_threaded_vs_paced": opened_episodes(sync_sigs) == opened_episodes(thr_sigs) == opened_episodes(paced_sigs),
            "posthoc_total_episodes": post_stats["posthoc_episodes"],
        },
        "queue": {"frame_queue_high_water": thr_stats["frame_queue"]["high_water"],
                  "frame_queue_capacity": thr_stats["frame_queue"]["capacity"],
                  "frames_dropped": thr_stats["engine"]["frames_dropped"]},
        "windows": len(sync_sigs), "tracks_created": sync_stats["tracks_created"],
        "outcomes_per_window": sync_out,
    }
    for k in ("sync_vs_threaded", "sync_vs_paced", "sync_vs_posthoc_batch"):
        d = report["part_a_cached"][k]
        print(f"  {k}: identical={d['identical']} differing_windows={d['differing_windows']} "
              f"field_diff_tracks={d['tracks_with_field_diffs']} (occlusion-state-only={d['occlusion_state_only_diffs']}) "
              f"only_a={d['tracks_only_in_a']} only_b={d['tracks_only_in_b']}")
    se = report["part_a_cached"]["stop_episodes"]
    print(f"  stop episodes: availability per window={se['availability_per_window_sync']} opened(sync)={len(se['opened_sync'])} "
          f"opened(posthoc)={len(se['opened_posthoc'])} identical sync/posthoc={se['opened_identical_sync_vs_posthoc']} "
          f"identical sync/threaded/paced={se['opened_identical_sync_vs_threaded_vs_paced']}")
    d = report["part_a_cached"]["sync_vs_posthoc_batch"]
    print(f"  posthoc downstream effect: outcome flips={d['reasoner_outcome_flips']}/{len(sync_out)}, windows with reliability delta="
          f"{d['windows_with_reliability_delta']}, max |reliability delta|={d['max_abs_reliability_delta']:.4f}")

    if not args.skip_real:
        n = int(args.real_seconds * 10)
        print(f"\nPART B: REAL YOLOv8n+ByteTrack on GPU, first {args.real_seconds:g}s ({n} frames)")
        real = lambda: ByteTrackYoloTracker(model_name=MODEL, confidence_threshold=0.25)  # noqa: E731
        a_sigs, a_out, _ = run_sync(video, real, profile, max_frames=n)
        b_sigs, b_out, _ = run_sync(video, real, profile, max_frames=n)
        c_sigs, c_out, c_stats = run_threaded(video, real, profile, max_frames=n)
        report["part_b_real"] = {
            "frames": n,
            "sync_run1_vs_sync_run2 (run-to-run nondeterminism)": {**diff_windows(a_sigs, b_sigs), "outcomes_identical": a_out == b_out},
            "sync_vs_threaded": {**diff_windows(a_sigs, c_sigs), "outcomes_identical": a_out == c_out},
            "frames_dropped": c_stats["engine"]["frames_dropped"],
        }
        for k, d in list(report["part_b_real"].items())[1:3]:
            print(f"  {k}: identical={d['identical']} differing_windows={d['differing_windows']} "
                  f"field_diff_tracks={d['tracks_with_field_diffs']} only_a={d['tracks_only_in_a']} only_b={d['tracks_only_in_b']} "
                  f"max|dspeed|={d['max_abs_speed_diff']:.3g} outcomes_identical={d['outcomes_identical']}")

    out_dir = args.out_dir if args.out_dir is not None else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "stream_parity.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
