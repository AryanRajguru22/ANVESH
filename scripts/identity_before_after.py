"""Duplicate-identity before/after on ANY video (no ground truth needed): run
the REAL tracker once, feed the identical per-frame tracker output to the
frozen legacy TrackManager and to the fixed one, and report the duplicate
statistics (`mot_metrics.duplicate_identity_stats`, per-window duplicate
entries) for each. Used for `demo.mp4`, which has no ground truth.

Usage:
    python scripts/identity_before_after.py --video demo.mp4 --seconds 240 --tag demo_240s
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stream_common import MODEL, OUT, REPO_ROOT  # noqa: E402

from eval_cityflow import _lifetime_stats, replay_legacy, replay_new  # noqa: E402

from anvesh.evaluation.mot_metrics import duplicate_identity_stats, duplicate_ids_per_window  # noqa: E402
from anvesh.perception.detection import COCO_CLASS_NAMES, COCO_TO_VEHICLE_CLASS  # noqa: E402
from anvesh.perception.ingestion import VideoFileFrameSource  # noqa: E402
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402
from anvesh.perception.track_manager import occlusion_horizon_s  # noqa: E402

NAME_TO_ID = {COCO_CLASS_NAMES[k]: k for k in COCO_TO_VEHICLE_CLASS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(REPO_ROOT / "demo.mp4"))
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--tag", default="demo_240s")
    args = ap.parse_args()

    src = VideoFileFrameSource(args.video)
    fps = src.fps()
    tracker = ByteTrackYoloTracker(model_name=MODEL, confidence_threshold=0.25)
    frames = []
    for frame in itertools.islice(src, int(args.seconds * fps)):
        raw, tracked = tracker.update(frame.image, frame.index, frame.timestamp)
        ids = {td.detection.bbox_xyxy: td.track_id for td in tracked}
        trk = []
        for d in raw:
            tid = ids.get(d.bbox_xyxy)
            trk.append([*d.bbox_xyxy, d.confidence, NAME_TO_ID.get(d.class_name, 2), None if tid is None else int(tid)])
        frames.append({"i": frame.index, "trk": trk})

    legacy = replay_legacy(frames, fps, 0, None)
    horizon = occlusion_horizon_s(tracker.lost_buffer_frames, fps)
    after = replay_new(frames, fps, 0, None, horizon, "fixed")

    def dup(d):
        s = duplicate_identity_stats(d["state_keys"])
        entries, dups = duplicate_ids_per_window(d["window_ids"])
        return {**dataclasses.asdict(s), "window_entries": entries, "window_duplicate_entries": dups,
                "window_duplicate_rate": dups / entries if entries else 0.0}

    report = {
        "video": Path(args.video).name, "seconds": args.seconds, "fps": fps, "frames": len(frames),
        "tracker_lost_buffer_frames": tracker.lost_buffer_frames, "fixed_horizon_s": horizon,
        "before_legacy": {**dup(legacy), "lifetimes": _lifetime_stats(legacy["states"])},
        "after_fixed": {**dup(after), "lifetimes": _lifetime_stats(after["states"])},
    }
    out = OUT / f"identity_before_after_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k in ("before_legacy", "after_fixed"):
        r = report[k]
        print(f"{k:<14} states={r['states']} unique_keys={r['identity_keys']} duplicated_keys={r['duplicated_keys']} "
              f"dup_rate={r['duplicate_rate']:.4f} | per-window duplicate entries {r['window_duplicate_entries']}/{r['window_entries']} "
              f"({100 * r['window_duplicate_rate']:.1f}%) | median life {r['lifetimes']['median_s']:.2f}s, <2s: {100 * r['lifetimes']['frac_lt_2s']:.0f}%")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
