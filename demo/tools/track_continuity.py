"""Track-continuity measurement on one segment of the demo video, for several tracker settings, with ANVESH's own tracker wrapper.

There is NO ground truth for this footage, so this reports fragmentation PROXIES only and never an ID-accuracy number:
  distinct_raw_ids        tracker IDs needed to cover the segment (more = more breaks, but also more vehicles)
  tracks_lt_1s            very short tracks (usually flicker or breaks)
  interior_births         tracks that START in the NEAR field (rows >= 200) away from the frame border (a vehicle there normally enters from an edge)
  interior_ends           tracks that END in the near field away from the frame border
  relink_candidates       an interior end followed within `RELINK_S` by an interior birth close to the predicted position.
                          A CANDIDATE only: in dense traffic that can be a different vehicle, so nothing is merged on this basis.
  identity_jumps          frame-to-frame centre moves > 0.5 box widths for one ID: a swap/teleport proxy. A looser matcher that lowers the
                          ID count by joining different vehicles shows up here as MORE jumps.
  size_jumps / tracks_with_class_change   a box that changes width by > 1.5x in one frame, or a track that changes vehicle class: the other
                          signature of one ID jumping between two different vehicles (e.g. a car and a lorry).
  mean_dets_per_frame, ms_per_frame  cost and volume.

  .venv\\Scripts\\python.exe demo/tools/track_continuity.py --start 60 --seconds 180 --out demo/runs/track_continuity.json
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402

TRACKERS = REPO / "demo" / "tools" / "trackers"
VARIANTS = {
    # name: (detector conf passed to model.track, tracker yaml)
    "baseline_conf025_default": (0.25, "bytetrack.yaml"),
    "conf010_default": (0.10, "bytetrack.yaml"),
    "conf010_buffer50": (0.10, str(TRACKERS / "anvesh_bytetrack_b50.yaml")),
    "conf010_match09": (0.10, str(TRACKERS / "anvesh_bytetrack_m09.yaml")),
    "conf010_buffer50_match09": (0.10, str(TRACKERS / "anvesh_bytetrack_b50_m09.yaml")),
}
BORDER = 24  # px from the frame edge that counts as "entering/leaving"
NEAR_Y = 200  # rows >= this are the near field (the speed zone starts at row 225). Far-field vehicles legitimately appear/vanish
#               in the interior at the horizon, so only near-field births/ends are counted as breaks.
RELINK_S = 1.5
RELINK_PX = 45.0


def interior(cx, cy, w, h):
    return BORDER < cx < w - BORDER and NEAR_Y < cy < h - BORDER


def run(model, conf, tracker_yaml, start, seconds):
    tr = ByteTrackYoloTracker(str(model), confidence_threshold=conf, tracker_config=tracker_yaml)
    cap = cv2.VideoCapture(str(REPO / "demo.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    first, last, per_frame = {}, {}, []
    prev, jumps, size_jumps = {}, 0, 0
    cls_seen = collections.defaultdict(set)  # identity-jump proxy: a track whose centre moves > 0.5 box widths in ONE frame is not one vehicle moving
    n, t0 = 0, time.time()
    while n < seconds * fps:
        ok, fr = cap.read()
        if not ok:
            break
        raw, tracked = tr.update(fr, n, n / fps)
        per_frame.append(len(raw))
        for d in tracked:
            x1, y1, x2, y2 = d.detection.bbox_xyxy
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            w = max(x2 - x1, 1.0)
            if d.track_id in prev and prev[d.track_id][0] == n - 1 and math.dist(prev[d.track_id][1], c) > 0.5 * w:
                jumps += 1
            if d.track_id in prev and prev[d.track_id][0] == n - 1:
                pw = prev[d.track_id][2]
                if max(w / pw, pw / w) > 1.5:
                    size_jumps += 1
            cls_seen[d.track_id].add(d.detection.vehicle_class.value)
            prev[d.track_id] = (n, c, w)
            first.setdefault(d.track_id, (n, c))
            last[d.track_id] = (n, c)
        n += 1
    wall = time.time() - t0
    lens = np.array([last[k][0] - first[k][0] + 1 for k in first]) / fps
    # a track still alive at the end of the segment is not a real "end"
    ib = {k for k, (f, c) in first.items() if f > 0 and interior(*c, W, H)}
    ie = {k for k, (f, c) in last.items() if f < n - 1 and interior(*c, W, H)}
    births = sorted((first[k][0], k) for k in ib)
    relink = 0
    used = set()
    for k in ie:
        fe, ce = last[k]
        for fb, kb in births:
            if kb in used or kb == k or fb <= fe or fb > fe + RELINK_S * fps:
                continue
            if math.dist(ce, first[kb][1]) <= RELINK_PX:
                relink += 1
                used.add(kb)
                break
    return {"frames": n, "tracker_yaml": str(tracker_yaml), "conf": conf, "ms_per_frame": round(1000 * wall / max(n, 1), 1),
            "mean_dets_per_frame": round(float(np.mean(per_frame)), 2), "distinct_raw_ids": len(first),
            "tracks_lt_1s": int((lens < 1).sum()), "tracks_ge_2s": int((lens >= 2).sum()),
            "median_track_len_s": round(float(np.median(lens)), 2), "mean_track_len_s": round(float(lens.mean()), 2),
            "interior_births": len(ib), "interior_ends": len(ie), "relink_candidates": relink, "identity_jumps": jumps, "size_jumps": size_jumps,
            "tracks_with_class_change": sum(1 for v in cls_seen.values() if len(v) > 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=60)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--model", default=str(REPO / "models" / "yolov8n.pt"))
    ap.add_argument("--out", default=str(REPO / "demo" / "runs" / "track_continuity.json"))
    a = ap.parse_args()
    res = {"segment": {"video": "demo.mp4", "start_s": a.start, "seconds": a.seconds}, "labels": "NONE (proxies only; no accuracy claim)",
           "model": a.model, "variants": {}}
    for name in a.variants.split(","):
        conf, yaml = VARIANTS[name]
        res["variants"][name] = run(a.model, conf, yaml, a.start, a.seconds)
        print(name, json.dumps(res["variants"][name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
