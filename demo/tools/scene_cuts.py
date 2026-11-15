"""Scene-cut scan: is a video one continuous camera shot? Samples every 0.5 s, compares colour histograms of consecutive samples (Bhattacharyya
distance) and reports the cuts (distance above a threshold). Used to confirm demo.mp4 (first N seconds) is one shot and traffic.mp4 is a montage.

   python demo/tools/scene_cuts.py --video demo.mp4 --seconds 1200 --out demo/runs/scene_cuts_demo.json
"""
import argparse
import json
from pathlib import Path

import cv2


def scan(path, seconds, step_s=0.5, thresh=0.45):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(n, int(seconds * fps)) if seconds else n
    step = max(1, int(round(step_s * fps)))
    prev, cuts, dists, i = None, [], [], 0
    while i < limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        h = cv2.calcHist([cv2.resize(fr, (160, 90))], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        cv2.normalize(h, h, 1.0, 0, cv2.NORM_L1)
        if prev is not None:
            d = cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)
            dists.append(d)
            if d > thresh:
                cuts.append({"t_s": round(i / fps, 1), "distance": round(float(d), 3)})
        prev = h
        i += step
    return {"video": str(path), "fps": fps, "frames_total": n, "scanned_s": round(limit / fps, 1), "samples": len(dists), "threshold": thresh,
            "max_distance": round(float(max(dists)), 3) if dists else None, "cuts": cuts, "n_cuts": len(cuts)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = scan(a.video, a.seconds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "cuts"}), "cuts:", r["cuts"][:12])
