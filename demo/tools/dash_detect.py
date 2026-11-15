"""Finds the lane-line dashes of the demo camera on a temporal-median background of the video (moving vehicles disappear) and writes their
sub-pixel end points. Purpose: measure MORE dashes than the original calibration used, so the far field can be calibrated instead of extrapolated.

    .venv\\Scripts\\python.exe demo/tools/dash_detect.py --video demo.mp4 --out demo/runs/dash_detect.json --png demo/runs/dash_detect.png

Method: median of N frames sampled across the video -> grayscale -> top-hat (bright thin structures) -> threshold -> connected components -> for each
component the principal axis through its pixels and the two extreme projections give sub-pixel end points. It reports every component in the road
region; deciding which are the dashes of which lane line (and their order) is done by the caller from the known near dashes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="demo.mp4")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--span-s", type=float, default=900.0)
    ap.add_argument("--out", default="demo/runs/dash_detect.json")
    ap.add_argument("--png", default="demo/runs/dash_detect.png")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idxs = np.linspace(0, int(a.span_s * fps), a.frames).astype(int)
    stack = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            stack.append(f)
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    _t, bw = cv2.threshold(th, 28, 255, cv2.THRESH_BINARY)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 6 or y < 100:
            continue
        ys, xs = np.nonzero(lab == k)
        pts = np.stack([xs, ys], 1).astype(float)
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        if d[1] < 0:
            d = -d                       # point downwards in the image: first end = far, second end = near
        proj = (pts - c) @ d
        far, near = c + d * proj.min(), c + d * proj.max()
        comps.append({"centre": [round(float(c[0]), 1), round(float(c[1]), 1)], "far_end": [round(float(far[0]), 1), round(float(far[1]), 1)],
                      "near_end": [round(float(near[0]), 1), round(float(near[1]), 1)], "length_px": round(float(proj.max() - proj.min()), 1), "area": int(area)})
    comps.sort(key=lambda c: (c["centre"][1], c["centre"][0]))
    Path(a.out).write_text(json.dumps({"frames_used": len(stack), "components": comps}, indent=1), encoding="utf-8")
    vis = cv2.resize(bg, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for c in comps:
        for e, col in (("far_end", (0, 0, 255)), ("near_end", (0, 255, 0))):
            cv2.circle(vis, (int(c[e][0] * 2), int(c[e][1] * 2)), 3, col, -1)
    cv2.imwrite(a.png, vis)
    cv2.imwrite(a.png.replace(".png", "_bg.png"), bg)
    print("components:", len(comps), "->", a.out)


if __name__ == "__main__":
    main()
