"""Collects the raw material for the speed audit: runs the REAL detector + tracker (the same ByteTrackYoloTracker the app uses, default
settings) over a stretch of the demo video and stores every tracked box, with the frame's video timestamp.

    .venv\\Scripts\\python.exe demo/tools/speed_audit_collect.py --video demo.mp4 --seconds 420 --out demo/runs/speed_audit_tracks.json

Also records the container's own presentation timestamps (CAP_PROP_POS_MSEC) next to index/fps, to test whether the video has a constant frame
rate (a variable-frame-rate file would make `index / fps` wrong)."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REPO / "demo.mp4"))
    ap.add_argument("--model", default=str(REPO / "models" / "yolov8n.pt"))
    ap.add_argument("--seconds", type=float, default=420.0)
    ap.add_argument("--out", default=str(REPO / "demo" / "runs" / "speed_audit_tracks.json"))
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_max = int(a.seconds * fps)
    tracker = ByteTrackYoloTracker(model_name=a.model)
    rows, pts_dev, t0 = [], [], time.time()
    for idx in range(n_max):
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # position of the frame about to be read
        ok, img = cap.read()
        if not ok:
            break
        ts = idx / fps
        pts_dev.append(round(pos_ms / 1000.0 - ts, 5))
        _raw, tracked = tracker.update(img, idx, ts)
        for td in tracked:
            d = td.detection
            x1, y1, x2, y2 = d.bbox_xyxy
            rows.append([idx, round(ts, 4), int(td.track_id), d.vehicle_class.value if hasattr(d.vehicle_class, "value") else str(d.vehicle_class),
                         round(float(d.confidence), 3), round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)])
        if idx % 1000 == 0:
            print(f"frame {idx}/{n_max}  {time.time() - t0:.0f}s", flush=True)
    out = {"video": Path(a.video).name, "fps": fps, "frame_size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
           "frames": idx + 1, "model": Path(a.model).name,
           "pos_msec_minus_index_over_fps_s": {"n": len(pts_dev), "min": min(pts_dev), "max": max(pts_dev), "first": pts_dev[:5], "last": pts_dev[-5:]},
           "columns": ["frame", "t", "raw_id", "class", "conf", "x1", "y1", "x2", "y2"], "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print("saved", a.out, len(rows), "boxes")


if __name__ == "__main__":
    main()
