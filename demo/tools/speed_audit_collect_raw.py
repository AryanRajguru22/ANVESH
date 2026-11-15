"""Collects BOTH the raw detector box and the tracker's Kalman-filtered box for every tracked vehicle.

Ultralytics' `model.track()` returns the tracker's (Kalman-filtered) box, not the detector's raw box. This script runs the same model and the same
ByteTrack class (`ultralytics.trackers.byte_tracker.BYTETracker`, config `bytetrack.yaml`, conf 0.25, same class filter) step by step so that the
detection index returned by the tracker can be used to read the RAW box of each tracked vehicle.

    .venv\\Scripts\\python.exe demo/tools/speed_audit_collect_raw.py --seconds 240 --out demo/runs/speed_audit_tracks_raw.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    from ultralytics import YOLO
    from ultralytics.cfg import IterableSimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    from anvesh.perception.detection import ClassMap

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REPO / "demo.mp4"))
    ap.add_argument("--model", default=str(REPO / "models" / "yolov8n.pt"))
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--out", default=str(REPO / "demo" / "runs" / "speed_audit_tracks_raw.json"))
    a = ap.parse_args()
    model = YOLO(a.model)
    classes = ClassMap.from_model_names(model.names).class_ids
    tracker = BYTETracker(args=IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))  # exactly what on_predict_start builds
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, t0 = [], time.time()
    for idx in range(int(a.seconds * fps)):
        ok, img = cap.read()
        if not ok:
            break
        res = model.predict(img, conf=0.25, classes=classes, verbose=False)[0]  # same detector call as track(), without the tracker callback
        det = res.boxes.cpu().numpy()
        tracks = tracker.update(det, img, feats=None)
        for tr in tracks:
            i = int(tr[-1])
            rx1, ry1, rx2, ry2 = (float(v) for v in det.xyxy[i])
            rows.append([idx, round(idx / fps, 4), int(tr[4]), int(tr[6]), round(float(tr[5]), 3),
                         round(float(tr[0]), 1), round(float(tr[1]), 1), round(float(tr[2]), 1), round(float(tr[3]), 1),
                         round(rx1, 1), round(ry1, 1), round(rx2, 1), round(ry2, 1)])
        if idx % 1000 == 0:
            print(f"frame {idx}  {time.time() - t0:.0f}s", flush=True)
    out = {"video": Path(a.video).name, "fps": fps, "frame_size": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))], "frames": idx + 1,
           "model": Path(a.model).name, "class_names": {int(k): v for k, v in model.names.items()},
           "columns": ["frame", "t", "raw_id", "cls_id", "conf", "kx1", "ky1", "kx2", "ky2", "rx1", "ry1", "rx2", "ry2"], "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print("saved", a.out, len(rows))


if __name__ == "__main__":
    main()
