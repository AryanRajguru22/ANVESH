"""Run the REAL detector and the REAL tracker once over a CityFlow camera
video and cache their raw outputs, so every downstream evaluation (detector /
tracker / TrackManager identity before-and-after / batch-vs-stream parity)
replays identical inputs and GPU nondeterminism cannot leak into a comparison.

Two independent passes per frame (two separate YOLO instances -- a shared one
would mix the tracker's persisted state into the detector-only pass):
  - DETECTOR pass : `model.predict` (no tracker) -> boxes, confidence, class.
  - TRACKER  pass : the production `ByteTrackYoloTracker.update` -> exactly
                    what `run_m1_pipeline` feeds `TrackManager`.

Cache format: gzip JSON-lines, one object per frame:
  {"i": frame_index(0-based), "t": timestamp, "det": [[x1,y1,x2,y2,conf,cls]...],
   "trk": [[x1,y1,x2,y2,conf,cls,raw_id_or_null]...]}
plus a first line {"meta": {...}} recording model, thresholds, fps, versions.

Usage:
    python scripts/cityflow_cache_perception.py --camera c006
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from anvesh.evaluation.cityflow_gt import camera_dir  # noqa: E402
from anvesh.perception.detection import COCO_TO_VEHICLE_CLASS  # noqa: E402
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "cityflow"


def _round_box(v):
    return [round(float(x), 2) for x in v]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="S02")
    ap.add_argument("--camera", default="c006")
    ap.add_argument("--model", default=str(REPO_ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch
    import ultralytics

    video = camera_dir(args.scenario, args.camera) / "vdo.avi"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = YOLO(args.model)
    tracker = ByteTrackYoloTracker(model_name=args.model, confidence_threshold=args.conf)
    classes = sorted(COCO_TO_VEHICLE_CLASS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.scenario.lower()}_{args.camera}_perception_cache.jsonl.gz"
    t0 = time.perf_counter()
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"meta": {
            "video": str(video), "fps": fps, "frame_count_header": n_total,
            "model": args.model, "conf": args.conf, "classes": classes,
            "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "tracker_config": "bytetrack.yaml",
        }}) + "\n")
        while True:
            ok, image = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            ts = n / fps
            res = detector.predict(image, classes=classes, conf=args.conf, verbose=False)[0]
            det = []
            if res.boxes is not None and len(res.boxes):
                for b, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                    det.append(_round_box(b) + [round(float(c), 4), int(k)])
            raw, tracked = tracker.update(image, n, ts)
            tracked_by_box = {td.detection.bbox_xyxy: td.track_id for td in tracked}
            trk = []
            for d in raw:
                tid = tracked_by_box.get(d.bbox_xyxy)
                trk.append(_round_box(d.bbox_xyxy) + [round(d.confidence, 4), _class_id(d), None if tid is None else int(tid)])
            out.write(json.dumps({"i": n, "t": ts, "det": det, "trk": trk}) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"  cached {n} frames ({n / (time.perf_counter() - t0):.1f} fps)", flush=True)
    cap.release()
    print(f"cached {n} frames -> {out_path} in {time.perf_counter() - t0:.1f}s (header says {n_total} frames, fps {fps})")
    return 0


def _class_id(raw_detection) -> int:
    for coco_id, vc in COCO_TO_VEHICLE_CLASS.items():
        if vc == raw_detection.vehicle_class and raw_detection.class_name == _coco_name(coco_id):
            return coco_id
    return -1


def _coco_name(coco_id: int) -> str:
    from anvesh.perception.detection import COCO_CLASS_NAMES

    return COCO_CLASS_NAMES.get(coco_id, str(coco_id))


if __name__ == "__main__":
    raise SystemExit(main())
