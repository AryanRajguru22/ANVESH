"""Compare detector checkpoints on representative ANVESH footage: CityFlowV2 S02 validation cameras (real traffic CCTV).

    .venv\\Scripts\\python.exe scripts\\compare_detectors_cityflow.py ^
        --model pretrained=models\\yolov8n.pt --model finetuned=models\\finetuned\\anvesh_mio5_yolov8n_best.pt ^
        --out outputs\\training\\compare_cityflow.json

What this can and cannot say (CityFlow ReadMe item 6): ground truth lists ONLY vehicles seen by >= 2 cameras, and has
NO class labels. Therefore:
  * RECALL @ IoU 0.5 (class-agnostic, GT inside roi.jpg) is valid;
  * PRECISION is not computable -- unannotated-but-correct vehicles would count as false positives. Raw false-positive
    counts are reported only as an upper bound on false alarms, clearly labelled;
  * the class mix per model is reported descriptively, with NO accuracy claim.
The detector runs at ANVESH's operating settings (conf 0.25, imgsz 640). CityFlow footage was never used for training
(and its licence is research-only).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.evaluation import cityflow_gt as gt_io  # noqa: E402
from anvesh.evaluation.mot_metrics import evaluate_detection  # noqa: E402
from anvesh.perception.detection import ClassMap  # noqa: E402
from anvesh.perception.weights import checkpoint_sha256, require_weights_file  # noqa: E402


def sampled_frames(video: Path, stride: int, max_frames: int):
    import cv2

    cap = cv2.VideoCapture(str(video))
    idx, taken = 0, 0
    while taken < max_frames:
        if idx % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            taken += 1
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--scenario", default="S02")
    ap.add_argument("--cameras", nargs="+", default=["c006", "c007", "c008", "c009"])
    ap.add_argument("--stride", type=int, default=10, help="every Nth video frame (10 fps footage -> 1 sample/s)")
    ap.add_argument("--max-frames", type=int, default=600, help="sampled frames per camera")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    models = {}
    for spec in args.model:
        name, _, p = spec.partition("=")
        path = require_weights_file(p)
        m = YOLO(str(path))
        models[name] = (path, m, ClassMap.from_model_names(m.names))

    out = {"utc": datetime.now(timezone.utc).isoformat(), "scenario": args.scenario, "stride": args.stride,
           "conf": args.conf, "imgsz": args.imgsz, "iou": 0.5,
           "caveat": "GT = vehicles seen by >=2 cameras only, class-agnostic: recall valid, precision NOT computable",
           "models": {n: {"checkpoint": str(p), "sha256": checkpoint_sha256(p)} for n, (p, _, _) in models.items()},
           "cameras": {}}
    for cam in args.cameras:
        d = gt_io.camera_dir(args.scenario, cam)
        mask = gt_io.load_roi_mask(d / "roi.jpg")
        gt = gt_io.filter_table_by_roi(gt_io.load_mot_file(d / "gt" / "gt.txt"), mask)
        preds = {n: {} for n in models}
        mix = {n: Counter() for n in models}
        frames = []
        for idx, frame in sampled_frames(d / "vdo.avi", args.stride, args.max_frames):
            fno = idx + gt_io.FRAME_OFFSET
            frames.append(fno)
            for n, (_, m, cmap) in models.items():
                r = m.predict(frame, conf=args.conf, imgsz=args.imgsz, classes=cmap.class_ids, verbose=False)[0]
                rows = []
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                    if gt_io.inside_roi(mask, x1, y1, x2 - x1, y2 - y1):
                        rows.append((-1, x1, y1, x2 - x1, y2 - y1, 1.0))
                        mix[n][cmap.id_to_vehicle_class[int(c)].value] += 1
                preds[n][fno] = rows
        cam_out = {"frames_sampled": len(frames), "gt_boxes_in_sampled_frames": sum(len(gt.get(f, [])) for f in frames),
                   "models": {}}
        for n in models:
            s = evaluate_detection(gt, preds[n], 0.5, frames=frames)
            cam_out["models"][n] = {
                "recall_class_agnostic": s.recall, "tp": s.tp, "fn": s.fn,
                "unmatched_predictions_UPPER_BOUND_on_false_alarms": s.fp,
                "detections_per_frame": sum(len(v) for v in preds[n].values()) / max(1, len(frames)),
                "class_mix_descriptive_only": dict(mix[n])}
        out["cameras"][cam] = cam_out
        print(cam, {n: round(v["recall_class_agnostic"], 3) for n, v in cam_out["models"].items()}, flush=True)

    out["pooled"] = {}
    for n in models:
        tp = sum(c["models"][n]["tp"] for c in out["cameras"].values())
        fn = sum(c["models"][n]["fn"] for c in out["cameras"].values())
        out["pooled"][n] = {"recall_class_agnostic": tp / (tp + fn) if tp + fn else None, "tp": tp, "fn": fn}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("pooled", {n: v["recall_class_agnostic"] for n, v in out["pooled"].items()}, "->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
