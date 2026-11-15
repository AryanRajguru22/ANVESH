"""Explain-only diagnostic for the CityFlow S02 recall comparison (does NOT replace compare_detectors_cityflow.py).

    .venv\\Scripts\\python.exe scripts\\diagnose_cityflow_recall.py --out outputs\\training\\diag_cityflow_recall.json

The fine-tuned detector has a 6th class, `other_vehicle`, that ANVESH's `ClassMap` ignores at run time. Any box the
model labels `other_vehicle` is therefore invisible to the app. This script scores the same frames, same GT, same
IoU 0.5 matching with three prediction sets per model:

  operational     exactly what compare_detectors_cityflow.py / the app use (ClassMap.class_ids filter)
  incl_other      operational + boxes of classes ClassMap ignores (diagnostic upper bound for the fine-tuned model)
  multi_label     operational path but multi-label NMS (the setting that explains the MIO-TCD cross-check gap)

`operational` must reproduce outputs/training/compare_cityflow.json (checked and recorded). The other two are
explanations only: the operational numbers remain the ones that count for the promotion decision.
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
from scripts.compare_detectors_cityflow import sampled_frames  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default=str(REPO_ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--finetuned", default=str(REPO_ROOT / "models" / "finetuned" / "anvesh_mio5_yolov8n_best.pt"))
    ap.add_argument("--reference", default=str(REPO_ROOT / "outputs" / "training" / "compare_cityflow.json"))
    ap.add_argument("--scenario", default="S02")
    ap.add_argument("--cameras", nargs="+", default=["c006", "c007", "c008", "c009"])
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from ultralytics.utils import nms

    models = {}
    for name, p in (("pretrained", args.pretrained), ("finetuned", args.finetuned)):
        path = require_weights_file(p)
        m = YOLO(str(path))
        models[name] = (path, m, ClassMap.from_model_names(m.names))

    orig_nms = nms.non_max_suppression
    modes = ("operational", "incl_other", "multi_label")
    pooled = {n: {md: [0, 0] for md in modes} for n in models}
    other_counts = {n: Counter() for n in models}
    cams_out = {}
    for cam in args.cameras:
        d = gt_io.camera_dir(args.scenario, cam)
        mask = gt_io.load_roi_mask(d / "roi.jpg")
        gt = gt_io.filter_table_by_roi(gt_io.load_mot_file(d / "gt" / "gt.txt"), mask)
        preds = {n: {md: {} for md in modes} for n in models}
        frames = []
        for idx, frame in sampled_frames(d / "vdo.avi", args.stride, args.max_frames):
            fno = idx + gt_io.FRAME_OFFSET
            frames.append(fno)
            for n, (_, m, cmap) in models.items():
                for md in modes:
                    nms.non_max_suppression = (lambda *a, **k: orig_nms(*a, **{**k, "multi_label": True})) if md == "multi_label" else orig_nms
                    try:
                        r = m.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                      classes=None if md == "incl_other" else cmap.class_ids, verbose=False)[0]
                    finally:
                        nms.non_max_suppression = orig_nms
                    rows = []
                    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                        ci = int(c)
                        if md == "incl_other" and ci not in cmap.id_to_vehicle_class:
                            if m.names[ci] == "person":  # COCO persons are not vehicles; do not count them
                                continue
                            if gt_io.inside_roi(mask, x1, y1, x2 - x1, y2 - y1):
                                other_counts[n][str(m.names[ci])] += 1
                                rows.append((-1, x1, y1, x2 - x1, y2 - y1, 1.0))
                        elif gt_io.inside_roi(mask, x1, y1, x2 - x1, y2 - y1):
                            rows.append((-1, x1, y1, x2 - x1, y2 - y1, 1.0))
                    preds[n][md][fno] = rows
        cams_out[cam] = {"frames_sampled": len(frames)}
        for n in models:
            cams_out[cam][n] = {}
            for md in modes:
                s = evaluate_detection(gt, preds[n][md], 0.5, frames=frames)
                cams_out[cam][n][md] = {"recall": s.recall, "tp": s.tp, "fn": s.fn}
                pooled[n][md][0] += s.tp
                pooled[n][md][1] += s.fn
        print(cam, {n: {md: round(cams_out[cam][n][md]["recall"], 3) for md in modes} for n in models}, flush=True)

    ref = json.loads(Path(args.reference).read_text()) if Path(args.reference).is_file() else {}
    out = {"utc": datetime.now(timezone.utc).isoformat(), "scenario": args.scenario, "stride": args.stride,
           "conf": args.conf, "imgsz": args.imgsz, "label": "EXPLANATION ONLY; operational recall is the figure that counts",
           "sha256": {n: checkpoint_sha256(p) for n, (p, _, _) in models.items()},
           "cameras": cams_out,
           "pooled": {n: {md: {"tp": tp, "fn": fn, "recall": tp / (tp + fn) if tp + fn else None}
                          for md, (tp, fn) in v.items()} for n, v in pooled.items()},
           "ignored_class_boxes_counted_in_incl_other": {n: dict(c) for n, c in other_counts.items()},
           "reproduces_reference": {n: {"reference_tp": ref.get("pooled", {}).get(n, {}).get("tp"),
                                        "this_run_tp": pooled[n]["operational"][0]} for n in models}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"pooled": out["pooled"], "ignored": out["ignored_class_boxes_counted_in_incl_other"],
                      "reproduces": out["reproduces_reference"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
