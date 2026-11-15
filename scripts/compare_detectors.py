"""Score one or more detector checkpoints on the SAME labelled images with ONE metric implementation.

    .venv\\Scripts\\python.exe scripts\\compare_detectors.py --split val ^
        --model pretrained=models\\yolov8n.pt --model finetuned=models\\finetuned\\anvesh_mio5_yolov8n_best.pt ^
        --out outputs\\training\\compare_val.json

Models are mapped BY CLASS NAME into ANVESH's shared 5-class taxonomy (bicycle, car, motorcycle, bus, truck), so the
COCO baseline (80 classes) and a fine-tuned 5-class model are scored identically (see anvesh/training/detector_eval.py).

HELD-OUT TEST DISCIPLINE: `--split test` may be run ONCE per set of checkpoints. A lock file records the checkpoint
hashes; a second run is refused unless `--allow-test-rerun` is given (and then the rerun is recorded). Select the
checkpoint on `--split val`, then evaluate on test exactly once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.perception.weights import checkpoint_sha256, require_weights_file  # noqa: E402
from anvesh.training.detector_eval import (  # noqa: E402
    SHARED_CLASSES, ImageResult, bootstrap, confusion_matrix, model_to_shared, operating_point, score)

DATASET = REPO_ROOT / "datasets" / "mio_tcd" / "yolo5"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _read_boxes(f: Path, w: int, h: int):
    out = []
    if f.is_file():
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
            out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def load_gt(label_file: Path, ignore_file: Path, w: int, h: int):
    """-> (gt_cls, gt_xyxy, ignore_xyxy). Label classes 0..4 are the shared taxonomy; the training label's extra
    class 5 (`other_vehicle`, MIO-TCD's unclassifiable "motorized_vehicle") and everything in the ignore file
    (pedestrian, non-motorized vehicle) are don't-care regions."""
    gt = [(c, b) for c, b in _read_boxes(label_file, w, h) if c < len(SHARED_CLASSES)]
    ign = [b for c, b in _read_boxes(label_file, w, h) if c >= len(SHARED_CLASSES)]
    ign += [b for _, b in _read_boxes(ignore_file, w, h)]
    return (np.array([c for c, _ in gt], dtype=int), np.array([b for _, b in gt], dtype=float).reshape(-1, 4),
            np.array(ign, dtype=float).reshape(-1, 4))


def run_model(name: str, path: Path, images: list, labels_dir: Path, ignore_dir: Path, imgsz: int, batch: int, device) -> tuple:
    from ultralytics import YOLO

    model = YOLO(str(path))
    id_map = model_to_shared(model.names)
    results, t0 = [], time.time()
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        preds = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                              classes=sorted(id_map), device=device, half=True, verbose=False)
        for img_path, r in zip(chunk, preds):
            h, w = r.orig_shape
            gt_cls, gt_xyxy, ign = load_gt(labels_dir / f"{img_path.stem}.txt", ignore_dir / f"{img_path.stem}.txt", w, h)
            b = r.boxes
            pc = np.array([id_map[int(c)] for c in b.cls.tolist()], dtype=int)
            results.append(ImageResult(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), pc, gt_xyxy, gt_cls, ign))
        if (start // batch) % 20 == 0:
            print(f"  [{name}] {start + len(chunk)}/{len(images)}", flush=True)
    return results, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N images (debugging only; recorded in the output)")
    ap.add_argument("--op-conf", type=float, default=0.25, help="operating confidence for fixed-threshold P/R")
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test-rerun", action="store_true")
    ap.add_argument("--crosscheck-val", action="store_true",
                    help="also run Ultralytics' own model.val() on models whose names are exactly the shared taxonomy")
    args = ap.parse_args(argv)

    models = {}
    for spec in args.model:
        name, _, p = spec.partition("=")
        models[name] = require_weights_file(p)
    hashes = {n: checkpoint_sha256(p) for n, p in models.items()}

    ds = Path(args.dataset)
    img_dir, lbl_dir, ign_dir = ds / "images" / args.split, ds / "labels" / args.split, ds / "ignore" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images in {img_dir}")

    lock = ds / "test_evaluation.lock.json"
    if args.split == "test":
        if lock.is_file() and not args.allow_test_rerun:
            raise SystemExit(f"held-out test split was already evaluated ({lock}); refusing to look at it again. "
                             "Pass --allow-test-rerun to override (the rerun is recorded).")
        prev = json.loads(lock.read_text()) if lock.is_file() else {"runs": []}
        prev["runs"].append({"utc": datetime.now(timezone.utc).isoformat(), "checkpoints": hashes,
                             "n_images": len(images), "limit": args.limit})
        lock.write_text(json.dumps(prev, indent=2))

    per_model, report = {}, {}
    for name, path in models.items():
        print(f"[{name}] {path}", flush=True)
        res, secs = run_model(name, path, images, lbl_dir, ign_dir, args.imgsz, args.batch, 0)
        per_model[name] = res
        report[name] = {
            "checkpoint": str(path), "sha256": hashes[name], "inference_seconds": round(secs, 1),
            "metrics_conf0.001": score(res), "operating_point": operating_point(res, args.op_conf),
            "confusion_matrix_conf0.25_iou0.45": {"labels": list(SHARED_CLASSES) + ["background/missed"],
                                                  "rows_pred_cols_true": confusion_matrix(res).tolist()},
        }
    out = {"split": args.split, "n_images": len(images), "limit": args.limit, "imgsz": args.imgsz,
           "dataset": str(ds), "shared_classes": list(SHARED_CLASSES), "utc": datetime.now(timezone.utc).isoformat(),
           "models": report}
    if len(models) > 1 and args.n_boot > 0:
        out["bootstrap_map50"] = bootstrap(per_model, args.n_boot, seed=0, metric="map50")
        out["bootstrap_map50_95"] = bootstrap(per_model, args.n_boot, seed=0, metric="map50_95")

    if args.crosscheck_val:
        # Ultralytics' own val() on a model whose class ids are the training taxonomy (ids 0..4 = shared classes).
        # Per-class AP depends only on same-class predictions and ground truth, so for the 5 shared classes it must
        # agree with our implementation when ours is run WITHOUT don't-care handling (val() has none).
        from ultralytics import YOLO
        out["ultralytics_val_crosscheck"] = {}
        for name, path in models.items():
            names = YOLO(str(path)).names
            if any(names.get(i) != n for i, n in enumerate(SHARED_CLASSES)):
                continue
            m = YOLO(str(path)).val(data=str(ds / "data.yaml"), split=args.split, imgsz=args.imgsz, batch=16,
                                    conf=0.001, iou=0.7, device=0, plots=False, verbose=False)
            ours = score(per_model[name], use_ignore=False)
            out["ultralytics_val_crosscheck"][name] = {
                "ultralytics_per_class_ap50": {SHARED_CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)
                                               if int(c) < len(SHARED_CLASSES)},
                "ours_no_ignore_per_class_ap50": {k: v.get("ap50") for k, v in ours["per_class"].items()},
                "note": "ultralytics val() runs the FULL split; ours runs the (possibly --limit-ed) image list"}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for n, r in report.items():
        m = r["metrics_conf0.001"]
        print(f"{n:12s} P={m['precision']:.3f} R={m['recall']:.3f} mAP50={m['map50']:.3f} mAP50-95={m['map50_95']:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
