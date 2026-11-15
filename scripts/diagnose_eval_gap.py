"""Explain the gap between our shared evaluator and Ultralytics' own `val()` -- on a VALIDATION subset only.

    .venv\\Scripts\\python.exe scripts\\diagnose_eval_gap.py --n 1500 --out outputs\\training\\diag_eval_gap.json

The held-out test split is never read (the script refuses `--split test`). It copies a stride-sampled subset of the
validation images/labels into a scratch directory, so Ultralytics' label cache next to the real dataset is untouched.

Factors toggled one at a time on the fine-tuned checkpoint (per-class AP50, our scorer WITHOUT don't-care regions,
which is what `val()` effectively does):

  U   Ultralytics val()                                      -- reference
  B   our prediction path as in compare_detectors.py          (predict, FP16, classes= filter, single-label NMS)
  C   B without the classes= filter (class 5 dropped after NMS)
  D   C + multi-label NMS (val() uses multi_label=True, predict() does not)
  E   D in FP32
  P   same four variants for the pretrained COCO model (ours only; val() cannot score it)

A cause is named only if toggling it closes the gap to U.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.perception.weights import checkpoint_sha256, require_weights_file  # noqa: E402
from anvesh.training.detector_eval import SHARED_CLASSES, ImageResult, model_to_shared, score  # noqa: E402
from scripts.compare_detectors import DATASET, IMG_EXT, load_gt  # noqa: E402


def make_subset(src: Path, dst: Path, n: int) -> list:
    images = sorted(p for p in (src / "images" / "val").iterdir() if p.suffix.lower() in IMG_EXT)
    step = max(1, len(images) // n)
    chosen = images[::step][:n]
    for sub in ("images", "labels", "ignore"):
        (dst / sub / "val").mkdir(parents=True, exist_ok=True)
    for p in chosen:
        shutil.copy2(p, dst / "images" / "val" / p.name)
        for sub in ("labels", "ignore"):
            f = src / sub / "val" / f"{p.stem}.txt"
            if f.is_file():
                shutil.copy2(f, dst / sub / "val" / f.name)
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\ntrain: images/val\nval: images/val\nnc: 6\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(("bicycle", "car", "motorcycle", "bus", "truck", "other_vehicle"))),
        encoding="utf-8")
    return [dst / "images" / "val" / p.name for p in chosen]


def duplicate_label_rows(labels_dir: Path) -> int:
    dup = 0
    for f in labels_dir.glob("*.txt"):
        rows = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        dup += len(rows) - len(set(rows))
    return dup


def predict_variant(model, images, dst, *, filter_classes: bool, multi_label: bool, half: bool, imgsz=640, batch=32):
    from ultralytics.utils import nms

    orig = nms.non_max_suppression
    if multi_label:
        nms.non_max_suppression = lambda *a, **k: orig(*a, **{**k, "multi_label": True})
    try:
        id_map = model_to_shared(model.names)
        out = []
        for s in range(0, len(images), batch):
            chunk = images[s:s + batch]
            preds = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                                  classes=sorted(id_map) if filter_classes else None, device=0,
                                  half=half, verbose=False)
            for p, r in zip(chunk, preds):
                h, w = r.orig_shape
                gt_cls, gt_xyxy, ign = load_gt(dst / "labels" / "val" / f"{p.stem}.txt", dst / "ignore" / "val" / f"{p.stem}.txt", w, h)
                b = r.boxes
                cls = b.cls.tolist()
                keep = [i for i, c in enumerate(cls) if int(c) in id_map]  # class 5 (other_vehicle) dropped AFTER nms
                out.append(ImageResult(b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep],
                                       np.array([id_map[int(cls[i])] for i in keep], dtype=int), gt_xyxy, gt_cls, ign))
        return out
    finally:
        nms.non_max_suppression = orig


def summary(results) -> dict:
    s = score(results, use_ignore=False)
    return {"map50_5class": s["map50"], "ap50": {k: v.get("ap50") for k, v in s["per_class"].items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=("val",), help="test is deliberately not selectable")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--pretrained", default=str(REPO_ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--finetuned", default=str(REPO_ROOT / "models" / "finetuned" / "anvesh_mio5_yolov8n_best.pt"))
    ap.add_argument("--scratch", default=str(REPO_ROOT / "outputs" / "training" / "diag" / "val_subset"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    dst = Path(args.scratch)
    if dst.exists():
        shutil.rmtree(dst)
    images = make_subset(Path(DATASET), dst, args.n)
    out = {"utc": datetime.now(timezone.utc).isoformat(), "split": "val", "n_images": len(images),
           "duplicate_label_rows_in_subset": duplicate_label_rows(dst / "labels" / "val"), "variants": {}}

    ft_path, pt_path = require_weights_file(args.finetuned), require_weights_file(args.pretrained)
    out["sha256"] = {"finetuned": checkpoint_sha256(ft_path), "pretrained": checkpoint_sha256(pt_path)}
    m = YOLO(str(ft_path)).val(data=str(dst / "data.yaml"), split="val", imgsz=640, batch=16, conf=0.001, iou=0.7, device=0,
               plots=False, verbose=False)
    out["variants"]["U_ultralytics_val"] = {
        "map50_5class": float(np.mean([a for c, a in zip(m.box.ap_class_index, m.box.ap50) if int(c) < 5])),
        "ap50": {SHARED_CLASSES[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50) if int(c) < 5}}
    print("U", out["variants"]["U_ultralytics_val"], flush=True)

    ft = YOLO(str(ft_path))  # fresh instance: after val() the shared one cannot predict ("Inference tensors" error)
    variants = {"B_ours_as_compare_detectors": dict(filter_classes=True, multi_label=False, half=True),
                "C_no_classes_filter": dict(filter_classes=False, multi_label=False, half=True),
                "D_plus_multi_label_nms": dict(filter_classes=False, multi_label=True, half=True),
                "E_plus_fp32": dict(filter_classes=False, multi_label=True, half=False)}
    for name, kw in variants.items():
        out["variants"][name] = summary(predict_variant(ft, images, dst, **kw))
        print(name, out["variants"][name], flush=True)

    pt = YOLO(str(pt_path))
    for name in ("B_ours_as_compare_detectors", "D_plus_multi_label_nms"):
        out["variants"]["PRETRAINED_" + name] = summary(predict_variant(pt, images, dst, **variants[name]))
        print("PRETRAINED", name, out["variants"]["PRETRAINED_" + name], flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
