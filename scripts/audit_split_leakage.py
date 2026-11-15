"""Near-duplicate leakage audit between the splits of a YOLO dataset (read-only).

    .venv\\Scripts\\python.exe -I scripts\\audit_split_leakage.py --data datasets\\plates_yolo\\data.yaml --out outputs\\exec4\\plates_leakage.json

For every val/test image: the RMS thumbnail distance (16x16 colour thumbnails, `anvesh.training.splitting`) to its nearest TRAIN image, and
to its nearest OTHER image of its own split. If held-out images are much closer to training images than to each other, the split
leaks (same scene / camera / adjacent frame on both sides) and held-out scores are inflated. This is a screening proxy for "same
scene", not proof: it will miss the same scene at a different time of day and can flag genuinely similar but distinct scenes.
Exit 0 always (it is a measurement); the numbers and the verdict text are in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvesh.training.splitting import nearest_distance, thumbnail_feature  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _features(img_dir: Path):
    import cv2

    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    feats = []
    for p in files:
        im = cv2.imread(str(p))
        feats.append(thumbnail_feature(im) if im is not None else np.zeros(16 * 16 * 3, np.uint8))
    return files, np.stack(feats) if feats else np.zeros((0, 16 * 16 * 3), np.uint8)


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--threshold", type=float, default=6.0, help="RMS pixel distance (0-255 scale) below which two thumbnails are 'near-identical'")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    data = Path(args.data)
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data.parent)
    root = root if root.is_absolute() else (data.parent / root).resolve()
    feats = {s: _features(root / cfg[s]) for s in ("train", "val", "test") if s in cfg}
    train = feats["train"][1]
    report = {"data": str(data), "threshold_rms": args.threshold, "n_images": {s: len(v[0]) for s, v in feats.items()}, "splits": {},
              "caveat": "16x16 thumbnail proxy for 'same scene'; misses the same scene at another time of day"}
    for s in ("val", "test"):
        if s not in feats or not len(feats[s][0]):
            continue
        q = feats[s][1]
        to_train = nearest_distance(q, train)
        to_self = nearest_distance(q, q, exclude_self=True) if len(q) > 1 else np.full(len(q), np.inf)
        report["splits"][s] = {
            "median_dist_to_nearest_train": float(np.median(to_train)), "median_dist_to_nearest_own_split": float(np.median(to_self)),
            "fraction_within_threshold_of_a_train_image": float((to_train <= args.threshold).mean()),
            "fraction_within_threshold_of_another_own_split_image": float((to_self <= args.threshold).mean()),
            "p10_dist_to_nearest_train": float(np.percentile(to_train, 10)),
        }
    leaky = [s for s, r in report["splits"].items() if r["fraction_within_threshold_of_a_train_image"] > 0.05]
    report["verdict"] = (f"split(s) {leaky}: more than 5% of held-out images have a near-identical training image - held-out scores are inflated by scene overlap"
                         if leaky else "no more than 5% of held-out images have a near-identical training image at this threshold (a screening result, not a guarantee)")
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
