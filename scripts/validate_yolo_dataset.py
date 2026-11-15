"""Validate a YOLO detection dataset (read-only) and print / save the report.

    .venv\\Scripts\\python.exe -I scripts\\validate_yolo_dataset.py --data datasets\\plates_yolo\\data.yaml --out outputs\\exec4\\plates_yolo_validation.json

Exit code 0 = no errors, 1 = errors (the dataset must not be trained on), 2 = bad usage. Warnings do not change the exit code;
`--strict` turns "images with no label file" into an error. The dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvesh.training.yolo_dataset import validate_yolo_dataset  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--out", help="write the full JSON report here")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-image-check", action="store_true", help="skip decoding image headers (faster; hashing and label checks still run)")
    args = ap.parse_args(argv)
    report = validate_yolo_dataset(args.data, strict=args.strict, check_images=not args.no_image_check)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"dataset: {report['data_yaml']}  classes: {report.get('classes')}")
    for name, s in report["splits"].items():
        print(f"  {name:5s} images {s['images']:6d}  labelled {s.get('labelled_images', 0):6d}  empty-label {s.get('empty_label_files', 0):5d}  "
              f"missing-label {s.get('missing_label_files', 0):5d}  boxes {s.get('boxes', 0):6d}")
    lk = report.get("leakage", {})
    print(f"  leakage: {lk.get('identical_images_across_splits')} identical images and {lk.get('base_ids_across_splits')} base ids across splits "
          f"({lk.get('scope')})")
    for w in report["warnings"]:
        print("  WARNING:", w)
    for e in report["errors"]:
        print("  ERROR:", e)
    print("RESULT:", "OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
