"""Run the plate reader (detector + OCR) on real photographs from the plates TEST split and save a contact sheet with what it read.
There is no ground-truth text in this dataset (only plate boxes), so the sheet is for a human to judge; the printed summary reports how often a
plate was found and how often the OCR text passed the acceptance rule.

  .venv\\Scripts\\python.exe demo/tools/plate_check.py --n 24 --out demo/runs/plate_check
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anvesh.app.plate_reader import PlateReader, _PLATE_RE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", default="demo/runs/plate_check")
ap.add_argument("--split", default="test")
ap.add_argument("--stride", type=int, default=0, help="take every k-th image (default: spread over the split)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
imgs = sorted((Path("datasets/plates_yolo/images") / a.split).glob("*.*"))
k = a.stride or max(1, len(imgs) // a.n)
imgs = imgs[::k][: a.n]
pr = PlateReader()
print("reader available:", pr.available, pr.reason)
rows, found, accepted = [], 0, 0
for p in imgs:
    im = cv2.imread(str(p))
    if im is None:
        continue
    got = pr.read_crop(im)
    tile = cv2.resize(im, (360, 270))
    label = "no plate found"
    if got:
        found += 1
        text, conf, det, pbox = got
        ok = conf >= pr.min_ocr_conf and bool(_PLATE_RE.match(text))
        accepted += ok
        label = f"{text} ocr={conf:.2f} det={det:.2f} {'ACCEPT' if ok else 'reject'}"
        x1, y1, x2, y2 = pbox
        sx, sy = 360 / im.shape[1], 270 / im.shape[0]
        cv2.rectangle(tile, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 255, 255), 2)
    cv2.rectangle(tile, (0, 0), (360, 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    rows.append(tile)
    print(p.name, "->", label)
while len(rows) % 4:
    rows.append(np.zeros((270, 360, 3), np.uint8))
sheet = np.vstack([np.hstack(rows[i:i + 4]) for i in range(0, len(rows), 4)])
cv2.imwrite(str(out / "plate_sheet.jpg"), sheet)
print(json.dumps({"images": len(imgs), "plate_found": found, "ocr_accepted": accepted}))
