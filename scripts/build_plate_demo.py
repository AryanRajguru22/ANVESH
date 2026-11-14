"""Build `demo/plates_demo.mp4`: real photographs of vehicles with readable plates (from the plates TEST split), shown one after another as a
video, so the live plate reader can be demonstrated. It is NOT CCTV footage and the screen says so.

Why it exists: the recorded motorway CCTV is 640x360 and its plates are ~8 px wide - no system can read them, and ours does not guess. This clip
shows the same reader working when the plate is large enough. Photos are chosen automatically: only those where the plate detector and the OCR agree
with high confidence are used, so the demo is stable; the choice is recorded in demo/plates_demo.json.

    .venv\\Scripts\\python.exe scripts\\build_plate_demo.py --count 12 --seconds-each 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from anvesh.app.plate_reader import PlateReader, _PLATE_RE  # noqa: E402

W, H, FPS = 1280, 720, 25


def crop_padding(im, thr: float = 1.0):
    """The dataset exporter padded many photos by repeating edge pixels (streaks). Cut that padding away so the clip shows only real picture."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.int16)
    rows = np.where(np.abs(np.diff(g, axis=0)).mean(axis=1) > thr)[0]
    cols = np.where(np.abs(np.diff(g, axis=1)).mean(axis=0) > thr)[0]
    if len(rows) < 50 or len(cols) < 50:
        return im
    y0, y1, x0, x1 = rows.min(), rows.max() + 2, cols.min(), cols.max() + 2
    sub = im[y0:y1, x0:x1]
    return sub if sub.shape[0] > 200 and sub.shape[1] > 200 else im


def letterbox(im):
    h, w = im.shape[:2]
    s = min(W / w, H / h)
    r = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((H, W, 3), (24, 22, 20), np.uint8)
    y, x = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def vehicle_boxes(model, im):
    """COCO vehicle boxes (car, motorcycle, bus, truck) the live pipeline's detector would find in this picture."""
    res = model.predict(letterbox(im), imgsz=640, conf=0.4, verbose=False)[0]
    out = []
    for b in res.boxes:
        if int(b.cls) in (2, 3, 5, 7):
            out.append([float(v) for v in b.xyxy[0].tolist()])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seconds-each", type=float, default=4.0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(REPO / "demo" / "plates_demo.mp4"))
    a = ap.parse_args()
    imgs = sorted((REPO / "datasets" / "plates_yolo" / "images" / a.split).glob("*.*"))
    pr = PlateReader()
    if not pr.available:
        raise SystemExit(f"plate reader unavailable: {pr.reason}")
    from ultralytics import YOLO

    vmodel = YOLO(str(REPO / "models" / "yolov8n.pt"))  # the same vehicle detector the live pipeline uses
    chosen, seen_text = [], set()
    step = max(1, len(imgs) // (a.count * 40))
    for p in imgs[::step]:
        if len(chosen) >= a.count:
            break
        im = cv2.imread(str(p))
        if im is None or min(im.shape[:2]) < 300:
            continue
        im = crop_padding(im)
        got = pr.read_crop(im)
        if not got:
            continue
        text, conf, det, pbox = got
        # the picture as it will be shown (letterboxed to 1280x720): the plate must be inside a detected vehicle and big enough to read
        sc = min(W / im.shape[1], H / im.shape[0])
        ox, oy = (W - im.shape[1] * sc) / 2, (H - im.shape[0] * sc) / 2
        cx, cy = (pbox[0] + pbox[2]) / 2 * sc + ox, (pbox[1] + pbox[3]) / 2 * sc + oy
        pw = (pbox[2] - pbox[0]) * sc
        veh = [v for v in vehicle_boxes(vmodel, im) if v[0] <= cx <= v[2] and v[1] <= cy <= v[3]]
        if (veh and pw >= 60 and conf >= 0.93 and det >= 0.65 and _PLATE_RE.match(text) and text not in seen_text and len(text) >= 6):
            chosen.append((p, text, conf))
            seen_text.add(text)
    if len(chosen) < 4:
        raise SystemExit("too few clean photos found")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for p, _t, _c in chosen:
        frame = letterbox(crop_padding(cv2.imread(str(p))))
        for _ in range(int(a.seconds_each * FPS)):
            vw.write(frame)
    vw.release()
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"note": "photos from datasets/plates_yolo test split (CC BY 4.0 per export README; attribution chain unverified); not CCTV footage",
         "seconds_each": a.seconds_each, "photos": [{"file": p.name, "reader_text": t, "ocr_conf": round(c, 2)} for p, t, c in chosen]}, indent=1), encoding="utf-8")
    print(f"wrote {a.out}: {len(chosen)} photos x {a.seconds_each}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
