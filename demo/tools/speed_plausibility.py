"""Independent plausibility check of the demo calibration scale (NOT a validation).

Runs YOLOv8n + ByteTrack over the first N seconds of demo.mp4 (unpaced, offline, for this check only) and reports the distribution
of per-track speeds inside the speed zone, using the same LiveSpeed estimator the dashboard uses, under the camera config's
calibration and under a hypothetical alternative lane-marking module (diagram 1005: 1 m mark + 5 m gap, 6 m pitch), so a reader can see
how much the headline numbers depend on the assumed 9 m pitch.

    .venv\Scripts\python.exe demo/tools/speed_plausibility.py --seconds 300 --out demo/runs/speed_plausibility.json
"""
import argparse, collections, json, sys, time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anvesh.app.cameras import build_calibration_profile  # noqa: E402
from anvesh.app.livespeed import LiveSpeed  # noqa: E402

S = collections.namedtuple("S", "t ground")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--out", default=str(REPO / "demo" / "runs" / "speed_plausibility.json"))
    a = ap.parse_args()
    cfg = json.loads((REPO / "demo" / "camera_config.json").read_text())
    prof = build_calibration_profile("p", "demo", cfg["calibration"])
    ls = LiveSpeed(prof, cfg["speed_zone"], ema=1.0)
    from ultralytics import YOLO
    model = YOLO(str(REPO / "models" / "yolov8n.pt"))
    cap = cv2.VideoCapture(str(REPO / "demo.mp4")); fps = cap.get(cv2.CAP_PROP_FPS)
    stat = collections.Counter(); hist = collections.defaultdict(list); per_track = collections.defaultdict(list); i = 0; t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok or i / fps > a.seconds:
            break
        r = model.track(fr, persist=True, tracker="bytetrack.yaml", classes=[2, 5, 7], conf=0.25, verbose=False)[0]
        t = i / fps
        if r.boxes is not None and r.boxes.id is not None:
            for b, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
                hist[tid].append(S(t, ((b[0] + b[2]) / 2, b[3])))
                hist[tid] = hist[tid][-60:]
                if i % 5 == 0:
                    e = ls.estimate(str(tid), hist[tid])
                    stat[e["status"]] += 1
                    if e["status"] == "ok":
                        per_track[tid].append(e["speed_kmh"])
        i += 1
    print("statuses", dict(stat))
    med = np.array([np.median(v) for v in per_track.values() if len(v) >= 5])
    res = {"seconds": a.seconds, "frames": i, "wall_s": round(time.time() - t0, 1), "tracks_with_speed": int(len(med)),
           "median_track_speed_kmh_9m_pitch": float(np.median(med)), "p10": float(np.percentile(med, 10)), "p90": float(np.percentile(med, 90)),
           "max": float(med.max()), "share_over_112.7": float((med > 112.7).mean()),
           "median_if_pitch_were_6m_kmh": float(np.median(med) * 6 / 9),
           "note": "Plausibility only. UK motorway car speeds are typically ~100-115 km/h in free flow, lower in congestion; this checks the order of magnitude of the assumed scale, nothing more."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
