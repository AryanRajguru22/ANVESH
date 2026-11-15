"""UNLABELLED comparison of the COCO-pretrained detector and the MIO-TCD fine-tuned checkpoint on a segment of the demo video, using ANVESH's own
tracker wrapper (so class mapping is the repo's). There is NO ground truth for this footage, so this reports behaviour PROXIES only - detections per
frame, how many tracks were needed to cover the traffic (fragmentation proxy: more tracks per unit of traffic = more ID breaks), track-length
distribution, class mix, speed - and never an accuracy figure. It is NOT a held-out evaluation and does not touch the MIO-TCD test lock.

  .venv\Scripts\python.exe demo/tools/compare_detectors_demo.py --start 60 --seconds 180 --out demo/runs/detector_compare.json
"""
import argparse, collections, json, sys, time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anvesh.perception.tracking import ByteTrackYoloTracker  # noqa: E402
from anvesh.perception.weights import checkpoint_sha256  # noqa: E402

MODELS = {"pretrained_yolov8n": REPO / "models" / "yolov8n.pt", "finetuned_mio5": REPO / "models" / "finetuned" / "anvesh_mio5_yolov8n_best.pt"}


def run(path, start, seconds):
    tr = ByteTrackYoloTracker(str(path))
    cap = cv2.VideoCapture(str(REPO / "demo.mp4")); fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n, per_frame, tracks, classes, conf = 0, [], collections.defaultdict(list), collections.Counter(), []
    t0 = time.time()
    while n < seconds * fps:
        ok, fr = cap.read()
        if not ok:
            break
        raw, tracked = tr.update(fr, n, n / fps)
        per_frame.append(len(raw))
        for d in tracked:
            tracks[d.track_id].append(n); classes[d.detection.vehicle_class.value] += 1; conf.append(d.detection.confidence)
        n += 1
    lens = np.array([len(v) for v in tracks.values()]) / fps
    # a vehicle that vanishes and reappears under a new id inflates the track count: count tracks starting well inside the frame lifetime? keep simple
    return {"frames": n, "wall_s": round(time.time() - t0, 1), "fps": round(n / (time.time() - t0), 1),
            "mean_detections_per_frame": round(float(np.mean(per_frame)), 2), "p95_detections_per_frame": float(np.percentile(per_frame, 95)),
            "distinct_track_ids": len(tracks), "tracks_ge_2s": int((lens >= 2).sum()), "tracks_lt_1s": int((lens < 1).sum()),
            "median_track_len_s": round(float(np.median(lens)), 2), "mean_track_len_s": round(float(lens.mean()), 2),
            "mean_conf": round(float(np.mean(conf)), 3), "class_counts": dict(classes)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", type=float, default=60); ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--out", default=str(REPO / "demo" / "runs" / "detector_compare.json")); a = ap.parse_args()
    res = {"segment": {"video": "demo.mp4", "start_s": a.start, "seconds": a.seconds}, "labels": "NONE (proxies only; no accuracy claim)", "models": {}}
    for name, p in MODELS.items():
        res["models"][name] = {"sha256": checkpoint_sha256(str(p)), **run(p, a.start, a.seconds)}
        print(name, json.dumps(res["models"][name]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
