"""Speed audit on REAL tracked boxes (collected by speed_audit_collect.py) with the SAME code the app uses (`LiveSpeed`, the camera calibration).

It answers, with numbers, what can and cannot be verified without ground-truth speeds:

  1. timing         - is `index / fps` the real video time? (container timestamps vs index/fps)
  2. what is shown  - distribution of the speeds LiveSpeed produces, per class / lane, and of the 'fastest vehicle' figure the officer tile shows
  3. reference point- bottom-centre vs box-centre: how much does the choice change the answer, and which is geometrically consistent
  4. consistency    - does a vehicle's speed depend on where in the picture it is measured? (near/far ratio over steady tracks)
  5. noise          - measured fit-residual size per image-row band (pixels and metres) vs the displayed error band
  6. lateral scale  - metric width of vehicles through the calibration (an independent check of the lateral half of the calibration)
  7. priors         - heavy goods vehicle speed distribution vs the UK limits (a plausibility check on the ASSUMED scale; NOT a measurement)

    .venv\\Scripts\\python.exe demo/tools/speed_audit_analyze.py --tracks demo/runs/speed_audit_tracks.json --out docs/benchmarks/speed_audit_summary.json
"""
import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from anvesh.app.cameras import build_calibration_profile  # noqa: E402
from anvesh.app.detectors import point_in_polygon  # noqa: E402
from anvesh.app.livespeed import LiveSpeed  # noqa: E402


class S:
    __slots__ = ("t", "ground", "bbox")

    def __init__(self, t, ground, bbox):
        self.t, self.ground, self.bbox = t, ground, bbox


def q(v, ps=(10, 25, 50, 75, 90, 99)):
    v = sorted(v)
    if not v:
        return None
    return {f"p{p}": round(float(np.percentile(v, p)), 1) for p in ps} | {"n": len(v), "max": round(max(v), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=str(REPO / "demo" / "runs" / "speed_audit_tracks.json"))
    ap.add_argument("--config", default=str(REPO / "demo" / "camera_config.json"))
    ap.add_argument("--out", default=str(REPO / "docs" / "benchmarks" / "speed_audit_summary.json"))
    ap.add_argument("--mode", choices=("app", "legacy"), default="app", help="app = the estimator settings the app uses now; legacy = window not limited to the zone (the behaviour before the fix)")
    a = ap.parse_args()
    data = json.loads(Path(a.tracks).read_text(encoding="utf-8"))
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    profile = build_calibration_profile("audit-cal", "demo-motorway", cfg["calibration"])
    zone = cfg["speed_zone"]
    lanes = {l["lane_id"]: l["vertices"] for l in cfg["lanes"]}
    W, Hh = data["frame_size"]
    fps = data["fps"]
    out = {"mode": a.mode, "video": data["video"], "fps": fps, "frames": data["frames"], "model": data["model"], "boxes": len(data["rows"])}

    # ---- 1 timing
    def mk(z):
        if a.mode == "legacy":
            return LiveSpeed(profile, z, frame_size=(W, Hh), in_zone_only=False, max_gap_s=1e9, max_jump_m=1e9, outlier_floor_m=1e9)
        return LiveSpeed(profile, z, frame_size=(W, Hh), min_span_s=0.4, min_samples=8, ema=1.0, max_error_kmh=10.0)

    out["timing"] = {"container_minus_index_over_fps_s": data["pos_msec_minus_index_over_fps_s"], "fps": fps}

    tracks = collections.defaultdict(list)
    cls_of = {}
    for f, t, rid, cls, conf, x1, y1, x2, y2 in data["rows"]:
        tracks[rid].append((t, (x1, y1, x2, y2), conf))
        cls_of[rid] = cls
    def ground(b, mode="bottom"):
        x1, y1, x2, y2 = b
        return {"bottom": ((x1 + x2) / 2, y2), "centre": ((x1 + x2) / 2, (y1 + y2) / 2), "bl": (x1, y2), "br": (x2, y2)}[mode]

    def lane_of(pt):
        for k, poly in lanes.items():
            if point_in_polygon(pt, poly):
                return k
        return None

    def series(rid, mode="bottom"):
        return [S(t, ground(b, mode), b) for t, b, _c in tracks[rid]]

    # ---- 2 what LiveSpeed shows (every 0.25 s per track, same estimator and thresholds as the app)
    est_rows = []  # (rid, t, kmh, err, row, class, lane)
    per_sec = collections.defaultdict(list)
    status_count = collections.Counter()
    for rid in tracks:
        ls = mk(zone)
        ser = series(rid)
        nxt = 0.0
        for i, s in enumerate(ser):
            if s.t < nxt:
                continue
            nxt = s.t + 0.25
            r = ls.estimate(str(rid), ser[: i + 1])
            status_count[r["status"]] += 1
            if r["status"] == "ok":
                est_rows.append((rid, s.t, r["speed_kmh"], r["error_kmh"], s.ground[1], cls_of[rid], lane_of(s.ground)))
                per_sec[int(s.t)].append(r["speed_kmh"])
    kmh = [r[2] for r in est_rows]
    out["live_speed_status_counts"] = dict(status_count)
    out["live_speed_all"] = q(kmh)
    out["by_class"] = {c: q([r[2] for r in est_rows if r[5] == c]) for c in sorted({r[5] for r in est_rows})}
    out["by_lane"] = {l: q([r[2] for r in est_rows if r[6] == l]) for l in sorted({r[6] for r in est_rows if r[6]})}
    seconds = [max(v) for v in per_sec.values() if v]
    med_seconds = [statistics.median(v) for v in per_sec.values() if v]
    out["tile_fastest_of_measured_per_second"] = q(seconds)
    out["median_of_measured_per_second"] = q(med_seconds)
    out["seconds_with_a_measured_vehicle"] = len(seconds)
    out["share_of_estimates_above_limit"] = round(sum(1 for v in kmh if v > cfg["speed_limit_kmh"]) / max(1, len(kmh)), 4)
    out["share_of_per_second_max_above_limit"] = round(sum(1 for v in seconds if v > cfg["speed_limit_kmh"]) / max(1, len(seconds)), 4)
    out["displayed_error_band_kmh"] = q([r[3] for r in est_rows])

    # ---- 3 reference point: speed from bottom-centre vs box centre (same estimator, zone test on the bottom-centre)
    ratios = collections.defaultdict(list)
    for rid in list(tracks)[:]:
        if len(tracks[rid]) < 40:
            continue
        base = series(rid, "bottom")
        for mode in ("centre", "bl", "br"):
            alt = series(rid, mode)
            ls_b, ls_a = mk(None), mk(None)
            for i in range(30, len(base), 25):
                zone_ok = point_in_polygon(base[i].ground, zone)
                if not zone_ok:
                    continue
                rb = ls_b.estimate("b", base[: i + 1]); ra = ls_a.estimate("a", alt[: i + 1])
                if rb["status"] == "ok" and ra["status"] == "ok" and rb["speed_kmh"] > 40:
                    ratios[mode].append(ra["speed_kmh"] / rb["speed_kmh"])
    out["reference_point_ratio_vs_bottom_centre"] = {m: q(v, (10, 50, 90)) for m, v in ratios.items()}

    # ---- 4 consistency across the picture: same steady track measured far (rows 225-260) vs near (rows 320-355)
    near_far = []
    for rid in tracks:
        ser = series(rid)
        ls = mk(zone)
        far, near = [], []
        for i in range(len(ser)):
            if i % 5:
                continue
            r = ls.estimate(str(rid), ser[: i + 1])
            if r["status"] != "ok":
                continue
            row = ser[i].ground[1]
            if 225 <= row <= 262:
                far.append(r["speed_kmh"])
            elif 318 <= row <= 358:
                near.append(r["speed_kmh"])
        if len(far) >= 3 and len(near) >= 3:
            near_far.append((statistics.median(near) / statistics.median(far), statistics.median(far), statistics.median(near)))
    out["near_over_far_speed_ratio"] = (q([x[0] * 100 for x in near_far], (10, 25, 50, 75, 90)) | {"note": "values are percent; 100 = identical speed near and far"}) if near_far else None

    # ---- 5 noise: residual of a straight-line fit of position against time, per image-row band
    res_by_band = collections.defaultdict(list)
    for rid in tracks:
        ser = series(rid)
        for i in range(0, len(ser) - 40, 20):
            win = ser[i:i + 40]
            if not point_in_polygon(win[-1].ground, zone) or any(w.bbox[3] >= Hh - 3 or w.bbox[2] >= W - 3 for w in win):
                continue
            ts = np.array([w.t for w in win]); gx = np.array([w.ground[0] for w in win]); gy = np.array([w.ground[1] for w in win])
            wx, wy = zip(*[profile.image_to_world(w.ground) for w in win])
            wx, wy = np.array(wx), np.array(wy)
            A = np.vstack([ts, np.ones_like(ts)]).T
            px = float(np.sqrt(np.mean((np.linalg.lstsq(A, gy, rcond=None)[0] @ A.T - gy) ** 2)))
            rm = float(np.sqrt(np.mean((A @ np.linalg.lstsq(A, wx, rcond=None)[0] - wx) ** 2 + (A @ np.linalg.lstsq(A, wy, rcond=None)[0] - wy) ** 2)))
            band = "225-270" if win[-1].ground[1] < 270 else ("270-320" if win[-1].ground[1] < 320 else "320-360")
            res_by_band[band].append((px, rm))
    out["fit_residual_by_row_band"] = {b: {"n": len(v), "median_px": round(statistics.median(x[0] for x in v), 2), "median_m": round(statistics.median(x[1] for x in v), 3)} for b, v in res_by_band.items()}

    # ---- 6 lateral scale: metric width of the box bottom edge through the calibration (rows >= 300, vehicle not clipped by the frame edge)
    widths = collections.defaultdict(list)
    for rid, rows in tracks.items():
        for t, b, _c in rows[::10]:
            x1, y1, x2, y2 = b
            if y2 < 300 or y2 >= Hh - 3 or x2 >= W - 3 or x1 <= 3:
                continue
            if not point_in_polygon(((x1 + x2) / 2, y2), zone):
                continue
            X1, _ = profile.image_to_world((x1, y2)); X2, _ = profile.image_to_world((x2, y2))
            widths[cls_of[rid]].append(abs(X2 - X1))
    out["box_bottom_width_m_by_class_rows_ge_300"] = {c: q(v, (10, 50, 90)) for c, v in widths.items()}

    # ---- 7 priors (plausibility only)
    trucks = [r[2] for r in est_rows if r[5] == "truck"]
    out["truck_class_speed"] = q(trucks)
    out["truck_class_share_above_100kmh"] = round(sum(1 for v in trucks if v > 100) / max(1, len(trucks)), 3) if trucks else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
