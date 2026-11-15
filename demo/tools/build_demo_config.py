"""Builds demo/camera_config.json for the demonstration video (demo.mp4, fixed overpass camera over a divided motorway).

CALIBRATION (read demo/README.md section "Speed calibration" before quoting any number):
  * Scale source: UK Traffic Signs Manual Chapter 5, Table 2-2 / Figure 2-3 (GOV.UK, DfT 2018): motorway lane lines are
    diagram 1005.1 = 2000 mm marking + 7000 mm gap, i.e. a 9 m pitch (stud spacing 18 m = 2 pitches, consistent).
    The road in the video LOOKS like a UK motorway (left-hand traffic, gantry, dashed lane lines) but its identity is NOT
    verified, so this is an ASSUMPTION and the speeds are APPROXIMATE (see ValidationEvidenceKind.MARKING_STANDARD_HOLDOUT).
  * Lane width across the pair of dashed lines is taken as 3.65 m (standard motorway lane); it only affects the lateral axis.
  * Dash end points below were measured on a temporal-median background of the video (vehicles removed) by
    thresholding the bright dashes and fitting each blob's principal axis (sub-pixel ends).
  * FIT on dash 0 (far end only: it is clipped by the frame edge), dash 1 and dash 2 of the two lines (10 points);
    HOLD OUT dash 3 (4 points) and check them against the fit. Hold-out RMSE is written into the validation record.
"""
import json
import pathlib

import cv2
import numpy as np

W = 3.65
# (image x, image y) -> (across m, along-road m, 0 = near end of dash 0); 9 m pitch, 2 m dashes
FIT = [((489.2, 336.8), (0, 2)), ((582.3, 336.7), (W, 2)),
       ((442.8, 276.1), (0, 9)), ((433.2, 263.9), (0, 11)), ((511.8, 276.2), (W, 9)), ((498.3, 264.7), (W, 11)),
       ((411.1, 234.9), (0, 18)), ((405.4, 227.7), (0, 20)), ((464.8, 236.2), (W, 18)), ((457.1, 229.9), (W, 20))]
HOLD = [((392.7, 210.3), (0, 27)), ((389.0, 206.0), (0, 29)), ((435.7, 211.4), (W, 27)), ((430.7, 207.4), (W, 29))]

# zones, image px (640x360 frame). Toward-camera carriageway = image right; away = image left.
TOWARD = [[337, 150], [395, 150], [640, 315], [640, 360], [397, 360]]
AWAY = [[231, 150], [302, 150], [245, 360], [0, 360], [0, 305]]
LANES = {
    "TOWARD-3-offside": [[344, 175], [365, 175], [507, 360], [397, 360]],
    "TOWARD-2": [[365, 175], [393, 175], [609, 360], [507, 360]],
    "TOWARD-1-nearside": [[393, 175], [432, 175], [640, 315], [640, 360], [609, 360]],
    "AWAY-1-nearside": [[231, 175], [242, 175], [20, 360], [0, 360], [0, 305]],
    "AWAY-2": [[242, 175], [268, 175], [129, 360], [20, 360]],
    "AWAY-3-offside": [[268, 175], [303, 175], [245, 360], [129, 360]],
}
SPEED_ZONE = [[358, 225], [506, 225], [640, 315], [640, 360], [397, 360]]  # near field of the toward carriageway only


def main() -> None:
    ip = np.array([p[0] for p in FIT]); wp = np.array([p[1] for p in FIT])
    H, _ = cv2.findHomography(ip, wp, 0)

    def tr(p):
        v = H @ np.array([p[0], p[1], 1.0]); return v[:2] / v[2]

    fit_rmse = float(np.sqrt(np.mean([np.sum((tr(a) - np.array(b)) ** 2) for a, b in FIT])))
    hold_rmse = float(np.sqrt(np.mean([np.sum((tr(a) - np.array(b)) ** 2) for a, b in HOLD])))
    print(f"fit RMSE {fit_rmse:.3f} m   hold-out RMSE {hold_rmse:.3f} m  (4 held-out dash ends)")
    cfg = {
        "max_stream_s": 1200, "realtime_factor": 1.0, "stop_criterion": "displacement",
        "speed_limit_kmh": 112.7,  # 70 mph, the UK national motorway limit, used as the DEMONSTRATION threshold (site signage unknown)
        "speed_zone": SPEED_ZONE,
        "calibration": {
            "reference_points": [{"image_xy": list(a), "world_xy": list(b)} for a, b in FIT],
            "validation": {"kind": "marking_standard_holdout", "n_holdout_points": len(HOLD), "holdout_rmse_m": round(hold_rmse, 3),
                           "tolerance_m": 1.0,
                           "evidence_ref": "demo/README.md#speed-calibration; UK TSM ch.5 Table 2-2 / Fig 2-3 diagram 1005.1 (2 m mark + 7 m gap); "
                                           "dash 3 (4 ends) held out; road identity and lane width (3.65 m) ASSUMED, site not surveyed"},
        },
        "direction_zones": [
            {"zone_id": "toward-camera", "polygon": TOWARD, "expected": [0.6, 0.8], "min_displacement_px": 60, "min_duration_s": 1.5},
            {"zone_id": "away-from-camera", "polygon": AWAY, "expected": [0.6, -0.8], "min_displacement_px": 60, "min_duration_s": 1.5},
        ],
        "lanes": [{"lane_id": k, "vertices": v} for k, v in LANES.items()],
        # congestion_close_grace_s 60: the replayed level sequence of this footage cycles building/congested/dissipating every ~50 s (stop-and-go waves);
        # a 30 s grace gives 7 episodes over the 20 minutes and 60 s gives 4 (the 13 separate episodes of the first run were one regime).
        # NOT set here: "tracker": {"conf": 0.10}. It lets ByteTrack's second association stage see low-score boxes (-8 % track ids, -15 % near-field
        # re-link candidates on 180 s, demo/runs/track_continuity.json) but adds ~10 % boxes per frame, which shifts the count-based congestion proxy
        # (flowing, dense traffic at ~230 s then reads "congested"), and the extra boxes cannot be validated without labels. Opt-in only.
        "thresholds": {"queue_peer_stops": 3, "congestion_close_grace_s": 60},
    }
    out = pathlib.Path(__file__).resolve().parent.parent / "camera_config.json"
    out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
