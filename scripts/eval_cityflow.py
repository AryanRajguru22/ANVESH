"""CityFlowV2 ground-truth evaluation of ANVESH's single-camera perception,
with the three tiers kept SEPARATE:

  TIER 1  DETECTOR     raw detector boxes (YOLOv8n `predict`, no tracker)
                       -> precision / recall @ IoU 0.5
  TIER 2  TRACKER      raw ByteTrack IDs as emitted by the production tracker
                       -> IDF1 / ID switches / fragmentation
  TIER 3  TRACKMANAGER the identity ANVESH's TrackManager assigns on top of the
                       tracker output, BEFORE (frozen legacy snapshot, raw-id
                       keyed, 5-frame occlusion clock) and AFTER (session-scoped
                       serial identity, time-based horizon aligned to ByteTrack's
                       buffer) -> the same identity metrics + duplicate-identity
                       statistics

All tiers replay ONE cached real run (`scripts/cityflow_cache_perception.py`),
so tier-3 before/after differ only in TrackManager logic, never in detector or
tracker noise.

GROUND-TRUTH CAVEAT (dataset ReadMe item 6): `gt.txt` annotates only vehicles
seen by >= 2 cameras. So:
  - recall, IDR, ID switches, fragmentation: VALID on the annotated vehicles;
  - precision, IDP, IDF1: LOWER BOUNDS (correct but unannotated vehicles count
    as false positives). True precision is UNAVAILABLE from this GT and is
    reported as such. The shipped baseline detectors/trackers are scored the same
    way as a sanity reference.
Predictions outside `roi.jpg` (by bottom-centre ground point) are ignored, per
the dataset's ROI definition.

Usage:
    python scripts/eval_cityflow.py --camera c006
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.evaluation import cityflow_gt as gt_io  # noqa: E402
from anvesh.evaluation.legacy_track_manager import TrackManager as LegacyTrackManager  # noqa: E402
from anvesh.evaluation.mot_metrics import (  # noqa: E402
    duplicate_identity_stats,
    duplicate_ids_per_window,
    evaluate_detection,
    evaluate_tracking,
    iou_matrix,
)
from anvesh.perception.detection import COCO_CLASS_NAMES, COCO_TO_VEHICLE_CLASS  # noqa: E402
from anvesh.perception.track_manager import TrackManager, occlusion_horizon_s  # noqa: E402
from anvesh.perception.types import RawDetection, TrackedDetection  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "cityflow"
WINDOW_S = 10.0


def load_cache(path: Path) -> tuple:
    frames = []
    meta = None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
            else:
                frames.append(obj)
    return meta, frames


def _xywh(b):
    return (b[0], b[1], b[2] - b[0], b[3] - b[1])


def tables_from_cache(frames: list, offset: int, roi) -> tuple:
    """-> (detector_table, tracker_table) as {gt_frame: [(id, x, y, w, h)]}, ROI-filtered."""
    det, trk = {}, {}
    for fr in frames:
        f = fr["i"] + offset
        d_rows = [(-1, *_xywh(b[:4])) for b in fr["det"]]
        t_rows = [(int(b[6]), *_xywh(b[:4])) for b in fr["trk"] if b[6] is not None]
        if roi is not None:
            d_rows = [r for r in d_rows if gt_io.inside_roi(roi, *r[1:5])]
            t_rows = [r for r in t_rows if gt_io.inside_roi(roi, *r[1:5])]
        det[f] = d_rows
        trk[f] = t_rows
    return det, trk


def alignment_check(gt: dict, frames: list) -> dict:
    """Mean best-IoU between GT boxes and detector boxes when the GT frame
    number is video-index + offset, offsets -1/0/+1/+2. The offset with the
    clearly highest value is the right alignment."""
    out = {}
    for off in (-1, 0, 1, 2):
        vals = []
        for fr in frames:
            g = [r[1:5] for r in gt.get(fr["i"] + off, [])]
            p = [_xywh(b[:4]) for b in fr["det"]]
            if not g or not p:
                continue
            iou = iou_matrix(g, p)
            vals.extend(iou.max(axis=1).tolist())
        out[off] = sum(vals) / len(vals) if vals else 0.0
    return out


def _detections_from_trk(frame: dict, ts: float, with_ids: bool = True) -> list:
    out = []
    for b in frame["trk"]:
        if b[6] is None:
            continue
        cls = int(b[5])
        vc = COCO_TO_VEHICLE_CLASS.get(cls)
        if vc is None:
            continue
        det = RawDetection(
            frame_index=frame["i"], timestamp=ts, class_name=COCO_CLASS_NAMES.get(cls, str(cls)),
            vehicle_class=vc, confidence=min(1.0, max(0.0, float(b[4]))), bbox_xyxy=tuple(float(v) for v in b[:4]),
        )
        out.append(TrackedDetection(detection=det, track_id=str(int(b[6]))))
    return out


def replay_legacy(frames: list, fps: float, offset: int, roi) -> dict:
    """Frozen pre-fix TrackManager (5-frame occlusion clock)."""
    mgr = LegacyTrackManager(max_missed_frames=5, partial_after_missed_frames=1)
    state_index = {}  # id(state) -> ordinal
    by_raw, by_state = {}, {}
    for fr in frames:
        ts = fr["i"] / fps
        tds = _detections_from_trk(fr, ts)
        mgr.update(ts, tds)
        rows_raw, rows_state = [], []
        for td in tds:
            st = mgr.active_tracks.get(td.track_id)
            ordinal = state_index.setdefault(id(st), len(state_index))
            x, y, w, h = _xywh(td.detection.bbox_xyxy)
            if roi is not None and not gt_io.inside_roi(roi, x, y, w, h):
                continue
            rows_raw.append((td.track_id, x, y, w, h))  # what downstream joins on
            rows_state.append((f"st{ordinal}", x, y, w, h))
        by_raw[fr["i"] + offset] = rows_raw
        by_state[fr["i"] + offset] = rows_state
    mgr.finalize()
    states = mgr.all_tracks()
    return {"by_raw_id": by_raw, "by_state": by_state, "states": states,
            "state_keys": [s.track_id for s in states], "window_ids": _window_id_lists(states, fps, lambda s: s.track_id)}


def replay_new(frames: list, fps: float, offset: int, roi, horizon_s: float, label: str) -> dict:
    mgr = TrackManager(max_occlusion_s=horizon_s, camera_id="cam", session_id="s000")
    table = {}
    for fr in frames:
        ts = fr["i"] / fps
        tds = _detections_from_trk(fr, ts)
        assignments = mgr.update(ts, tds)
        rows = []
        for a in assignments:
            x, y, w, h = _xywh(a.tracked.detection.bbox_xyxy)
            if roi is not None and not gt_io.inside_roi(roi, x, y, w, h):
                continue
            rows.append((a.track_id, x, y, w, h))
        table[fr["i"] + offset] = rows
    mgr.finalize()
    states = mgr.all_tracks()
    return {"table": table, "states": states, "state_keys": [s.identity_key for s in states],
            "horizon_s": horizon_s, "label": label,
            "window_ids": _window_id_lists(states, fps, lambda s: s.track_id)}


def _window_id_lists(states: list, fps: float, key) -> list:
    """Per 10 s window, the ids of states overlapping it (audit's framing)."""
    if not states:
        return []
    end = max(s.last_seen for s in states)
    out, start = [], 0.0
    while start < end:
        w_end = start + WINDOW_S
        out.append([key(s) for s in states if s.first_seen < w_end and s.last_seen > start])
        start = w_end
    return out


def _score_dict(score) -> dict:
    return dataclasses.asdict(score)


def baseline_scores(cam_dir: Path, gt: dict, roi, n_frames: int) -> dict:
    out = {}
    frames = range(1, n_frames + 1)
    for name in ("det_yolo3", "det_ssd512", "det_mask_rcnn"):
        p = cam_dir / "det" / f"{name}.txt"
        if p.is_file():
            table = gt_io.filter_table_by_roi(gt_io.load_mot_file(p), roi)
            out[name] = _score_dict(evaluate_detection(gt, table, frames=frames))
    for name in ("mtsc_tc_yolo3", "mtsc_deepsort_yolo3", "mtsc_moana_yolo3"):
        p = cam_dir / "mtsc" / f"{name}.txt"
        if p.is_file():
            table = gt_io.filter_table_by_roi(gt_io.load_mot_file(p), roi)
            out[name] = _score_dict(evaluate_tracking(gt, table, frames=frames))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="S02")
    ap.add_argument("--camera", default="c006")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    cam_dir = gt_io.camera_dir(args.scenario, args.camera)
    cache_path = OUT_DIR / f"{args.scenario.lower()}_{args.camera}_perception_cache.jsonl.gz"
    meta, frames = load_cache(cache_path)
    fps = float(meta["fps"])
    n_frames = len(frames)

    gt = gt_io.load_mot_file(cam_dir / "gt" / "gt.txt")
    roi = gt_io.load_roi_mask(cam_dir / "roi.jpg")
    gt_boxes = sum(len(v) for v in gt.values())
    gt_ids = len({r[0] for v in gt.values() for r in v})

    align = alignment_check(gt, frames)
    offset = max(align, key=align.get)
    eval_frames = range(1, n_frames + 1)

    det_table, trk_table = tables_from_cache(frames, offset, roi)
    det_table_noroi, _ = tables_from_cache(frames, offset, None)

    report = {
        "camera": f"{args.scenario}/{args.camera}",
        "video_frames": n_frames,
        "fps": fps,
        "model": meta["model"], "conf": meta["conf"], "ultralytics": meta["ultralytics"], "torch": meta["torch"],
        "gt": {"boxes": gt_boxes, "tracks": gt_ids, "caveat": "gt.txt annotates only vehicles seen by >=2 cameras"},
        "frame_alignment_mean_best_iou_by_offset": align, "frame_offset_used": offset,
    }

    # TIER 1 -- detector
    d_roi = evaluate_detection(gt, det_table, frames=eval_frames)
    d_all = evaluate_detection(gt, det_table_noroi, frames=eval_frames)
    report["tier1_detector"] = {
        "iou": 0.5, "roi_filtered": _score_dict(d_roi), "unfiltered": _score_dict(d_all),
        "note": "recall valid on annotated vehicles; precision is a LOWER BOUND (unannotated single-camera vehicles count as FP); true precision UNAVAILABLE",
        "precision_lower_bound": d_roi.precision, "recall": d_roi.recall,
    }

    # TIER 2 -- tracker (raw ByteTrack ids)
    report["tier2_tracker_raw_bytetrack_ids"] = _score_dict(evaluate_tracking(gt, trk_table, frames=eval_frames))

    # TIER 3 -- TrackManager identity, before vs after
    legacy = replay_legacy(frames, fps, offset, roi)
    horizon = occlusion_horizon_s(meta.get("tracker_buffer", 30), fps)
    after = replay_new(frames, fps, offset, roi, horizon, "serial identity, ByteTrack-aligned horizon")
    serial_only = replay_new(frames, fps, offset, roi, 5 / fps, "serial identity, legacy 5-frame horizon (isolates the serial mechanism)")

    def dup(d):
        s = duplicate_identity_stats(d["state_keys"])
        entries, dups = duplicate_ids_per_window(d["window_ids"])
        return {**dataclasses.asdict(s), "window_entries": entries, "window_duplicate_entries": dups,
                "window_duplicate_rate": dups / entries if entries else 0.0}

    report["tier3_trackmanager"] = {
        "before_legacy": {
            "description": "frozen pre-fix TrackManager: raw tracker id is the identity; 5-frame occlusion clock; a returning id opens a SECOND state under the same id",
            "downstream_view_keyed_by_raw_id": _score_dict(evaluate_tracking(gt, legacy["by_raw_id"], frames=eval_frames)),
            "per_state_view": _score_dict(evaluate_tracking(gt, legacy["by_state"], frames=eval_frames)),
            "duplicates": dup(legacy),
        },
        "after_fixed": {
            "description": after["label"], "horizon_s": after["horizon_s"],
            "identity": _score_dict(evaluate_tracking(gt, after["table"], frames=eval_frames)),
            "duplicates": dup(after),
        },
        "after_serial_only": {
            "description": serial_only["label"], "horizon_s": serial_only["horizon_s"],
            "identity": _score_dict(evaluate_tracking(gt, serial_only["table"], frames=eval_frames)),
            "duplicates": dup(serial_only),
        },
        "tracklet_lifetimes_s": {
            "legacy_states": _lifetime_stats(legacy["states"]), "after_states": _lifetime_stats(after["states"]),
        },
    }

    if not args.no_baselines:
        report["shipped_baselines_same_harness"] = baseline_scores(cam_dir, gt, roi, n_frames)

    out = OUT_DIR / f"{args.scenario.lower()}_{args.camera}_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"\nwrote {out}")
    return 0


def _lifetime_stats(states: list) -> dict:
    import statistics

    life = sorted(s.last_seen - s.first_seen for s in states)
    return {"states": len(states), "median_s": statistics.median(life) if life else 0.0,
            "frac_lt_2s": sum(1 for v in life if v < 2.0) / len(life) if life else 0.0}


def _print_summary(r: dict) -> None:
    def trk(label, d):
        print(f"  {label:<46} IDF1={d['idf1']:.3f} IDP={d['idp']:.3f} IDR={d['idr']:.3f} IDSW={d['id_switches']} FM={d['fragmentations']} "
              f"pred_tracks={d['pred_tracks']}")

    print(f"== {r['camera']}  frames={r['video_frames']}  GT boxes={r['gt']['boxes']} tracks={r['gt']['tracks']}  offset={r['frame_offset_used']}")
    print("alignment (mean best IoU):", {k: round(v, 3) for k, v in r["frame_alignment_mean_best_iou_by_offset"].items()})
    t1 = r["tier1_detector"]["roi_filtered"]
    print(f"TIER1 detector  P(lower bound)={t1['precision']:.3f} R={t1['recall']:.3f} TP={t1['tp']} FP={t1['fp']} FN={t1['fn']}")
    trk("TIER2 tracker (raw ByteTrack ids)", r["tier2_tracker_raw_bytetrack_ids"])
    t3 = r["tier3_trackmanager"]
    trk("TIER3 before  (raw-id keyed, what downstream joins)", t3["before_legacy"]["downstream_view_keyed_by_raw_id"])
    trk("TIER3 before  (per TrackState)", t3["before_legacy"]["per_state_view"])
    trk("TIER3 after   (serial identity, aligned horizon)", t3["after_fixed"]["identity"])
    trk("TIER3 after   (serial identity, 5-frame horizon)", t3["after_serial_only"]["identity"])
    for k in ("before_legacy", "after_fixed", "after_serial_only"):
        d = t3[k]["duplicates"]
        print(f"  duplicates [{k}]: states={d['states']} keys={d['identity_keys']} dup_keys={d['duplicated_keys']} "
              f"dup_rate={d['duplicate_rate']:.4f}  per-window dup entries={d['window_duplicate_entries']}/{d['window_entries']}")


if __name__ == "__main__":
    raise SystemExit(main())
