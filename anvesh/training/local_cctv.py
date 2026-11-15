"""Local-CCTV detector benchmark: dataset layout, annotation QA, group-wise splitting and group-level statistics.

Layout of one local dataset directory (everything is plain files; nothing here needs a new dependency):

    manifest.json          one record per frame (see MANIFEST_FIELDS); the single source of truth
    frames/<frame_id>.jpg  extracted frames (never overwritten)
    labels/<frame_id>.txt  YOLO rows `cls cx cy w h` (normalised), classes = LOCAL_CLASSES
    ignore/<frame_id>.txt  YOLO rows (class column ignored): don't-care regions -- difficult, unclassifiable
    labels_second/<annotator>/<frame_id>.txt   optional second-annotator labels, used ONLY for agreement checks
    annotation_log.jsonl   append-only record of every import

Label-file semantics (this is what keeps precision honest):
    MISSING label file  -> the frame is UNLABELLED and is excluded from every metric (`labelled` = false).
    EMPTY label file    -> a human verified there are no (non-difficult) vehicles in the frame; counts as a negative frame.

Splitting is by SOURCE GROUP (recording, camera, or camera-day), never by frame, so adjacent frames cannot straddle
a split. Scoring reuses `detector_eval` (the same matcher/AP code as the MIO-TCD comparisons); statistical uncertainty is
estimated by resampling groups, not frames, because frames of one recording are strongly correlated.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from anvesh.training.detector_eval import SHARED_CLASSES, ImageResult, box_iou_np, score
from anvesh.training.splitting import assign_groups_to_splits

# id = index. 0..4 are exactly detector_eval.SHARED_CLASSES (both candidate checkpoints can output them).
LOCAL_CLASSES = SHARED_CLASSES + ("auto_rickshaw",)
UNSUPPORTED_CLASSES = ("auto_rickshaw",)  # labelled, but neither candidate model can output it
IGNORE_NAMES = ("difficult", "ignore", "other_vehicle")  # export class names that mean "don't-care region"
SPLITS = ("train", "val", "test", "eval")  # `eval` = evaluation-only footage that will never be trained on
GROUP_BY = ("recording", "camera", "camera_day")

REQUIRED_FIELDS = ("frame_id", "image", "width", "height", "image_sha256", "camera_id", "recording_id",
                   "labelled", "split")
RECOMMENDED_FIELDS = ("recording_start_utc", "frame_time_s", "source_video", "source_video_sha256", "conditions",
                      "annotator", "annotation_tool", "annotation_tool_version", "annotated_utc", "reviewed")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_frame_id(camera_id: str, recording_id: str, frame_index: int) -> str:
    for part in (camera_id, recording_id):
        if not part or any(c in part for c in "/\\ :*?\"<>|") or "__" in part:
            raise ValueError(f"camera_id/recording_id {part!r} must be non-empty, without separators, spaces or '__'")
    return f"{camera_id}__{recording_id}__{int(frame_index):07d}"


def group_key(rec: dict, group_by: str) -> str:
    if group_by == "recording":
        return f"{rec['camera_id']}/{rec['recording_id']}"
    if group_by == "camera":
        return rec["camera_id"]
    if group_by == "camera_day":
        day = (rec.get("recording_start_utc") or "")[:10]
        if not day:
            raise ValueError(f"group_by=camera_day needs recording_start_utc on every frame ({rec['frame_id']} has none)")
        return f"{rec['camera_id']}/{day}"
    raise ValueError(f"group_by must be one of {GROUP_BY}")


# ---------------------------------------------------------------- manifest io

def load_manifest(root) -> dict:
    p = Path(root) / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; run `local_cctv.py extract` first")
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(root, manifest: dict) -> None:
    """Write atomically and keep the previous revision under manifest_history/ (never lose an earlier manifest)."""
    root = Path(root)
    cur = root / "manifest.json"
    if cur.is_file():
        hist = root / "manifest_history"
        hist.mkdir(exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        (hist / f"manifest-{stamp}.json").write_bytes(cur.read_bytes())
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(cur)


# ---------------------------------------------------------------- label parsing and QA

def parse_yolo_rows(text: str, n_classes: int = len(LOCAL_CLASSES)):
    """-> (rows [(cls, cx, cy, w, h)], errors [str]). Never raises on malformed content; reports it."""
    rows, errors = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {i}: expected 5 fields, got {len(parts)}")
            continue
        try:
            c, vals = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line {i}: non-numeric field")
            continue
        if not 0 <= c < n_classes:
            errors.append(f"line {i}: class id {c} outside 0..{n_classes - 1}")
            continue
        if any(not np.isfinite(v) for v in vals):
            errors.append(f"line {i}: non-finite value")
            continue
        rows.append((c, *vals))
    return rows, errors


def rows_to_xyxy(rows, w: int, h: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, 4))
    a = np.array([r[1:] for r in rows], dtype=float)
    return np.stack([(a[:, 0] - a[:, 2] / 2) * w, (a[:, 1] - a[:, 3] / 2) * h,
                     (a[:, 0] + a[:, 2] / 2) * w, (a[:, 1] + a[:, 3] / 2) * h], axis=1)


def qa_frame(rows, width: int, height: int, min_side_px: float = 8.0, dup_iou: float = 0.9, tol: float = 1e-3):
    """Per-frame annotation checks -> list of {"code", "detail"}. Codes: out_of_bounds, degenerate, tiny_box, duplicate_box."""
    issues = []
    for i, (c, cx, cy, bw, bh) in enumerate(rows):
        if bw <= 0 or bh <= 0:
            issues.append({"code": "degenerate", "detail": f"row {i + 1}: non-positive width/height"})
            continue
        if cx - bw / 2 < -tol or cy - bh / 2 < -tol or cx + bw / 2 > 1 + tol or cy + bh / 2 > 1 + tol:
            issues.append({"code": "out_of_bounds", "detail": f"row {i + 1}: box leaves the image"})
        if bw * width < min_side_px or bh * height < min_side_px:
            issues.append({"code": "tiny_box",
                           "detail": f"row {i + 1}: {bw * width:.1f}x{bh * height:.1f}px < {min_side_px}px; label it as difficult/ignore or remove"})
    xyxy = rows_to_xyxy([r for r in rows if r[3] > 0 and r[4] > 0], width, height)
    if len(xyxy) > 1:
        iou = box_iou_np(xyxy, xyxy)
        np.fill_diagonal(iou, 0.0)
        for a, b in zip(*np.nonzero(np.triu(iou >= dup_iou))):
            issues.append({"code": "duplicate_box", "detail": f"rows {a + 1} and {b + 1} overlap with IoU {iou[a, b]:.2f}"})
    return issues


def agreement(rows_a, rows_b, width: int, height: int, iou_thr: float = 0.5) -> dict:
    """Two annotators on one frame. Greedy one-to-one IoU matching, class-agnostic for localisation, then class agreement."""
    a, b = rows_to_xyxy(rows_a, width, height), rows_to_xyxy(rows_b, width, height)
    matched, same_class = 0, 0
    if len(a) and len(b):
        iou = box_iou_np(a, b)
        pairs = sorted(((iou[i, j], i, j) for i in range(len(a)) for j in range(len(b)) if iou[i, j] >= iou_thr), reverse=True)
        ua, ub = set(), set()
        for _, i, j in pairs:
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            matched += 1
            same_class += int(rows_a[i][0] == rows_b[j][0])
    return {"n_a": len(rows_a), "n_b": len(rows_b), "matched": matched, "same_class": same_class}


def summarise_agreement(per_frame: list) -> dict:
    na, nb = sum(x["n_a"] for x in per_frame), sum(x["n_b"] for x in per_frame)
    m, sc = sum(x["matched"] for x in per_frame), sum(x["same_class"] for x in per_frame)
    f1 = 2 * m / (na + nb) if na + nb else None
    return {"frames": len(per_frame), "boxes_a": na, "boxes_b": nb, "matched": m, "box_agreement_f1": f1,
            "class_agreement_given_match": sc / m if m else None}


def check_dataset(root, manifest: dict | None = None, min_side_px: float = 8.0, near_time_s: float = 600.0) -> dict:
    """Validate a local dataset. -> {"errors": [...], "warnings": [...], "counts": {...}}. Errors block benchmarking."""
    root = Path(root)
    manifest = manifest or load_manifest(root)
    errors, warnings = [], []
    frames = manifest.get("frames", [])
    seen, split_of_group, missing_time = set(), {}, 0
    counts = {"frames": len(frames), "labelled": 0, "unlabelled": 0, "empty_negative_frames": 0, "boxes": 0,
              "ignore_regions": 0, "per_class": {c: 0 for c in LOCAL_CLASSES}}
    for rec in frames:
        fid = rec.get("frame_id", "?")
        miss = [f for f in REQUIRED_FIELDS if f not in rec]
        if miss:
            errors.append(f"{fid}: missing manifest fields {miss}")
            continue
        if fid in seen:
            errors.append(f"{fid}: duplicate frame_id")
        seen.add(fid)
        img = root / rec["image"]
        if not img.is_file():
            errors.append(f"{fid}: image file {rec['image']} not found")
        elif sha256_file(img) != rec["image_sha256"]:
            errors.append(f"{fid}: image SHA-256 differs from the manifest (file changed after import)")
        if rec["split"] not in SPLITS and rec["split"] is not None:
            errors.append(f"{fid}: split {rec['split']!r} not in {SPLITS}")
        if not rec.get("recording_start_utc"):
            missing_time += 1
        lbl, ign = root / "labels" / f"{fid}.txt", root / "ignore" / f"{fid}.txt"
        if rec["labelled"]:
            if not lbl.is_file():
                errors.append(f"{fid}: labelled=true but labels/{fid}.txt is missing (missing = unlabelled; use an empty file for 'no vehicles')")
                continue
            counts["labelled"] += 1
            rows, perr = parse_yolo_rows(lbl.read_text(encoding="utf-8"))
            errors += [f"{fid}: labels: {e}" for e in perr]
            irows, ierr = parse_yolo_rows(ign.read_text(encoding="utf-8")) if ign.is_file() else ([], [])
            errors += [f"{fid}: ignore: {e}" for e in ierr]
            if not rows:
                counts["empty_negative_frames"] += 1
            counts["boxes"] += len(rows)
            counts["ignore_regions"] += len(irows)
            for r in rows:
                counts["per_class"][LOCAL_CLASSES[r[0]]] += 1
            for q in qa_frame(rows, rec["width"], rec["height"], min_side_px):
                (errors if q["code"] in ("degenerate", "out_of_bounds") else warnings).append(f"{fid}: {q['code']}: {q['detail']}")
            if not rec.get("annotator"):
                warnings.append(f"{fid}: labelled frame has no annotator recorded")
        else:
            counts["unlabelled"] += 1
            if lbl.is_file():
                errors.append(f"{fid}: labelled=false but labels/{fid}.txt exists (re-run import or fix the manifest)")
        if rec["split"]:
            gk = (rec["camera_id"], rec["recording_id"])
            if split_of_group.setdefault(gk, rec["split"]) != rec["split"]:
                errors.append(f"LEAKAGE: recording {gk[0]}/{gk[1]} appears in splits {split_of_group[gk]!r} and {rec['split']!r}")
    if missing_time:
        warnings.append(f"{missing_time} frame(s) have no recording_start_utc: time-adjacency leakage cannot be audited for them")
    # time adjacency: different recordings of the same camera that end/start close together but sit in different splits
    spans = {}
    for rec in frames:
        if rec.get("recording_start_utc") and rec.get("split") and rec.get("frame_time_s") is not None:
            t0 = datetime.fromisoformat(rec["recording_start_utc"].replace("Z", "+00:00")).timestamp() + float(rec["frame_time_s"])
            key = (rec["camera_id"], rec["recording_id"], rec["split"])
            lo, hi = spans.get(key, (t0, t0))
            spans[key] = (min(lo, t0), max(hi, t0))
    for (cam_a, rec_a, sp_a), (lo_a, hi_a) in spans.items():
        for (cam_b, rec_b, sp_b), (lo_b, hi_b) in spans.items():
            if cam_a == cam_b and rec_a < rec_b and sp_a != sp_b and max(lo_a, lo_b) - min(hi_a, hi_b) < near_time_s:
                warnings.append(f"ADJACENT: {cam_a}/{rec_a} ({sp_a}) and {cam_a}/{rec_b} ({sp_b}) are within {near_time_s:.0f}s; "
                                "same scene, same vehicles may recur across the split boundary")
    return {"errors": errors, "warnings": warnings, "counts": counts}


# ---------------------------------------------------------------- group-wise splitting

def assign_splits(frames: list, fractions: dict, seed: int, group_by: str = "recording") -> dict:
    """-> {frame_id: split}. Whole groups go to one split. Raises if a requested split would receive no group."""
    keys = [group_key(r, group_by) for r in frames]
    uniq = sorted(set(keys))
    labels = np.array([uniq.index(k) for k in keys])
    if len(uniq) < sum(1 for v in fractions.values() if v > 0):
        raise ValueError(f"only {len(uniq)} {group_by} group(s) for {sum(1 for v in fractions.values() if v > 0)} non-empty splits")
    assigned = assign_groups_to_splits(labels, {k: v for k, v in fractions.items() if v > 0}, seed)
    empty = set(k for k, v in fractions.items() if v > 0) - set(assigned.tolist())
    if empty:
        raise ValueError(f"split(s) {sorted(empty)} received no group; supply more recordings or change the fractions")
    return {r["frame_id"]: str(s) for r, s in zip(frames, assigned)}


# ---------------------------------------------------------------- scoring on labelled local frames

def load_frame_gt(root, rec: dict):
    """-> (gt_cls, gt_xyxy, ignore_xyxy, unsupported_xyxy), pixel coordinates.
    gt = classes 0..4. Ignore regions (difficult / unclassifiable) AND unsupported-class boxes both act as don't-care for
    per-class precision; unsupported boxes are additionally returned so their class-agnostic recall can be reported."""
    root, w, h = Path(root), rec["width"], rec["height"]
    rows, _ = parse_yolo_rows((root / "labels" / f"{rec['frame_id']}.txt").read_text(encoding="utf-8"))
    ign_f = root / "ignore" / f"{rec['frame_id']}.txt"
    irows, _ = parse_yolo_rows(ign_f.read_text(encoding="utf-8")) if ign_f.is_file() else ([], [])
    sup = [r for r in rows if r[0] < len(SHARED_CLASSES)]
    uns = [r for r in rows if r[0] >= len(SHARED_CLASSES)]
    return (np.array([r[0] for r in sup], dtype=int), rows_to_xyxy(sup, w, h),
            np.concatenate([rows_to_xyxy(irows, w, h), rows_to_xyxy(uns, w, h)]), rows_to_xyxy(uns, w, h))


def agnostic_counts(r: ImageResult, conf_thr: float) -> dict:
    """Class-agnostic detection counts at IoU 0.5 for ONE frame (supported-class GT only), predictions >= conf_thr.
    tp/fn: ground truth found/missed. fp: predictions matching no GT and not inside a don't-care region."""
    keep = r.pred_conf >= conf_thr
    px = r.pred_xyxy[keep]
    gx = r.gt_xyxy
    matched_p, tp = set(), 0
    if len(px) and len(gx):
        iou = box_iou_np(gx, px)
        used_g = set()
        for _, gi, pi in sorted(((iou[g, p], g, p) for g, p in zip(*np.nonzero(iou >= 0.5))), reverse=True):
            if gi in used_g or pi in matched_p:
                continue
            used_g.add(gi)
            matched_p.add(pi)
            tp += 1
    fp = 0
    ign = r.ignore_xyxy
    for pi in range(len(px)):
        if pi in matched_p:
            continue
        if len(ign) and box_iou_np(px[pi:pi + 1], ign).max() >= 0.5:
            continue
        fp += 1
    return {"tp": tp, "fn": len(gx) - tp, "fp": fp, "n_gt": len(gx), "n_pred": int(keep.sum())}


def unsupported_recall(r: ImageResult, unsupported_xyxy: np.ndarray, conf_thr: float) -> dict:
    """Class-agnostic: did the detector put ANY box (any shared class) on an auto-rickshaw? Reported separately."""
    keep = r.pred_conf >= conf_thr
    n = len(unsupported_xyxy)
    if n == 0:
        return {"found": 0, "total": 0}
    if not keep.any():
        return {"found": 0, "total": n}
    iou = box_iou_np(unsupported_xyxy, r.pred_xyxy[keep])
    return {"found": int((iou.max(1) >= 0.5).sum()), "total": n}


def sum_counts(counts: list) -> dict:
    out = {k: sum(c[k] for c in counts) for k in ("tp", "fn", "fp", "n_gt", "n_pred")}
    out["recall"] = out["tp"] / out["n_gt"] if out["n_gt"] else None
    out["precision"] = out["tp"] / (out["tp"] + out["fp"]) if out["tp"] + out["fp"] else None
    out["fp_per_frame"] = out["fp"] / len(counts) if counts else None
    return out


def group_bootstrap(per_frame: dict, groups: list, stat, n_boot: int = 200, seed: int = 0) -> dict:
    """Resample GROUPS (recordings) with replacement. `per_frame`: {model: [x_i]} aligned with `groups` (group id per
    frame); `stat(list_of_x) -> float`. -> per-model mean + 95% percentile CI and the paired difference for each pair."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    idx_by_group = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}
    draws = {m: [] for m in per_frame}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = [i for k in pick for i in idx_by_group[uniq[k]]]
        for m, xs in per_frame.items():
            draws[m].append(stat([xs[i] for i in idx]))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    names = list(per_frame)
    out = {"n_groups": len(uniq), "n_boot": n_boot, "seed": seed,
           "models": {m: {"mean": float(np.nanmean(d)), "ci95": ci(d)} for m, d in draws.items()}, "paired_difference": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.array(draws[b]) - np.array(draws[a])
            out["paired_difference"][f"{b} - {a}"] = {"mean": float(np.nanmean(diff)), "ci95": ci(diff)}
    return out


def class_aware_map(results: list) -> float:
    return score(results)["map50"]
