"""One detection-metric implementation for comparing ANY checkpoints on the same labelled images.

Why this exists: the COCO-pretrained model's class indices (bicycle=1, car=2, motorcycle=3, bus=5,
truck=7) do not match a fine-tuned 5-class model's (0..4), so `model.val(data=<our yaml>)` cannot
score the baseline. Both models are therefore run through the same code path: predictions are mapped
BY NAME into one shared taxonomy, matched to ground truth with the same procedure Ultralytics uses
(`DetectionValidator.match_predictions`, greedy one-to-one by IoU, 10 thresholds 0.50:0.95), and
scored with Ultralytics' own `ap_per_class` (101-point interpolated AP, as in its `val`). No new
dependency.

Reported numbers:
  * P / R / F1 "at max-F1" and mAP50 / mAP50-95 -- Ultralytics' convention, from conf >= 0.001 predictions.
  * P / R at a FIXED operating confidence (ANVESH runs at 0.25) -- recomputed by thresholding BEFORE
    matching, so low-confidence boxes cannot steal ground-truth matches.
  * Optional image-level bootstrap CIs, including a paired CI for the difference between two models.

Don't-care regions: MIO-TCD also labels classes that are not in ANVESH's taxonomy (pedestrian, non-motorized
vehicle, and "motorized vehicle" = a vehicle too occluded/oblique to classify). A prediction that overlaps such a
box (IoU >= 0.5) and did NOT match a real ground-truth box is neither a hit nor a false alarm (KITTI-style
"DontCare"); it is dropped for every model alike. Pass `use_ignore=False` to score without this.
"""

from __future__ import annotations

import numpy as np

SHARED_CLASSES = ("bicycle", "car", "motorcycle", "bus", "truck")  # index = shared class id
IOUV = np.linspace(0.5, 0.95, 10)


def model_to_shared(names) -> dict:
    """model class id -> shared class id, matched by NAME. Classes outside the shared set are dropped."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    shared = {n: i for i, n in enumerate(SHARED_CLASSES)}
    return {int(k): shared[str(v).strip().lower()] for k, v in items if str(v).strip().lower() in shared}


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)


def match_image(pred_xyxy, pred_cls, gt_xyxy, gt_cls) -> np.ndarray:
    """-> (n_pred, 10) bool: is each prediction a true positive at each IoU threshold.
    Greedy one-to-one by descending IoU, class-aware (same as Ultralytics' non-scipy matcher)."""
    correct = np.zeros((len(pred_xyxy), len(IOUV)), dtype=bool)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return correct
    iou = box_iou_np(gt_xyxy, pred_xyxy) * (gt_cls[:, None] == pred_cls[None, :])
    for i, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T  # (k, 2) = (gt_idx, pred_idx)
        if len(m) == 0:
            continue
        if len(m) > 1:
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1], i] = True
    return correct


class ImageResult:
    """Raw predictions + ground truth for one image, already in shared class ids."""
    __slots__ = ("pred_xyxy", "pred_conf", "pred_cls", "gt_xyxy", "gt_cls", "ignore_xyxy")

    def __init__(self, pred_xyxy, pred_conf, pred_cls, gt_xyxy, gt_cls, ignore_xyxy=()):
        self.pred_xyxy = np.asarray(pred_xyxy, dtype=float).reshape(-1, 4)
        self.pred_conf = np.asarray(pred_conf, dtype=float).reshape(-1)
        self.pred_cls = np.asarray(pred_cls, dtype=int).reshape(-1)
        self.gt_xyxy = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
        self.gt_cls = np.asarray(gt_cls, dtype=int).reshape(-1)
        self.ignore_xyxy = np.asarray(ignore_xyxy, dtype=float).reshape(-1, 4)


def _drop_ignored(px, tp, ignore_xyxy):
    """Boolean keep-mask: drop unmatched predictions that overlap a don't-care box."""
    if len(px) == 0 or len(ignore_xyxy) == 0:
        return np.ones(len(px), dtype=bool)
    overlaps_ignore = box_iou_np(px, ignore_xyxy).max(1) >= 0.5
    return ~(overlaps_ignore & ~tp[:, 0])


def _stack(results, min_conf: float = 0.0, use_ignore: bool = True):
    tps, confs, pcls, tcls = [], [], [], []
    for r in results:
        keep = r.pred_conf >= min_conf
        px, pc, pk = r.pred_xyxy[keep], r.pred_conf[keep], r.pred_cls[keep]
        tp = match_image(px, pk, r.gt_xyxy, r.gt_cls)
        if use_ignore:
            k = _drop_ignored(px, tp, r.ignore_xyxy)
            tp, pc, pk = tp[k], pc[k], pk[k]
        tps.append(tp)
        confs.append(pc)
        pcls.append(pk)
        tcls.append(r.gt_cls)
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)  # noqa: E731
    return (cat(tps, (0, 10)).astype(bool), cat(confs, (0,)), cat(pcls, (0,)).astype(int), cat(tcls, (0,)).astype(int))


def score(results, names=SHARED_CLASSES, use_ignore: bool = True) -> dict:
    """Ultralytics-convention metrics over a list of `ImageResult` (use predictions at conf >= 0.001)."""
    from ultralytics.utils.metrics import ap_per_class

    tp, conf, pcls, tcls = _stack(results, use_ignore=use_ignore)
    n_gt = np.bincount(tcls, minlength=len(names))
    out = {"n_images": len(results), "n_gt": int(n_gt.sum()), "per_class": {}}
    if len(tp) == 0 or n_gt.sum() == 0:
        out.update(precision=0.0, recall=0.0, f1=0.0, map50=0.0, map50_95=0.0)
        return out
    _, _, p, r, f1, ap, uc, *_ = ap_per_class(tp, conf, pcls, tcls, names={i: n for i, n in enumerate(names)})
    out.update(precision=float(p.mean()), recall=float(r.mean()), f1=float(f1.mean()),
               map50=float(ap[:, 0].mean()), map50_95=float(ap.mean()))
    for j, c in enumerate(uc):
        out["per_class"][names[int(c)]] = {
            "n_gt": int(n_gt[c]), "precision": float(p[j]), "recall": float(r[j]), "f1": float(f1[j]),
            "ap50": float(ap[j, 0]), "ap50_95": float(ap[j].mean())}
    for c, n in enumerate(names):  # classes with zero ground truth are reported as such, never as 0.0 accuracy
        out["per_class"].setdefault(n, {"n_gt": int(n_gt[c]), "note": "no ground-truth instances in this split"})
    return out


def operating_point(results, conf_thr: float, names=SHARED_CLASSES, use_ignore: bool = True) -> dict:
    """P / R / F1 at IoU 0.5 for predictions with conf >= `conf_thr` (thresholded before matching)."""
    tp, conf, pcls, tcls = _stack(results, min_conf=conf_thr, use_ignore=use_ignore)
    n_gt = np.bincount(tcls, minlength=len(names))
    res = {"conf_threshold": conf_thr, "per_class": {}}
    TP = FP = FN = 0
    for c, n in enumerate(names):
        sel = pcls == c
        tpc = int(tp[sel, 0].sum())
        fpc = int(sel.sum()) - tpc
        fnc = int(n_gt[c]) - tpc
        TP, FP, FN = TP + tpc, FP + fpc, FN + fnc
        res["per_class"][n] = {"tp": tpc, "fp": fpc, "fn": fnc, **_prf(tpc, fpc, fnc)}
    res.update(tp=TP, fp=FP, fn=FN, **_prf(TP, FP, FN))  # micro-averaged over instances
    return res


def _prf(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1}


def confusion_matrix(results, conf_thr: float = 0.25, iou_thr: float = 0.45, names=SHARED_CLASSES) -> np.ndarray:
    """(nc+1, nc+1) matrix [pred, true]; last row = missed (false negative), last column = background (false positive).
    Class-AGNOSTIC IoU matching so confusions between classes are visible (same idea as Ultralytics' ConfusionMatrix)."""
    nc = len(names)
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for r in results:
        keep = r.pred_conf >= conf_thr
        px, pk = r.pred_xyxy[keep], r.pred_cls[keep]
        gx, gk = r.gt_xyxy, r.gt_cls
        iou = box_iou_np(gx, px)
        matched_g, matched_p = set(), set()
        if iou.size:
            cand = np.array(np.nonzero(iou >= iou_thr)).T
            if len(cand):
                cand = cand[iou[cand[:, 0], cand[:, 1]].argsort()[::-1]]
                for gi, pi in cand:
                    if gi in matched_g or pi in matched_p:
                        continue
                    matched_g.add(gi)
                    matched_p.add(pi)
                    cm[pk[pi], gk[gi]] += 1
        for gi in range(len(gx)):
            if gi not in matched_g:
                cm[nc, gk[gi]] += 1
        ign = box_iou_np(px, r.ignore_xyxy).max(1) >= 0.5 if len(px) and len(r.ignore_xyxy) else np.zeros(len(px), bool)
        for pi in range(len(px)):
            if pi not in matched_p and not ign[pi]:
                cm[pk[pi], nc] += 1
    return cm


def bootstrap(per_model: dict, n_boot: int = 100, seed: int = 0, metric: str = "map50") -> dict:
    """Image-level bootstrap. `per_model`: {name: [ImageResult,...]} over the SAME images in the SAME order.
    Returns each model's mean + 95% percentile CI for `metric` and, for every pair, the paired difference CI
    (same resamples for both models, so the CI reflects the model difference, not image-sampling noise)."""
    rng = np.random.default_rng(seed)
    names = list(per_model)
    n = len(per_model[names[0]])
    assert all(len(v) == n for v in per_model.values()), "models must be scored on the same images"
    draws = {m: [] for m in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in names:
            draws[m].append(score([per_model[m][i] for i in idx])[metric])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    out = {"metric": metric, "n_boot": n_boot, "seed": seed,
           "models": {m: {"mean": float(np.mean(d)), "ci95": ci(d)} for m, d in draws.items()}, "paired_difference": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.array(draws[b]) - np.array(draws[a])
            out["paired_difference"][f"{b} - {a}"] = {"mean": float(diff.mean()), "ci95": ci(diff),
                                                     "fraction_of_resamples_b_better": float((diff > 0).mean())}
    return out
