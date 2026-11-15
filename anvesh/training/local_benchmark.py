"""Benchmark detector checkpoints on a labelled local-CCTV dataset, and the detector-promotion gate.

`evaluate` is model-agnostic: it takes `predictors = {name: fn(list_of_image_paths) -> [(xyxy, conf, shared_cls_ids)]}`
so tests can inject fakes and the CLI (scripts/benchmark_local_cctv.py) injects Ultralytics models. Predictions must
already be mapped to the shared class ids (detector_eval.model_to_shared) and be produced at conf >= 0.001.

Honesty rules enforced here: unlabelled frames are never scored; a class with fewer than `MIN_CLASS_GT` ground-truth boxes
is reported as "insufficient", not as a number; an empty split yields NotEvaluable, never zeros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anvesh.training import local_cctv as lc
from anvesh.training.detector_eval import SHARED_CLASSES, ImageResult, operating_point, score

MIN_CLASS_GT = 30
CONF_GRID = (0.10, 0.25, 0.40, 0.50)

# Promotion criteria. The numeric values are PROPOSALS (docs/ANVESH_LOCAL_CCTV_BENCHMARK.md section 6): they must be
# confirmed by the operator before the gate is treated as binding. Changing them is a deliberate, reviewed edit.
PROMOTION_CRITERIA = {
    "min_cameras": 3, "min_recordings": 8, "min_labelled_frames": 400, "min_boxes": 1500, "min_lighting_conditions": 2,
    "op_conf": 0.25,
    "recall_floor_point": 0.85, "recall_floor_ci_lower": 0.80,
    "recall_noninferiority_margin": -0.02,
    "fp_per_frame_ceiling": 0.5, "fp_per_frame_worse_than_baseline_max": 0.10,
    "per_camera_recall_regression_max": -0.05, "per_camera_min_gt": 100,
    "p95_latency_ms_ceiling": 33.0, "p95_latency_ratio_to_baseline_max": 1.25,
    "cityflow_recall_noninferiority_margin": -0.02,
    "downstream_idf1_mota_regression_max": -0.01,
}
LIGHTING_TAGS = ("day", "night", "dusk", "dawn")


class NotEvaluable(Exception):
    pass


def labelled_frames(manifest: dict, split: str | None) -> list:
    fr = [r for r in manifest["frames"] if r.get("labelled") and (split is None or r.get("split") == split)]
    if not fr:
        raise NotEvaluable(f"no labelled frames in split {split!r}: nothing to benchmark (unlabelled frames are never scored)")
    return fr


def dataset_coverage(frames: list, root) -> dict:
    per_class = {c: 0 for c in lc.LOCAL_CLASSES}
    for r in frames:
        rows, _ = lc.parse_yolo_rows((Path(root) / "labels" / f"{r['frame_id']}.txt").read_text(encoding="utf-8"))
        for row in rows:
            per_class[lc.LOCAL_CLASSES[row[0]]] += 1
    tags = {t for r in frames for t in r.get("conditions", [])}
    return {"frames": len(frames), "cameras": sorted({r["camera_id"] for r in frames}),
            "recordings": len({(r["camera_id"], r["recording_id"]) for r in frames}),
            "boxes": sum(per_class.values()), "per_class_boxes": per_class,
            "conditions": sorted(tags), "lighting_conditions": sorted(t for t in tags if t in LIGHTING_TAGS)}


def evaluate(root, frames: list, predictors: dict, op_conf: float = 0.25, n_boot: int = 200, seed: int = 0, batch: int = 8) -> dict:
    root = Path(root)
    gts = [lc.load_frame_gt(root, r) for r in frames]
    groups = [f"{r['camera_id']}/{r['recording_id']}" for r in frames]
    results = {}
    for name, fn in predictors.items():
        res = []
        for s in range(0, len(frames), batch):
            chunk = frames[s:s + batch]
            preds = fn([root / r["image"] for r in chunk])
            for (gc, gx, ign, _), (px, pc, pk) in zip(gts[s:s + batch], preds):
                res.append(ImageResult(px, pc, pk, gx, gc, ign))
        results[name] = res

    out = {"coverage": dataset_coverage(frames, root), "op_conf": op_conf, "models": {}}
    cams = sorted({r["camera_id"] for r in frames})
    for name, res in results.items():
        sc = score(res)
        per_class = {}
        n_gt = np.bincount(np.concatenate([r.gt_cls for r in res]) if res else np.zeros(0, int), minlength=len(SHARED_CLASSES))
        for cls_i, cls in enumerate(SHARED_CLASSES):
            # score() returns no per-class rows at all when a model emits nothing; rebuild so ground truth is never silently dropped
            v = sc["per_class"].get(cls)
            if not n_gt[cls_i]:
                v = {"n_gt": 0}
            elif v is None or "ap50" not in v:
                v = {"n_gt": int(n_gt[cls_i]), "precision": None, "recall": 0.0, "f1": None, "ap50": 0.0, "ap50_95": 0.0}
            if v["n_gt"] < MIN_CLASS_GT:
                per_class[cls] = {"n_gt": v["n_gt"], "status": f"insufficient (< {MIN_CLASS_GT} ground-truth boxes): no number reported"}
            else:
                per_class[cls] = v
        agn = {c: lc.sum_counts([lc.agnostic_counts(r, c) for r in res]) for c in CONF_GRID}
        by_cam = {}
        for cam in cams:
            sel = [r for r, f in zip(res, frames) if f["camera_id"] == cam]
            by_cam[cam] = lc.sum_counts([lc.agnostic_counts(r, op_conf) for r in sel]) | {"frames": len(sel)}
        uns = [lc.unsupported_recall(r, g[3], op_conf) for r, g in zip(res, gts)]
        out["models"][name] = {
            "map50_conf0.001": sc["map50"], "map50_95_conf0.001": sc["map50_95"], "per_class": per_class,
            "class_aware_operating_points": {str(c): operating_point(res, c) for c in CONF_GRID},
            "class_agnostic_by_conf": {str(c): v for c, v in agn.items()},
            "class_agnostic_by_camera_at_op_conf": by_cam,
            "unsupported_class_recall_class_agnostic": {
                "classes": list(lc.UNSUPPORTED_CLASSES), "found": sum(u["found"] for u in uns), "total": sum(u["total"] for u in uns)},
        }
    # group-level uncertainty (recordings are resampled, not frames)
    cnt = {n: [lc.agnostic_counts(r, op_conf) for r in res] for n, res in results.items()}
    rec = lambda xs: (sum(x["tp"] for x in xs) / max(1, sum(x["n_gt"] for x in xs)))  # noqa: E731
    fpf = lambda xs: sum(x["fp"] for x in xs) / max(1, len(xs))  # noqa: E731
    out["bootstrap_by_recording"] = {
        "recall_class_agnostic": lc.group_bootstrap(cnt, groups, rec, n_boot, seed),
        "fp_per_frame": lc.group_bootstrap(cnt, groups, fpf, n_boot, seed),
        "map50": lc.group_bootstrap(results, groups, lc.class_aware_map, max(20, n_boot // 4), seed)}
    out["_results"] = results  # in-memory only; the CLI strips it before writing JSON
    return out


def _lighting_ok(cov, need):
    return len(cov["lighting_conditions"]) >= need


def evaluate_gate(report: dict | None, candidate: str, baseline: str, latency: dict | None = None,
                  cityflow: dict | None = None, downstream: dict | None = None, criteria: dict | None = None) -> dict:
    """-> {"verdict": PROMOTE | DO_NOT_PROMOTE | INCONCLUSIVE, "criteria": [{id, status, detail}]}.
    status: pass | fail | not_evaluable. PROMOTE requires every criterion to pass; any fail -> DO_NOT_PROMOTE;
    otherwise (some not_evaluable, none failed) -> INCONCLUSIVE. The gate NEVER changes the app default."""
    c = {**PROMOTION_CRITERIA, **(criteria or {})}
    rows = []

    def add(cid, status, detail):
        rows.append({"id": cid, "status": status, "detail": detail})

    if report is None:
        for cid in ("data_sufficiency", "recall_floor", "recall_noninferiority", "fp_per_frame", "map50_superiority", "per_camera_regression"):
            add(cid, "not_evaluable", "no labelled local benchmark has been run")
    elif report.get("split") != "test":
        add("held_out_test_split", "not_evaluable",
            f"report was computed on split {report.get('split')!r}; promotion needs the locked, group-disjoint 'test' split (a run on all/train/val/eval frames may have informed settings or training)")
        for cid in ("data_sufficiency", "recall_floor", "recall_noninferiority", "fp_per_frame", "map50_superiority", "per_camera_regression"):
            add(cid, "not_evaluable", "report is not from the held-out test split")
    else:
        cov, m, b = report["coverage"], report["models"][candidate], report["models"][baseline]
        op = str(c["op_conf"])
        bs = report["bootstrap_by_recording"]
        problems = []
        if len(cov["cameras"]) < c["min_cameras"]:
            problems.append(f"cameras {len(cov['cameras'])} < {c['min_cameras']}")
        if cov["recordings"] < c["min_recordings"]:
            problems.append(f"recordings {cov['recordings']} < {c['min_recordings']}")
        if cov["frames"] < c["min_labelled_frames"]:
            problems.append(f"labelled frames {cov['frames']} < {c['min_labelled_frames']}")
        if cov["boxes"] < c["min_boxes"]:
            problems.append(f"boxes {cov['boxes']} < {c['min_boxes']}")
        if not _lighting_ok(cov, c["min_lighting_conditions"]):
            problems.append(f"lighting conditions {cov['lighting_conditions']} < {c['min_lighting_conditions']} distinct")
        add("data_sufficiency", "fail" if problems else "pass", "; ".join(problems) or "coverage meets the minimums")
        if problems:
            # Without enough data every statistical criterion below is advisory only: report it, do not let it pass.
            for cid in ("recall_floor", "recall_noninferiority", "fp_per_frame", "map50_superiority", "per_camera_regression"):
                add(cid, "not_evaluable", "benchmark data below the minimum coverage; numbers would not support a promotion claim")
        else:
            rc = bs["recall_class_agnostic"]
            point, lo = m["class_agnostic_by_conf"][op]["recall"], rc["models"][candidate]["ci95"][0]
            add("recall_floor", "pass" if point >= c["recall_floor_point"] and lo >= c["recall_floor_ci_lower"] else "fail",
                f"candidate recall {point:.3f}, 95% CI lower {lo:.3f} (need >= {c['recall_floor_point']} and >= {c['recall_floor_ci_lower']})")
            d = rc["paired_difference"].get(f"{candidate} - {baseline}") or rc["paired_difference"].get(f"{baseline} - {candidate}")
            sign = 1 if f"{candidate} - {baseline}" in rc["paired_difference"] else -1
            dlo = d["ci95"][0] if sign == 1 else -d["ci95"][1]
            add("recall_noninferiority", "pass" if dlo >= c["recall_noninferiority_margin"] else "fail",
                f"paired recall difference CI lower {dlo:.3f} (need >= {c['recall_noninferiority_margin']})")
            fp = bs["fp_per_frame"]
            fpc = m["class_agnostic_by_conf"][op]["fp_per_frame"]
            fd = fp["paired_difference"].get(f"{candidate} - {baseline}")
            fup = fd["ci95"][1] if fd else -fp["paired_difference"][f"{baseline} - {candidate}"]["ci95"][0]
            add("fp_per_frame", "pass" if fpc <= c["fp_per_frame_ceiling"] and fup <= c["fp_per_frame_worse_than_baseline_max"] else "fail",
                f"candidate {fpc:.3f} FP/frame (ceiling {c['fp_per_frame_ceiling']}); difference CI upper {fup:.3f} (max {c['fp_per_frame_worse_than_baseline_max']})")
            mp = bs["map50"]["paired_difference"]
            mk = f"{candidate} - {baseline}"
            mlo = mp[mk]["ci95"][0] if mk in mp else -mp[f"{baseline} - {candidate}"]["ci95"][1]
            add("map50_superiority", "pass" if mlo > 0 else "fail", f"class-aware mAP50 paired difference CI lower {mlo:.3f} (need > 0)")
            bad = []
            for cam, v in m["class_agnostic_by_camera_at_op_conf"].items():
                bv = b["class_agnostic_by_camera_at_op_conf"][cam]
                if v["n_gt"] >= c["per_camera_min_gt"] and v["recall"] is not None and bv["recall"] is not None \
                        and v["recall"] - bv["recall"] < c["per_camera_recall_regression_max"]:
                    bad.append(f"{cam}: {v['recall']:.3f} vs {bv['recall']:.3f}")
            add("per_camera_regression", "fail" if bad else "pass", "; ".join(bad) or f"no camera (>= {c['per_camera_min_gt']} GT) regresses by more than {abs(c['per_camera_recall_regression_max'])}")
    if latency and candidate in latency and baseline in latency:
        p, pb = latency[candidate]["p95_ms"], latency[baseline]["p95_ms"]
        ok = p <= c["p95_latency_ms_ceiling"] and p <= pb * c["p95_latency_ratio_to_baseline_max"]
        add("latency", "pass" if ok else "fail", f"p95 {p:.1f} ms vs baseline {pb:.1f} ms (ceiling {c['p95_latency_ms_ceiling']} ms, ratio <= {c['p95_latency_ratio_to_baseline_max']})")
    else:
        add("latency", "not_evaluable", "no batch-1 latency measurement for both checkpoints")
    if cityflow and candidate in cityflow.get("pooled", {}) and baseline in cityflow["pooled"]:
        diff = cityflow["pooled"][candidate]["recall_class_agnostic"] - cityflow["pooled"][baseline]["recall_class_agnostic"]
        add("cityflow_noninferiority", "pass" if diff >= c["cityflow_recall_noninferiority_margin"] else "fail",
            f"pooled CityFlow recall difference {diff:+.3f} (point estimate; no CI available; need >= {c['cityflow_recall_noninferiority_margin']})")
    else:
        add("cityflow_noninferiority", "not_evaluable", "no CityFlow comparison supplied")
    if downstream:
        worst = min(downstream.get("idf1_diff", 0.0), downstream.get("mota_diff", 0.0))
        add("downstream_tracking", "pass" if worst >= c["downstream_idf1_mota_regression_max"] else "fail",
            f"worst of IDF1/MOTA difference {worst:+.3f} (need >= {c['downstream_idf1_mota_regression_max']})")
    else:
        add("downstream_tracking", "not_evaluable", "no downstream tracking/speed/incident comparison supplied (run eval with ANVESH_FINETUNED_MODEL set)")
    statuses = {r["status"] for r in rows}
    verdict = "DO_NOT_PROMOTE" if "fail" in statuses else ("INCONCLUSIVE" if "not_evaluable" in statuses else "PROMOTE")
    return {"candidate": candidate, "baseline": baseline, "verdict": verdict, "criteria": rows, "thresholds": c,
            "note": "The gate only reports. Changing the app default model is a separate, explicit, human-approved step."}
