"""Independent cross-check of `anvesh.evaluation.mot_metrics` against the
reference `motmetrics` package on REAL CityFlow c006 data.

`motmetrics` (and its pandas/scipy dependencies) is deliberately NOT a project
dependency. Install it into a throwaway directory and point at it:

    pip install --target <dir> motmetrics scipy pandas
    python scripts/crosscheck_metrics_motmetrics.py --extra-path <dir>

Compared on identical GT / prediction tables (IoU >= 0.5, ROI-filtered):
the shipped `mtsc_tc_yolo3` baseline and ANVESH's raw ByteTrack output.
IDF1/IDP/IDR and the identity-count metrics are expected to agree closely;
ID switches / fragmentations can differ slightly because the two libraries
break matching ties differently (both are CLEAR-MOT style), which is reported
rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--extra-path", required=True)
ap.add_argument("--camera", default="c006")
args = ap.parse_args()
sys.path.insert(0, args.extra_path)

import numpy as np  # noqa: E402

if not hasattr(np, "asfarray"):  # motmetrics 1.4.0 predates NumPy 2 (np.asfarray was removed)
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402
from _stream_common import OUT, REPO_ROOT  # noqa: E402
from eval_cityflow import load_cache, tables_from_cache  # noqa: E402

from anvesh.evaluation import cityflow_gt as gt_io  # noqa: E402
from anvesh.evaluation.mot_metrics import evaluate_tracking  # noqa: E402


def motmetrics_scores(gt: dict, pred: dict, frames) -> dict:
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        dist = mm.distances.iou_matrix([r[1:5] for r in g], [r[1:5] for r in p], max_iou=0.5)
        acc.update([r[0] for r in g], [r[0] for r in p], dist, frameid=f)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "idp", "idr", "num_switches", "num_fragmentations", "num_matches",
                                 "num_misses", "num_false_positives"], name="x").iloc[0]
    return {k: float(v) for k, v in s.items()}


def main() -> int:
    cam = gt_io.camera_dir("S02", args.camera)
    gt = gt_io.load_mot_file(cam / "gt" / "gt.txt")
    roi = gt_io.load_roi_mask(cam / "roi.jpg")
    meta, frames = load_cache(REPO_ROOT / "outputs" / "cityflow" / f"s02_{args.camera}_perception_cache.jsonl.gz")
    n = len(frames)
    eval_frames = range(1, n + 1)
    _, anvesh_trk = tables_from_cache(frames, 1, roi)
    baseline = gt_io.filter_table_by_roi(gt_io.load_mot_file(cam / "mtsc" / "mtsc_tc_yolo3.txt"), roi)

    report = {}
    for name, table in (("mtsc_tc_yolo3 (shipped baseline)", baseline), ("ANVESH raw ByteTrack ids", anvesh_trk)):
        mine = evaluate_tracking(gt, table, frames=eval_frames)
        ref = motmetrics_scores(gt, table, eval_frames)
        report[name] = {
            "ours": {"idf1": mine.idf1, "idp": mine.idp, "idr": mine.idr, "id_switches": mine.id_switches,
                     "fragmentations": mine.fragmentations, "matches": mine.frame_tp, "misses": mine.frame_fn,
                     "false_positives": mine.frame_fp},
            "motmetrics": ref,
        }
        print(f"{name}")
        print(f"   ours       IDF1={mine.idf1:.4f} IDP={mine.idp:.4f} IDR={mine.idr:.4f} IDSW={mine.id_switches} FM={mine.fragmentations} "
              f"TP={mine.frame_tp} FN={mine.frame_fn} FP={mine.frame_fp}")
        print(f"   motmetrics IDF1={ref['idf1']:.4f} IDP={ref['idp']:.4f} IDR={ref['idr']:.4f} IDSW={ref['num_switches']:.0f} "
              f"FM={ref['num_fragmentations']:.0f} TP={ref['num_matches']:.0f} FN={ref['num_misses']:.0f} FP={ref['num_false_positives']:.0f}")
    (OUT / "metrics_crosscheck_motmetrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
