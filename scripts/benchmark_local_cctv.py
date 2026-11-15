"""Benchmark detector checkpoints on a labelled local-CCTV dataset (see scripts/local_cctv.py to build one).

    .venv\\Scripts\\python.exe scripts\\benchmark_local_cctv.py run --dataset datasets\\local_cctv\\v1 --split test ^
        --model pretrained=models\\yolov8n.pt --model finetuned=models\\finetuned\\anvesh_mio5_yolov8n_best.pt ^
        --out outputs\\local_benchmark\\v1_test.json --visuals 24
    .venv\\Scripts\\python.exe scripts\\benchmark_local_cctv.py latency --images <dir or video> ^
        --model pretrained=models\\yolov8n.pt --model finetuned=... --out outputs\\local_benchmark\\latency.json
    .venv\\Scripts\\python.exe scripts\\benchmark_local_cctv.py gate --report outputs\\local_benchmark\\v1_test.json ^
        --latency outputs\\local_benchmark\\latency.json --cityflow outputs\\training\\compare_cityflow.json

Same prediction settings for every model (conf 0.001 for AP, then thresholded for operating points; iou 0.7, max_det 300,
single-label NMS, FP16, imgsz 640) -- exactly the `compare_detectors.py` path, so numbers are comparable between models
but are conservative relative to Ultralytics' multi-label `val()` (docs/ANVESH_MODEL_TRAINING_REPORT.md section 4).
If the split has no labelled frames the command exits non-zero and writes NO results.

HELD-OUT DISCIPLINE: `--split test` and `--split eval` are locked per dataset after one run (like the MIO-TCD test); pass
--allow-test-rerun to override, and the rerun is recorded. The promotion gate accepts only a report computed on `test`.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.perception.weights import checkpoint_sha256, require_weights_file  # noqa: E402
from anvesh.training import local_benchmark as lb  # noqa: E402
from anvesh.training import local_cctv as lc  # noqa: E402
from anvesh.training.detector_eval import SHARED_CLASSES, model_to_shared  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def environment(imgsz: int, batch: int, half: bool) -> dict:
    import torch
    import ultralytics

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"utc": datetime.now(timezone.utc).isoformat(), "device": f"cuda:0 ({gpu})" if gpu else "cpu", "imgsz": imgsz,
            "batch": batch, "precision": "fp16" if half and gpu else "fp32", "python": platform.python_version(),
            "torch": torch.__version__, "ultralytics": ultralytics.__version__, "os": platform.platform()}


def ultralytics_predictor(path: Path, imgsz: int, half: bool, device):
    from ultralytics import YOLO

    model = YOLO(str(path))
    id_map = model_to_shared(model.names)

    def predict(paths):
        out = []
        for r in model.predict([str(p) for p in paths], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                               classes=sorted(id_map), device=device, half=half, verbose=False):
            b = r.boxes
            out.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), np.array([id_map[int(c)] for c in b.cls.tolist()], dtype=int)))
        return out
    return predict


def draw_errors(root: Path, frames, results, op_conf: float, model: str, outdir: Path, n: int) -> list:
    """Save annotated images of the frames with the most errors: GT green, matched pred blue, FP orange, FN red."""
    import cv2

    from anvesh.training.detector_eval import box_iou_np

    scored = []
    for rec, r in zip(frames, results):
        c = lc.agnostic_counts(r, op_conf)
        scored.append((c["fn"] + c["fp"], rec, r))
    scored.sort(key=lambda t: -t[0])
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for errs, rec, r in scored[:n]:
        if errs == 0:
            break
        img = cv2.imread(str(root / rec["image"]))
        keep = r.pred_conf >= op_conf
        px, pc, pk = r.pred_xyxy[keep], r.pred_conf[keep], r.pred_cls[keep]
        iou = box_iou_np(r.gt_xyxy, px) if len(px) and len(r.gt_xyxy) else np.zeros((len(r.gt_xyxy), len(px)))
        gt_hit = iou.max(1) >= 0.5 if iou.size else np.zeros(len(r.gt_xyxy), bool)
        p_hit = iou.max(0) >= 0.5 if iou.size else np.zeros(len(px), bool)
        for g, hit in zip(r.gt_xyxy, gt_hit):
            cv2.rectangle(img, tuple(int(v) for v in g[:2]), tuple(int(v) for v in g[2:]), (0, 200, 0) if hit else (0, 0, 255), 2 if not hit else 1)
        for b_, c_, k_, hit in zip(px, pc, pk, p_hit):
            ign = len(r.ignore_xyxy) and box_iou_np(b_[None], r.ignore_xyxy).max() >= 0.5
            if hit or ign:
                color = (255, 128, 0)
            else:
                color = (0, 140, 255)
                cv2.putText(img, f"FP {SHARED_CLASSES[int(k_)]} {c_:.2f}", (int(b_[0]), max(12, int(b_[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(img, tuple(int(v) for v in b_[:2]), tuple(int(v) for v in b_[2:]), color, 1)
        name = outdir / f"{model}__{rec['frame_id']}.jpg"
        cv2.imwrite(str(name), img)
        saved.append({"file": str(name), "frame_id": rec["frame_id"], "errors_fn_plus_fp": errs})
    return saved


def cmd_run(a) -> int:
    root = Path(a.dataset)
    manifest = lc.load_manifest(root)
    qa = lc.check_dataset(root, manifest)
    if qa["errors"]:
        print(f"dataset QA has {len(qa['errors'])} error(s); fix them first, e.g. {qa['errors'][:3]}", file=sys.stderr)
        return 2
    try:
        frames = lb.labelled_frames(manifest, None if a.split == "all" else a.split)
    except lb.NotEvaluable as e:
        print(f"NOT RUN: {e}", file=sys.stderr)
        return 3
    lock = root / f"{a.split}_evaluation.lock.json"
    locked = a.split in ("test", "eval")  # benchmark-only (`eval`) footage is held out too
    models = {}
    for spec in a.model:
        name, _, p = spec.partition("=")
        models[name] = require_weights_file(p)
    hashes = {n: checkpoint_sha256(p) for n, p in models.items()}
    if locked:
        if lock.is_file() and not a.allow_test_rerun:
            print(f"the {a.split} split of this dataset was already evaluated ({lock}); pass --allow-test-rerun to override", file=sys.stderr)
            return 4
        prev = json.loads(lock.read_text()) if lock.is_file() else {"runs": []}
        prev["runs"].append({"utc": datetime.now(timezone.utc).isoformat(), "checkpoints": hashes, "frames": len(frames)})
        lock.write_text(json.dumps(prev, indent=2))
    preds = {n: ultralytics_predictor(p, a.imgsz, True, 0) for n, p in models.items()}
    t0 = time.time()
    rep = lb.evaluate(root, frames, preds, op_conf=a.op_conf, n_boot=a.n_boot, batch=a.batch)
    results = rep.pop("_results")
    rep.update(dataset=str(root), split=a.split, labelled_frames=len(frames), wall_seconds=round(time.time() - t0, 1),
               environment=environment(a.imgsz, a.batch, True), checkpoints={n: {"path": str(p), "sha256": hashes[n]} for n, p in models.items()},
               qa_warnings=len(qa["warnings"]),
               scope="Results describe THIS labelled sample of THESE cameras/recordings only; they are not a general real-world accuracy claim.",
               timing_note="wall_seconds covers batched evaluation incl. disk reads; it is NOT a latency measurement (use the `latency` command).")
    if a.visuals:
        out_dir = Path(a.out).parent / (Path(a.out).stem + "_errors")
        rep["error_visuals"] = {n: draw_errors(root, frames, res, a.op_conf, n, out_dir, a.visuals) for n, res in results.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for n, m in rep["models"].items():
        ag = m["class_agnostic_by_conf"][str(a.op_conf)]
        print(f"{n:12s} mAP50={m['map50_conf0.001']:.3f} recall@{a.op_conf}={ag['recall']:.3f} FP/frame={ag['fp_per_frame']:.3f}")
    print("wrote", a.out)
    return 0


def _images(src: Path, limit: int):
    if src.is_dir():
        return [str(p) for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXT][:limit]
    import cv2

    cap, out, i = cv2.VideoCapture(str(src)), [], 0
    while len(out) < limit:
        ok, fr = cap.read()
        if not ok:
            break
        if i % 10 == 0:
            out.append(fr)
        i += 1
    return out


def cmd_latency(a) -> int:
    """Batch-1 end-to-end latency per frame: warm-up, CUDA-synchronised wall clock, median/p95, plus Ultralytics' own breakdown."""
    import torch
    from ultralytics import YOLO

    frames = _images(Path(a.images), a.n + a.warmup)
    if len(frames) < a.warmup + 10:
        print("NOT RUN: too few frames for a latency measurement", file=sys.stderr)
        return 3
    out = {"environment": environment(a.imgsz, 1, not a.fp32), "n_timed": len(frames) - a.warmup, "warmup": a.warmup,
           "source": str(a.images), "method": "model.predict(single frame) at conf 0.25; wall clock around the call with "
           "torch.cuda.synchronize() before/after; first `warmup` frames discarded; includes preprocess+inference+NMS, excludes decode of the source. "
           "Models are INTERLEAVED frame by frame (rotating order) so clock/power/thermal drift hits all models equally; "
           "single runs on this laptop GPU varied up to ~3.7x between invocations, so compare models only within one run.",
           "models": {}}
    loaded = {}
    for spec in a.model:
        name, _, p = spec.partition("=")
        path = require_weights_file(p)
        loaded[name] = (path, YOLO(str(path)))
    wall = {n: [] for n in loaded}
    parts = {n: [] for n in loaded}
    names = list(loaded)
    dev = 0 if torch.cuda.is_available() else "cpu"
    for i, fr in enumerate(frames):
        order = names[i % len(names):] + names[:i % len(names)]
        for n in order:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            r = loaded[n][1].predict(fr, imgsz=a.imgsz, conf=0.25, half=not a.fp32, device=dev, verbose=False)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= a.warmup:
                wall[n].append((time.perf_counter() - t) * 1000)
                parts[n].append(r.speed)
    for n, (path, _) in loaded.items():
        w = np.array(wall[n])
        out["models"][n] = {"sha256": checkpoint_sha256(path), "median_ms": float(np.median(w)), "p95_ms": float(np.percentile(w, 95)),
                            "mean_ms": float(w.mean()), "max_ms": float(w.max()),
                            "ultralytics_median_ms": {k: statistics.median(x[k] for x in parts[n]) for k in ("preprocess", "inference", "postprocess")}}
        print(n, {k: round(v, 2) for k, v in out["models"][n].items() if k.endswith("_ms") and not isinstance(v, dict)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", a.out)
    return 0


def cmd_gate(a) -> int:
    rep = json.loads(Path(a.report).read_text()) if a.report and Path(a.report).is_file() else None
    lat = json.loads(Path(a.latency).read_text())["models"] if a.latency and Path(a.latency).is_file() else None
    cf = json.loads(Path(a.cityflow).read_text()) if a.cityflow and Path(a.cityflow).is_file() else None
    ds = json.loads(Path(a.downstream).read_text()) if a.downstream and Path(a.downstream).is_file() else None
    g = lb.evaluate_gate(rep, a.candidate, a.baseline, lat, cf, ds)
    print(json.dumps({"verdict": g["verdict"], "criteria": [(c["id"], c["status"], c["detail"]) for c in g["criteria"]]}, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(g, indent=2), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--split", default="test", help="train|val|test|eval|all")
    r.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--op-conf", type=float, default=0.25, choices=lb.CONF_GRID)
    r.add_argument("--n-boot", type=int, default=200)
    r.add_argument("--visuals", type=int, default=0, help="save the N worst frames per model as annotated images")
    r.add_argument("--allow-test-rerun", action="store_true")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    l = sub.add_parser("latency")
    l.add_argument("--images", required=True, help="directory of images or a video file")
    l.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    l.add_argument("--imgsz", type=int, default=640)
    l.add_argument("--n", type=int, default=200)
    l.add_argument("--warmup", type=int, default=20)
    l.add_argument("--fp32", action="store_true", help="time FP32 instead of the default FP16")
    l.add_argument("--out", required=True)
    l.set_defaults(fn=cmd_latency)
    g = sub.add_parser("gate")
    g.add_argument("--report", default=None)
    g.add_argument("--latency", default=None)
    g.add_argument("--cityflow", default=None)
    g.add_argument("--downstream", default=None, help='JSON {"idf1_diff": x, "mota_diff": y} (candidate minus baseline)')
    g.add_argument("--candidate", default="finetuned")
    g.add_argument("--baseline", default="pretrained")
    g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_gate)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
