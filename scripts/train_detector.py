"""Fine-tune the COCO-pretrained YOLOv8n on the ANVESH 6-class MIO-TCD dataset (transfer learning).

Classes: bicycle, car, motorcycle, bus, truck + `other_vehicle` (MIO-TCD motorized_vehicle). ANVESH ignores
`other_vehicle` at inference, so five classes reach the application.

    .venv\\Scripts\\python.exe scripts\\train_detector.py --name anvesh_mio5_v1 --epochs 15 --imgsz 640 --batch 16

Every setting is an explicit argument with a recorded default (nothing is left to Ultralytics' "auto" choices),
the seed is fixed, and a `training_manifest.json` is written next to the run with the settings, hardware, package
versions, base-checkpoint SHA-256, dataset manifest, timing and the number of epochs ACTUALLY completed (read back
from Ultralytics' results.csv, not from the requested epoch count).

Before anything heavy starts a PRE-FLIGHT runs (free RAM / disk / VRAM, dataset size, iterations per epoch) and the run is
refused if the machine cannot sustain it; `--preflight-only` prints the report and stops, `--skip-preflight` is the explicit
override. `--resume RUN_NAME` continues an interrupted run from its own `weights/last.pt` and refuses (with the reason) when that
file does not exist - a run that never finished an epoch has nothing to resume.

The base checkpoint is the local `models/yolov8n.pt`; training never starts from random weights and never
downloads the base model. Ultralytics itself may fetch auxiliary files (a font, the AMP self-check model); those are
visible in the run log and are listed in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.perception.weights import checkpoint_sha256, require_weights_file  # noqa: E402
from anvesh.training import preflight  # noqa: E402

RUNS_DIR = REPO_ROOT / "models" / "finetuned" / "runs"


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def hardware() -> dict:
    import torch

    hw = {"platform": platform.platform(), "cpu": platform.processor(), "python": platform.python_version(),
          "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        hw.update(gpu=p.name, gpu_memory_gib=round(p.total_memory / 2**30, 2), cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version())
    return hw


def versions() -> dict:
    import numpy
    import torch
    import ultralytics

    return {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "numpy": numpy.__version__}


def tempfile_dir() -> str:
    import tempfile

    return tempfile.gettempdir()


def count_train_images(data_yaml: Path) -> int:
    """Number of images in the train split of a YOLO data.yaml (0 if it cannot be determined)."""
    try:
        import yaml

        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        root = Path(cfg.get("path") or data_yaml.parent)
        d = (root if root.is_absolute() else (data_yaml.parent / root)) / cfg["train"]
        return sum(1 for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    except Exception:  # noqa: BLE001
        return 0


def completed_epochs(run_dir: Path) -> int:
    f = run_dir / "results.csv"
    if not f.is_file():
        return 0
    with f.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO_ROOT / "datasets" / "mio_tcd" / "yolo5" / "data.yaml"))
    ap.add_argument("--base", default=str(REPO_ROOT / "models" / "yolov8n.pt"), help="pretrained checkpoint to fine-tune")
    ap.add_argument("--name", help="new run name (required unless --resume)")
    ap.add_argument("--epochs", type=int, help="required unless --resume (Ultralytics restores the original value)")
    ap.add_argument("--resume", metavar="RUN_NAME", help="continue the interrupted run RUN_NAME from its weights/last.pt")
    ap.add_argument("--preflight-only", action="store_true", help="print the resource/dataset report and exit (0 = ok, 2 = would be refused)")
    ap.add_argument("--skip-preflight", action="store_true", help="start even if the pre-flight fails (you accept the risk)")
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    ap.add_argument("--min-free-disk-gb", type=float, default=5.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train split to use (smoke tests)")
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--lrf", type=float, default=0.01, help="final LR = lr0 * lrf")
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5, help="early stopping: epochs without val fitness improvement")
    ap.add_argument("--close-mosaic", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=None, help="optional wall-clock cap in hours (Ultralytics `time`)")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args(argv)
    if not args.resume and (not args.name or args.epochs is None):
        ap.error("--name and --epochs are required unless --resume is given")
    resume_ckpt = preflight.resolve_resume_checkpoint(RUNS_DIR, args.resume) if args.resume else None
    if resume_ckpt is None and (RUNS_DIR / args.name).exists():
        raise SystemExit(f"run directory {RUNS_DIR / args.name} already exists; choose a new --name (existing runs are never overwritten) or use --resume")

    base = require_weights_file(args.base)
    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"dataset yaml not found: {data} (run scripts/prepare_mio_tcd.py first)")

    # PRE-FLIGHT: refuse to start a long job on a machine that cannot finish it.
    snap = preflight.resource_snapshot([RUNS_DIR if RUNS_DIR.exists() else REPO_ROOT])
    verdict = preflight.evaluate(snap, args.min_free_ram_gb, args.min_free_disk_gb)
    temp_free = preflight.resource_snapshot([tempfile_dir()])["disk_free_gb"].get(tempfile_dir())
    if temp_free is not None and temp_free < args.min_free_disk_gb:  # a warning, not a refusal: only small temp/cache files land there
        verdict["warnings"].append(f"the system temp drive has only {temp_free} GB free (pagefile / temp pressure can kill a long run)")
    n_train = count_train_images(data)
    plan = preflight.estimate_runtime(n_train, args.batch, args.epochs or 0)
    print("[preflight]", json.dumps({"resources": snap, "dataset_train_images": n_train, "plan": plan, **verdict}, indent=2))
    if args.preflight_only:
        return 0 if verdict["ok"] else 2
    if not verdict["ok"] and not args.skip_preflight:
        print("[preflight] REFUSED:", "; ".join(verdict["failures"]), "(use --skip-preflight to override)", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    settings = dict(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, fraction=args.fraction,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, patience=args.patience, close_mosaic=args.close_mosaic,
        workers=args.workers, seed=args.seed, deterministic=True, amp=not args.no_amp, cos_lr=False,
        cache=False, pretrained=True, project=str(RUNS_DIR), name=args.name, exist_ok=False, plots=True, device=0,
        val=True, save_period=-1, time=args.time)
    t0, started = time.time(), datetime.now(timezone.utc).isoformat()
    model = YOLO(str(resume_ckpt if resume_ckpt else base))
    status, error = "completed", None
    try:
        if resume_ckpt:
            model.train(resume=True)  # Ultralytics restores the optimizer, epoch counter and original arguments from last.pt
        else:
            model.train(**settings)
    except BaseException as exc:  # noqa: BLE001 - a failed run must still leave a manifest
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    trainer = getattr(model, "trainer", None)
    run_dir = Path(trainer.save_dir) if trainer is not None and getattr(trainer, "save_dir", None) else RUNS_DIR / (args.resume or args.name)
    manifest = {
        "status": status, "error": error, "started_utc": started,
        "elapsed_seconds": round(time.time() - t0, 1),
        "epochs_requested": args.epochs if args.epochs is not None else "restored from last.pt", "epochs_completed": completed_epochs(run_dir),
        "settings": settings, "base_checkpoint": {"path": str(base), "sha256": checkpoint_sha256(base)},
        "resumed_from": {"path": str(resume_ckpt), "sha256": checkpoint_sha256(resume_ckpt)} if resume_ckpt else None,
        "hardware": hardware(), "versions": versions(), "git_head": _git_head(),
        "command": "python scripts/train_detector.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "weights": {k: str(run_dir / "weights" / f"{k}.pt") for k in ("best", "last")},
    }
    dm = data.parent / "dataset_manifest.json"
    if dm.is_file():
        manifest["dataset_manifest"] = json.loads(dm.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath = run_dir / "training_manifest.json"
    if mpath.exists():  # keep the earlier attempt's record
        mpath = run_dir / f"training_manifest.{int(time.time())}.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[train] {status}: {manifest['epochs_completed']}/{args.epochs if args.epochs is not None else '?'} epochs in {manifest['elapsed_seconds']}s -> {run_dir}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
