"""Pre-flight checks for a training run: do not start a long, memory-hungry job on a machine that cannot finish it, and do not
claim to resume a run that has nothing to resume.

Nothing here starts training or downloads anything. The functions are pure (injectable inputs) so they are unit-tested without a GPU.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

GIB = 1024 ** 3


def resource_snapshot(paths=(".",)) -> dict:
    """Free RAM, free disk for each path's drive, and free VRAM when CUDA is available. Never raises on a missing GPU/psutil."""
    snap = {"ram_total_gb": None, "ram_free_gb": None, "disk_free_gb": {}, "gpu": None}
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["ram_total_gb"], snap["ram_free_gb"] = round(vm.total / GIB, 2), round(vm.available / GIB, 2)
    except Exception:  # noqa: BLE001
        pass
    for p in paths:
        try:
            snap["disk_free_gb"][str(p)] = round(shutil.disk_usage(p).free / GIB, 2)
        except OSError:
            snap["disk_free_gb"][str(p)] = None
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snap["gpu"] = {"name": torch.cuda.get_device_name(0), "vram_free_gb": round(free / GIB, 2), "vram_total_gb": round(total / GIB, 2)}
    except Exception:  # noqa: BLE001
        pass
    return snap


def evaluate(snapshot: dict, min_free_ram_gb: float = 3.0, min_free_disk_gb: float = 5.0, min_free_vram_gb: float = 2.0,
             need_gpu: bool = True) -> dict:
    """{ok, failures, warnings}. An unknown quantity is a failure, never a pass: if free RAM cannot be measured we do not guess."""
    failures, warnings = [], []
    ram = snapshot.get("ram_free_gb")
    if ram is None:
        failures.append("free RAM could not be measured")
    elif ram < min_free_ram_gb:
        failures.append(f"free RAM {ram} GB is below the {min_free_ram_gb} GB minimum (dataloader workers + caches need it; a low-memory kill mid-epoch wastes the run)")
    for path, free in snapshot.get("disk_free_gb", {}).items():
        if free is None:
            failures.append(f"free disk space on {path} could not be measured")
        elif free < min_free_disk_gb:
            failures.append(f"free disk on {path} is {free} GB, below the {min_free_disk_gb} GB minimum (checkpoints, plots and caches)")
        elif free < 2 * min_free_disk_gb:
            warnings.append(f"free disk on {path} is only {free} GB")
    gpu = snapshot.get("gpu")
    if need_gpu and gpu is None:
        failures.append("no CUDA GPU is available (CPU training of a detector is not practical here)")
    elif gpu is not None and gpu["vram_free_gb"] < min_free_vram_gb:
        failures.append(f"free VRAM {gpu['vram_free_gb']} GB is below the {min_free_vram_gb} GB minimum")
    return {"ok": not failures, "failures": failures, "warnings": warnings}


def iterations_per_epoch(n_train_images: int, batch: int) -> int:
    if n_train_images < 0 or batch < 1:
        raise ValueError("n_train_images must be >= 0 and batch >= 1")
    return math.ceil(n_train_images / batch)


def estimate_runtime(n_train_images: int, batch: int, epochs: int, seconds_per_iteration: float = None) -> dict:
    """Runtime estimate. With no MEASURED seconds_per_iteration there is no estimate: the honest answer is 'measure it in epoch 1'."""
    its = iterations_per_epoch(n_train_images, batch)
    out = {"iterations_per_epoch": its, "epochs": epochs, "total_iterations": its * epochs}
    if seconds_per_iteration is None or not seconds_per_iteration > 0:
        out.update(estimated_hours=None, note="no measured throughput supplied: runtime is not predicted. Watch it/s in epoch 1 and extrapolate.")
    else:
        out.update(estimated_hours=round(its * epochs * seconds_per_iteration / 3600.0, 2),
                   note=f"extrapolated from a supplied {seconds_per_iteration:.3f} s/iteration; throughput on this laptop varies several-fold between runs")
    return out


def resolve_resume_checkpoint(runs_dir, run_name: str) -> Path:
    """The `last.pt` of an interrupted run, or SystemExit with a precise reason. Resume needs Ultralytics' own last.pt (it carries the
    optimizer and epoch state); a run that never finished an epoch has none, and best.pt is not a substitute."""
    if not run_name or "/" in run_name or "\\" in run_name or run_name in (".", ".."):
        raise SystemExit(f"invalid run name {run_name!r}")
    run = Path(runs_dir) / run_name
    if not run.is_dir():
        raise SystemExit(f"nothing to resume: no run directory {run}")
    last = run / "weights" / "last.pt"
    if not last.is_file():
        present = sorted(p.name for p in (run / "weights").glob("*")) if (run / "weights").is_dir() else []
        raise SystemExit(f"nothing to resume: {last} does not exist (weights present: {present or 'none'}). A run that never completed an "
                         "epoch writes no checkpoint; start a new run under a new name instead.")
    return last
