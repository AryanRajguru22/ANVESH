"""The held-out-split one-run locks. These tests use temporary datasets only; the real MIO-TCD test split is never evaluated or modified here
(the last test only READS the real lock file)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from anvesh.training import local_benchmark as lb
from anvesh.training import local_cctv as lc

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts import benchmark_local_cctv as bench  # noqa: E402

from .test_local_cctv import build_dataset, perfect_predictor  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mio_tcd_style_test_split_is_refused_on_a_second_run_and_the_lock_is_untouched(tmp_path):
    cmp = _load("compare_detectors")
    ds = tmp_path / "yolo5"
    (ds / "images" / "test").mkdir(parents=True)
    (ds / "images" / "test" / "a.jpg").write_bytes(b"x")
    lock = ds / "test_evaluation.lock.json"
    lock.write_text(json.dumps({"runs": [{"utc": "2026-01-01T00:00:00+00:00", "checkpoints": {"m": "h"}, "n_images": 1, "limit": None}]}))
    before = lock.read_bytes()
    weights = tmp_path / "m.pt"
    weights.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit, match="already evaluated"):
        cmp.main(["--split", "test", "--model", f"m={weights}", "--dataset", str(ds), "--out", str(tmp_path / "o.json")])
    assert lock.read_bytes() == before and not (tmp_path / "o.json").exists()


def test_the_real_mio_tcd_test_lock_still_records_exactly_one_run():
    """Evidence guard: the held-out MIO-TCD test split was evaluated once (2026-10-18). A second entry means someone overrode the lock."""
    lock = REPO / "datasets" / "mio_tcd" / "yolo5" / "test_evaluation.lock.json"
    if not lock.is_file():
        pytest.skip("MIO-TCD dataset not present on this machine")
    runs = json.loads(lock.read_text())["runs"]
    assert len(runs) == 1, f"the held-out test split has been evaluated {len(runs)} times"


def _locked_dataset(tmp_path, split):
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 3, ("c2", "r1"): 3, ("c3", "r1"): 3})
    for r in m["frames"]:
        r["split"] = split
    lc.save_manifest(root, m)
    return root, [r for r in lc.load_manifest(root)["frames"]]


@pytest.mark.parametrize("split", ["test", "eval"])
def test_local_benchmark_held_out_splits_run_once_then_refuse_unless_overridden(tmp_path, monkeypatch, split):
    root, frames = _locked_dataset(tmp_path, split)
    monkeypatch.setattr(bench, "ultralytics_predictor", lambda p, imgsz, half, dev: perfect_predictor(root, frames))
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    args = ["run", "--dataset", str(root), "--split", split, "--model", f"m={weights}", "--out", str(tmp_path / "r1.json"), "--n-boot", "10"]
    assert bench.main(args) == 0
    lock = root / f"{split}_evaluation.lock.json"
    assert len(json.loads(lock.read_text())["runs"]) == 1
    assert bench.main(args[:-4] + ["--out", str(tmp_path / "r2.json"), "--n-boot", "10"]) == 4  # refused
    assert not (tmp_path / "r2.json").exists() and len(json.loads(lock.read_text())["runs"]) == 1
    assert bench.main(args[:-4] + ["--out", str(tmp_path / "r3.json"), "--n-boot", "10", "--allow-test-rerun"]) == 0  # override is explicit and recorded
    assert len(json.loads(lock.read_text())["runs"]) == 2


def test_local_benchmark_validation_split_is_not_locked(tmp_path, monkeypatch):
    root, frames = _locked_dataset(tmp_path, "val")
    monkeypatch.setattr(bench, "ultralytics_predictor", lambda p, imgsz, half, dev: perfect_predictor(root, frames))
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    base = ["run", "--dataset", str(root), "--split", "val", "--model", f"m={weights}", "--n-boot", "10"]
    assert bench.main(base + ["--out", str(tmp_path / "a.json")]) == 0 and bench.main(base + ["--out", str(tmp_path / "b.json")]) == 0
    assert not (root / "val_evaluation.lock.json").exists()


def test_a_dataset_with_frames_but_no_labels_never_produces_an_accuracy_verdict(tmp_path):
    """Unlabelled != verified empty: all-unlabelled footage cannot be 'perfect' or 'zero recall' - it cannot be scored at all."""
    root = tmp_path / "ds"
    build_dataset(root, {("c1", "r1"): 5}, labelled=False)
    m = lc.load_manifest(root)
    with pytest.raises(lb.NotEvaluable):
        lb.labelled_frames(m, None)
    # a gate fed nothing is INCONCLUSIVE, never PROMOTE
    assert lb.evaluate_gate(None, "finetuned", "pretrained")["verdict"] == "INCONCLUSIVE"
