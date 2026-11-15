"""YOLO dataset validator, pre-flight and resume resolution. Synthetic tiny datasets only: nothing here says anything about a real dataset."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from anvesh.training import preflight
from anvesh.training.yolo_dataset import base_id, validate_yolo_dataset

REPO = Path(__file__).resolve().parents[2]


def _img(path: Path, seed: int, size=(48, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (*size, 3), dtype=np.uint8))


def _build(tmp_path, spec, names="  0: plate\n  1: car\n"):
    """spec: {split: {stem: label_text or None (=no label file)}}"""
    root = tmp_path / "ds"
    seed = 0
    for split, items in spec.items():
        for stem, label in items.items():
            seed += 1
            _img(root / "images" / split / f"{stem}.jpg", seed)
            if label is not None:
                (root / "labels" / split).mkdir(parents=True, exist_ok=True)
                (root / "labels" / split / f"{stem}.txt").write_text(label, encoding="utf-8")
    lines = [f"path: {root.as_posix()}"] + [f"{s}: images/{s}" for s in spec] + ["names:", names.rstrip("\n")]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "data.yaml"


GOOD = "0 0.5 0.5 0.2 0.1\n"


def test_a_clean_dataset_passes_and_counts_are_right(tmp_path):
    y = _build(tmp_path, {"train": {"a": GOOD, "b": GOOD + "1 0.3 0.3 0.1 0.1\n", "c": ""}, "val": {"d": GOOD}})
    r = validate_yolo_dataset(y)
    assert r["ok"] and r["errors"] == []
    t = r["splits"]["train"]
    assert t["images"] == 3 and t["labelled_images"] == 2 and t["empty_label_files"] == 1 and t["boxes"] == 3
    assert t["class_boxes"] == {0: 2, 1: 1}
    assert r["leakage"]["identical_images_across_splits"] == 0


@pytest.mark.parametrize("row,why", [
    ("0 0.5 0.5 0.2", "4 fields"), ("x 0.5 0.5 0.2 0.1", "non-numeric"), ("5 0.5 0.5 0.2 0.1", "class id out of range"),
    ("0 1.5 0.5 0.2 0.1", "centre outside"), ("0 0.5 0.5 0 0.1", "zero width"), ("0 0.5 0.5 -0.2 0.1", "negative width"),
    ("0 0.95 0.5 0.2 0.1", "box beyond the right edge"), ("0 0.5 0.5 nan 0.1", "nan"),
])
def test_bad_label_rows_are_errors(tmp_path, row, why):
    r = validate_yolo_dataset(_build(tmp_path, {"train": {"a": row + "\n"}}))
    assert not r["ok"], why
    assert r["splits"]["train"]["bad_label_rows"] == 1


def test_missing_label_file_is_not_an_empty_label_file(tmp_path):
    spec = {"train": {"a": GOOD, "unlabelled": None, "verified_empty": ""}}
    r = validate_yolo_dataset(_build(tmp_path, spec))
    t = r["splits"]["train"]
    assert t["missing_label_files"] == 1 and t["empty_label_files"] == 1  # reported separately, never merged
    assert r["ok"] and any("NO label file" in w for w in r["warnings"])
    strict = validate_yolo_dataset(tmp_path / "ds" / "data.yaml", strict=True)
    assert not strict["ok"] and any("never labelled" in e for e in strict["errors"])


def test_orphan_label_without_an_image_is_an_error(tmp_path):
    y = _build(tmp_path, {"train": {"a": GOOD}})
    (tmp_path / "ds" / "labels" / "train" / "ghost.txt").write_text(GOOD)
    r = validate_yolo_dataset(y)
    assert not r["ok"] and r["splits"]["train"]["orphan_label_files"] == 1


def test_unreadable_image_is_an_error(tmp_path):
    y = _build(tmp_path, {"train": {"a": GOOD, "b": GOOD}})
    (tmp_path / "ds" / "images" / "train" / "b.jpg").write_bytes(b"not an image")
    r = validate_yolo_dataset(y)
    assert not r["ok"] and r["splits"]["train"]["unreadable_images"] == 1


def test_duplicate_rows_are_a_warning(tmp_path):
    r = validate_yolo_dataset(_build(tmp_path, {"train": {"a": GOOD + GOOD}}))
    assert r["ok"] and r["splits"]["train"]["duplicate_label_rows"] == 1 and any("duplicate" in w for w in r["warnings"])


def test_byte_identical_image_in_two_splits_is_reported_as_leakage(tmp_path):
    y = _build(tmp_path, {"train": {"a": GOOD}, "val": {"b": GOOD}, "test": {"c": GOOD}})
    (tmp_path / "ds" / "images" / "test" / "c.jpg").write_bytes((tmp_path / "ds" / "images" / "train" / "a.jpg").read_bytes())
    r = validate_yolo_dataset(y)
    assert not r["ok"] and r["leakage"]["identical_images_across_splits"] == 1
    assert "exact" in r["leakage"]["scope"] and "near-duplicate" in r["leakage"]["scope"]  # the limits of the check are stated


def test_same_base_id_across_splits_is_reported_even_when_pixels_differ(tmp_path):
    y = _build(tmp_path, {"train": {"car1_jpg.rf.aaaa": GOOD}, "val": {"car1_jpg.rf.bbbb": GOOD}})
    r = validate_yolo_dataset(y)
    assert not r["ok"] and r["leakage"]["base_ids_across_splits"] == 1
    assert base_id("car1_jpg.rf.0a1b2c") == "car1" and base_id("plain") == "plain"


def test_missing_split_directory_and_bad_names(tmp_path):
    y = _build(tmp_path, {"train": {"a": GOOD}}, names="  0: a\n  2: c\n")
    r = validate_yolo_dataset(y)
    assert not r["ok"] and any("contiguous" in e for e in r["errors"])
    assert not validate_yolo_dataset(tmp_path / "nope.yaml")["ok"]


def test_the_cli_exit_codes(tmp_path):
    spec = importlib.util.spec_from_file_location("vyd", REPO / "scripts" / "validate_yolo_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = _build(tmp_path, {"train": {"a": GOOD}})
    assert mod.main(["--data", str(good), "--out", str(tmp_path / "r.json")]) == 0 and (tmp_path / "r.json").is_file()
    (tmp_path / "ds" / "labels" / "train" / "a.txt").write_text("9 0.5 0.5 0.1 0.1\n")
    assert mod.main(["--data", str(good)]) == 1


# -- pre-flight and resume ---------------------------------------------------------------------------------------------------


def snap(ram=8.0, disk=50.0, vram=5.0, gpu=True):
    return {"ram_free_gb": ram, "ram_total_gb": 16, "disk_free_gb": {"D": disk},
            "gpu": {"name": "x", "vram_free_gb": vram, "vram_total_gb": 6} if gpu else None}


def test_preflight_passes_with_headroom_and_fails_for_each_shortage():
    assert preflight.evaluate(snap())["ok"]
    for kw, word in ((dict(ram=1.0), "RAM"), (dict(disk=1.0), "disk"), (dict(vram=0.5), "VRAM"), (dict(gpu=False), "CUDA")):
        v = preflight.evaluate(snap(**kw))
        assert not v["ok"] and any(word in f for f in v["failures"]), kw
    assert preflight.evaluate(snap(gpu=False), need_gpu=False)["ok"]


def test_preflight_never_passes_on_an_unmeasurable_quantity():
    v = preflight.evaluate({"ram_free_gb": None, "disk_free_gb": {"D": None}, "gpu": None}, need_gpu=False)
    assert not v["ok"] and len(v["failures"]) == 2


def test_runtime_is_not_predicted_without_a_measured_throughput():
    est = preflight.estimate_runtime(88270, 16, 8)
    assert est["iterations_per_epoch"] == 5517 and est["estimated_hours"] is None and "not predicted" in est["note"]
    assert preflight.estimate_runtime(88270, 16, 8, seconds_per_iteration=0.33)["estimated_hours"] == pytest.approx(5517 * 8 * 0.33 / 3600, abs=0.01)
    with pytest.raises(ValueError):
        preflight.iterations_per_epoch(10, 0)


def test_resume_needs_a_real_last_checkpoint(tmp_path):
    runs = tmp_path / "runs"
    (runs / "never_finished" / "weights").mkdir(parents=True)
    (runs / "ok" / "weights").mkdir(parents=True)
    (runs / "ok" / "weights" / "last.pt").write_bytes(b"x")
    (runs / "only_best" / "weights").mkdir(parents=True)
    (runs / "only_best" / "weights" / "best.pt").write_bytes(b"x")
    assert preflight.resolve_resume_checkpoint(runs, "ok") == runs / "ok" / "weights" / "last.pt"
    with pytest.raises(SystemExit, match="never completed an epoch|nothing to resume"):
        preflight.resolve_resume_checkpoint(runs, "never_finished")
    with pytest.raises(SystemExit, match="best.pt"):
        preflight.resolve_resume_checkpoint(runs, "only_best")  # best.pt is not a substitute, and the message says what exists
    with pytest.raises(SystemExit, match="no run directory"):
        preflight.resolve_resume_checkpoint(runs, "missing")
    for bad in ("", "..", "../ok", "a/b", "a\\b"):
        with pytest.raises(SystemExit):
            preflight.resolve_resume_checkpoint(runs, bad)


def _train_module():
    spec = importlib.util.spec_from_file_location("train_detector_under_test", REPO / "scripts" / "train_detector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_script_refuses_to_resume_the_real_interrupted_plate_run():
    """models/finetuned/runs/plates_hf_v1 died during epoch 1: it holds only args.yaml and preview images, no weights. (Skipped where the run is absent.)"""
    mod = _train_module()
    run = mod.RUNS_DIR / "plates_hf_v1"
    if not run.is_dir():
        pytest.skip("run directory not present on this machine")
    with pytest.raises(SystemExit, match="nothing to resume"):
        mod.main(["--resume", "plates_hf_v1"])


def test_train_script_preflight_only_makes_no_run_and_refuses_when_resources_are_short(tmp_path, capsys):
    mod = _train_module()
    name = "pytest_preflight_never_created"
    rc = mod.main(["--name", name, "--epochs", "1", "--preflight-only", "--min-free-ram-gb", "100000"])
    assert rc == 2 and not (mod.RUNS_DIR / name).exists()
    out = capsys.readouterr().out
    assert "[preflight]" in out and "free RAM" in out
    with pytest.raises(SystemExit):
        mod.main(["--epochs", "1"])  # --name is required for a new run (argparse error)
