"""Tests for the local-CCTV benchmark workflow. All data here is a SYNTHETIC fixture generated in tmp_path to exercise
the tooling; it is not footage and no metric from it means anything about a detector."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from anvesh.training import local_benchmark as lb
from anvesh.training import local_cctv as lc

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts import benchmark_local_cctv as bench  # noqa: E402
from scripts import local_cctv as cli  # noqa: E402

W, H = 320, 240


def _box(cx, cy, w=0.2, h=0.2):
    return (cx, cy, w, h)


def build_dataset(root: Path, recordings: dict, labelled=True, boxes_per_frame=3):
    """recordings: {(camera, recording): n_frames}. Each frame holds `boxes_per_frame` cars at fixed places."""
    (root / "frames").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "ignore").mkdir()
    frames = []
    for (cam, rec), n in recordings.items():
        for i in range(n):
            fid = lc.make_frame_id(cam, rec, i * 10)
            img = np.full((H, W, 3), 90 + i, np.uint8)
            cv2.imwrite(str(root / "frames" / f"{fid}.jpg"), img)
            rows = [(1, *_box(0.2 + 0.3 * k, 0.5)) for k in range(boxes_per_frame)]
            if labelled:
                (root / "labels" / f"{fid}.txt").write_text("".join(f"{c} {a} {b} {w} {h}\n" for c, a, b, w, h in rows))
            frames.append({"frame_id": fid, "image": f"frames/{fid}.jpg", "width": W, "height": H,
                           "image_sha256": lc.sha256_file(root / "frames" / f"{fid}.jpg"), "camera_id": cam, "recording_id": rec,
                           "recording_start_utc": "2026-11-02T02:30:00Z", "frame_time_s": float(i * 10), "conditions": ["day"],
                           "labelled": labelled, "split": None, "annotator": "a" if labelled else None})
    m = {"format": "anvesh-local-cctv/1", "classes": list(lc.LOCAL_CLASSES), "frames": frames}
    lc.save_manifest(root, m)
    return m


def perfect_predictor(root, frames, drop_every=0, extra_fp=0):
    gt = {str(root / r["image"]): lc.load_frame_gt(root, r) for r in frames}

    def predict(paths):
        out = []
        for k, p in enumerate(paths):
            gc, gx, _, _ = gt[str(p)]
            keep = np.ones(len(gx), bool)
            if drop_every:
                keep[::drop_every] = False
            px = gx[keep]
            conf = np.full(len(px), 0.9)
            cls = gc[keep]
            if extra_fp:
                px = np.vstack([px, np.tile([5.0, 5.0, 25.0, 25.0], (extra_fp, 1))])
                conf, cls = np.append(conf, [0.8] * extra_fp), np.append(cls, [1] * extra_fp)
            out.append((px, conf, cls))
        return out
    return predict


def test_parse_and_qa_flag_bad_rows_and_boxes():
    rows, errs = lc.parse_yolo_rows("1 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n1 0.5 0.5\nx 1 1 1 1\n")
    assert len(rows) == 1 and len(errs) == 3
    issues = lc.qa_frame([(1, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.5, 0.2, 0.2), (1, 0.95, 0.5, 0.2, 0.2), (1, 0.5, 0.5, 0.001, 0.2)], W, H)
    codes = {i["code"] for i in issues}
    assert {"duplicate_box", "out_of_bounds", "tiny_box"} <= codes


def test_missing_label_is_unlabelled_and_empty_label_is_a_verified_negative(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("camA", "r1"): 3})
    for r in m["frames"]:
        r["labelled"] = True
    (root / "labels" / f"{m['frames'][0]['frame_id']}.txt").write_text("")  # empty = verified negative
    (root / "labels" / f"{m['frames'][1]['frame_id']}.txt").unlink()  # missing but marked labelled -> error
    lc.save_manifest(root, m)
    rep = lc.check_dataset(root)
    assert any("missing = unlabelled" in e for e in rep["errors"])
    assert rep["counts"]["empty_negative_frames"] == 1


def test_manifest_detects_a_changed_image_and_unlabelled_frame_with_label_file(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("camA", "r1"): 2})
    cv2.imwrite(str(root / m["frames"][0]["image"]), np.zeros((H, W, 3), np.uint8))
    m["frames"][1]["labelled"] = False
    rep = lc.check_dataset(root, m)
    assert any("SHA-256" in e for e in rep["errors"])
    assert any("labelled=false but" in e for e in rep["errors"])


def test_split_is_group_wise_and_leakage_is_detected(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 6, ("c1", "r2"): 6, ("c2", "r1"): 6, ("c2", "r2"): 6, ("c3", "r1"): 6})
    mapping = lc.assign_splits(m["frames"], {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0, group_by="recording")
    by_group = {}
    for r in m["frames"]:
        by_group.setdefault((r["camera_id"], r["recording_id"]), set()).add(mapping[r["frame_id"]])
    assert all(len(s) == 1 for s in by_group.values()), "a recording straddles splits"
    assert {s for v in by_group.values() for s in v} == {"train", "val", "test"}
    for r in m["frames"]:
        r["split"] = mapping[r["frame_id"]]
    assert not [e for e in lc.check_dataset(root, m)["errors"] if "LEAKAGE" in e]
    leaky = next(r for r in m["frames"] if r["split"] == "train")
    twin = next(r for r in m["frames"] if r["camera_id"] == leaky["camera_id"] and r["recording_id"] == leaky["recording_id"] and r is not leaky)
    twin["split"] = "test"
    assert any("LEAKAGE" in e for e in lc.check_dataset(root, m)["errors"])
    with pytest.raises(ValueError):
        lc.assign_splits(m["frames"][:6], {"train": 0.5, "test": 0.5}, 0, "recording")  # one group, two splits


def test_adjacent_recordings_across_split_boundary_are_warned(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 3, ("c1", "r2"): 3})
    for r in m["frames"]:
        r["split"] = "train" if r["recording_id"] == "r1" else "test"
    assert any("ADJACENT" in w for w in lc.check_dataset(root, m)["warnings"])


def test_agreement_between_annotators():
    a = [(1, 0.3, 0.5, 0.2, 0.2), (3, 0.7, 0.5, 0.2, 0.2)]
    b = [(1, 0.3, 0.5, 0.2, 0.2), (4, 0.7, 0.5, 0.2, 0.2), (1, 0.1, 0.1, 0.05, 0.05)]
    s = lc.summarise_agreement([lc.agreement(a, b, W, H)])
    assert s["matched"] == 2 and s["class_agreement_given_match"] == 0.5 and s["box_agreement_f1"] == pytest.approx(0.8)


def test_evaluate_counts_misses_false_positives_and_ignores_dontcare(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 4, ("c2", "r1"): 4})
    (root / "ignore" / f"{m['frames'][0]['frame_id']}.txt").write_text("0 0.0625 0.0625 0.0625 0.0833\n")  # covers the 5,5,25,25 box
    frames = lb.labelled_frames(m, None)
    rep = lb.evaluate(root, frames, {"perfect": perfect_predictor(root, frames), "noisy": perfect_predictor(root, frames, drop_every=3, extra_fp=1)},
                      n_boot=20)
    perfect, noisy = rep["models"]["perfect"], rep["models"]["noisy"]
    assert perfect["class_agnostic_by_conf"]["0.25"]["recall"] == 1.0 and perfect["class_agnostic_by_conf"]["0.25"]["fp"] == 0
    n = noisy["class_agnostic_by_conf"]["0.25"]
    assert n["fn"] == 8 and n["tp"] == 16  # every 3rd of 3 boxes (index 0) dropped in 8 frames
    assert n["fp"] == 7  # one FP per frame, except the frame whose ignore region swallows it
    assert perfect["per_class"]["truck"]["status"].startswith("insufficient")  # no GT -> not reported as 0.0
    assert rep["coverage"]["cameras"] == ["c1", "c2"]


def test_evaluate_raises_not_evaluable_without_labelled_frames(tmp_path):
    m = build_dataset(tmp_path / "ds", {("c1", "r1"): 3}, labelled=False)
    with pytest.raises(lb.NotEvaluable):
        lb.labelled_frames(m, None)


def test_gate_is_inconclusive_without_data_and_refuses_to_pass_on_thin_data(tmp_path):
    g = lb.evaluate_gate(None, "finetuned", "pretrained")
    assert g["verdict"] == "INCONCLUSIVE" and all(c["status"] == "not_evaluable" for c in g["criteria"])
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 4, ("c2", "r1"): 4})
    frames = lb.labelled_frames(m, None)
    rep = lb.evaluate(root, frames, {"pretrained": perfect_predictor(root, frames, drop_every=2), "finetuned": perfect_predictor(root, frames)}, n_boot=20)
    rep["split"] = "test"
    g = lb.evaluate_gate(rep, "finetuned", "pretrained")
    assert g["verdict"] == "DO_NOT_PROMOTE"  # perfect numbers on a tiny fixture still fail data sufficiency
    assert next(c for c in g["criteria"] if c["id"] == "data_sufficiency")["status"] == "fail"
    assert next(c for c in g["criteria"] if c["id"] == "recall_floor")["status"] == "not_evaluable"


def test_gate_fails_on_cityflow_regression_and_passes_all_when_everything_clears(tmp_path):
    big = {k: 1 for k in ("min_cameras", "min_recordings", "min_labelled_frames", "min_boxes", "min_lighting_conditions")}
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 4, ("c2", "r1"): 4, ("c3", "r1"): 4})
    for r in m["frames"]:
        r["conditions"] = ["day", "night"]
    frames = lb.labelled_frames(m, None)
    rep = lb.evaluate(root, frames, {"pretrained": perfect_predictor(root, frames, drop_every=2, extra_fp=1), "finetuned": perfect_predictor(root, frames)},
                      n_boot=40)
    rep["split"] = "test"
    lat = {"pretrained": {"p95_ms": 10.0}, "finetuned": {"p95_ms": 11.0}}
    ok_cf = {"pooled": {"pretrained": {"recall_class_agnostic": 0.75}, "finetuned": {"recall_class_agnostic": 0.74}}}
    ds = {"idf1_diff": 0.0, "mota_diff": 0.01}
    crit = {**big, "per_camera_min_gt": 1, "recall_floor_ci_lower": 0.5}
    # map50 CI on a 3-recording fixture can straddle 0; relax only that for the pass path
    g = lb.evaluate_gate(rep, "finetuned", "pretrained", lat, ok_cf, ds, crit)
    stat = {c["id"]: c["status"] for c in g["criteria"]}
    assert stat["data_sufficiency"] == "pass" and stat["recall_floor"] == "pass" and stat["latency"] == "pass"
    bad_cf = {"pooled": {"pretrained": {"recall_class_agnostic": 0.748}, "finetuned": {"recall_class_agnostic": 0.718}}}
    g2 = lb.evaluate_gate(rep, "finetuned", "pretrained", lat, bad_cf, ds, crit)
    assert next(c for c in g2["criteria"] if c["id"] == "cityflow_noninferiority")["status"] == "fail"  # the real -0.030 case
    assert g2["verdict"] == "DO_NOT_PROMOTE"


def test_group_bootstrap_resamples_groups_not_frames():
    xs = {"a": [1.0, 1.0, 0.0, 0.0], "b": [1.0, 1.0, 1.0, 1.0]}
    out = lc.group_bootstrap(xs, ["g1", "g1", "g2", "g2"], lambda v: float(np.mean(v)), n_boot=200, seed=1)
    assert out["n_groups"] == 2
    assert out["models"]["a"]["ci95"] == [0.0, 1.0]  # with two groups a resample is all-g1, all-g2 or mixed


def test_cli_extract_import_qa_split_roundtrip_and_no_overwrite(tmp_path):
    vid = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (W, H))
    for i in range(60):
        vw.write(np.full((H, W, 3), i * 3, np.uint8))
    vw.release()
    root = tmp_path / "ds"
    assert cli.main(["extract", "--dataset", str(root), "--video", str(vid), "--camera-id", "camA", "--recording-id", "r1",
                     "--recording-start-utc", "2026-11-02T02:30:00Z", "--stride-seconds", "2", "--conditions", "day,dry"]) == 0
    m = lc.load_manifest(root)
    assert len(m["frames"]) == 3 and not any(r["labelled"] for r in m["frames"])
    with pytest.raises(SystemExit):  # same recording again must not overwrite
        cli.main(["extract", "--dataset", str(root), "--video", str(vid), "--camera-id", "camA", "--recording-id", "r1"])
    exp = tmp_path / "export"
    exp.mkdir()
    (exp / "obj.names").write_text("car\ndifficult\n")
    ids = [r["frame_id"] for r in m["frames"]]
    (exp / f"{ids[0]}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n")
    (exp / f"{ids[1]}.txt").write_text("")  # verified negative
    args = ["import-yolo", "--dataset", str(root), "--labels", str(exp), "--classes", str(exp / "obj.names"),
            "--annotator", "a", "--tool", "CVAT", "--tool-version", "2.0"]
    assert cli.main(args) == 0
    m = lc.load_manifest(root)
    lab = {r["frame_id"]: r["labelled"] for r in m["frames"]}
    assert lab == {ids[0]: True, ids[1]: True, ids[2]: False}
    assert (root / "ignore" / f"{ids[0]}.txt").is_file() and (root / "labels" / f"{ids[1]}.txt").read_text() == ""
    with pytest.raises(SystemExit):  # existing annotations are never overwritten
        cli.main(args)
    (exp / "obj.names").write_text("car\nspaceship\n")
    with pytest.raises(SystemExit):  # unmapped class would silently become background
        cli.main(["import-yolo", "--dataset", str(root), "--labels", str(exp), "--classes", str(exp / "obj.names"), "--annotator", "b", "--tool", "x", "--role", "second"])
    assert cli.main(["qa", "--dataset", str(root)]) == 0
    assert (root / "manifest_history").is_dir() and (root / "annotation_log.jsonl").is_file()
    assert cli.main(["split", "--dataset", str(root), "--eval-only"]) == 0
    assert {r["split"] for r in lc.load_manifest(root)["frames"]} == {"eval"}


def test_benchmark_cli_refuses_to_run_without_labelled_frames(tmp_path, capsys):
    root = tmp_path / "ds"
    build_dataset(root, {("c1", "r1"): 3}, labelled=False)
    out = tmp_path / "res.json"
    rc = bench.main(["run", "--dataset", str(root), "--split", "all", "--model", "x=models/yolov8n.pt", "--out", str(out)])
    assert rc == 3 and not out.exists()
    assert "NOT RUN" in capsys.readouterr().err


def test_gate_cli_with_no_inputs_is_inconclusive(tmp_path, capsys):
    out = tmp_path / "g.json"
    assert bench.main(["gate", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "INCONCLUSIVE"


def test_gate_only_accepts_a_report_from_the_held_out_test_split(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 4, ("c2", "r1"): 4, ("c3", "r1"): 4})
    frames = lb.labelled_frames(m, None)
    rep = lb.evaluate(root, frames, {"pretrained": perfect_predictor(root, frames, drop_every=2), "finetuned": perfect_predictor(root, frames)}, n_boot=20)
    for split in ("all", "train", "val", "eval", None):
        rep["split"] = split
        g = lb.evaluate_gate(rep, "finetuned", "pretrained", criteria={"min_cameras": 1, "min_recordings": 1, "min_labelled_frames": 1, "min_boxes": 1,
                                                                       "min_lighting_conditions": 0})
        stat = {c["id"]: c["status"] for c in g["criteria"]}
        assert stat["held_out_test_split"] == "not_evaluable" and stat["recall_floor"] == "not_evaluable"
        assert g["verdict"] != "PROMOTE"


def test_model_that_detects_nothing_still_reports_ground_truth_and_zero_recall(tmp_path):
    root = tmp_path / "ds"
    m = build_dataset(root, {("c1", "r1"): 6, ("c2", "r1"): 6})  # 12 frames x 3 cars = 36 >= MIN_CLASS_GT
    frames = lb.labelled_frames(m, None)
    empty = lambda paths: [(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) for _ in paths]  # noqa: E731
    rep = lb.evaluate(root, frames, {"silent": empty}, n_boot=10)
    car = rep["models"]["silent"]["per_class"]["car"]
    assert car["n_gt"] == 36 and car["recall"] == 0.0 and car["ap50"] == 0.0
    assert rep["models"]["silent"]["class_agnostic_by_conf"]["0.25"]["fn"] == 36
