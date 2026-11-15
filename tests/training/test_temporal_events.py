"""Temporal-event tooling. ALL data here is synthetic and exists only to check the software: no accident dataset has been downloaded,
no detector exists, and no number in this file says anything about real accident detection."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from anvesh.training import temporal_events as te

REPO = Path(__file__).resolve().parents[2]


def vid(i, group=None, dur=3600.0, full=True, split=None):
    v = {"video_id": f"v{i}", "source_group": group or f"g{i}", "fps": 25, "duration_s": dur, "license": "synthetic-test", "fully_annotated": full}
    if split:
        v["split"] = split
    return v


def ev(i, video, start, end, certainty="certain"):
    return {"event_id": f"e{i}", "video_id": video, "type": "accident", "start_s": start, "end_s": end, "annotator": "a", "certainty": certainty}


def manifest(n_videos=6, events_per=2, **kw):
    videos = [vid(i, **kw) for i in range(n_videos)]
    events = [ev(f"{i}_{j}", f"v{i}", 100 + 400 * j, 108 + 400 * j) for i in range(n_videos) for j in range(events_per)]
    return {"schema_version": 1, "dataset": "synthetic", "videos": videos, "events": events}


# -- schema validation -------------------------------------------------------------------------------------------------


def test_valid_manifest_passes_and_reports_negative_hours():
    r = te.validate_manifest(manifest())
    assert r["ok"] and r["stats"]["events"] == 12 and r["stats"]["negative_hours_available"] == 6.0


def _mutate(fn):
    m = manifest(3)
    fn(m)
    return te.validate_manifest(m)


@pytest.mark.parametrize("fn,needle", [
    (lambda m: m.update(schema_version=2), "schema_version"),
    (lambda m: m["videos"][0].update(license=""), "license"),
    (lambda m: m["videos"][0].update(source_group=""), "source_group"),
    (lambda m: m["videos"][0].update(fully_annotated=None), "fully_annotated"),
    (lambda m: m["videos"][0].update(fully_annotated="yes"), "fully_annotated"),
    (lambda m: m["videos"][0].update(fps=0), "fps"),
    (lambda m: m["videos"][0].update(duration_s=float("nan")), "duration_s"),
    (lambda m: m["videos"][1].update(video_id="v0"), "duplicate video_id"),
    (lambda m: m["events"][0].update(video_id="nope"), "unknown video_id"),
    (lambda m: m["events"][0].update(start_s=9, end_s=9), "start_s < end_s"),
    (lambda m: m["events"][0].update(end_s=99999), "beyond the video duration"),
    (lambda m: m["events"][0].update(certainty="maybe"), "certainty"),
    (lambda m: m["events"][0].update(annotator=""), "annotator"),
    (lambda m: m["events"][1].update(start_s=105), "overlap"),
    (lambda m: m["events"][1].update(event_id="e0_0"), "event_id"),
])
def test_each_schema_defect_is_reported(fn, needle):
    r = _mutate(fn)
    assert not r["ok"] and any(needle in e for e in r["errors"]), r["errors"]


@pytest.mark.parametrize("bad", [None, [], "x", {"schema_version": 1}, {"schema_version": 1, "videos": "x", "events": 5}])
def test_garbage_manifests_are_reported_not_raised(bad):
    assert te.validate_manifest(bad)["ok"] is False


def test_no_fully_annotated_video_warns_that_false_alarm_rate_is_unmeasurable():
    r = te.validate_manifest(manifest(full=False))
    assert r["ok"] and any("false alarms per hour cannot be measured" in w for w in r["warnings"])


# -- source-grouped splitting -------------------------------------------------------------------------------------------------


def test_split_never_separates_a_source_group_and_puts_events_in_every_split():
    m = manifest(12)
    for i in range(12):  # three clips per source recording
        m["videos"][i]["source_group"] = f"rec{i // 3}"
    assign = te.split_by_source_group(m, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=1)
    by_group = {}
    for v in m["videos"]:
        by_group.setdefault(v["source_group"], set()).add(assign[v["video_id"]])
    assert all(len(s) == 1 for s in by_group.values())  # whole groups only
    for v in m["videos"]:
        v["split"] = assign[v["video_id"]]
    r = te.validate_manifest(m)
    assert r["ok"] and all(r["stats"]["events_per_split"].get(s, 0) > 0 for s in te.SPLITS)


def test_split_is_deterministic_per_seed_and_differs_across_seeds():
    m = manifest(24)
    a = te.split_by_source_group(m, {"train": 0.6, "val": 0.2, "test": 0.2}, 7)
    assert a == te.split_by_source_group(m, {"train": 0.6, "val": 0.2, "test": 0.2}, 7)
    assert a != te.split_by_source_group(m, {"train": 0.6, "val": 0.2, "test": 0.2}, 8)


def test_split_refuses_rather_than_return_an_empty_split():
    m = manifest(3)
    for v in m["videos"]:
        v["source_group"] = "only_one_recording"
    with pytest.raises(ValueError, match="received no events"):
        te.split_by_source_group(m, {"train": 0.7, "val": 0.15, "test": 0.15}, 0)
    with pytest.raises(ValueError):
        te.split_by_source_group(manifest(3), {"train": 0.5, "val": 0.2}, 0)


def test_a_group_in_two_splits_or_a_partial_split_is_a_validation_error():
    m = manifest(4)
    m["videos"][0].update(source_group="shared", split="train")
    m["videos"][1].update(source_group="shared", split="test")
    r = te.validate_manifest(m)
    assert not r["ok"] and any("shared" in e and "leak" in e for e in r["errors"]) and any("have no split" in e for e in r["errors"])


# -- evaluation ------------------------------------------------------------------------------------------------------------


def preds(*items):
    return [{"video_id": v, "time_s": t, **({"score": s} if s is not None else {})} for v, t, s in items]


def test_recall_delay_and_false_alarms_per_hour_on_a_hand_checkable_case():
    m = manifest(2, events_per=1)  # v0: event 100-108, v1: event 100-108; 1 h each, fully annotated
    p = preds(("v0", 103.0, 0.9), ("v0", 500.0, 0.8), ("v0", 900.0, 0.7), ("v1", 50.0, 0.6))
    r = te.evaluate_event_detector(m, p, early_tol_s=2.0, late_tol_s=2.0, cooldown_s=5.0, n_boot=0)
    assert r["events"] == 2 and r["hits"] == 1 and r["recall"] == 0.5
    assert r["false_alarms"] == 3 and r["negative_hours"] == 2.0 and r["false_alarms_per_hour"] == 1.5
    assert r["detection_delay"]["median_s"] == pytest.approx(3.0) and r["detection_delay"]["n"] == 1


def test_early_alarm_inside_tolerance_counts_and_is_reported_as_early_not_clipped():
    m = manifest(1, events_per=1)
    r = te.evaluate_event_detector(m, preds(("v0", 98.5, None)), early_tol_s=2.0, n_boot=0)
    assert r["hits"] == 1 and r["detection_delay"]["median_s"] == pytest.approx(-1.5) and r["detection_delay"]["early_alarms"] == 1
    r2 = te.evaluate_event_detector(m, preds(("v0", 97.0, None)), early_tol_s=2.0, n_boot=0)
    assert r2["hits"] == 0 and r2["false_alarms"] == 1


def test_repeat_alarms_for_one_event_are_one_detection_and_cooldown_merges_bursts():
    m = manifest(1, events_per=1)
    p = preds(("v0", 101, 0.9), ("v0", 104, 0.9), ("v0", 107, 0.9), ("v0", 800, 0.9), ("v0", 802, 0.9), ("v0", 803, 0.9))
    r = te.evaluate_event_detector(m, p, cooldown_s=5.0, n_boot=0)
    assert r["hits"] == 1 and r["false_alarms"] == 1  # the 800/802/803 burst is ONE false alarm
    assert r["repeat_alarms_merged"] == 3  # 104 and 802 and 803 merged away (101->104 is <5s apart too)


def test_alarms_in_unannotated_videos_are_neither_hits_nor_false_alarms():
    m = manifest(2, events_per=1)
    m["videos"][1]["fully_annotated"] = False
    r = te.evaluate_event_detector(m, preds(("v1", 5000.0, None), ("v1", 101.0, None)), n_boot=0)
    assert r["false_alarms"] == 0 and r["alarms_in_unannotated_videos_ignored"] == 1  # the one outside the event window
    assert r["hits"] == 1  # annotated events still count for recall
    assert r["negative_hours"] == 1.0  # the unannotated video contributes no negative time


def test_zero_predictions_gives_zero_recall_and_zero_false_alarms():
    r = te.evaluate_event_detector(manifest(2), [], n_boot=0)
    assert r["recall"] == 0.0 and r["false_alarms"] == 0 and r["false_alarms_per_hour"] == 0.0 and r["detection_delay"]["n"] == 0


def test_claims_are_not_allowed_on_a_small_or_non_test_evaluation_and_the_text_says_why():
    r = te.evaluate_event_detector(manifest(2), [], n_boot=0)
    assert r["claims_allowed"] is False and "NOT_EVALUABLE_FOR_CLAIMS" in r["verdict"] and "events" in r["verdict"]
    m = manifest(40, events_per=1, split="val")  # 40 events, 40 h, but the VAL split
    r = te.evaluate_event_detector(m, [], split="val", n_boot=0)
    assert not r["claims_allowed"] and "val" in r["verdict"]
    m = manifest(40, events_per=1, split="test")
    r = te.evaluate_event_detector(m, [], split="test", n_boot=0)
    assert r["claims_allowed"] is True and r["verdict"] == "evaluable"


def test_bad_inputs_are_refused():
    m = manifest(2)
    with pytest.raises(ValueError, match="unknown video"):
        te.evaluate_event_detector(m, preds(("zzz", 1.0, None)))
    with pytest.raises(ValueError, match="invalid prediction time"):
        te.evaluate_event_detector(m, preds(("v0", float("nan"), None)))
    with pytest.raises(ValueError, match="invalid prediction time"):
        te.evaluate_event_detector(m, preds(("v0", -1.0, None)))
    with pytest.raises(ValueError, match="no videos in split"):
        te.evaluate_event_detector(m, [], split="test")
    bad = copy.deepcopy(m)
    bad["videos"][0]["license"] = ""
    with pytest.raises(ValueError, match="invalid manifest"):
        te.evaluate_event_detector(bad, [])


def test_bootstrap_interval_brackets_the_point_estimate_and_is_seeded():
    m = manifest(30, events_per=1)
    p = preds(*[(f"v{i}", 103.0, 0.9) for i in range(0, 30, 2)] + [(f"v{i}", 2000.0, 0.5) for i in range(0, 30, 3)])
    a = te.evaluate_event_detector(m, p, n_boot=300, seed=3)
    b = te.evaluate_event_detector(m, p, n_boot=300, seed=3)
    assert a["uncertainty"] == b["uncertainty"]
    lo, hi = a["uncertainty"]["recall_ci95"]
    assert lo <= a["recall"] <= hi and hi - lo > 0
    lo, hi = a["uncertainty"]["false_alarms_per_hour_ci95"]
    assert lo <= a["false_alarms_per_hour"] <= hi


def test_operating_curve_trades_recall_for_false_alarms():
    m = manifest(10, events_per=1)
    p = preds(*([(f"v{i}", 103.0, 0.9) for i in range(10)] + [(f"v{i}", 2000.0, 0.3) for i in range(10)]))
    curve = te.operating_curve(m, p, thresholds=5)
    assert curve[0]["false_alarms_per_hour"] > curve[-1]["false_alarms_per_hour"]
    assert curve[0]["recall"] == curve[-1]["recall"] == 1.0  # the real alarms score high, the spurious ones low
    with pytest.raises(ValueError):
        te.operating_curve(m, preds(("v0", 1.0, None)))


def test_end_to_end_with_a_synthetic_feature_rule_on_synthetic_temporal_features():
    """A throw-away rule (sudden speed drop) over SYNTHETIC speed series stands in for 'a detector' so the plumbing runs end to end.
    It is not a model and its score means nothing."""
    m = manifest(8, events_per=1)
    predictions = []
    for i in range(8):
        speed = [10.0] * 3600
        for t in range(100, 108):
            speed[t] = 1.0  # synthetic crash signature
        if i % 2:
            speed[2500] = 0.5  # a synthetic nuisance dip: a false alarm
        for t in range(1, 3600):
            if speed[t - 1] - speed[t] > 5:
                predictions.append({"video_id": f"v{i}", "time_s": float(t), "score": speed[t - 1] - speed[t]})
    r = te.evaluate_event_detector(m, predictions, n_boot=0)
    assert r["recall"] == 1.0 and r["false_alarms"] == 4 and r["negative_hours"] == 8.0
    assert r["claims_allowed"] is False  # 8 synthetic events is never a claim


def test_cli_validate_split_evaluate_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location("event_manifest_cli", REPO / "scripts" / "event_manifest.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    m = manifest(12)
    for i, v in enumerate(m["videos"]):
        v["source_group"] = f"rec{i // 2}"
    (tmp_path / "m.json").write_text(json.dumps(m))
    assert cli.main(["validate", "--manifest", str(tmp_path / "m.json")]) == 0
    assert cli.main(["split", "--manifest", str(tmp_path / "m.json"), "--out", str(tmp_path / "s.json")]) == 0
    assert cli.main(["split", "--manifest", str(tmp_path / "m.json"), "--out", str(tmp_path / "s.json")]) == 1  # never overwrites
    assert cli.main(["split", "--manifest", str(tmp_path / "s.json"), "--out", str(tmp_path / "s2.json")]) == 1  # already split
    (tmp_path / "p.json").write_text(json.dumps([]))
    assert cli.main(["evaluate", "--manifest", str(tmp_path / "s.json"), "--predictions", str(tmp_path / "p.json"), "--split", "test",
                     "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["claims_allowed"] is False
    (tmp_path / "bad.json").write_text(json.dumps({"schema_version": 1, "videos": [], "events": []}))
    assert cli.main(["validate", "--manifest", str(tmp_path / "bad.json")]) == 1
