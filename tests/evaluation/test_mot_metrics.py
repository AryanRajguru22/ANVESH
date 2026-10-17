"""Toy cases with hand-computed answers for the GT scoring primitives.
The harness is only trustworthy if these exact numbers hold."""

import pytest

from anvesh.evaluation.mot_metrics import (
    duplicate_identity_stats,
    duplicate_ids_per_window,
    evaluate_detection,
    evaluate_tracking,
    iou_matrix,
)


def _table(rows_by_frame):
    return {f: list(rows) for f, rows in rows_by_frame.items()}


def _line(obj_id, frames, x0=0.0, dx=2.0, w=50.0, h=50.0, y=100.0):
    """A moving box present in `frames`; returns {frame: [(id, x, y, w, h)]}."""
    out = {}
    for f in frames:
        out[f] = [(obj_id, x0 + dx * f, y, w, h)]
    return out


def _merge(*tables):
    merged = {}
    for t in tables:
        for f, rows in t.items():
            merged.setdefault(f, []).extend(rows)
    return merged


def test_iou_known_values():
    iou = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 10, 10), (20, 20, 5, 5)])
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert iou[0, 2] == 0.0


def test_detection_precision_recall():
    gt = {0: [(1, 0, 0, 10, 10), (2, 100, 100, 10, 10)]}
    pred = {0: [(0, 0, 0, 10, 10), (0, 500, 500, 10, 10), (0, 300, 300, 10, 10)]}
    s = evaluate_detection(gt, pred)
    assert (s.tp, s.fp, s.fn) == (1, 2, 1)
    assert s.precision == pytest.approx(1 / 3)
    assert s.recall == pytest.approx(0.5)


def test_detection_is_one_to_one():
    gt = {0: [(1, 0, 0, 10, 10)]}
    pred = {0: [(0, 0, 0, 10, 10), (0, 0, 0, 10, 10)]}  # duplicate detection of one object
    s = evaluate_detection(gt, pred)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_perfect_tracking():
    frames = range(10)
    gt = _line(1, frames)
    pred = _line(7, frames)  # different label, same trajectory
    s = evaluate_tracking(gt, pred)
    assert s.idf1 == pytest.approx(1.0)
    assert s.id_switches == 0 and s.fragmentations == 0
    assert s.mostly_tracked == 1 and s.mostly_lost == 0


def test_single_id_switch():
    gt = _line(1, range(10))
    pred = _merge(_line("A", range(0, 5)), _line("B", range(5, 10)))
    s = evaluate_tracking(gt, pred)
    assert s.id_switches == 1
    assert s.fragmentations == 0
    # GT -> A (5 frames); B is 5 FP frames, the other 5 GT frames FN
    assert (s.idtp, s.idfp, s.idfn) == (5, 5, 5)
    assert s.idf1 == pytest.approx(0.5)


def test_fragmentation_without_switch():
    gt = _line(1, range(10))
    pred = _merge(_line("A", range(0, 3)), _line("A", range(5, 10)))
    s = evaluate_tracking(gt, pred)
    assert s.id_switches == 0
    assert s.fragmentations == 1
    assert (s.idtp, s.idfp, s.idfn) == (8, 0, 2)
    assert s.idf1 == pytest.approx(16 / 18)


def test_split_after_gap_counts_switch_and_fragmentation():
    gt = _line(1, range(10))
    pred = _merge(_line("A", range(0, 3)), _line("B", range(6, 10)))
    s = evaluate_tracking(gt, pred)
    assert s.id_switches == 1
    assert s.fragmentations == 1


def test_extra_unannotated_prediction_lowers_precision_side_only():
    """The CityFlow GT is incomplete (only multi-camera vehicles): an extra
    predicted track must hurt IDP/IDF1 but leave IDR and switch/fragment
    counts untouched -- the property that makes IDR/IDSW/FM the trustworthy
    metrics under partial GT."""
    gt = _line(1, range(10))
    clean = evaluate_tracking(gt, _line("A", range(10)))
    extra = evaluate_tracking(gt, _merge(_line("A", range(10)), _line("Z", range(10), y=700.0)))
    assert extra.idr == pytest.approx(clean.idr) == pytest.approx(1.0)
    assert extra.idp < clean.idp
    assert extra.idf1 < clean.idf1
    assert extra.id_switches == clean.id_switches == 0
    assert extra.fragmentations == clean.fragmentations == 0


def test_two_objects_swapping_identities_is_counted():
    gt = _merge(_line(1, range(10), y=100.0), _line(2, range(10), y=400.0))
    pred = {}
    for f in range(10):
        a, b = ("P", "Q") if f < 5 else ("Q", "P")
        pred[f] = [(a, 2.0 * f, 100.0, 50.0, 50.0), (b, 2.0 * f, 400.0, 50.0, 50.0)]
    s = evaluate_tracking(gt, pred)
    assert s.id_switches == 2


def test_empty_inputs():
    s = evaluate_tracking({}, {})
    assert s.idf1 == 0.0 and s.id_switches == 0
    d = evaluate_detection({}, {})
    assert d.precision == 0.0 and d.recall == 0.0


def test_duplicate_identity_stats():
    s = duplicate_identity_stats(["a", "b", "b", "c", "b"])
    assert s.states == 5 and s.identity_keys == 3
    assert s.duplicated_keys == 1
    assert s.duplicate_rate == pytest.approx(2 / 5)
    assert duplicate_identity_stats(["a", "b"]).duplicate_rate == 0.0
    assert duplicate_identity_stats([]).duplicate_rate == 0.0


def test_duplicate_ids_per_window():
    entries, dup = duplicate_ids_per_window([["a", "b", "a"], ["c"], []])
    assert (entries, dup) == (4, 1)
