import numpy as np
import pytest

from anvesh.training.detector_eval import (
    SHARED_CLASSES, ImageResult, bootstrap, confusion_matrix, match_image, model_to_shared, operating_point, score)

CAR, BUS = SHARED_CLASSES.index("car"), SHARED_CLASSES.index("bus")
BOX = [10.0, 10.0, 50.0, 50.0]


def img(pred, gt):
    """pred: [(xyxy, conf, cls)], gt: [(xyxy, cls)]"""
    return ImageResult([p[0] for p in pred], [p[1] for p in pred], [p[2] for p in pred],
                       [g[0] for g in gt], [g[1] for g in gt])


def test_model_to_shared_maps_by_name_for_coco_and_anvesh_layouts():
    coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert model_to_shared(coco) == {1: 0, 2: 1, 3: 2, 5: 3, 7: 4}
    assert model_to_shared({0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_perfect_predictions_score_one_and_missing_classes_are_reported_as_absent_not_zero():
    res = [img([(BOX, 0.9, CAR)], [(BOX, CAR)]) for _ in range(5)]
    s = score(res)
    # Ultralytics' 101-point interpolated AP tops out at 0.995 for a perfect detector; same convention here.
    assert s["map50"] == pytest.approx(0.995, abs=1e-3) and s["map50_95"] == pytest.approx(0.995, abs=1e-3)
    assert s["per_class"]["car"]["recall"] == pytest.approx(1.0)
    assert "note" in s["per_class"]["truck"] and "ap50" not in s["per_class"]["truck"]


def test_wrong_class_and_no_prediction_score_zero():
    assert score([img([(BOX, 0.9, BUS)], [(BOX, CAR)])])["map50"] == 0.0
    assert score([img([], [(BOX, CAR)])])["map50"] == 0.0


def test_match_is_one_to_one_so_a_duplicate_box_is_a_false_positive():
    pred_xyxy = np.array([BOX, BOX])
    correct = match_image(pred_xyxy, np.array([CAR, CAR]), np.array([BOX]), np.array([CAR]))
    assert correct[:, 0].sum() == 1


def test_iou_threshold_gating_across_the_0_5_to_0_95_range():
    shifted = [10.0, 10.0, 50.0, 62.0]  # IoU = 40*40 / (40*52) ~ 0.77
    correct = match_image(np.array([shifted]), np.array([CAR]), np.array([BOX]), np.array([CAR]))[0]
    assert correct[0] and not correct[-1]


def test_operating_point_thresholds_before_matching():
    # A confident correct box plus a low-confidence duplicate: at conf 0.25 only the first counts.
    res = [img([(BOX, 0.9, CAR), (BOX, 0.1, CAR)], [(BOX, CAR)])]
    op = operating_point(res, 0.25)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 0, 0)
    assert operating_point(res, 0.05)["fp"] == 1


def test_confusion_matrix_shows_class_confusion_missed_and_background():
    nc = len(SHARED_CLASSES)
    far = [200.0, 200.0, 240.0, 240.0]
    res = [img([(BOX, 0.9, BUS), (far, 0.9, CAR)], [(BOX, CAR)])]
    cm = confusion_matrix(res)
    assert cm[BUS, CAR] == 1  # a car predicted as bus
    assert cm[CAR, nc] == 1  # a car prediction on background
    assert cm.sum() == 2


def test_bootstrap_is_seeded_and_paired_difference_favours_the_better_model():
    good = [img([(BOX, 0.9, CAR)], [(BOX, CAR)]) for _ in range(20)]
    bad = [img([], [(BOX, CAR)]) for _ in range(20)]
    a = bootstrap({"bad": bad, "good": good}, n_boot=10, seed=3)
    b = bootstrap({"bad": bad, "good": good}, n_boot=10, seed=3)
    assert a == b
    d = a["paired_difference"]["good - bad"]
    assert d["mean"] > 0.5 and d["fraction_of_resamples_b_better"] == 1.0


def test_unmatched_predictions_on_dont_care_regions_are_not_false_positives():
    far = [200.0, 200.0, 240.0, 240.0]
    res = [ImageResult([BOX, far], [0.9, 0.9], [CAR, CAR], [BOX], [CAR], ignore_xyxy=[far])]
    assert operating_point(res, 0.25)["fp"] == 0  # the box on the don't-care region is dropped
    assert operating_point(res, 0.25, use_ignore=False)["fp"] == 1
    assert confusion_matrix(res)[CAR, len(SHARED_CLASSES)] == 0
    # ...but a prediction that matches a real GT box is never dropped just because an ignore box overlaps it
    res2 = [ImageResult([BOX], [0.9], [CAR], [BOX], [CAR], ignore_xyxy=[BOX])]
    assert operating_point(res2, 0.25)["tp"] == 1
