import pytest

from anvesh.training.mio_tcd import CLASS_MAP, IGNORE_ONLY, NATIVE_CLASSES, TRAIN_CLASSES, convert_rows
from anvesh.training.detector_eval import SHARED_CLASSES


def test_every_native_class_is_accounted_for_exactly_once():
    assert set(NATIVE_CLASSES) == {
        "articulated_truck", "bicycle", "bus", "car", "motorcycle", "motorized_vehicle", "non-motorized_vehicle",
        "pedestrian", "pickup_truck", "single_unit_truck", "work_van"}
    assert not set(CLASS_MAP) & set(IGNORE_ONLY)


def test_training_class_ids_0_to_4_are_the_shared_evaluation_taxonomy_in_order():
    assert TRAIN_CLASSES[:5] == SHARED_CLASSES and TRAIN_CLASSES[5] == "other_vehicle"


def test_mapping_of_the_judgement_call_classes():
    assert CLASS_MAP["pickup_truck"] == "truck" and CLASS_MAP["work_van"] == "car"
    assert CLASS_MAP["motorized_vehicle"] == "other_vehicle"


def test_convert_rows_yolo_normalisation_ignore_and_clipping():
    rows = [("car", 10, 20, 50, 60), ("pedestrian", 0, 0, 10, 10), ("bus", -5, 90, 30, 130)]
    labels, ignore, dropped = convert_rows(rows, w=100, h=100)
    assert labels[0] == "1 0.300000 0.400000 0.400000 0.400000"
    assert labels[1].startswith("3 ")  # bus clipped to y in [90, 100]
    assert labels[1] == "3 0.150000 0.950000 0.300000 0.100000"
    assert ignore == ["0 0.050000 0.050000 0.100000 0.100000"] and dropped == 0


def test_degenerate_boxes_are_dropped_and_counted_and_unknown_classes_raise():
    _, _, dropped = convert_rows([("car", 10, 10, 10, 50), ("car", 10, 10, 40, 40)], 100, 100)
    assert dropped == 1
    with pytest.raises(ValueError):
        convert_rows([("tuk_tuk", 1, 1, 9, 9)], 100, 100)
