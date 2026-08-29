from anvesh.perception.detection import parse_detection_results
from anvesh.storage.schemas import VehicleClass
from tests.perception.fakes import FakeBoxes, FakeResult


def test_parses_relevant_classes_only():
    # COCO ids: 2=car, 3=motorcycle, 0=person (irrelevant, must be dropped)
    boxes = FakeBoxes(
        xyxy=[[10.0, 10.0, 50.0, 50.0], [5.0, 5.0, 15.0, 15.0], [0.0, 0.0, 8.0, 8.0]],
        cls=[2.0, 3.0, 0.0],
        conf=[0.9, 0.6, 0.8],
    )
    detections = parse_detection_results(FakeResult(boxes), frame_index=3, timestamp=0.3)

    assert len(detections) == 2
    assert {d.vehicle_class for d in detections} == {VehicleClass.CAR, VehicleClass.MOTORCYCLE}
    for d in detections:
        assert d.frame_index == 3
        assert d.timestamp == 0.3


def test_no_boxes_returns_empty_list():
    detections = parse_detection_results(FakeResult(None), frame_index=0, timestamp=0.0)
    assert detections == []


def test_bbox_and_confidence_are_preserved():
    boxes = FakeBoxes(xyxy=[[1.0, 2.0, 3.0, 4.0]], cls=[2.0], conf=[0.77])
    detections = parse_detection_results(FakeResult(boxes), frame_index=1, timestamp=0.1)

    assert len(detections) == 1
    d = detections[0]
    assert d.bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert d.confidence == 0.77
    assert d.class_name == "car"
    assert d.centroid == (2.0, 3.0)
