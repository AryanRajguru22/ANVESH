from anvesh.perception.tracking import parse_tracked_results
from anvesh.storage.schemas import VehicleClass
from tests.perception.fakes import FakeBoxes, FakeResult


def test_confirmed_track_ids_produce_tracked_detections():
    boxes = FakeBoxes(
        xyxy=[[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]],
        cls=[2.0, 7.0],
        conf=[0.9, 0.8],
        ids=[1.0, 2.0],
    )
    raw, tracked = parse_tracked_results(FakeResult(boxes), frame_index=0, timestamp=0.0)

    assert len(raw) == 2
    assert len(tracked) == 2
    assert {t.track_id for t in tracked} == {"1", "2"}
    assert {t.detection.vehicle_class for t in tracked} == {VehicleClass.CAR, VehicleClass.TRUCK}


def test_unconfirmed_detection_kept_as_raw_but_not_tracked():
    boxes = FakeBoxes(
        xyxy=[[10.0, 10.0, 20.0, 20.0]],
        cls=[2.0],
        conf=[0.5],
        ids=[None],
    )
    raw, tracked = parse_tracked_results(FakeResult(boxes), frame_index=0, timestamp=0.0)

    assert len(raw) == 1
    assert len(tracked) == 0


def test_entirely_unconfirmed_frame_has_no_id_tensor_at_all():
    # Real Ultralytics behavior: when nothing in the frame has a confirmed
    # track yet, `boxes.id` is None entirely (not a per-box None list).
    boxes = FakeBoxes(
        xyxy=[[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]],
        cls=[2.0, 3.0],
        conf=[0.5, 0.6],
        ids=None,
    )
    raw, tracked = parse_tracked_results(FakeResult(boxes), frame_index=0, timestamp=0.0)

    assert len(raw) == 2
    assert len(tracked) == 0


def test_irrelevant_class_dropped_from_both_raw_and_tracked():
    boxes = FakeBoxes(
        xyxy=[[0.0, 0.0, 5.0, 5.0]],
        cls=[0.0],  # person
        conf=[0.9],
        ids=[1.0],
    )
    raw, tracked = parse_tracked_results(FakeResult(boxes), frame_index=0, timestamp=0.0)

    assert raw == []
    assert tracked == []


def test_no_boxes_returns_empty_lists():
    raw, tracked = parse_tracked_results(FakeResult(None), frame_index=0, timestamp=0.0)
    assert raw == []
    assert tracked == []
