import pytest

from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import VehicleClass


def _detection(bbox_xyxy: tuple) -> RawDetection:
    return RawDetection(
        frame_index=0,
        timestamp=0.0,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.9,
        bbox_xyxy=bbox_xyxy,
    )


def test_ground_point_is_bottom_center_of_normal_box():
    # x1=10, y1=20, x2=30, y2=60 -> width=20, height=40
    detection = _detection((10.0, 20.0, 30.0, 60.0))

    assert detection.ground_point == (20.0, 60.0)


def test_ground_point_bottom_center_differs_from_centroid_for_tall_box():
    # A tall box (bus/truck-like aspect ratio): centroid sits well above
    # ground_point, which is exactly the bias ground_point exists to avoid.
    detection = _detection((10.0, 0.0, 30.0, 200.0))

    assert detection.centroid == (20.0, 100.0)
    assert detection.ground_point == (20.0, 200.0)
    assert detection.ground_point[1] > detection.centroid[1]


def test_ground_point_supports_fractional_coordinates():
    detection = _detection((10.25, 20.5, 30.75, 61.1))

    x, y = detection.ground_point
    assert x == pytest.approx((10.25 + 30.75) / 2.0)
    assert y == pytest.approx(61.1)


def test_ground_point_on_zero_area_box():
    # x1 == x2 and y1 == y2 is a degenerate but currently VALID box under
    # RawDetection's own __post_init__ (it only rejects x2 < x1 / y2 < y1,
    # not equality) -- ground_point must not raise or misbehave for it.
    detection = _detection((15.0, 40.0, 15.0, 40.0))

    assert detection.ground_point == (15.0, 40.0)


def test_invalid_box_still_rejected_before_ground_point_is_reachable():
    # Existing ANVESH convention (RawDetection.__post_init__): a box with
    # x2 < x1 or y2 < y1 is rejected at construction time, not silently
    # accepted and left for a downstream property to misinterpret.
    with pytest.raises(ValueError):
        _detection((30.0, 20.0, 10.0, 60.0))  # x2 < x1

    with pytest.raises(ValueError):
        _detection((10.0, 60.0, 30.0, 20.0))  # y2 < y1


def test_centroid_behavior_is_unchanged():
    detection = _detection((10.0, 20.0, 30.0, 60.0))

    assert detection.centroid == (20.0, 40.0)
