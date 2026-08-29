"""Vehicle detection (blueprint Part 3, module 3): per-frame detection with
class + confidence, using Ultralytics YOLO, pretrained (blueprint Part 4 --
fine-tuning is explicitly later).

Detections are filtered to ANVESH's road-user taxonomy
(`anvesh.storage.schemas.VehicleClass`). The pretrained COCO model does not
cover every V1 taxonomy class -- notably `auto_rickshaw` has no COCO
equivalent and will never be produced until a fine-tuned model exists.
"""

from __future__ import annotations

from typing import Protocol

from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import VehicleClass

# Standard COCO class indices, restricted to ANVESH's road-user taxonomy.
COCO_TO_VEHICLE_CLASS: dict = {
    1: VehicleClass.BICYCLE,
    2: VehicleClass.CAR,
    3: VehicleClass.MOTORCYCLE,
    5: VehicleClass.BUS,
    7: VehicleClass.TRUCK,
}

COCO_CLASS_NAMES: dict = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class Detector(Protocol):
    def detect(self, image, frame_index: int, timestamp: float) -> list: ...


def parse_detection_results(result, frame_index: int, timestamp: float) -> list:
    """Convert one Ultralytics `Results` object into `RawDetection`s.

    Pure/standalone so it is testable against a lightweight fake `result`
    without loading a real YOLO model (see tests/perception/test_detection.py).
    """
    detections: list = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.tolist()
    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist()

    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        cls_id_int = int(cls_id)
        vehicle_class = COCO_TO_VEHICLE_CLASS.get(cls_id_int)
        if vehicle_class is None:
            continue
        detections.append(
            RawDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                class_name=COCO_CLASS_NAMES.get(cls_id_int, str(cls_id_int)),
                vehicle_class=vehicle_class,
                confidence=float(conf),
                bbox_xyxy=tuple(float(v) for v in box),
            )
        )
    return detections


class YoloDetector:
    """Ultralytics YOLO, pretrained, filtered to the relevant vehicle classes.

    Detection-only (no tracking) -- kept as its own, independently usable
    stage per blueprint Part 3's module 3 / module 4 split, even though the
    M1 pipeline itself uses `tracking.ByteTrackYoloTracker` (which performs
    detection internally as part of Ultralytics' `model.track()` call).
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25) -> None:
        from ultralytics import YOLO  # lazy import: importing this module should not require torch/ultralytics

        self._model = YOLO(model_name)
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._classes = sorted(COCO_TO_VEHICLE_CLASS)

    @property
    def model_id(self) -> str:
        return self._model_name

    def detect(self, image, frame_index: int, timestamp: float) -> list:
        results = self._model.predict(
            image,
            conf=self._confidence_threshold,
            classes=self._classes,
            verbose=False,
        )
        return parse_detection_results(results[0], frame_index, timestamp)
