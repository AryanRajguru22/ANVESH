"""Vehicle detection (blueprint Part 3, module 3): per-frame detection with
class + confidence, using Ultralytics YOLO.

Detections are filtered to ANVESH's road-user taxonomy
(`anvesh.storage.schemas.VehicleClass`). The pretrained COCO model does not
cover every V1 taxonomy class -- notably `auto_rickshaw` has no COCO
equivalent and is only produced by a checkpoint trained with that class.
Class indices are resolved per checkpoint from `model.names` (`ClassMap`), so
both the COCO model and an ANVESH fine-tuned checkpoint work.
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


class ClassMap:
    """Model class index -> ANVESH `VehicleClass`, derived from a model's own `names`.

    The COCO model has 80 classes (car=2, bus=5, ...) while an ANVESH fine-tuned
    model has 5 (car=1, ...). Indices therefore differ per checkpoint, so the
    mapping is resolved by class NAME from `model.names`, never by hard-coded
    index. `class_ids` is the list passed to Ultralytics' `classes=` filter.
    """

    def __init__(self, id_to_vehicle_class: dict, id_to_name: dict) -> None:
        self.id_to_vehicle_class = dict(id_to_vehicle_class)
        self.id_to_name = dict(id_to_name)
        self.class_ids = sorted(self.id_to_vehicle_class)

    @classmethod
    def coco(cls) -> "ClassMap":
        return cls(COCO_TO_VEHICLE_CLASS, COCO_CLASS_NAMES)

    @classmethod
    def from_model_names(cls, names) -> "ClassMap":
        """Build from an Ultralytics `model.names` (dict id->name, or a list).

        Class names are matched case-insensitively to `VehicleClass` values
        (`car`, `motorcycle`, `bus`, `truck`, `bicycle`, `auto_rickshaw`). Names
        that are not road-user classes (COCO's `person`, `traffic light`, ...)
        are ignored. A model with NO recognisable vehicle class is rejected: it
        cannot feed ANVESH, and silently producing zero detections would look
        like an empty road.
        """
        items = names.items() if isinstance(names, dict) else enumerate(names)
        known = {vc.value: vc for vc in VehicleClass}
        id_to_vc: dict = {}
        id_to_name: dict = {}
        for idx, name in items:
            vc = known.get(str(name).strip().lower().replace(" ", "_"))
            if vc is not None:
                id_to_vc[int(idx)] = vc
                id_to_name[int(idx)] = str(name)
        if not id_to_vc:
            raise ValueError(
                f"model has no recognised ANVESH vehicle classes in its names {names!r}; expected some of {sorted(known)}")
        return cls(id_to_vc, id_to_name)


def parse_detection_results(result, frame_index: int, timestamp: float, class_map: ClassMap = None) -> list:
    """Convert one Ultralytics `Results` object into `RawDetection`s.

    `class_map` defaults to the COCO mapping (the original behaviour). Pure/standalone
    so it is testable against a lightweight fake `result` without loading a real YOLO
    model (see tests/perception/test_detection.py).
    """
    class_map = class_map or ClassMap.coco()
    detections: list = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.tolist()
    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist()

    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        cls_id_int = int(cls_id)
        vehicle_class = class_map.id_to_vehicle_class.get(cls_id_int)
        if vehicle_class is None:
            continue
        detections.append(
            RawDetection(
                frame_index=frame_index,
                timestamp=timestamp,
                class_name=class_map.id_to_name.get(cls_id_int, str(cls_id_int)),
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

        from anvesh.perception.weights import require_weights_file

        require_weights_file(model_name)  # never let Ultralytics download a missing recognised name
        self._model = YOLO(model_name)
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._class_map = ClassMap.from_model_names(self._model.names)
        self._classes = self._class_map.class_ids

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
        return parse_detection_results(results[0], frame_index, timestamp, self._class_map)
