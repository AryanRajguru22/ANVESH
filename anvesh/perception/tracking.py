"""Multi-object tracking (blueprint Part 3, module 4): frame-to-frame
association, persistent per-camera track IDs.

Uses Ultralytics' built-in ByteTrack (`bytetrack.yaml`), per blueprint
Part 4: "simplest tracker that's good enough for V1." No custom
association/tracking algorithm is implemented here.
"""

from __future__ import annotations

from typing import Protocol

from anvesh.perception.detection import COCO_CLASS_NAMES, COCO_TO_VEHICLE_CLASS
from anvesh.perception.types import RawDetection, TrackedDetection


class Tracker(Protocol):
    def update(self, image, frame_index: int, timestamp: float) -> tuple: ...

    def reset(self) -> None: ...


def parse_tracked_results(result, frame_index: int, timestamp: float) -> tuple:
    """Convert one Ultralytics `Results` object (from `model.track()`) into
    `(raw_detections, tracked_detections)`.

    ByteTrack reports `track_id=None` for a detection that has not yet
    accumulated enough consecutive hits to be confirmed as a track; such
    detections are kept as raw detections (they still count as an
    observation for that frame) but are not fed into track management
    until a persistent ID exists.

    Pure/standalone so it is testable against a lightweight fake `result`
    without loading a real YOLO model or ByteTrack
    (see tests/perception/test_tracking.py).
    """
    raw_detections: list = []
    tracked_detections: list = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return raw_detections, tracked_detections

    xyxy = boxes.xyxy.tolist()
    cls_ids = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(xyxy)

    for box, cls_id, conf, track_id in zip(xyxy, cls_ids, confs, track_ids):
        cls_id_int = int(cls_id)
        vehicle_class = COCO_TO_VEHICLE_CLASS.get(cls_id_int)
        if vehicle_class is None:
            continue

        detection = RawDetection(
            frame_index=frame_index,
            timestamp=timestamp,
            class_name=COCO_CLASS_NAMES.get(cls_id_int, str(cls_id_int)),
            vehicle_class=vehicle_class,
            confidence=float(conf),
            bbox_xyxy=tuple(float(v) for v in box),
        )
        raw_detections.append(detection)

        if track_id is not None:
            tracked_detections.append(TrackedDetection(detection=detection, track_id=str(int(track_id))))

    return raw_detections, tracked_detections


class ByteTrackYoloTracker:
    """Ultralytics YOLO + built-in ByteTrack, run frame-by-frame with
    persisted tracker state (`persist=True`)."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO  # lazy import: importing this module should not require torch/ultralytics

        self._model = YOLO(model_name)
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._tracker_config = tracker_config
        self._classes = sorted(COCO_TO_VEHICLE_CLASS)

    @property
    def model_id(self) -> str:
        return self._model_name

    def reset(self) -> None:
        """Drop persisted tracker state so the next `update()` starts a
        fresh track-ID sequence (documented Ultralytics reset pattern)."""
        predictor = getattr(self._model, "predictor", None)
        if predictor is not None:
            predictor.trackers = None

    def update(self, image, frame_index: int, timestamp: float) -> tuple:
        results = self._model.track(
            image,
            persist=True,
            tracker=self._tracker_config,
            classes=self._classes,
            conf=self._confidence_threshold,
            verbose=False,
        )
        return parse_tracked_results(results[0], frame_index, timestamp)
