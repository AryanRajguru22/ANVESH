"""Internal perception types shared across the M1 pipeline stages.

These are intentionally distinct from `anvesh.storage.schemas` -- they are
the pipeline's own working representation (raw detections, tracked
detections) before `schema_conversion.py` turns them into the frozen
Part 5 contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.storage.schemas import VehicleClass


class Frame:
    """One decoded video frame. Not a dataclass: holding a numpy array in a
    dataclass makes the generated __eq__ compare arrays elementwise, which
    raises when Python tries to reduce that to a single bool -- plain
    attributes avoid the trap entirely."""

    __slots__ = ("index", "timestamp", "image")

    def __init__(self, index: int, timestamp: float, image) -> None:
        self.index = index
        self.timestamp = timestamp
        self.image = image


@dataclass(frozen=True)
class RawDetection:
    """One detector output for one frame -- module 3's raw result.

    `bbox_xyxy` is in source-frame pixel coordinates. No calibration is
    applied anywhere in this type.
    """

    frame_index: int
    timestamp: float
    class_name: str
    vehicle_class: VehicleClass
    confidence: float
    bbox_xyxy: tuple

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be within [0, 1], got {self.confidence}")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must be an (x1, y1, x2, y2) tuple")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox_xyxy must have x2 >= x1 and y2 >= y1")

    @property
    def centroid(self) -> tuple:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrackedDetection:
    """A RawDetection that has been associated with a persistent track ID."""

    detection: RawDetection
    track_id: str

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be a non-empty string")
