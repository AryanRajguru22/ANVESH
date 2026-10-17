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
    attributes avoid the trap entirely.

    TIMING (streaming, M-A). `timestamp` is STREAM TIME in seconds relative
    to the start of the source -- the clock the whole world model (tracks,
    windows, evidence) runs on. For a recorded file it is `index / fps`
    (deterministic, so batch and stream agree); for a live source it is the
    monotonic capture clock relative to the first frame. The extra fields are
    for observability and never feed the world model:
      - `capture_wall_s`: process-monotonic wall clock when the frame was
        decoded (latency accounting only; None if the source does not record it)
      - `decode_s`: seconds spent decoding this frame
      - `epoch`: connection/session epoch -- increments each time a live
        source reconnects; a new epoch means tracker IDs are unrelated to the
        previous epoch's and a new identity session must begin
      - `gap_before`: number of frames the SOURCE knows it skipped before this
        one (network stalls, source-side skips); 0 for an intact file
    """

    __slots__ = ("index", "timestamp", "image", "capture_wall_s", "decode_s", "epoch", "gap_before")

    def __init__(
        self,
        index: int,
        timestamp: float,
        image,
        capture_wall_s: float = None,
        decode_s: float = None,
        epoch: int = 0,
        gap_before: int = 0,
    ) -> None:
        self.index = index
        self.timestamp = timestamp
        self.image = image
        self.capture_wall_s = capture_wall_s
        self.decode_s = decode_s
        self.epoch = epoch
        self.gap_before = gap_before


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

    @property
    def ground_point(self) -> tuple:
        """The vehicle's estimated ground-contact point: bottom-center of
        the bounding box (`bbox.x + bbox.width / 2`, `bbox.y + bbox.height`),
        in the same source-frame pixel coordinates as `bbox_xyxy`.

        WHY BOTTOM-CENTER FOR ROAD/LANE REASONING:
        `centroid` (the box center) sits above the road plane for any
        vehicle taller than it is wide in screen space -- a bus or truck's
        box center can be well above the pavement, and how far above scales
        with the vehicle's apparent height, which itself varies with
        distance from the camera. Bottom-center is the point on the box
        closest to where the vehicle's wheels actually meet the road, which
        is the physically meaningful point for anything that reasons about
        *where on the road* a vehicle is -- e.g. a future lane-polygon test
        (lane boundaries exist on the road plane, not floating above it) or
        a homography/world-space projection, which assumes the projected
        point lies on the road plane (see
        `anvesh.world.calibration.CalibrationProfile`).

        WHY `centroid` REMAINS:
        `centroid` is unchanged and still what `TrackManager`/
        `schema_conversion` use for trajectory and speed today (see
        `anvesh.perception.track_manager`). This property is purely
        additive, not a replacement -- switching speed/trajectory math to
        `ground_point` is a separate, not-yet-made decision (see
        `ANVESH_LANE_SPEED_FORENSIC_AUDIT.md`, which scopes this property to
        Commit 1 only: no lane assignment, no speed change).

        LIMITATIONS (explicit, not silently assumed away):
        - Still a 2D image-space point, not a measured 3D contact point --
          occlusion (the true bottom edge hidden behind another vehicle or
          object) biases this point upward, same as it would bias any
          other box-derived proxy.
        - For a box touching the bottom edge of the frame, this point is
          the frame boundary, not necessarily the vehicle's true contact
          point (the vehicle may extend further out of frame).
        - Computed purely from the 2D box; it does not know vehicle
          orientation, so for a vehicle at a steep side angle the
          bottom-center of its box is not necessarily under its true
          centerline.
        - Performs no lane assignment, calibration, or coordinate
          transform itself -- it only derives a pixel point from an
          existing pixel box.
        """
        x1, _y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)


@dataclass(frozen=True)
class TrackedDetection:
    """A RawDetection that has been associated with a persistent track ID."""

    detection: RawDetection
    track_id: str

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be a non-empty string")
