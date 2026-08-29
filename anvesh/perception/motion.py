"""Motion measurement (blueprint Part 3, module 5 -- image-space only): the
simplest defensible displacement/velocity measurement computable without
camera calibration.

This intentionally stops short of claiming road speed: without a
CalibrationProfile (`world/calibration.py`, milestone M2) there is no
homography to convert pixel distances into metres, so every value produced
here is explicitly pixels-per-second, never metres-per-second. See the M1
report's "Limitations" section, and `schema_conversion.py`'s module
docstring, for how this interacts with
`anvesh.storage.schemas.VehicleTrack.speed_estimate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageSpaceSpeedEstimate:
    """A pixels-per-second speed estimate -- explicitly image-space, never metric."""

    speed_px_per_s: float
    error_px_per_s: float
    sample_count: int
    space: str = "image"

    def __post_init__(self) -> None:
        if self.space != "image":
            raise ValueError("ImageSpaceSpeedEstimate.space must be 'image'")
        if self.speed_px_per_s < 0:
            raise ValueError("speed_px_per_s must be >= 0")
        if self.error_px_per_s < 0:
            raise ValueError("error_px_per_s must be >= 0")
        if self.sample_count < 0:
            raise ValueError("sample_count must be >= 0")


def compute_image_space_speed(trajectory: list) -> ImageSpaceSpeedEstimate:
    """Mean image-space speed (px/s) across consecutive trajectory samples.

    `trajectory` is an ordered list of (x, y, timestamp) image-space
    samples, e.g. `TrackState.trajectory`. A trajectory with fewer than 2
    samples, or with no positive-duration consecutive pair, has no
    measurable motion yet and returns a zero estimate rather than raising.
    """
    if len(trajectory) < 2:
        return ImageSpaceSpeedEstimate(speed_px_per_s=0.0, error_px_per_s=0.0, sample_count=len(trajectory))

    per_step_speeds = []
    for (x0, y0, t0), (x1, y1, t1) in zip(trajectory, trajectory[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        distance = math.hypot(x1 - x0, y1 - y0)
        per_step_speeds.append(distance / dt)

    if not per_step_speeds:
        return ImageSpaceSpeedEstimate(speed_px_per_s=0.0, error_px_per_s=0.0, sample_count=len(trajectory))

    mean_speed = sum(per_step_speeds) / len(per_step_speeds)
    if len(per_step_speeds) > 1:
        variance = sum((s - mean_speed) ** 2 for s in per_step_speeds) / (len(per_step_speeds) - 1)
        error = math.sqrt(variance)
    else:
        error = 0.0

    return ImageSpaceSpeedEstimate(
        speed_px_per_s=mean_speed,
        error_px_per_s=error,
        sample_count=len(per_step_speeds),
    )
