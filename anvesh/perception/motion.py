"""Motion measurement (blueprint Part 3, module 5): displacement/velocity
from a trajectory, kept strictly separated by coordinate space.

Two parallel, non-interchangeable estimate types exist on purpose:

- `ImageSpaceSpeedEstimate` (pixels/second) -- always computable, requires
  no calibration. This was the *only* option in M1.
- `WorldSpaceSpeedEstimate` (metres/second) -- only computable once a
  camera has a valid `CalibrationProfile` (`world/calibration.py`, M2) and
  its trajectory has already been projected to the road plane via
  `CalibrationProfile.image_to_world`.

Nothing in this module ever converts one into the other, and no function
here labels a pixel displacement as metric. See
`anvesh/perception/schema_conversion.py` for how a track's
`anvesh.storage.schemas.MotionSpace` is chosen between the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _step_speeds(trajectory: list) -> list:
    speeds = []
    for (x0, y0, t0), (x1, y1, t1) in zip(trajectory, trajectory[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        distance = math.hypot(x1 - x0, y1 - y0)
        speeds.append(distance / dt)
    return speeds


def _mean_and_error(speeds: list) -> tuple:
    if not speeds:
        return 0.0, 0.0
    mean_speed = sum(speeds) / len(speeds)
    if len(speeds) > 1:
        variance = sum((s - mean_speed) ** 2 for s in speeds) / (len(speeds) - 1)
        error = math.sqrt(variance)
    else:
        error = 0.0
    return mean_speed, error


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


@dataclass(frozen=True)
class WorldSpaceSpeedEstimate:
    """A metres-per-second speed estimate -- only valid once a trajectory
    has been projected to the road plane by a CALIBRATED CalibrationProfile."""

    speed_m_per_s: float
    error_m_per_s: float
    sample_count: int
    space: str = "world"

    def __post_init__(self) -> None:
        if self.space != "world":
            raise ValueError("WorldSpaceSpeedEstimate.space must be 'world'")
        if self.speed_m_per_s < 0:
            raise ValueError("speed_m_per_s must be >= 0")
        if self.error_m_per_s < 0:
            raise ValueError("error_m_per_s must be >= 0")
        if self.sample_count < 0:
            raise ValueError("sample_count must be >= 0")


def compute_image_space_speed(trajectory: list) -> ImageSpaceSpeedEstimate:
    """Mean image-space speed (px/s) across consecutive trajectory samples.

    `trajectory` is an ordered list of (x, y, timestamp) IMAGE-SPACE
    (pixel) samples. A trajectory with fewer than 2 samples, or with no
    positive-duration consecutive pair, has no measurable motion yet and
    returns a zero estimate rather than raising.
    """
    if len(trajectory) < 2:
        return ImageSpaceSpeedEstimate(speed_px_per_s=0.0, error_px_per_s=0.0, sample_count=len(trajectory))

    speeds = _step_speeds(trajectory)
    if not speeds:
        return ImageSpaceSpeedEstimate(speed_px_per_s=0.0, error_px_per_s=0.0, sample_count=len(trajectory))

    mean_speed, error = _mean_and_error(speeds)
    return ImageSpaceSpeedEstimate(speed_px_per_s=mean_speed, error_px_per_s=error, sample_count=len(speeds))


def compute_world_space_speed(trajectory_world: list) -> WorldSpaceSpeedEstimate:
    """Mean world-space (road-plane) speed (m/s) across consecutive samples.

    `trajectory_world` is an ordered list of (world_x, world_y, timestamp)
    METRE samples -- the caller is responsible for having already run each
    image-space point through `CalibrationProfile.image_to_world` first;
    this function does not calibrate anything itself, it only measures
    speed given points that are already in metres.
    """
    if len(trajectory_world) < 2:
        return WorldSpaceSpeedEstimate(speed_m_per_s=0.0, error_m_per_s=0.0, sample_count=len(trajectory_world))

    speeds = _step_speeds(trajectory_world)
    if not speeds:
        return WorldSpaceSpeedEstimate(speed_m_per_s=0.0, error_m_per_s=0.0, sample_count=len(trajectory_world))

    mean_speed, error = _mean_and_error(speeds)
    return WorldSpaceSpeedEstimate(speed_m_per_s=mean_speed, error_m_per_s=error, sample_count=len(speeds))
