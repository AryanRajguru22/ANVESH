"""Camera calibration (blueprint Part 3, module 2): image <-> world/road
coordinate transformation via a planar-road homography.

V1 scope, per blueprint Part 4/Part 16: calibration is MANUAL. Reference
point correspondences (an image pixel and the corresponding measured
road-plane position, in meters) are supplied by the caller -- e.g. from a
one-time site survey. No automatic camera calibration (checkerboard
detection, self-calibration, bundle adjustment, etc.) is implemented here.

A camera with no valid reference points has status UNCALIBRATED: its
measurements must never be presented as metric. `anvesh.perception.motion`
and `anvesh.perception.schema_conversion` both branch on this status so an
uncalibrated camera's output stays explicitly image-space (pixels), never
mislabeled as world-space (metres). See `docs/ANVESH_V1_IMPLEMENTATION_BLUEPRINT.md`
Part 3 module 2 and Part 16 ("camera calibration" risk).

`CalibrationProfile` is not itself a Part 5 schema entity -- Part 5 only
references `Camera.calibration_profile_id` as an opaque foreign key and
never defines this table. This module owns the concrete type, per Part 3's
module table ("world/calibration.py" -> `CalibrationProfile`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2


class CalibrationStatus(str, Enum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"


@dataclass(frozen=True)
class CalibrationPoint:
    """One manually-surveyed image<->world correspondence used to fit the homography.

    `world_xy` is in metres, on the (assumed planar) road surface: x runs
    along the corridor's direction of travel, y runs across it. This is a
    local, per-corridor convention -- not a georeferenced coordinate
    system (blueprint Part 2 explicitly excludes GIS/city-scale modeling).
    """

    image_xy: tuple
    world_xy: tuple

    def __post_init__(self) -> None:
        if len(self.image_xy) != 2:
            raise ValueError("image_xy must be an (x, y) pair")
        if len(self.world_xy) != 2:
            raise ValueError("world_xy must be an (x, y) pair, in metres")


@dataclass(frozen=True)
class CalibrationProfile:
    """Image <-> world/road-plane transform for one camera."""

    calibration_profile_id: str
    camera_id: str
    status: CalibrationStatus
    reference_points: tuple
    homography: object  # 3x3 numpy array, image -> world; None if uncalibrated
    inverse_homography: object  # 3x3 numpy array, world -> image; None if uncalibrated
    reprojection_error_m: object  # float; mean fit error on the reference points, or None if uncalibrated
    units: str = "meters"
    method: str = "manual_planar_homography_v1"

    def __post_init__(self) -> None:
        if not self.calibration_profile_id:
            raise ValueError("calibration_profile_id must be a non-empty string")
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        object.__setattr__(self, "status", CalibrationStatus(self.status))

        if self.status == CalibrationStatus.CALIBRATED:
            if self.homography is None or self.inverse_homography is None:
                raise ValueError("a CALIBRATED profile must carry both homography and inverse_homography")
            if len(self.reference_points) < 4:
                raise ValueError("a CALIBRATED profile requires at least 4 reference points")
        else:
            if self.homography is not None or self.inverse_homography is not None:
                raise ValueError("an UNCALIBRATED profile must not carry a homography")
            if self.reprojection_error_m is not None:
                raise ValueError("an UNCALIBRATED profile has no reprojection error to report")

    @property
    def is_calibrated(self) -> bool:
        return self.status == CalibrationStatus.CALIBRATED

    def image_to_world(self, image_xy: tuple) -> tuple:
        """Project an image-space pixel onto the road plane, in metres.

        Raises if this profile is UNCALIBRATED -- callers must check
        `is_calibrated` (or catch this) and fall back to image-space-only
        handling, never silently substitute pixels for metres.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                f"camera '{self.camera_id}' has no valid calibration -- "
                "cannot produce a world-space position from this profile"
            )
        return _apply_homography(self.homography, image_xy)

    def world_to_image(self, world_xy: tuple) -> tuple:
        if not self.is_calibrated:
            raise RuntimeError(f"camera '{self.camera_id}' has no valid calibration")
        return _apply_homography(self.inverse_homography, world_xy)


def _apply_homography(h, point: tuple) -> tuple:
    x, y = point
    vec = h @ np.array([x, y, 1.0])
    if abs(vec[2]) < 1e-12:
        raise ValueError("degenerate homography application: w-component near zero")
    return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


def fit_planar_homography(
    calibration_profile_id: str,
    camera_id: str,
    reference_points: list,
) -> CalibrationProfile:
    """Fit a planar-road homography from >=4 manually-surveyed correspondences.

    Raises `ValueError` for degenerate input (fewer than 4 points,
    collinear points, or a numerically singular fit) instead of silently
    returning an unreliable transform -- `cv2.findHomography` itself
    returns `None` for a degenerate/collinear point set, which this
    function turns into an explicit, documented error.
    """
    if len(reference_points) < 4:
        raise ValueError(f"at least 4 reference points are required, got {len(reference_points)}")

    image_pts = np.array([p.image_xy for p in reference_points], dtype=np.float64)
    world_pts = np.array([p.world_xy for p in reference_points], dtype=np.float64)

    homography, _mask = cv2.findHomography(image_pts, world_pts, method=0)
    if homography is None:
        raise ValueError(
            "failed to fit a homography from the given reference points "
            "(degenerate/collinear configuration?)"
        )

    try:
        inverse_homography = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fitted homography is singular and cannot be inverted") from exc

    reprojection_error_m = _mean_reprojection_error(homography, image_pts, world_pts)

    return CalibrationProfile(
        calibration_profile_id=calibration_profile_id,
        camera_id=camera_id,
        status=CalibrationStatus.CALIBRATED,
        reference_points=tuple(reference_points),
        homography=homography,
        inverse_homography=inverse_homography,
        reprojection_error_m=reprojection_error_m,
    )


def _mean_reprojection_error(homography, image_pts: np.ndarray, world_pts: np.ndarray) -> float:
    errors = []
    for img_pt, world_pt in zip(image_pts, world_pts):
        projected = _apply_homography(homography, (float(img_pt[0]), float(img_pt[1])))
        errors.append(float(np.hypot(projected[0] - world_pt[0], projected[1] - world_pt[1])))
    return sum(errors) / len(errors)


def uncalibrated_profile(calibration_profile_id: str, camera_id: str) -> CalibrationProfile:
    """The explicit "no calibration available" profile -- image-space only."""
    return CalibrationProfile(
        calibration_profile_id=calibration_profile_id,
        camera_id=camera_id,
        status=CalibrationStatus.UNCALIBRATED,
        reference_points=(),
        homography=None,
        inverse_homography=None,
        reprojection_error_m=None,
    )
