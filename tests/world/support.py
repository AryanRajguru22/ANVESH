"""Shared deterministic calibration fixtures for tests (M2).

`GROUND_TRUTH_HOMOGRAPHY` is a fabricated (not measured) 3x3 projective
transform used only to generate internally-consistent synthetic
image<->world correspondences for tests. It does not correspond to any
real camera or site survey -- see the "quality rule" in the M2 report
about not claiming real-world accuracy from synthetic fixtures.
"""

from __future__ import annotations

import numpy as np

from anvesh.world.calibration import CalibrationPoint

GROUND_TRUTH_HOMOGRAPHY = np.array(
    [
        [0.05, 0.0, -2.0],
        [0.0, 0.04, -1.0],
        [0.0002, 0.0001, 1.0],
    ]
)


def apply_ground_truth(image_xy: tuple) -> tuple:
    x, y = image_xy
    vec = GROUND_TRUTH_HOMOGRAPHY @ np.array([x, y, 1.0])
    return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


def make_reference_points() -> list:
    """5 well-conditioned (non-collinear) synthetic correspondences."""
    image_points = [(10.0, 10.0), (500.0, 10.0), (500.0, 400.0), (10.0, 400.0), (250.0, 200.0)]
    return [CalibrationPoint(image_xy=p, world_xy=apply_ground_truth(p)) for p in image_points]


def make_collinear_reference_points() -> list:
    """4 degenerate (collinear) correspondences -- must fail to calibrate."""
    image_points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    return [CalibrationPoint(image_xy=p, world_xy=(p[0] / 10.0, 0.0)) for p in image_points]
