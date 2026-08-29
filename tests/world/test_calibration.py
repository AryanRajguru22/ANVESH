import math

import numpy as np
import pytest

from anvesh.world.calibration import (
    CalibrationPoint,
    CalibrationProfile,
    CalibrationStatus,
    fit_planar_homography,
    uncalibrated_profile,
)
from tests.world.support import apply_ground_truth, make_collinear_reference_points, make_reference_points


def test_fit_produces_calibrated_profile():
    profile = fit_planar_homography("calib-01", "cam-A", make_reference_points())
    assert profile.status == CalibrationStatus.CALIBRATED
    assert profile.is_calibrated is True
    assert profile.camera_id == "cam-A"
    assert profile.homography is not None
    assert profile.inverse_homography is not None
    assert profile.units == "meters"


def test_image_to_world_matches_ground_truth_on_held_out_point():
    """Known-point calibration accuracy: a point NOT used for fitting must
    still project close to the ground-truth transform, not just the fitted
    points themselves."""
    profile = fit_planar_homography("calib-01", "cam-A", make_reference_points())

    held_out_image_point = (300.0, 150.0)
    expected_world = apply_ground_truth(held_out_image_point)
    got_world = profile.image_to_world(held_out_image_point)

    assert got_world[0] == pytest.approx(expected_world[0], abs=1e-4)
    assert got_world[1] == pytest.approx(expected_world[1], abs=1e-4)


def test_reprojection_error_is_near_zero_for_exact_synthetic_points():
    profile = fit_planar_homography("calib-01", "cam-A", make_reference_points())
    assert profile.reprojection_error_m < 1e-6


def test_world_to_image_is_the_inverse_transform():
    profile = fit_planar_homography("calib-01", "cam-A", make_reference_points())

    original_image_point = (123.0, 77.0)
    world_point = profile.image_to_world(original_image_point)
    recovered_image_point = profile.world_to_image(world_point)

    assert recovered_image_point[0] == pytest.approx(original_image_point[0], abs=1e-6)
    assert recovered_image_point[1] == pytest.approx(original_image_point[1], abs=1e-6)


def test_fitting_with_fewer_than_four_points_raises():
    points = make_reference_points()[:3]
    with pytest.raises(ValueError):
        fit_planar_homography("calib-01", "cam-A", points)


def test_fitting_with_collinear_points_raises():
    with pytest.raises(ValueError):
        fit_planar_homography("calib-01", "cam-A", make_collinear_reference_points())


def test_uncalibrated_profile_rejects_projection():
    profile = uncalibrated_profile("calib-none", "cam-B")
    assert profile.status == CalibrationStatus.UNCALIBRATED
    assert profile.is_calibrated is False
    assert profile.homography is None
    assert profile.reprojection_error_m is None

    with pytest.raises(RuntimeError):
        profile.image_to_world((10.0, 10.0))
    with pytest.raises(RuntimeError):
        profile.world_to_image((1.0, 1.0))


def test_calibrated_profile_requires_homography_and_reference_points():
    with pytest.raises(ValueError):
        CalibrationProfile(
            calibration_profile_id="bad",
            camera_id="cam-A",
            status=CalibrationStatus.CALIBRATED,
            reference_points=(),
            homography=None,
            inverse_homography=None,
            reprojection_error_m=None,
        )


def test_uncalibrated_profile_must_not_carry_homography():
    with pytest.raises(ValueError):
        CalibrationProfile(
            calibration_profile_id="bad",
            camera_id="cam-A",
            status=CalibrationStatus.UNCALIBRATED,
            reference_points=(),
            homography=np.eye(3),
            inverse_homography=np.eye(3),
            reprojection_error_m=None,
        )


def test_calibration_point_requires_xy_pairs():
    with pytest.raises(ValueError):
        CalibrationPoint(image_xy=(1.0,), world_xy=(1.0, 2.0))
    with pytest.raises(ValueError):
        CalibrationPoint(image_xy=(1.0, 2.0), world_xy=(1.0, 2.0, 3.0))


def test_empty_ids_are_rejected():
    with pytest.raises(ValueError):
        uncalibrated_profile("", "cam-A")
    with pytest.raises(ValueError):
        uncalibrated_profile("calib-none", "")
