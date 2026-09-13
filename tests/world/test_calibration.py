import math

import numpy as np
import pytest

from anvesh.world.calibration import (
    CalibrationPoint,
    CalibrationProfile,
    CalibrationProvenance,
    CalibrationStatus,
    MetricValidationStatus,
    fit_planar_homography,
    from_cityflow_homography,
    load_cityflow_calibration,
    parse_cityflow_calibration_text,
    uncalibrated_profile,
)
from tests.world.support import apply_ground_truth, make_collinear_reference_points, make_reference_points

# Exact contents of the real CityFlowV2 S02 calibration files, copied here
# so these tests never depend on the (16GB, not always mounted) dataset
# being present -- requirement 9, Commit 5. Verbatim from:
#   E:\traffic dataset\validation\S02\c006\calibration.txt
#   E:\traffic dataset\validation\S02\c009\calibration.txt
C006_CALIBRATION_TEXT = (
    "Homography matrix: -39.660220243283760 15.592282221103229 3099.834691537730578;"
    "-1.073038463974085 0.181163893667474 62.034492542754101;"
    "0.001914912485368 0.011919314669326 1.000000000000000\n"
    "Reprojection error: 10.254867218805751\n"
)
C009_CALIBRATION_TEXT = (
    "Homography matrix: -39.931762346875395 15.097050319357958 3066.432457736477318;"
    "1.564708487880704 1.493191280606009 68.977560416958141;"
    "0.001539740258468 0.011743673340779 1.000000000000000\n"
    "Reprojection error: 7.105744643288634\n"
)
# Verbatim from E:\traffic dataset\validation\S05\c035\calibration.txt --
# the only S02/S05 camera whose file actually carries the optional
# "Intrinsic parameter matrix:"/"Distortion coefficients:" lines.
C035_CALIBRATION_TEXT_WITH_OPTIONAL_FIELDS = (
    "Homography matrix: -18.811203584846680 -237.488629837366716 -20739.839301692441950;"
    "64.304824084169979 55.653194328254109 2314.620526920503380;"
    "-0.166408512433775 -0.066952682842798 1.000000000000000\n"
    "Intrinsic parameter matrix: 1920.000000000000000 0.000000000000000 960.000000000000000;"
    "0.000000000000000 1920.000000000000000 540.000000000000000;"
    "0.000000000000000 0.000000000000000 1.000000000000000\n"
    "Distortion coefficients: -0.600000023841858 0.000000000000000 0.000000000000000 0.000000000000000\n"
    "Reprojection error: 11.620452065017947\n"
)


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


# --- Commit 5: CityFlow homography ingestion ---


def test_parse_c006_calibration_text():
    data = parse_cityflow_calibration_text(C006_CALIBRATION_TEXT)

    assert data.homography.shape == (3, 3)
    assert data.reprojection_error == pytest.approx(10.254867218805751)
    assert data.intrinsic_matrix is None
    assert data.distortion_coefficients is None


def test_parse_c009_calibration_text():
    data = parse_cityflow_calibration_text(C009_CALIBRATION_TEXT)

    assert data.homography.shape == (3, 3)
    assert data.reprojection_error == pytest.approx(7.105744643288634)
    assert data.intrinsic_matrix is None
    assert data.distortion_coefficients is None


def test_c006_homography_values_preserved_exactly():
    """No rescaling/re-fitting -- every one of the 9 parsed values must
    match the source file's literal digits exactly (requirement 3)."""
    data = parse_cityflow_calibration_text(C006_CALIBRATION_TEXT)
    expected = np.array(
        [
            [-39.660220243283760, 15.592282221103229, 3099.834691537730578],
            [-1.073038463974085, 0.181163893667474, 62.034492542754101],
            [0.001914912485368, 0.011919314669326, 1.000000000000000],
        ]
    )
    assert np.array_equal(data.homography, expected)


def test_c009_homography_values_preserved_exactly():
    data = parse_cityflow_calibration_text(C009_CALIBRATION_TEXT)
    expected = np.array(
        [
            [-39.931762346875395, 15.097050319357958, 3066.432457736477318],
            [1.564708487880704, 1.493191280606009, 68.977560416958141],
            [0.001539740258468, 0.011743673340779, 1.000000000000000],
        ]
    )
    assert np.array_equal(data.homography, expected)


def test_intrinsic_matrix_and_distortion_coefficients_parsed_when_present():
    data = parse_cityflow_calibration_text(C035_CALIBRATION_TEXT_WITH_OPTIONAL_FIELDS)

    assert data.intrinsic_matrix is not None
    assert data.intrinsic_matrix.shape == (3, 3)
    assert data.intrinsic_matrix[0, 0] == pytest.approx(1920.0)
    assert data.distortion_coefficients == pytest.approx(
        (-0.600000023841858, 0.0, 0.0, 0.0)
    )
    assert data.reprojection_error == pytest.approx(11.620452065017947)


def test_missing_homography_line_raises_value_error():
    with pytest.raises(ValueError):
        parse_cityflow_calibration_text("Reprojection error: 5.0\n")


def test_missing_reprojection_error_line_raises_value_error():
    with pytest.raises(ValueError):
        parse_cityflow_calibration_text("Homography matrix: 1 0 0;0 1 0;0 0 1\n")


def test_malformed_homography_wrong_row_count_raises_value_error():
    with pytest.raises(ValueError):
        parse_cityflow_calibration_text("Homography matrix: 1 0 0;0 1 0\nReprojection error: 5.0\n")


def test_malformed_homography_wrong_column_count_raises_value_error():
    with pytest.raises(ValueError):
        parse_cityflow_calibration_text("Homography matrix: 1 0;0 1 0;0 0 1\nReprojection error: 5.0\n")


def test_malformed_numeric_value_in_homography_raises_value_error():
    with pytest.raises(ValueError):
        parse_cityflow_calibration_text("Homography matrix: 1 0 x;0 1 0;0 0 1\nReprojection error: 5.0\n")


def test_malformed_numeric_reprojection_error_raises_value_error():
    with pytest.raises(ValueError):
        parse_cityflow_calibration_text("Homography matrix: 1 0 0;0 1 0;0 0 1\nReprojection error: not-a-number\n")


def test_from_cityflow_homography_provenance_is_cityflow_supplied():
    data = parse_cityflow_calibration_text(C006_CALIBRATION_TEXT)
    profile = from_cityflow_homography("calib-c006", "s02_c006", data.homography, data.reprojection_error)

    assert profile.provenance == CalibrationProvenance.CITYFLOW_SUPPLIED
    assert profile.method == "cityflow_precomputed_v1"
    assert profile.status == CalibrationStatus.CALIBRATED
    assert profile.is_calibrated is True
    assert profile.reference_points == ()


def test_from_cityflow_homography_does_not_falsely_assert_physical_units():
    """Requirement 4: a CityFlow-supplied profile must NOT be
    auto-labeled meters/GPS/etc, and requirement 6: it must NOT claim
    metric speed is calibrated (VALIDATED) just because a homography
    exists."""
    data = parse_cityflow_calibration_text(C006_CALIBRATION_TEXT)
    profile = from_cityflow_homography("calib-c006", "s02_c006", data.homography, data.reprojection_error)

    assert profile.units != "meters"
    assert profile.units == "unknown"
    assert profile.metric_validation == MetricValidationStatus.UNVALIDATED


def test_from_cityflow_homography_preserves_reprojection_error():
    data = parse_cityflow_calibration_text(C009_CALIBRATION_TEXT)
    profile = from_cityflow_homography("calib-c009", "s02_c009", data.homography, data.reprojection_error)

    assert profile.reprojection_error_m == pytest.approx(7.105744643288634)


def test_from_cityflow_homography_projects_and_round_trips():
    """The ingested homography must be genuinely usable via the existing
    CalibrationProfile.image_to_world/world_to_image methods, same as any
    other CALIBRATED profile."""
    data = parse_cityflow_calibration_text(C006_CALIBRATION_TEXT)
    profile = from_cityflow_homography("calib-c006", "s02_c006", data.homography, data.reprojection_error)

    image_point = (960.0, 540.0)
    projected = profile.image_to_world(image_point)
    recovered = profile.world_to_image(projected)

    assert recovered[0] == pytest.approx(image_point[0], abs=1e-6)
    assert recovered[1] == pytest.approx(image_point[1], abs=1e-6)


def test_from_cityflow_homography_rejects_non_3x3_shape():
    with pytest.raises(ValueError):
        from_cityflow_homography("bad", "cam-A", np.eye(2), 5.0)


def test_from_cityflow_homography_rejects_non_finite_values():
    bad_homography = np.array([[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        from_cityflow_homography("bad", "cam-A", bad_homography, 5.0)


def test_from_cityflow_homography_rejects_singular_matrix():
    singular = np.zeros((3, 3))
    with pytest.raises(ValueError):
        from_cityflow_homography("bad", "cam-A", singular, 5.0)


def test_load_cityflow_calibration_reads_and_parses_file(tmp_path):
    calibration_file = tmp_path / "calibration.txt"
    calibration_file.write_text(C006_CALIBRATION_TEXT)

    profile = load_cityflow_calibration("calib-c006", "s02_c006", calibration_file)

    assert profile.camera_id == "s02_c006"
    assert profile.provenance == CalibrationProvenance.CITYFLOW_SUPPLIED
    assert profile.reprojection_error_m == pytest.approx(10.254867218805751)


def test_load_cityflow_calibration_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_cityflow_calibration("calib-x", "cam-x", "does_not_exist_calibration.txt")


def test_cityflow_supplied_profile_without_four_reference_points_does_not_raise():
    """Requirement 1: no fabricated reference points are needed -- unlike
    a MANUAL_SURVEY profile, an empty reference_points tuple is valid for
    CITYFLOW_SUPPLIED provenance."""
    data = parse_cityflow_calibration_text(C006_CALIBRATION_TEXT)
    profile = CalibrationProfile(
        calibration_profile_id="calib-c006",
        camera_id="s02_c006",
        status=CalibrationStatus.CALIBRATED,
        reference_points=(),
        homography=data.homography,
        inverse_homography=np.linalg.inv(data.homography),
        reprojection_error_m=data.reprojection_error,
        provenance=CalibrationProvenance.CITYFLOW_SUPPLIED,
    )
    assert profile.reference_points == ()


# --- Commit 5: existing calibration workflows must remain unchanged ---


def test_fit_planar_homography_is_explicitly_manual_survey_and_validated():
    """New provenance/metric_validation fields must not change
    fit_planar_homography's pre-Commit-5 behavior, only make its already
    -trusted status explicit."""
    profile = fit_planar_homography("calib-01", "cam-A", make_reference_points())

    assert profile.provenance == CalibrationProvenance.MANUAL_SURVEY
    assert profile.metric_validation == MetricValidationStatus.VALIDATED
    assert profile.units == "meters"
    assert profile.status == CalibrationStatus.CALIBRATED


def test_uncalibrated_profile_is_explicitly_unknown_provenance():
    profile = uncalibrated_profile("calib-none", "cam-B")

    assert profile.provenance == CalibrationProvenance.UNKNOWN
    assert profile.metric_validation == MetricValidationStatus.UNVALIDATED
    assert profile.status == CalibrationStatus.UNCALIBRATED


def test_manual_survey_still_requires_four_reference_points():
    """The relaxed reference-point check (Commit 5) must apply ONLY to
    non-MANUAL_SURVEY provenance -- the original MANUAL_SURVEY invariant
    is unchanged."""
    with pytest.raises(ValueError):
        CalibrationProfile(
            calibration_profile_id="bad",
            camera_id="cam-A",
            status=CalibrationStatus.CALIBRATED,
            reference_points=(),
            homography=np.eye(3),
            inverse_homography=np.eye(3),
            reprojection_error_m=0.0,
            provenance=CalibrationProvenance.MANUAL_SURVEY,
        )
