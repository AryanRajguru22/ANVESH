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

COMMIT 5 ADDITION (`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` SS4/SS11):
`from_cityflow_homography`/`load_cityflow_calibration` ingest a SECOND,
different kind of calibration input -- CityFlowV2's `calibration.txt`, an
already-fitted 3x3 homography with no correspondence points at all, not a
site survey. This is deliberately not routed through
`fit_planar_homography` (which fits its own homography from points; it
has none to fit from here). Two new explicit fields distinguish this from
the MANUAL_SURVEY path everywhere above: `CalibrationProvenance` (who
produced the homography) and `MetricValidationStatus` (whether the
claimed output units have been independently confirmed against a known
real-world distance -- NOT the same question as "is a homography
present"). A CityFlow-supplied profile is CALIBRATED (a usable transform
exists) but `metric_validation=UNVALIDATED` and `units="unknown"` until
that separate check happens -- this is calibration INGESTION only; no
such validation step exists anywhere in the codebase yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import cv2

from anvesh.storage.schemas import GeometryValidity  # noqa: F401  (re-exported: the capability enum lives with the schemas)


class CalibrationStatus(str, Enum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATED = "calibrated"


class CalibrationProvenance(str, Enum):
    """Where a CALIBRATED profile's homography actually came from --
    distinct from `CalibrationStatus`, which only says whether a usable
    transform exists at all. Added in Commit 5 (`ANVESH_LANE_SPEED_
    FORENSIC_AUDIT.md` SS4.5/SS11) so a precomputed, externally-supplied
    homography (e.g. CityFlowV2's `calibration.txt`) is never presented,
    or silently treated, as if it were this project's own manually
    surveyed fit."""

    MANUAL_SURVEY = "manual_survey"  # fit_planar_homography, from real operator-supplied correspondences
    CITYFLOW_SUPPLIED = "cityflow_supplied"  # from_cityflow_homography, an already-fitted external matrix
    UNKNOWN = "unknown"  # no meaningful provenance (e.g. an UNCALIBRATED profile)


class MetricValidationStatus(str, Enum):
    """Whether a CALIBRATED profile's claimed output units (`units`, e.g.
    "meters") have been independently confirmed against a known
    real-world reference distance -- NOT whether the homography fit
    itself converged. Added in Commit 5 so a precomputed external
    homography (CityFlow) can never silently upgrade to "metric speed is
    calibrated" just because a matrix exists; only an explicit,
    independent validation step (not implemented anywhere yet) may set
    `VALIDATED`. Downstream speed code (a later commit) must read this,
    not just `CalibrationProfile.is_calibrated`, before presenting a
    speed as confirmed-metric rather than approximate."""

    VALIDATED = "validated"
    UNVALIDATED = "unvalidated"


class ValidationEvidenceKind(str, Enum):
    """What a `MetricValidationRecord` is evidence OF.

      SURVEY_HOLDOUT       -- a real site survey: reference points/distances that were NOT used
                              to fit the homography, compared against it.
      SYNTHETIC_TEST_ONLY  -- a fixture or demo. Declares "treat this as metric for the test";
                              measures nothing and is never evidence about a real camera.
    """

    SURVEY_HOLDOUT = "survey_holdout"
    SYNTHETIC_TEST_ONLY = "synthetic_test_only"
    # Scale taken from a PUBLISHED road-marking module (e.g. UK TSRGD diagram 1005.1: 2 m mark + 7 m gap) and
    # checked against marking segments that were NOT used in the fit. Weaker than a site survey: it assumes the
    # road is built to the standard and says nothing about the identity of the road. Speeds derived from it are
    # APPROXIMATE and must be presented as such.
    MARKING_STANDARD_HOLDOUT = "marking_standard_holdout"


# A real hold-out must contain at least this many independent points. This is a STRUCTURAL
# minimum (a homography itself needs 4), not a statistical guarantee of accuracy.
MIN_HOLDOUT_POINTS = 4


@dataclass(frozen=True)
class MetricValidationRecord:
    """The auditable evidence behind `MetricValidationStatus.VALIDATED` (hardening A1).

    A homography fitted from `n` correspondences reproduces those same `n` points almost
    exactly (an exact fit with 4), so the in-sample `reprojection_error_m` says nothing about
    whether the output is metric. `VALIDATED` therefore requires a record of an INDEPENDENT
    check. This class validates the record's structure and internal consistency only; it
    cannot prove the evidence is true. The procedure that produces real evidence (surveyed
    hold-out points, reference speeds) is not implemented here (M-B)."""

    kind: ValidationEvidenceKind
    evidence_ref: str  # where the evidence lives (survey id / file) -- must be non-empty
    n_holdout_points: int
    holdout_rmse_m: float
    tolerance_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ValidationEvidenceKind(self.kind))
        if not isinstance(self.evidence_ref, str) or not self.evidence_ref.strip():
            raise ValueError("a MetricValidationRecord needs a non-empty evidence_ref")
        for name in ("holdout_rmse_m", "tolerance_m"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if self.holdout_rmse_m < 0:
            raise ValueError("holdout_rmse_m must be >= 0")
        if self.tolerance_m <= 0:
            raise ValueError("tolerance_m must be > 0")
        if self.holdout_rmse_m > self.tolerance_m:
            raise ValueError(
                f"hold-out error {self.holdout_rmse_m} m exceeds the declared tolerance {self.tolerance_m} m: "
                "this evidence does not support VALIDATED"
            )
        if not isinstance(self.n_holdout_points, int) or isinstance(self.n_holdout_points, bool) or self.n_holdout_points < 0:
            raise ValueError("n_holdout_points must be a non-negative integer")
        if self.kind in (ValidationEvidenceKind.SURVEY_HOLDOUT, ValidationEvidenceKind.MARKING_STANDARD_HOLDOUT)                 and self.n_holdout_points < MIN_HOLDOUT_POINTS:
            raise ValueError(
                f"a {self.kind.name} record needs at least {MIN_HOLDOUT_POINTS} independent hold-out points, "
                f"got {self.n_holdout_points}"
            )


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
    provenance: CalibrationProvenance = CalibrationProvenance.MANUAL_SURVEY
    metric_validation: MetricValidationStatus = MetricValidationStatus.UNVALIDATED
    # Hardening A1: the evidence behind VALIDATED. Required when (and only when) metric_validation
    # is VALIDATED; see `MetricValidationRecord`.
    validation: MetricValidationRecord = None

    def __post_init__(self) -> None:
        if not self.calibration_profile_id:
            raise ValueError("calibration_profile_id must be a non-empty string")
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        object.__setattr__(self, "status", CalibrationStatus(self.status))
        object.__setattr__(self, "provenance", CalibrationProvenance(self.provenance))
        object.__setattr__(self, "metric_validation", MetricValidationStatus(self.metric_validation))

        if self.status == CalibrationStatus.CALIBRATED:
            if self.homography is None or self.inverse_homography is None:
                raise ValueError("a CALIBRATED profile must carry both homography and inverse_homography")
            # The >=4 reference-point invariant only makes sense for a
            # profile that claims to come from real surveyed
            # correspondences (MANUAL_SURVEY). A CITYFLOW_SUPPLIED (or
            # otherwise externally-sourced) profile honestly has none --
            # requiring/fabricating points here would be exactly the
            # fabrication requirement 1 (Commit 5) forbids.
            if self.provenance == CalibrationProvenance.MANUAL_SURVEY and len(self.reference_points) < 4:
                raise ValueError("a CALIBRATED MANUAL_SURVEY profile requires at least 4 reference points")
            if self.metric_validation == MetricValidationStatus.VALIDATED and self.units != "meters":
                # Structural guard: a profile whose output units are not
                # metres can never be "metric validated".
                raise ValueError(
                    f"a VALIDATED profile must have units == 'meters', got {self.units!r}"
                )
            if self.metric_validation == MetricValidationStatus.VALIDATED:
                # Hardening A1: VALIDATED is not a self-declaration. Only a surveyed profile that
                # carries an auditable validation record may claim it; an externally supplied
                # homography (e.g. CityFlow) never can.
                if self.provenance != CalibrationProvenance.MANUAL_SURVEY:
                    raise ValueError(
                        f"a {self.provenance.value} profile cannot be metric-VALIDATED: only MANUAL_SURVEY "
                        "profiles can carry validation evidence"
                    )
                if not isinstance(self.validation, MetricValidationRecord):
                    raise ValueError("a VALIDATED profile requires a MetricValidationRecord (validation=...)")
            elif self.validation is not None:
                raise ValueError("a validation record was supplied but metric_validation is not VALIDATED")
        else:
            if self.metric_validation == MetricValidationStatus.VALIDATED:
                raise ValueError("an UNCALIBRATED profile cannot be metric-VALIDATED")
            if self.validation is not None:
                raise ValueError("an UNCALIBRATED profile cannot carry a validation record")
            if self.homography is not None or self.inverse_homography is not None:
                raise ValueError("an UNCALIBRATED profile must not carry a homography")
            if self.reprojection_error_m is not None:
                raise ValueError("an UNCALIBRATED profile has no reprojection error to report")

    @property
    def is_calibrated(self) -> bool:
        """A transform matrix exists. NOT a statement that its output is
        metric -- use `geometry_validity` / `supports_metric_speed`."""
        return self.status == CalibrationStatus.CALIBRATED

    @property
    def geometry_validity(self) -> GeometryValidity:
        if not self.is_calibrated:
            return GeometryValidity.IMAGE
        if self.metric_validation == MetricValidationStatus.VALIDATED and self.units == "meters":
            return GeometryValidity.WORLD_VALIDATED
        return GeometryValidity.WORLD_UNVALIDATED

    @property
    def supports_metric_speed(self) -> bool:
        """True iff m/s / km/h may be derived from this profile."""
        return self.geometry_validity == GeometryValidity.WORLD_VALIDATED

    @property
    def is_synthetic_validated(self) -> bool:
        """VALIDATED only by a test/demo opt-in (`SYNTHETIC_TEST_ONLY` record)."""
        return (
            self.geometry_validity == GeometryValidity.WORLD_VALIDATED
            and self.validation is not None
            and self.validation.kind == ValidationEvidenceKind.SYNTHETIC_TEST_ONLY
        )

    @property
    def is_real_camera_validated(self) -> bool:
        """VALIDATED on the strength of survey hold-out evidence (as recorded -- not independently
        re-verified by this code). Always False for a synthetic opt-in and for external homographies."""
        return (
            self.geometry_validity == GeometryValidity.WORLD_VALIDATED
            and self.validation is not None
            and self.validation.kind == ValidationEvidenceKind.SURVEY_HOLDOUT
        )

    @property
    def is_marking_standard_validated(self) -> bool:
        """VALIDATED only against a published road-marking module (see `MARKING_STANDARD_HOLDOUT`):
        approximate metric scale, site not surveyed."""
        return (
            self.geometry_validity == GeometryValidity.WORLD_VALIDATED
            and self.validation is not None
            and self.validation.kind == ValidationEvidenceKind.MARKING_STANDARD_HOLDOUT
        )

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
    validation: MetricValidationRecord = None,
) -> CalibrationProfile:
    """Fit a planar-road homography from >=4 manually-surveyed correspondences.

    FAILS CLOSED (hardening A1): the returned profile is CALIBRATED but `UNVALIDATED` -- its
    geometry is `WORLD_UNVALIDATED`, so tracks stay image-space and no m/s or km/h is derived --
    unless the caller supplies a `MetricValidationRecord` (independent hold-out evidence). The
    fit's own `reprojection_error_m` is in-sample and is NOT validation (a 4-point fit is exact).
    Test/demo code that needs metric geometry uses the explicitly named
    `anvesh.world.synthetic_fixture.fit_synthetic_validated_homography`.

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
        provenance=CalibrationProvenance.MANUAL_SURVEY,
        # Hardening A1: VALIDATED needs evidence. Without a record the profile is UNVALIDATED.
        # (Before this change VALIDATED was asserted "by construction" for any 4+ points.) No code
        # in this repository produces a real SURVEY_HOLDOUT record yet: that is M-B.
        metric_validation=MetricValidationStatus.VALIDATED if validation is not None else MetricValidationStatus.UNVALIDATED,
        validation=validation,
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
        provenance=CalibrationProvenance.UNKNOWN,
        metric_validation=MetricValidationStatus.UNVALIDATED,
    )


@dataclass(frozen=True)
class CityFlowCalibrationData:
    """Raw fields parsed from one CityFlowV2 `calibration.txt` file, before
    they become a `CalibrationProfile`. Pure data, no validation beyond
    what `parse_cityflow_calibration_text` already performs (shape/type
    only) -- this is not itself a calibration profile and carries no
    provenance/status opinion."""

    homography: object  # 3x3 numpy array, exactly as parsed -- never rescaled or re-fit
    reprojection_error: float
    intrinsic_matrix: object  # 3x3 numpy array, or None if the file has no "Intrinsic parameter matrix:" line
    distortion_coefficients: object  # tuple of floats, or None if the file has no "Distortion coefficients:" line


def parse_cityflow_calibration_text(text: str) -> CityFlowCalibrationData:
    """Parse CityFlowV2's `calibration.txt` format:

        Homography matrix: a b c;d e f;g h i
        Intrinsic parameter matrix: a b c;d e f;g h i    (optional)
        Distortion coefficients: c0 c1 c2 c3             (optional)
        Reprojection error: <float>

    Each matrix line is 3 semicolon-separated rows of 3 whitespace-
    separated floats (confirmed against the real files at
    `E:\\traffic dataset\\validation\\S02\\{c006,c009}\\calibration.txt`,
    and against `S05\\c035\\calibration.txt`, which is the only S02/S05
    camera whose file actually carries the optional intrinsic/distortion
    lines -- see Commit 5 notes in
    `docs/LANE_SPEED_IMPLEMENTATION_NOTES.md`). The "Homography matrix"
    values are returned exactly as parsed -- no rescaling, normalization,
    or re-fitting (requirement 3, Commit 5).

    Raises `ValueError` for a missing/malformed "Homography matrix:"
    line, a malformed optional intrinsic-matrix/distortion-coefficients
    line, a non-numeric value anywhere, or a missing/malformed
    "Reprojection error:" line.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    homography = None
    intrinsic_matrix = None
    distortion_coefficients = None
    reprojection_error = None

    for line in lines:
        if line.startswith("Homography matrix:"):
            homography = _parse_matrix_3x3(line[len("Homography matrix:"):])
        elif line.startswith("Intrinsic parameter matrix:"):
            intrinsic_matrix = _parse_matrix_3x3(line[len("Intrinsic parameter matrix:"):])
        elif line.startswith("Distortion coefficients:"):
            distortion_coefficients = tuple(_parse_float_list(line[len("Distortion coefficients:"):]))
        elif line.startswith("Reprojection error:"):
            reprojection_error = _parse_float(line[len("Reprojection error:"):])

    if homography is None:
        raise ValueError("CityFlow calibration text is missing a 'Homography matrix:' line")
    if reprojection_error is None:
        raise ValueError("CityFlow calibration text is missing a 'Reprojection error:' line")

    return CityFlowCalibrationData(
        homography=homography,
        reprojection_error=reprojection_error,
        intrinsic_matrix=intrinsic_matrix,
        distortion_coefficients=distortion_coefficients,
    )


def _parse_matrix_3x3(raw: str) -> np.ndarray:
    rows = raw.strip().split(";")
    if len(rows) != 3:
        raise ValueError(f"expected 3 semicolon-separated matrix rows, got {len(rows)}: {raw!r}")
    values = []
    for row in rows:
        row_values = _parse_float_list(row)
        if len(row_values) != 3:
            raise ValueError(f"expected 3 values per matrix row, got {len(row_values)}: {row!r}")
        values.append(row_values)
    return np.array(values, dtype=np.float64)


def _parse_float_list(raw: str) -> list:
    tokens = raw.strip().split()
    if not tokens:
        raise ValueError("expected at least one numeric value, got an empty field")
    return [_parse_float(tok) for tok in tokens]


def _parse_float(raw: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"expected a numeric value, got {raw!r}") from exc


def from_cityflow_homography(
    calibration_profile_id: str,
    camera_id: str,
    homography,
    reprojection_error: float,
) -> CalibrationProfile:
    """Build a CALIBRATED `CalibrationProfile` from an already-fitted,
    externally-supplied homography (e.g. CityFlowV2's `calibration.txt`,
    via `parse_cityflow_calibration_text`), WITHOUT fabricating reference
    -point correspondences the dataset never provided (requirement 1,
    Commit 5) and WITHOUT asserting the output space is metres, GPS, or
    any other physical unit (requirement 4) -- CityFlow's own
    documentation does not establish that, and the forensic audit's own
    numerical check (`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` SS4.3) found
    the projected values are plausible-but-unconfirmed as some local
    metric frame, never proven.

    Always sets:
      - `provenance=CalibrationProvenance.CITYFLOW_SUPPLIED` (never
        `MANUAL_SURVEY` -- these are two different trust levels and must
        never be conflated, requirement 5).
      - `metric_validation=MetricValidationStatus.UNVALIDATED` (requirement
        6 -- a precomputed external homography does not, by itself, make
        ANVESH claim metric speed is calibrated; only an independent
        check against a known real-world reference distance could move a
        profile to VALIDATED, and no such check happens here or anywhere
        in the codebase yet).
      - `units="unknown"` (requirement 4 -- overrides `CalibrationProfile`
        's own `"meters"` default, which is only appropriate for the
        MANUAL_SURVEY path where a human supplied real-metre
        correspondences).
      - `reference_points=()` -- honestly empty; no correspondence points
        exist for this provenance. `CalibrationProfile.__post_init__`'s
        `>=4 reference points` invariant is relaxed specifically for
        non-MANUAL_SURVEY provenance, so this is not a validation
        loophole -- it is the documented exception.
      - `method="cityflow_precomputed_v1"` (distinct from
        `fit_planar_homography`'s `"manual_planar_homography_v1"`, so
        provenance is legible even from `method` alone).

    `reprojection_error` is stored, unmodified, into `reprojection_error_m`
    -- the field name's "_m" suffix is inherited naming from the
    MANUAL_SURVEY path and is NOT a units claim here; this value's actual
    units are exactly as unvalidated as the homography's own output space
    (see `units`/`metric_validation` above).

    Raises `ValueError` if `homography` is not a 3x3 array of finite
    numbers, or is singular (cannot be inverted) -- the only validation
    performed; the values themselves are never rescaled or re-fit.
    """
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3):
        raise ValueError(f"homography must be a 3x3 matrix, got shape {homography.shape}")
    if not np.all(np.isfinite(homography)):
        raise ValueError("homography must contain only finite numbers")

    try:
        inverse_homography = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("supplied homography is singular and cannot be inverted") from exc

    return CalibrationProfile(
        calibration_profile_id=calibration_profile_id,
        camera_id=camera_id,
        status=CalibrationStatus.CALIBRATED,
        reference_points=(),
        homography=homography,
        inverse_homography=inverse_homography,
        reprojection_error_m=float(reprojection_error),
        units="unknown",
        method="cityflow_precomputed_v1",
        provenance=CalibrationProvenance.CITYFLOW_SUPPLIED,
        metric_validation=MetricValidationStatus.UNVALIDATED,
    )


def load_cityflow_calibration(calibration_profile_id: str, camera_id: str, path) -> CalibrationProfile:
    """Read and parse one CityFlowV2 `calibration.txt` file and build a
    `CalibrationProfile` from it (`parse_cityflow_calibration_text` +
    `from_cityflow_homography`).

    Raises `FileNotFoundError` if `path` does not exist, and `ValueError`
    for anything `parse_cityflow_calibration_text`/
    `from_cityflow_homography` already raise `ValueError` for.
    """
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"CityFlow calibration file not found: {calibration_path}")

    data = parse_cityflow_calibration_text(calibration_path.read_text())
    return from_cityflow_homography(
        calibration_profile_id=calibration_profile_id,
        camera_id=camera_id,
        homography=data.homography,
        reprojection_error=data.reprojection_error,
    )
