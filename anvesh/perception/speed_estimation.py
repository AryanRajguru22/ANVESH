"""Provenance-aware approximate vehicle speed estimation (Commit 6, see
`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` SS8.2/SS11). Composes:

  - `TrackState.ground_point_trajectory` (Commit 4) -- ground-contact
    points, NEVER `TrackState.trajectory` (centroid). See
    `RawDetection.ground_point`'s own rationale for why.
  - `CalibrationProfile.image_to_world` (`world/calibration.py`) --
    projects each ground point through the camera's homography, only
    when the profile `is_calibrated`.
  - `motion.compute_robust_speed` (Commit 6 addition to `motion.py`) -- a
    robust, explicit-status, timestamp-based speed computation, unit
    -agnostic by itself.

SPEED ESTIMATION ONLY (this commit's own scope): no lane-change
detection, no candidate-vehicle ranking, no dashboard/export change, and
-- the rule this whole module exists to enforce -- NO claim that a
CityFlow-supplied homography's speed is validated-metric. A
`CalibrationProfile` with `provenance=CalibrationProvenance.
CITYFLOW_SUPPLIED` always produces `SpeedConfidence.APPROXIMATE`, never
`VALIDATED`, regardless of how clean the underlying trajectory math is --
see `SpeedConfidence` and `estimate_vehicle_speed` below. There is no
parameter anywhere in this module that can override that.

Never mutates `RawDetection`, `TrackState`, `CalibrationProfile`, or the
frozen `storage.schemas.VehicleTrack` -- `VehicleSpeedEstimate` is a
derived, non-schema companion result, the same pattern
`anvesh.perception.lane_assignment.TrackLaneHistory` already established
in Commit 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anvesh.perception.motion import RobustSpeedStatus, compute_robust_speed
from anvesh.perception.track_manager import TrackState
from anvesh.storage.schemas import Measurement
from anvesh.world.calibration import CalibrationProfile, MetricValidationStatus

# m/s -> km/h. Applied identically regardless of provenance -- what
# differs between a MANUAL_SURVEY and a CITYFLOW_SUPPLIED profile is
# `SpeedConfidence`, never the conversion arithmetic itself.
MPS_TO_KMH = 3.6

# CityFlowV2's confirmed frame rate for the showcase cameras (c006, c009):
# 10.0 fps, i.e. a 0.1s frame interval -- see
# `ANVESH_S02_CAMERA_PAIR_ANALYSIS.md`'s per-camera baseline table. These
# defaults are DERIVED from that, not arbitrary, and exist only as a
# reasonable starting point -- a caller for a different camera/frame rate
# should pass its own values to `estimate_vehicle_speed`.
CITYFLOW_FRAME_INTERVAL_S = 0.1
DEFAULT_MIN_SAMPLES = 3
# 3 frame-intervals: enough to span more than two adjacent, highly
# -correlated frames (requirement 7's "robust window", not first-vs-last).
DEFAULT_MIN_ELAPSED_S = 3 * CITYFLOW_FRAME_INTERVAL_S
# 6 frame-intervals: comfortably above TrackManager's own default
# `max_missed_frames=5` (0.5s at 10fps) -- the longest gap a track can
# have while still PARTIAL rather than finalized LOST, so a gap at or
# below that is normal occlusion, not a discontinuity worth excluding.
DEFAULT_MAX_GAP_S = 6 * CITYFLOW_FRAME_INTERVAL_S


class SpeedConfidence(str, Enum):
    """Whether a speed's physical-unit (km/h) claim is independently
    confirmed, only a provisional/approximate assumption, or not
    available at all -- a SEPARATE question from `RobustSpeedStatus`
    (whether the underlying displacement/time math itself succeeded). A
    mathematically `OK` speed can still be merely `APPROXIMATE` if the
    coordinate system's scale is unvalidated
    (`MetricValidationStatus.UNVALIDATED`, `world/calibration.py`)."""

    VALIDATED = "validated"
    APPROXIMATE = "approximate"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VehicleSpeedEstimate:
    """A track's speed, as a non-schema companion result -- same pattern
    as `lane_assignment.TrackLaneHistory` (Commit 4): never folded into
    the frozen `storage.schemas.VehicleTrack`, which carries no
    speed-confidence concept of its own.

    `measurement` is `Measurement(value=km/h, error=km/h)` if and only if
    `confidence != SpeedConfidence.UNAVAILABLE` -- never a fabricated
    placeholder number for an unavailable/failed estimate.
    """

    track_id: str
    camera_id: str
    status: RobustSpeedStatus
    confidence: SpeedConfidence
    measurement: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RobustSpeedStatus(self.status))
        object.__setattr__(self, "confidence", SpeedConfidence(self.confidence))
        if self.confidence == SpeedConfidence.UNAVAILABLE:
            if self.measurement is not None:
                raise ValueError("an UNAVAILABLE VehicleSpeedEstimate must not carry a measurement")
        else:
            if not isinstance(self.measurement, Measurement):
                raise ValueError(f"a {self.confidence.value} VehicleSpeedEstimate must carry a Measurement")
            if self.status != RobustSpeedStatus.OK:
                raise ValueError("a non-UNAVAILABLE VehicleSpeedEstimate must have status == RobustSpeedStatus.OK")


def _unavailable(camera_id: str, track_id: str, status: RobustSpeedStatus) -> VehicleSpeedEstimate:
    return VehicleSpeedEstimate(
        track_id=f"{camera_id}:{track_id}",
        camera_id=camera_id,
        status=status,
        confidence=SpeedConfidence.UNAVAILABLE,
        measurement=None,
    )


def estimate_vehicle_speed(
    camera_id: str,
    state: TrackState,
    calibration_profile: CalibrationProfile = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_elapsed_s: float = DEFAULT_MIN_ELAPSED_S,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> VehicleSpeedEstimate:
    """Estimate one track's speed from its GROUND-POINT trajectory
    (requirement 1 -- never `state.trajectory`, which is centroid-based;
    see `state.ground_point_trajectory`, the Commit 4 addition to
    `TrackState`).

    Returns `confidence=UNAVAILABLE` (no `measurement`) when
    `calibration_profile` is missing or not `is_calibrated` -- a pixel
    displacement has no physical speed meaning, and this module never
    guesses one (requirement 6). Otherwise projects every ground-point
    sample through `calibration_profile.image_to_world`, runs `motion.
    compute_robust_speed` on the projected trajectory, and -- only if
    that reaches `RobustSpeedStatus.OK` -- converts to km/h via
    `MPS_TO_KMH`, ALWAYS under the same assumption: that the projected
    distance unit is metres. That assumption has a real basis for both
    provenances a `CalibrationProfile` can currently carry --
    `MANUAL_SURVEY`'s correspondences are real measured metres;
    `CITYFLOW_SUPPLIED`'s scale is merely plausible-as-metres per the
    forensic audit's own numerical check (`ANVESH_LANE_SPEED_FORENSIC_
    AUDIT.md` SS4.3), never proven -- so the SAME arithmetic runs either
    way. What differs is `confidence`:
      - `VALIDATED` only when `calibration_profile.metric_validation ==
        MetricValidationStatus.VALIDATED` (today: `MANUAL_SURVEY` only).
      - `APPROXIMATE` when the math reached `OK` but `metric_validation
        == UNVALIDATED` (today: `CITYFLOW_SUPPLIED` always lands here --
        requirements 5/6; there is no parameter to override this).
      - `UNAVAILABLE` when calibration is missing/invalid, or the
        underlying math did not reach `OK` (see `RobustSpeedStatus` for
        every specific reason: insufficient observations, insufficient
        elapsed time, an invalid/non-finite trajectory, or an excluded
        temporal gap).

    `min_samples`/`min_elapsed_s`/`max_gap_s` default to values derived
    from CityFlowV2's confirmed 10fps frame rate (see the module-level
    constants above) but are overridable for any other camera.
    """
    track_id = state.track_id

    if calibration_profile is None or not calibration_profile.is_calibrated:
        return _unavailable(camera_id, track_id, RobustSpeedStatus.CALIBRATION_UNAVAILABLE)

    try:
        world_trajectory = [
            (*calibration_profile.image_to_world((x, y)), t) for (x, y, t) in state.ground_point_trajectory
        ]
    except ValueError:
        # `CalibrationProfile.image_to_world` raises ValueError for a
        # degenerate application of an otherwise-CALIBRATED homography
        # (w-component near zero for this specific point) -- an
        # unusable calibration for this data, not a crash.
        return _unavailable(camera_id, track_id, RobustSpeedStatus.INVALID_CALIBRATION)

    result = compute_robust_speed(
        world_trajectory, min_samples=min_samples, min_elapsed_s=min_elapsed_s, max_gap_s=max_gap_s
    )

    if result.status != RobustSpeedStatus.OK:
        return _unavailable(camera_id, track_id, result.status)

    speed_kmh = result.speed * MPS_TO_KMH
    error_kmh = result.error * MPS_TO_KMH
    confidence = (
        SpeedConfidence.VALIDATED
        if calibration_profile.metric_validation == MetricValidationStatus.VALIDATED
        else SpeedConfidence.APPROXIMATE
    )

    return VehicleSpeedEstimate(
        track_id=f"{camera_id}:{track_id}",
        camera_id=camera_id,
        status=RobustSpeedStatus.OK,
        confidence=confidence,
        measurement=Measurement(value=speed_kmh, error=error_kmh),
    )
