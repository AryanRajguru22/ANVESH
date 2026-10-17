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

M-A SPEED-SAFETY INVARIANT (ANVESH_PRODUCT_GAP_AUDIT.md finding 2): a km/h or
m/s number exists on a `VehicleSpeedEstimate` ONLY when the calibration's
geometry is `GeometryValidity.WORLD_VALIDATED` (`CalibrationProfile.
supports_metric_speed`). For a WORLD_UNVALIDATED profile (every CityFlow
camera) the projected displacement/time is carried ONLY as
`unvalidated_world_speed`, in unvalidated "world-frame units per second", a
different field that downstream code cannot mistake for metric speed, and
`measurement` (km/h) is None. `VehicleSpeedEstimate.metric_speed_kmh` is the
single accessor a consumer that needs a real speed may use; it returns None
unless the speed is validated, plausible, metric.

SPEED ESTIMATION ONLY (this commit's own scope): no lane-change
detection, no candidate-vehicle ranking, no dashboard/export change, and
-- the rule this whole module exists to enforce -- NO claim that a
CityFlow-supplied homography's speed is validated-metric. A
`CalibrationProfile` with `provenance=CalibrationProvenance.
CITYFLOW_SUPPLIED` always produces `SpeedConfidence.APPROXIMATE`, never
`VALIDATED`, regardless of how clean the underlying trajectory math is --
see `SpeedConfidence` and `estimate_vehicle_speed` below. There is no
parameter anywhere in this module that can override that.

PHYSICAL PLAUSIBILITY GATE (V1 Showcase speed-validation pass): a
CityFlow-supplied homography's OUTPUT UNITS are unvalidated
(`CalibrationProfile.units == "unknown"`, `metric_validation ==
UNVALIDATED` -- see `world/calibration.py`'s own `from_cityflow_
homography` docstring). Investigating an observed "~2084 km/h" c006
showcase value traced this to a real, measured property of that specific
homography, not a bug in the displacement/gap math (`motion.
compute_robust_speed`, unchanged, already excludes large temporal gaps
and requires a minimum sample count/elapsed time): probing
`calibration.image_to_world` at the c006 homography's own coordinates
shows its implied world-scale is on the order of several "units" per
IMAGE PIXEL, and grows sharply for points nearer the frame's vanishing
line (perspective foreshortening) -- two orders of magnitude too large
for those units to plausibly be metres, given this camera's real lane
widths. Under that scale, ordinary ground-point measurement noise of a
few pixels (inherent to any bounding-box detector, not a tracking defect)
-- or even entirely ordinary real vehicle motion -- is amplified into a
derived "speed" of hundreds to thousands of km/h once multiplied through
`MPS_TO_KMH` as if the units were confirmed metres.

`MAX_PLAUSIBLE_SPEED_KMH` below is the resulting safeguard: a generous
PHYSICAL ceiling on what any road vehicle could plausibly be doing in a
fixed traffic-camera scene, not a measured accuracy claim and not this
camera's real (much lower) legal speed limit. A derived speed at or below
it is not thereby proven correct -- it is merely not provably impossible,
which is exactly the same epistemic status `SpeedConfidence.APPROXIMATE`
already carries. A derived speed ABOVE it gets `SpeedConfidence.OUTLIER`
instead of `APPROXIMATE`: the underlying math still reached
`RobustSpeedStatus.OK` (nothing was numerically wrong) and the
measurement is still carried (never discarded -- available for
diagnostics, per this module's existing "never fabricate, never silently
drop" discipline), but it must never be presented, formatted, or consumed
as an ordinary observed speed. See `anvesh.rendering.format_speed_text`
for how `OUTLIER` renders ("UNRELIABLE — OUTLIER", never a number), and
`tests/perception/test_speed_evidence_safety.py` for why this path was
already structurally incapable of reaching the evidence/ranking/
candidate-vehicle/propagation layers even before this gate existed.

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
from anvesh.storage.schemas import GeometryValidity, Measurement
from anvesh.world.calibration import CalibrationProfile

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

# A generous PHYSICAL plausibility ceiling for a road vehicle observed by
# a fixed traffic camera -- see the module docstring's "PHYSICAL
# PLAUSIBILITY GATE" section for the full rationale and the measured
# c006-homography evidence that motivated it. This is deliberately NOT:
#   - a measured accuracy bound (it says nothing about how close an
#     APPROXIMATE speed at or below it is to the truth);
#   - this camera's real, much lower legal speed limit (that would be a
#     context-specific policy threshold, not a physical-plausibility one,
#     and would reject legitimate-if-unusual fast driving along with
#     genuine artifacts);
#   - vehicle-class-specific (a truck and a motorcycle share one ceiling
#     here -- V1 has no per-class plausibility model, and inventing one
#     was not justified by the root-cause investigation).
# 200 km/h safely exceeds any real-world production road vehicle's
# sustained speed in ordinary traffic-camera footage while sitting
# comfortably below every outlier this investigation actually observed in
# the c006 sample (roughly 450-3200 km/h) -- see
# `ANVESH_C006_SPEED_VALIDATION_AUDIT.md` for the full numeric trace.
MAX_PLAUSIBLE_SPEED_KMH = 200.0


class SpeedConfidence(str, Enum):
    """Whether a speed's physical-unit (km/h) claim is independently
    confirmed, only a provisional/approximate assumption, physically
    implausible, or not available at all -- a SEPARATE question from
    `RobustSpeedStatus` (whether the underlying displacement/time math
    itself succeeded). A mathematically `OK` speed can still be merely
    `APPROXIMATE` if the coordinate system's scale is unvalidated
    (`MetricValidationStatus.UNVALIDATED`, `world/calibration.py`), and a
    mathematically `OK` speed that additionally exceeds
    `MAX_PLAUSIBLE_SPEED_KMH` is `OUTLIER` instead -- never silently
    presented as an ordinary `APPROXIMATE` observation."""

    VALIDATED = "validated"
    APPROXIMATE = "approximate"
    OUTLIER = "outlier"  # RobustSpeedStatus.OK, but exceeds MAX_PLAUSIBLE_SPEED_KMH -- see module docstring
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VehicleSpeedEstimate:
    """A track's speed, as a non-schema companion result -- same pattern
    as `lane_assignment.TrackLaneHistory` (Commit 4): never folded into
    the frozen `storage.schemas.VehicleTrack`'s metric `speed_estimate`.

    Exactly one of these holds (enforced in `__post_init__`):
      - UNAVAILABLE : no number of any kind.
      - VALIDATED   : `measurement` = Measurement(km/h) from validated metric
                      geometry; `unvalidated_world_speed` is None.
      - APPROXIMATE : `unvalidated_world_speed` = Measurement(world-frame
                      units/s) from an UNVALIDATED transform; `measurement`
                      (km/h) is None -- no metric claim exists.
      - OUTLIER     : an `OK` computation rejected as physically implausible.
                      Carried for diagnostics in whichever field matches its
                      geometry (`measurement` if validated geometry,
                      `unvalidated_world_speed` otherwise), never usable.

    `geometry_validity` records which kind of transform produced it
    (IMAGE for "no transform"); `None` derives it from which field is set.
    Callers must branch on `confidence` and read speeds ONLY through
    `metric_speed_kmh`.
    """

    track_id: str
    camera_id: str
    status: RobustSpeedStatus
    confidence: SpeedConfidence
    measurement: object
    unvalidated_world_speed: object = None
    geometry_validity: GeometryValidity = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RobustSpeedStatus(self.status))
        object.__setattr__(self, "confidence", SpeedConfidence(self.confidence))
        for name in ("measurement", "unvalidated_world_speed"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Measurement):
                raise ValueError(f"{name} must be a Measurement or None")

        if self.confidence == SpeedConfidence.UNAVAILABLE:
            if self.measurement is not None or self.unvalidated_world_speed is not None:
                raise ValueError("an UNAVAILABLE VehicleSpeedEstimate must not carry any speed value")
        else:
            if self.status != RobustSpeedStatus.OK:
                raise ValueError("a non-UNAVAILABLE VehicleSpeedEstimate must have status == RobustSpeedStatus.OK")
            if (self.measurement is None) == (self.unvalidated_world_speed is None):
                raise ValueError(
                    f"a {self.confidence.value} VehicleSpeedEstimate must carry exactly one of "
                    "`measurement` (validated km/h) or `unvalidated_world_speed`"
                )
            if self.confidence == SpeedConfidence.VALIDATED and self.measurement is None:
                raise ValueError("a VALIDATED VehicleSpeedEstimate must carry a metric `measurement`")
            if self.confidence == SpeedConfidence.APPROXIMATE and self.unvalidated_world_speed is None:
                raise ValueError(
                    "an APPROXIMATE VehicleSpeedEstimate is by definition NON-metric: it must carry "
                    "`unvalidated_world_speed`, never a km/h `measurement`"
                )

        if self.geometry_validity is None:
            if self.measurement is not None:
                # M-A.1: a km/h figure never implies its own validity.
                raise ValueError(
                    "a VehicleSpeedEstimate carrying a km/h `measurement` must declare "
                    "geometry_validity=WORLD_VALIDATED explicitly"
                )
            elif self.unvalidated_world_speed is not None:
                derived = GeometryValidity.WORLD_UNVALIDATED
            else:
                derived = GeometryValidity.IMAGE
            object.__setattr__(self, "geometry_validity", derived)
        else:
            object.__setattr__(self, "geometry_validity", GeometryValidity(self.geometry_validity))
        if self.measurement is not None and self.geometry_validity != GeometryValidity.WORLD_VALIDATED:
            raise ValueError("a km/h `measurement` requires geometry_validity == WORLD_VALIDATED")

    @property
    def metric_speed_kmh(self):
        """The ONLY sanctioned way to read a km/h speed: `Measurement(km/h)`
        iff this estimate is VALIDATED metric (and therefore plausible);
        otherwise None -- unavailable, unvalidated, or an outlier."""
        if self.confidence == SpeedConfidence.VALIDATED:
            return self.measurement
        return None


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

    Returns `confidence=UNAVAILABLE` (no speed value of any kind) when
    `calibration_profile` is missing or not `is_calibrated` -- a pixel
    displacement has no physical speed meaning, and this module never
    guesses one. Otherwise projects every ground-point sample through
    `calibration_profile.image_to_world` and runs `motion.
    compute_robust_speed` on the projected trajectory. What the result IS
    depends on `calibration_profile.geometry_validity`, never on the mere
    existence of a homography (M-A):
      - WORLD_VALIDATED (`supports_metric_speed`): the projected unit is
        metres, so the speed is converted with `MPS_TO_KMH` into
        `measurement` and is `VALIDATED` -- unless it exceeds
        `MAX_PLAUSIBLE_SPEED_KMH`, in which case it is `OUTLIER` (a value
        that fails a basic physical-plausibility check cannot
        simultaneously be an independently confirmed metric observation).
      - WORLD_UNVALIDATED (every CityFlow-supplied profile): the unit is
        NOT known to be metres. The value is carried ONLY as
        `unvalidated_world_speed` (world-frame units/s), `measurement`
        (km/h) stays None, and the confidence is `APPROXIMATE` -- or
        `OUTLIER` if, read as if it were metres, it would exceed
        `MAX_PLAUSIBLE_SPEED_KMH`. There is no parameter that can turn an
        unvalidated profile's value into km/h.
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

    metric = calibration_profile.supports_metric_speed
    # `speed`/`error` are in the projected frame's distance unit per second:
    # metres/s ONLY when `metric`; otherwise unvalidated world-frame units/s.
    speed_kmh_if_metres = result.speed * MPS_TO_KMH
    implausible = speed_kmh_if_metres > MAX_PLAUSIBLE_SPEED_KMH

    if metric:
        confidence = SpeedConfidence.OUTLIER if implausible else SpeedConfidence.VALIDATED
        return VehicleSpeedEstimate(
            track_id=f"{camera_id}:{track_id}",
            camera_id=camera_id,
            status=RobustSpeedStatus.OK,
            confidence=confidence,
            measurement=Measurement(value=speed_kmh_if_metres, error=result.error * MPS_TO_KMH),
            geometry_validity=GeometryValidity.WORLD_VALIDATED,
        )

    # Unvalidated geometry: NEVER a km/h value. A physically implausible
    # magnitude (the c006 homography's oversized scale) is flagged OUTLIER;
    # anything else is APPROXIMATE -- a non-metric diagnostic only.
    confidence = SpeedConfidence.OUTLIER if implausible else SpeedConfidence.APPROXIMATE
    return VehicleSpeedEstimate(
        track_id=f"{camera_id}:{track_id}",
        camera_id=camera_id,
        status=RobustSpeedStatus.OK,
        confidence=confidence,
        measurement=None,
        unvalidated_world_speed=Measurement(value=result.speed, error=result.error),
        geometry_validity=GeometryValidity.WORLD_UNVALIDATED,
    )
