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

COMMIT 6 ADDITION (`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` SS8.2/SS11):
`compute_robust_speed`/`RobustSpeedResult`/`RobustSpeedStatus` add a
third, unit-AGNOSTIC computation (works on pixels, metres, or an
unvalidated externally-supplied coordinate space -- this module still
never itself decides which). Unlike the two functions above, it never
silently substitutes a zero-speed estimate for "no real measurement" --
every degenerate case (too few samples, too little elapsed time, a
non-finite coordinate, an excluded temporal gap) is its own explicit
`RobustSpeedStatus`, and it uses a time-weighted (distance/time) central
estimate rather than either "first vs. last point" or an unweighted
average of per-step speeds, so a single near-zero-duration jittery step
cannot dominate the result. See `anvesh.perception.speed_estimation` for
how this composes with `world.calibration.CalibrationProfile` to produce
a provenance-aware km/h estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


def _consecutive_steps(trajectory: list) -> list:
    """(distance, dt) for every consecutive pair with dt > 0 -- shared by
    both the legacy per-step-average functions below and
    `compute_robust_speed`."""
    steps = []
    for (x0, y0, t0), (x1, y1, t1) in zip(trajectory, trajectory[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        distance = math.hypot(x1 - x0, y1 - y0)
        steps.append((distance, dt))
    return steps


def _step_speeds(trajectory: list) -> list:
    return [distance / dt for distance, dt in _consecutive_steps(trajectory)]


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


class RobustSpeedStatus(str, Enum):
    """Explicit outcome of a robust speed computation -- OK plus every
    reason it might not be, so a caller never has to infer "no
    measurement" from a fabricated zero (requirement 9, Commit 6).

    `compute_robust_speed` below only ever returns the first five values.
    `CALIBRATION_UNAVAILABLE`/`INVALID_CALIBRATION` are reserved for a
    composition-layer caller with calibration knowledge this module
    deliberately does not have (see `anvesh.perception.speed_estimation.
    estimate_vehicle_speed`) -- reusing this one enum there keeps a single
    shared status vocabulary instead of two parallel ones that would need
    to stay in sync by hand.
    """

    OK = "ok"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    INSUFFICIENT_ELAPSED_TIME = "insufficient_elapsed_time"
    INVALID_TRAJECTORY = "invalid_trajectory"
    LARGE_TEMPORAL_GAP = "large_temporal_gap"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    INVALID_CALIBRATION = "invalid_calibration"


@dataclass(frozen=True)
class RobustSpeedResult:
    """Result of `compute_robust_speed` -- unit-agnostic (the caller
    knows, this function does not, whether `speed`/`error` are px/s, m/s,
    or an unvalidated externally-supplied distance-unit/s).

    `speed`/`error`/`elapsed_s` are populated if and only if
    `status == RobustSpeedStatus.OK`; every other status carries `None`
    for all three and `sample_count == 0` -- enforced below so a caller
    can never accidentally read a meaningless 0.0 as a real measurement.
    """

    status: RobustSpeedStatus
    speed: object
    error: object
    sample_count: int
    elapsed_s: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RobustSpeedStatus(self.status))
        if self.status == RobustSpeedStatus.OK:
            if self.speed is None or self.error is None or self.elapsed_s is None:
                raise ValueError("an OK RobustSpeedResult must carry speed, error, and elapsed_s")
            if self.speed < 0:
                raise ValueError("speed must be >= 0")
            if self.error < 0:
                raise ValueError("error must be >= 0")
            if self.sample_count < 1:
                raise ValueError("an OK RobustSpeedResult must have sample_count >= 1")
        else:
            if self.speed is not None or self.error is not None or self.elapsed_s is not None:
                raise ValueError(f"a {self.status.value} result must not carry speed/error/elapsed_s")
            if self.sample_count != 0:
                raise ValueError(f"a {self.status.value} result must have sample_count == 0")


def _no_measurement(status: RobustSpeedStatus) -> RobustSpeedResult:
    return RobustSpeedResult(status=status, speed=None, error=None, sample_count=0, elapsed_s=None)


def compute_robust_speed(
    trajectory: list,
    *,
    min_samples: int = 2,
    min_elapsed_s: float = 0.0,
    max_gap_s: float = None,
) -> RobustSpeedResult:
    """Robust, timestamp-based, explicit-status speed from an (x, y,
    timestamp) trajectory in ANY coordinate space -- this function has no
    opinion about units; the caller (e.g. `perception.speed_estimation`)
    attaches that meaning.

    ROBUST TEMPORAL WINDOW (requirement 7, Commit 6): speed is
    `total_distance / total_elapsed_time` across every valid consecutive
    step, i.e. each step's contribution is naturally WEIGHTED by its own
    duration -- not a naive first-vs-last-point measurement, and not an
    unweighted average of per-step instantaneous speeds
    (`compute_image_space_speed`'s approach, unchanged above). This
    matters for noisy/jittery trajectories: a near-duplicate-timestamp
    step contributes a small increment to BOTH the distance and time
    sums (jitter is small-magnitude by nature), so it cannot dominate the
    result the way it would as "one vote out of N" in a plain per-step
    average -- see `tests/perception/test_motion.py::
    test_noisy_outlier_step_does_not_dominate_robust_speed`. The
    reported `error` reuses `_mean_and_error`'s existing variance
    machinery over the same per-step speeds `compute_image_space_speed`/
    `compute_world_space_speed` already use, so a genuinely noisy
    trajectory still shows up as a large uncertainty, honestly, rather
    than being hidden (requirement 10 -- report value + error, never a
    falsely precise single number).

    Parameters are generic; a caller with camera-specific knowledge
    (e.g. a known frame rate) should derive values from that, not this
    function -- see `anvesh.perception.speed_estimation`'s CityFlow
    -derived defaults for an example.
      - `min_samples`: minimum trajectory points required even to
        attempt a measurement (default 2 -- the bare minimum for any
        displacement at all).
      - `min_elapsed_s`: minimum total time the retained steps must span
        (default 0.0 -- no minimum).
      - `max_gap_s`: a consecutive step whose own dt exceeds this is
        excluded from the totals (default `None` -- no gap filtering). A
        large gap usually means the vehicle was briefly undetected/
        occluded; excluding it avoids quietly averaging a long,
        unobserved interval in with genuinely continuous motion.

    Handles (requirement 8, Commit 6): insufficient observations
    (`INSUFFICIENT_OBSERVATIONS`), insufficient elapsed time after
    filtering (`INSUFFICIENT_ELAPSED_TIME`), a non-finite x/y/timestamp
    anywhere in `trajectory` (`INVALID_TRAJECTORY`), and a temporal gap
    that leaves no usable steps once excluded (`LARGE_TEMPORAL_GAP`). A
    stationary vehicle (zero net displacement) is NOT an error -- it
    produces a legitimate `OK` result with `speed=0.0`, matching how
    `compute_image_space_speed`/`compute_world_space_speed` already treat
    zero as a real measurement, not a failure.
    """
    if len(trajectory) < min_samples:
        return _no_measurement(RobustSpeedStatus.INSUFFICIENT_OBSERVATIONS)

    for x, y, t in trajectory:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)):
            return _no_measurement(RobustSpeedStatus.INVALID_TRAJECTORY)

    had_excluded_gap = False
    steps = []
    for (x0, y0, t0), (x1, y1, t1) in zip(trajectory, trajectory[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        if max_gap_s is not None and dt > max_gap_s:
            had_excluded_gap = True
            continue
        distance = math.hypot(x1 - x0, y1 - y0)
        steps.append((distance, dt))

    if not steps:
        status = RobustSpeedStatus.LARGE_TEMPORAL_GAP if had_excluded_gap else RobustSpeedStatus.INSUFFICIENT_OBSERVATIONS
        return _no_measurement(status)

    total_distance = sum(distance for distance, _ in steps)
    total_time = sum(dt for _, dt in steps)

    if total_time < min_elapsed_s:
        return _no_measurement(RobustSpeedStatus.INSUFFICIENT_ELAPSED_TIME)

    speed = total_distance / total_time
    _, error = _mean_and_error([distance / dt for distance, dt in steps])

    return RobustSpeedResult(
        status=RobustSpeedStatus.OK,
        speed=speed,
        error=error,
        sample_count=len(steps),
        elapsed_s=total_time,
    )
