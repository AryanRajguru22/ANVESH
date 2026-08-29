"""Conversion from perception-internal types into the frozen ANVESH schemas
(`anvesh.storage.schemas`), per blueprint Part 5 / Part 3 module 7.

FLAGGED INCOMPATIBILITY (reported to the user rather than silently patched):

Part 5 types `VehicleTrack.position_history` as a list of `WorldPosition`
samples ("via CalibrationProfile projection") and `VehicleTrack.speed_estimate`
as a generic `Measurement` (a bare float +/- error, with no unit field).
Both presuppose a `CalibrationProfile` (`world/calibration.py`), which does
not exist until milestone M2. M1 has no calibration.

Rather than editing the frozen schema or fabricating a fake calibration,
this module makes the gap visible in the data instead of hiding it:

  - `position_history` is populated with plain `(x, y, timestamp)` tuples
    in IMAGE-SPACE pixel coordinates -- deliberately NOT `WorldPosition`
    instances. `VehicleTrack.__post_init__` does not validate
    `position_history`'s element type, so this is schema-legal, but this
    module avoids the `WorldPosition` type on purpose: that type's own
    name and docstring mean "calibrated world coordinates," which pixels
    are not.
  - `speed_estimate` is populated with a `Measurement` whose `value` is in
    PIXELS PER SECOND (from `anvesh.perception.motion`), never m/s. There
    is no per-field unit tag on the frozen `Measurement` type to record
    this, so the unit is only recoverable from this module's
    documentation and from `Camera.calibration_profile_id` being the
    placeholder `UNCALIBRATED_PROFILE_ID` below.

M2 (camera calibration) is expected to either populate a real
`CalibrationProfile` and re-run this conversion to produce true
`WorldPosition`/metric `Measurement` values, or the schema may need a
`calibration_status`/unit field added at that point -- a decision for
whoever owns `storage/schemas.py`, not made unilaterally here.
"""

from __future__ import annotations

from anvesh.perception.motion import compute_image_space_speed
from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import CameraObservation, Measurement, VehicleTrack

UNCALIBRATED_PROFILE_ID = "uncalibrated-identity-v1"


def build_camera_observation(
    camera_id: str,
    frame_index: int,
    timestamp: float,
    detections: list,
    model_id: str,
    model_version: str,
) -> CameraObservation:
    return CameraObservation(
        observation_id=f"{camera_id}:{frame_index}",
        camera_id=camera_id,
        frame_timestamp=timestamp,
        detections=[_detection_to_dict(d) for d in detections],
        model_id=model_id,
        model_version=model_version,
    )


def _detection_to_dict(detection: RawDetection) -> dict:
    return {
        "frame_index": detection.frame_index,
        "timestamp": detection.timestamp,
        "class_name": detection.class_name,
        "vehicle_class": detection.vehicle_class.value,
        "confidence": detection.confidence,
        "bbox_xyxy": detection.bbox_xyxy,
    }


def build_vehicle_track(camera_id: str, state: TrackState) -> VehicleTrack:
    """Convert one finished/active TrackState into a VehicleTrack.

    See the module docstring for why `position_history` holds image-space
    pixel tuples (not `WorldPosition`) and `speed_estimate` is in
    pixels/second (not m/s).
    """
    speed = compute_image_space_speed(state.trajectory)
    return VehicleTrack(
        track_id=f"{camera_id}:{state.track_id}",
        camera_id=camera_id,
        first_seen=state.first_seen,
        last_seen=state.last_seen,
        vehicle_class=state.vehicle_class,
        occlusion_state=state.occlusion_state,
        position_history=list(state.trajectory),
        speed_estimate=Measurement(value=speed.speed_px_per_s, error=speed.error_px_per_s),
    )
