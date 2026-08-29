"""Conversion from perception-internal types into the frozen ANVESH schemas
(`anvesh.storage.schemas`), per blueprint Part 5 / Part 3 module 7.

M2 UPDATE (resolves the M1-flagged incompatibility):

M1 flagged that `VehicleTrack.position_history`/`speed_estimate` presuppose
a `CalibrationProfile` that did not exist yet, and worked around it by
always populating image-space pixels, documented only in prose. M2 adds
real calibration (`world/calibration.py`) and a schema-level
`MotionSpace` field (see `anvesh/storage/schemas.py`'s "M2 addition" note),
so this module now branches explicitly:

  - `calibration_profile` given and CALIBRATED: `position_history` is
    built from real `WorldPosition` instances (metres, via
    `CalibrationProfile.image_to_world`), `speed_estimate` is a metric
    `Measurement` (m/s, via `motion.compute_world_space_speed`), and
    `motion_space=MotionSpace.WORLD`.
  - `calibration_profile` omitted or UNCALIBRATED: unchanged from M1 --
    `position_history` stays plain `(x, y, t)` pixel tuples (never
    `WorldPosition`), `speed_estimate` is pixels/second (via
    `motion.compute_image_space_speed`), and
    `motion_space=MotionSpace.IMAGE`.

The previous M1 pixel-space convention is used ONLY in the second branch
now -- i.e. only where no valid calibration exists -- per the M2
instruction that it "must no longer be used where valid calibration
exists."
"""

from __future__ import annotations

from anvesh.perception.motion import compute_image_space_speed, compute_world_space_speed
from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import CameraObservation, Measurement, MotionSpace, VehicleTrack, WorldPosition
from anvesh.world.calibration import CalibrationProfile


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


def build_vehicle_track(
    camera_id: str,
    state: TrackState,
    calibration_profile: CalibrationProfile = None,
) -> VehicleTrack:
    """Convert one finished/active TrackState into a VehicleTrack.

    Produces a WORLD-space track (metres, `MotionSpace.WORLD`) if
    `calibration_profile` is given and calibrated; otherwise falls back to
    the M1 IMAGE-space convention (`MotionSpace.IMAGE`). See the module
    docstring for the full rationale.
    """
    if calibration_profile is not None and calibration_profile.is_calibrated:
        world_trajectory = [
            (*calibration_profile.image_to_world((x, y)), t) for (x, y, t) in state.trajectory
        ]
        speed = compute_world_space_speed(world_trajectory)
        position_history = [WorldPosition(world_x=wx, world_y=wy, timestamp=t) for (wx, wy, t) in world_trajectory]
        speed_estimate = Measurement(value=speed.speed_m_per_s, error=speed.error_m_per_s)
        motion_space = MotionSpace.WORLD
    else:
        speed = compute_image_space_speed(state.trajectory)
        position_history = list(state.trajectory)
        speed_estimate = Measurement(value=speed.speed_px_per_s, error=speed.error_px_per_s)
        motion_space = MotionSpace.IMAGE

    return VehicleTrack(
        track_id=f"{camera_id}:{state.track_id}",
        camera_id=camera_id,
        first_seen=state.first_seen,
        last_seen=state.last_seen,
        vehicle_class=state.vehicle_class,
        occlusion_state=state.occlusion_state,
        position_history=position_history,
        speed_estimate=speed_estimate,
        motion_space=motion_space,
    )
