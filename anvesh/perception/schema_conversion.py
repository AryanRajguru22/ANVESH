"""Conversion from perception-internal types into the frozen ANVESH schemas
(`anvesh.storage.schemas`), per blueprint Part 5 / Part 3 module 7.

M-A SPEED-SAFETY (ANVESH_PRODUCT_GAP_AUDIT.md finding 2). This module used to
treat `calibration_profile.is_calibrated` as "metric speed is validated": it
projected the CENTROID trajectory through whatever homography existed and
labelled the result m/s with `MotionSpace.WORLD`. For CityFlow's supplied
homography (output scale 2-28 "units" per pixel, units unknown) that produced
median "speeds" of hundreds of m/s which then drove congestion, H2/H3 evidence
and H1 ranking. That path is deleted. The branch is now on
`CalibrationProfile.geometry_validity`:

  - IMAGE (no profile / uncalibrated): `MotionSpace.IMAGE`, pixel centroid
    history, px/s -- unchanged from M1.
  - WORLD_UNVALIDATED (a transform exists, scale not validated): ALSO
    `MotionSpace.IMAGE` with px/s kinematics -- the world transform is NOT
    applied to the track at all -- and `geometry_validity=WORLD_UNVALIDATED`
    recorded on the track so a consumer can tell "a transform exists but is
    not trusted" from "no transform". (Unvalidated world-frame speeds exist
    only on `perception.speed_estimation.VehicleSpeedEstimate.
    unvalidated_world_speed`, never as an m/s quantity.)
  - WORLD_VALIDATED: `MotionSpace.WORLD`; position_history is `WorldPosition`
    (metres) from the GROUND point, and the speed comes from the one
    provenance-aware estimator (`speed_estimation.estimate_vehicle_speed`) --
    there is no second speed computation. If that estimator cannot produce a
    VALIDATED, plausible speed (too few samples, a gap, an outlier) the track
    is still returned but with `speed_available=False`: its
    `speed_estimate` is a placeholder no consumer may read, never a
    fabricated 0 m/s.

TRACK IDENTITY: `VehicleTrack.track_id` is `"<camera_id>:<state.track_id>"`
where `state.track_id` is the TrackManager-assigned session-scoped identity
(`"<session_id>-t<serial>"`), never a raw tracker ID.
"""

from __future__ import annotations

from anvesh.perception.motion import compute_image_space_speed
from anvesh.perception.speed_estimation import MPS_TO_KMH, SpeedConfidence, estimate_vehicle_speed
from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection
from anvesh.storage.schemas import (
    CameraObservation,
    GeometryValidity,
    Measurement,
    MotionSpace,
    VehicleTrack,
    WorldPosition,
)
from anvesh.world.calibration import CalibrationProfile


def geometry_validity_for(calibration_profile: CalibrationProfile = None) -> GeometryValidity:
    """The camera's geometry capability. The ONE place callers derive it:
    never from `is_calibrated`."""
    if calibration_profile is None:
        return GeometryValidity.IMAGE
    return calibration_profile.geometry_validity


def motion_space_for(calibration_profile: CalibrationProfile = None) -> MotionSpace:
    """WORLD (metres) iff the geometry is validated metric; IMAGE otherwise
    -- including for a calibrated-but-UNVALIDATED profile."""
    if geometry_validity_for(calibration_profile) == GeometryValidity.WORLD_VALIDATED:
        return MotionSpace.WORLD
    return MotionSpace.IMAGE


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
    """Convert one finished/active TrackState into a VehicleTrack. See the
    module docstring for the three-way branch on geometry validity."""
    validity = geometry_validity_for(calibration_profile)
    track_id = f"{camera_id}:{state.track_id}"

    if validity == GeometryValidity.WORLD_VALIDATED:
        estimate = estimate_vehicle_speed(camera_id, state, calibration_profile)
        try:
            world_trajectory = [
                (*calibration_profile.image_to_world((x, y)), t) for (x, y, t) in state.ground_point_trajectory
            ]
        except ValueError:
            world_trajectory = []
        if estimate.confidence == SpeedConfidence.VALIDATED and world_trajectory:
            metric = estimate.metric_speed_kmh
            speed_estimate = Measurement(value=metric.value / MPS_TO_KMH, error=metric.error / MPS_TO_KMH)
            speed_available = True
        else:
            speed_estimate = Measurement(value=0.0, error=0.0)  # placeholder; speed_available=False
            speed_available = False
        return VehicleTrack(
            track_id=track_id,
            camera_id=camera_id,
            first_seen=state.first_seen,
            last_seen=state.last_seen,
            vehicle_class=state.vehicle_class,
            occlusion_state=state.occlusion_state,
            position_history=[WorldPosition(world_x=wx, world_y=wy, timestamp=t) for (wx, wy, t) in world_trajectory],
            speed_estimate=speed_estimate,
            motion_space=MotionSpace.WORLD,
            geometry_validity=GeometryValidity.WORLD_VALIDATED,
            speed_available=speed_available,
        )

    # IMAGE or WORLD_UNVALIDATED: image-space kinematics only (px, px/s).
    speed = compute_image_space_speed(state.trajectory)
    return VehicleTrack(
        track_id=track_id,
        camera_id=camera_id,
        first_seen=state.first_seen,
        last_seen=state.last_seen,
        vehicle_class=state.vehicle_class,
        occlusion_state=state.occlusion_state,
        position_history=list(state.trajectory),
        speed_estimate=Measurement(value=speed.speed_px_per_s, error=speed.error_px_per_s),
        motion_space=MotionSpace.IMAGE,
        geometry_validity=validity,
    )
