"""Synthetic two-camera development fixture (M2 demo/testing aid).

NOT real calibration data and NOT real two-camera video. This module
fabricates a small, internally-consistent two-camera corridor scenario --
manually-authored reference points, a manually-chosen inter-camera
distance, and a scripted vehicle trajectory -- purely to exercise
calibration -> corridor -> world-space motion end-to-end without needing
a second real recorded clip (see the M2 report's "development data"
section for why this was chosen over sourcing/recording a second video).

Every number in this module is invented for demonstration purposes. None
of it is a measured or validated real corridor -- do not read any output
derived from this fixture as evidence of real-world calibration accuracy.

Both cameras' CalibrationProfiles map into ONE shared corridor world frame
(world_x runs along the corridor from the upstream reference point; camera
B's own reference points are authored `INTER_CAMERA_DISTANCE_M` further
along that same axis) -- this is what makes positions from the two
cameras directly comparable once projected to world space.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.storage.schemas import CameraRole, VehicleClass
from anvesh.world.calibration import CalibrationPoint, CalibrationProfile, fit_planar_homography
from anvesh.world.corridor import CameraPlacement, CorridorTopology

CAMERA_A_ID = "cam-A-upstream"
CAMERA_B_ID = "cam-B-downstream"
CORRIDOR_ID = "dev-corridor-01"
INTER_CAMERA_DISTANCE_M = 40.0
VEHICLE_SPEED_M_PER_S = 8.0  # arbitrary constant (~29 km/h) used only to script this fixture


def build_camera_a_calibration() -> CalibrationProfile:
    """A simple, close-to-fronto-parallel mapping for camera A's frame."""
    reference_points = [
        CalibrationPoint(image_xy=(0.0, 400.0), world_xy=(0.0, 0.0)),
        CalibrationPoint(image_xy=(640.0, 400.0), world_xy=(32.0, 0.0)),
        CalibrationPoint(image_xy=(640.0, 0.0), world_xy=(32.0, 8.0)),
        CalibrationPoint(image_xy=(0.0, 0.0), world_xy=(0.0, 8.0)),
    ]
    return fit_planar_homography("calib-cam-A-dev", CAMERA_A_ID, reference_points)


def build_camera_b_calibration() -> CalibrationProfile:
    """Same style of mapping as camera A, but its road-plane reference
    points are authored INTER_CAMERA_DISTANCE_M further along the shared
    corridor world-x axis -- see module docstring."""
    reference_points = [
        CalibrationPoint(image_xy=(0.0, 400.0), world_xy=(INTER_CAMERA_DISTANCE_M + 0.0, 0.0)),
        CalibrationPoint(image_xy=(640.0, 400.0), world_xy=(INTER_CAMERA_DISTANCE_M + 32.0, 0.0)),
        CalibrationPoint(image_xy=(640.0, 0.0), world_xy=(INTER_CAMERA_DISTANCE_M + 32.0, 8.0)),
        CalibrationPoint(image_xy=(0.0, 0.0), world_xy=(INTER_CAMERA_DISTANCE_M + 0.0, 8.0)),
    ]
    return fit_planar_homography("calib-cam-B-dev", CAMERA_B_ID, reference_points)


def build_corridor_topology() -> CorridorTopology:
    return CorridorTopology(
        corridor_id=CORRIDOR_ID,
        placements=(
            CameraPlacement(camera_id=CAMERA_A_ID, role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(
                camera_id=CAMERA_B_ID,
                role=CameraRole.DOWNSTREAM,
                distance_from_upstream_m=INTER_CAMERA_DISTANCE_M,
            ),
        ),
    )


@dataclass(frozen=True)
class SyntheticFixture:
    corridor: CorridorTopology
    camera_a_calibration: CalibrationProfile
    camera_b_calibration: CalibrationProfile
    camera_a_tracked_detections: tuple
    camera_b_tracked_detections: tuple


def build_fixture() -> SyntheticFixture:
    """One vehicle, constant VEHICLE_SPEED_M_PER_S, sampled 3 times in
    camera A's FOV, then (INTER_CAMERA_DISTANCE_M / VEHICLE_SPEED_M_PER_S)
    seconds later, sampled 3 times in camera B's FOV -- world-space ground
    truth is generated first, then projected back to each camera's image
    plane via its own inverse homography, so calibration correctly
    recovers the known truth."""
    camera_a_calibration = build_camera_a_calibration()
    camera_b_calibration = build_camera_b_calibration()

    sample_offsets_s = (0.0, 0.5, 1.0)
    camera_a_world_xs = [4.0 + VEHICLE_SPEED_M_PER_S * t for t in sample_offsets_s]
    camera_a_tracked = tuple(
        _synthetic_tracked_detection(
            camera_a_calibration, world_x=wx, world_y=4.0, timestamp=t, frame_index=i, track_id="1"
        )
        for i, (wx, t) in enumerate(zip(camera_a_world_xs, sample_offsets_s))
    )

    arrival_delay_s = INTER_CAMERA_DISTANCE_M / VEHICLE_SPEED_M_PER_S
    camera_b_world_xs = [INTER_CAMERA_DISTANCE_M + 4.0 + VEHICLE_SPEED_M_PER_S * t for t in sample_offsets_s]
    camera_b_timestamps = [arrival_delay_s + t for t in sample_offsets_s]
    camera_b_tracked = tuple(
        _synthetic_tracked_detection(
            camera_b_calibration, world_x=wx, world_y=4.0, timestamp=ts, frame_index=i, track_id="1"
        )
        for i, (wx, ts) in enumerate(zip(camera_b_world_xs, camera_b_timestamps))
    )

    return SyntheticFixture(
        corridor=build_corridor_topology(),
        camera_a_calibration=camera_a_calibration,
        camera_b_calibration=camera_b_calibration,
        camera_a_tracked_detections=camera_a_tracked,
        camera_b_tracked_detections=camera_b_tracked,
    )


def _synthetic_tracked_detection(
    calibration: CalibrationProfile,
    world_x: float,
    world_y: float,
    timestamp: float,
    frame_index: int,
    track_id: str,
) -> TrackedDetection:
    image_x, image_y = calibration.world_to_image((world_x, world_y))
    half_box = 15.0
    detection = RawDetection(
        frame_index=frame_index,
        timestamp=timestamp,
        class_name="car",
        vehicle_class=VehicleClass.CAR,
        confidence=0.95,
        bbox_xyxy=(image_x - half_box, image_y - half_box, image_x + half_box, image_y + half_box),
    )
    return TrackedDetection(detection=detection, track_id=track_id)
