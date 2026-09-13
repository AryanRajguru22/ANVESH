"""Characterization tests for `anvesh.perception.visualize.draw_frame` --
its behavior is unchanged by Commit 8 (`anvesh/rendering.py`), which adds
a separate, new overlay function rather than touching this one. These
tests exist to make that "unchanged" claim checkable, since this module
had no dedicated test file before."""

import numpy as np

from anvesh.perception.track_manager import TrackState
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.perception.visualize import draw_frame
from anvesh.storage.schemas import OcclusionState, VehicleClass


def _tracked_detection(track_id="1"):
    detection = RawDetection(
        frame_index=0, timestamp=0.0, class_name="car", vehicle_class=VehicleClass.CAR,
        confidence=0.9, bbox_xyxy=(10.0, 20.0, 50.0, 80.0),
    )
    return TrackedDetection(detection=detection, track_id=track_id)


def test_draw_frame_returns_same_shape_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_frame(image, frame_index=0, timestamp=0.0, tracked_detections=[], active_tracks={})
    assert annotated.shape == image.shape


def test_draw_frame_draws_green_box_for_every_detection():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td = _tracked_detection()
    annotated = draw_frame(image, frame_index=0, timestamp=0.0, tracked_detections=[td], active_tracks={})
    # bottom edge of the box, at (30, 80) -- clear of the label text drawn
    # above the box -- box color is (60, 200, 60) BGR
    assert tuple(annotated[80, 30]) == (60, 200, 60)


def test_draw_frame_does_not_mutate_input_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()
    draw_frame(image, frame_index=0, timestamp=0.0, tracked_detections=[_tracked_detection()], active_tracks={})
    assert np.array_equal(image, original)


def test_draw_frame_draws_trajectory_trail_when_track_state_present():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td = _tracked_detection()
    state = TrackState(
        track_id="1", vehicle_class=VehicleClass.CAR, first_seen=0.0, last_seen=1.0,
        occlusion_state=OcclusionState.VISIBLE, trajectory=[(10.0, 10.0, 0.0), (40.0, 10.0, 0.5)],
    )
    annotated = draw_frame(
        image, frame_index=0, timestamp=0.0, tracked_detections=[td], active_tracks={"1": state}
    )
    # trail color (60, 160, 255) BGR should appear somewhere along the line
    assert (annotated == np.array([60, 160, 255])).all(axis=-1).any()


def test_draw_frame_with_no_detections_still_draws_header_and_is_unchanged_elsewhere():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_frame(image, frame_index=5, timestamp=1.23, tracked_detections=[], active_tracks={})
    # header text is drawn near (10, 24); background elsewhere stays black
    assert tuple(annotated[90, 90]) == (0, 0, 0)
