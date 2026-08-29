"""Single-camera pipeline orchestration: wires ingestion -> tracking ->
track management -> schema conversion (+ optional visualization) together.

Deliberately thin -- each stage lives in its own, independently testable
module (ingestion.py, tracking.py, track_manager.py, motion.py,
schema_conversion.py, visualize.py). This module only sequences them; it
is not itself the entry point (see scripts/run_pipeline.py).

`calibration_profile` is optional (M2): when given and calibrated, tracks
come out in world-space (metres); otherwise this camera's tracks stay
image-space, exactly as in M1. See `schema_conversion.build_vehicle_track`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from anvesh.perception.ingestion import VideoFileFrameSource
from anvesh.perception.schema_conversion import build_camera_observation, build_vehicle_track
from anvesh.perception.track_manager import TrackManager
from anvesh.perception.visualize import VideoWriter, draw_frame


@dataclass
class M1PipelineResult:
    camera_observations: list = field(default_factory=list)
    vehicle_tracks: list = field(default_factory=list)
    frame_count: int = 0


def run_m1_pipeline(
    video_path,
    camera_id: str,
    tracker,
    model_id: str,
    model_version: str,
    max_missed_frames: int = 5,
    partial_after_missed_frames: int = 1,
    visualization_output_path=None,
    calibration_profile=None,
) -> M1PipelineResult:
    frame_source = VideoFileFrameSource(video_path)
    track_manager = TrackManager(
        max_missed_frames=max_missed_frames,
        partial_after_missed_frames=partial_after_missed_frames,
    )
    result = M1PipelineResult()

    writer = None
    try:
        for frame in frame_source:
            raw_detections, tracked_detections = tracker.update(frame.image, frame.index, frame.timestamp)
            track_manager.update(frame.timestamp, tracked_detections)

            result.camera_observations.append(
                build_camera_observation(
                    camera_id=camera_id,
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    detections=raw_detections,
                    model_id=model_id,
                    model_version=model_version,
                )
            )
            result.frame_count += 1

            if visualization_output_path is not None:
                if writer is None:
                    height, width = frame.image.shape[:2]
                    writer = VideoWriter(
                        visualization_output_path, fps=frame_source.fps(), frame_size=(width, height)
                    )
                annotated = draw_frame(
                    frame.image,
                    frame.index,
                    frame.timestamp,
                    tracked_detections,
                    track_manager.active_tracks,
                )
                writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()

    track_manager.finalize()
    result.vehicle_tracks = [
        build_vehicle_track(camera_id, state, calibration_profile=calibration_profile)
        for state in track_manager.all_tracks()
    ]
    return result
