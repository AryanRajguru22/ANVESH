"""M1 demo entry point.

Usage (from the repository root, with the venv active):

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --video path/to/other.mp4
    python scripts/run_pipeline.py --no-visualize

Runs the M1 perception pipeline (YOLO detection + ByteTrack tracking +
track management + schema conversion) on a single camera's recorded video
and, by default, writes an annotated debug video to
configs/experiment/m1_pipeline.toml's `output_video_path` showing bounding
boxes, vehicle class, persistent track ID, frame/time, and trajectory
trails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.config import load_config
from anvesh.logging_setup import configure_logging, get_logger
from anvesh.perception.config import load_pipeline_config
from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.tracking import ByteTrackYoloTracker

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ANVESH M1 single-camera perception pipeline.")
    parser.add_argument("--video", type=str, default=None, help="Override the configured video path.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip writing the annotated debug video.")
    args = parser.parse_args()

    app_config = load_config()
    configure_logging(app_config.logging)

    pipeline_config = load_pipeline_config()
    video_path = Path(args.video) if args.video else REPO_ROOT / pipeline_config.video_path
    output_path = None if args.no_visualize else REPO_ROOT / pipeline_config.output_video_path

    logger.info("Running M1 pipeline on %s", video_path)

    tracker = ByteTrackYoloTracker(
        model_name=str(REPO_ROOT / pipeline_config.model_name),
        confidence_threshold=pipeline_config.confidence_threshold,
        tracker_config=pipeline_config.tracker_config,
    )

    result = run_m1_pipeline(
        video_path=video_path,
        camera_id=pipeline_config.camera_id,
        tracker=tracker,
        model_id=pipeline_config.model_name,
        model_version="pretrained",
        max_missed_frames=pipeline_config.max_missed_frames,
        partial_after_missed_frames=pipeline_config.partial_after_missed_frames,
        visualization_output_path=output_path,
    )

    logger.info(
        "Processed %d frames -> %d CameraObservations, %d VehicleTracks",
        result.frame_count,
        len(result.camera_observations),
        len(result.vehicle_tracks),
    )
    if output_path is not None:
        logger.info("Annotated debug video written to %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
