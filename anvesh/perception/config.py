"""M1 pipeline configuration loading (video path, model, thresholds).

Mirrors the stdlib-tomllib pattern already used by `anvesh.config` for the
app-level config -- kept as a separate loader because this config is
perception-pipeline-specific (video path, detector/tracker settings), not
application-wide.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PIPELINE_CONFIG_PATH = REPO_ROOT / "configs" / "experiment" / "m1_pipeline.toml"


@dataclass(frozen=True)
class PipelineConfig:
    camera_id: str
    video_path: str
    model_name: str
    confidence_threshold: float
    tracker_config: str
    max_missed_frames: int
    partial_after_missed_frames: int
    output_video_path: str


def load_pipeline_config(path: Path | str | None = None) -> PipelineConfig:
    config_path = Path(path) if path is not None else DEFAULT_PIPELINE_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config file not found: {config_path}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    try:
        p = data["pipeline"]
    except KeyError as exc:
        raise ValueError(f"Pipeline config missing required section: {exc}") from exc

    try:
        return PipelineConfig(
            camera_id=p["camera_id"],
            video_path=p["video_path"],
            model_name=p["model_name"],
            confidence_threshold=p["confidence_threshold"],
            tracker_config=p["tracker_config"],
            max_missed_frames=p["max_missed_frames"],
            partial_after_missed_frames=p["partial_after_missed_frames"],
            output_video_path=p["output_video_path"],
        )
    except KeyError as exc:
        raise ValueError(f"Pipeline config missing required key: {exc}") from exc
