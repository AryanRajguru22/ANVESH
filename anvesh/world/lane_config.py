"""Per-camera lane geometry configuration loading (blueprint-adjacent; see
`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` SS8.1/SS11 Commit 3).

Mirrors the stdlib-tomllib pattern already used by `anvesh.config` and
`anvesh.perception.config` -- kept as a separate loader because lane
geometry is camera-specific, hand-authored configuration, not
application-wide or pipeline-stage configuration.

This module only turns a TOML file into `anvesh.world.lane_geometry`
objects. It performs no vehicle-to-lane assignment and is not imported by
`anvesh/perception/` or `anvesh/pipeline.py` -- see `configs/lanes/*.toml`
for the "configuration only, not wired in yet" scope of this commit.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from anvesh.world.lane_geometry import LaneRegion

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LANE_CONFIG_DIR = REPO_ROOT / "configs" / "lanes"


@dataclass(frozen=True)
class LaneCameraConfig:
    """One camera's lane geometry, as loaded from `configs/lanes/*.toml`."""

    camera_id: str
    source_video: str
    image_width: int
    image_height: int
    regions: tuple

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"image_width/image_height must be positive, got {self.image_width}x{self.image_height}"
            )
        lane_ids = [region.lane_id for region in self.regions]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError(f"camera '{self.camera_id}' lane config has duplicate lane_id(s): {lane_ids}")


def load_lane_config(path: Path | str) -> LaneCameraConfig:
    """Load one camera's lane geometry from a TOML file.

    Raises `FileNotFoundError` if `path` does not exist, and `ValueError`
    for a missing required key or an invalid polygon (delegated to
    `LaneRegion.__post_init__`, so the same fail-fast rules apply here as
    everywhere else `LaneRegion` is constructed).
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Lane config file not found: {config_path}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    try:
        cam = data["camera"]
    except KeyError as exc:
        raise ValueError(f"Lane config missing required section: {exc}") from exc

    try:
        camera_id = cam["camera_id"]
        source_video = cam["source_video"]
        image_width = cam["image_width"]
        image_height = cam["image_height"]
    except KeyError as exc:
        raise ValueError(f"Lane config [camera] section missing required key: {exc}") from exc

    lanes = data.get("lane", [])
    regions = []
    for lane in lanes:
        try:
            lane_id = lane["lane_id"]
            vertices = lane["vertices"]
        except KeyError as exc:
            raise ValueError(f"Lane config [[lane]] entry missing required key: {exc}") from exc
        regions.append(LaneRegion(lane_id=lane_id, vertices=tuple(tuple(v) for v in vertices)))

    return LaneCameraConfig(
        camera_id=camera_id,
        source_video=source_video,
        image_width=image_width,
        image_height=image_height,
        regions=tuple(regions),
    )


def default_lane_config_path(camera_id: str) -> Path:
    """The conventional `configs/lanes/<camera_id>.toml` path for a camera_id
    of the form used by these configs (e.g. `s02_c006`). Does not check the
    file exists -- callers pass the result straight to `load_lane_config`,
    which raises `FileNotFoundError` if it doesn't."""
    return DEFAULT_LANE_CONFIG_DIR / f"{camera_id}.toml"
