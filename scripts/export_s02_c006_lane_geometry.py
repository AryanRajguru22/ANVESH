"""Presentation-layer sidecar: serialize c006's ALREADY-CONFIGURED lane
geometry (`configs/lanes/s02_c006.toml`, loaded through the existing,
frozen `anvesh.world.lane_config.load_lane_config`) into a small JSON file
the V1 Dataset Showcase UI can fetch and draw -- so the dashboard never
hand-invents lane polygons of its own (see
`ui/anvesh_dashboard.template.html`'s showcase view).

This performs NO lane assignment, no detection, no re-derivation of the
polygons -- it only reads the same config file
`scripts/run_s02_c006_showcase.py` already loads and reshapes it into
plain JSON. It never touches `ui/data/anvesh_s02_c006_data.json` (the
expensive, GPU-produced detection/tracking/evidence artifact) or reruns
any part of the perception/evidence/ranking pipeline.

The one piece of information this file adds beyond a literal dump of the
TOML is the `provisional` flag on `L2`. `anvesh.world.lane_geometry.
LaneRegion`/`anvesh.world.lane_config.LaneCameraConfig` carry no such
field (checked directly against their source) -- "L2 is PROVISIONAL" is
today only a human-readable comment in the TOML file, not machine-
readable data. `_PROVISIONAL_LANE_IDS` below is this script's own
presentation-layer transcription of that comment, not a new engine rule;
if `configs/lanes/s02_c006.toml` is ever revised to make this
machine-readable, this constant should be replaced by reading it from
there instead.

Usage (from the repository root, with the venv active):

    python scripts/export_s02_c006_lane_geometry.py

Writes: ui/data/anvesh_s02_c006_lane_geometry.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.world.lane_config import default_lane_config_path, load_lane_config  # noqa: E402

CAMERA_ID = "s02_c006"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "ui" / "data" / "anvesh_s02_c006_lane_geometry.json"

# Transcribed from configs/lanes/s02_c006.toml's own REVISION HISTORY /
# KNOWN LIMITATIONS prose (see this module's docstring) -- not derived
# from any loader field, because none exists yet.
_PROVISIONAL_LANE_IDS = {"L2"}
_SOURCE_CONFIG_RELPATH = "configs/lanes/s02_c006.toml"


def export_lane_geometry(camera_id: str = CAMERA_ID, output_path: Path = DEFAULT_OUTPUT_JSON) -> dict:
    lane_config = load_lane_config(default_lane_config_path(camera_id))

    data = {
        "camera_id": lane_config.camera_id,
        "image_width": lane_config.image_width,
        "image_height": lane_config.image_height,
        "source_config": _SOURCE_CONFIG_RELPATH,
        "provenance": "Camera-specific configured geometry -- valid only for this fixed camera's framing, not a general-purpose lane model.",
        "lanes": [
            {
                "lane_id": region.lane_id,
                "vertices": [list(v) for v in region.vertices],
                "provisional": region.lane_id in _PROVISIONAL_LANE_IDS,
            }
            for region in lane_config.regions
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")
    return data


def main() -> int:
    export_lane_geometry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
