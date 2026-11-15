"""Rebuild ui/live/anvesh_dashboard_live.html -- the FULL real-video live
dashboard shell that points at the existing, already-computed demo.mp4
export (`ui/data/anvesh_demo_data.json`) and `demo.mp4` itself, both left
completely untouched. This is the "current localhost dashboard" served by
scripts/serve_presentation.py at `/ui/live/anvesh_dashboard_live.html`.

Until this script existed, this file had been hand-built once via
`build_ui_artifact.build_live_dashboard_artifact` with no script to redo
it -- use this after editing `ui/anvesh_dashboard.template.html` (e.g. to
add/change a dashboard feature) so the change actually reaches this file,
without re-running any GPU pipeline (demo.mp4's own data is never
recomputed here).

Also refreshes `window.ANVESH_LIVE_CONFIG.showcaseDataUrl/showcaseLaneUrl/
showcaseImageUrl` (see `build_ui_artifact.build_live_dashboard_artifact`)
whenever the V1 Dataset Showcase's own artifacts
(`ui/data/anvesh_s02_c006_data.json`, etc.) already exist on disk.

Usage (from the repository root):

    python scripts/rebuild_dashboard_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_ui_artifact  # noqa: E402
import analyze_video  # noqa: E402 (for _ensure_vendor_assets)

DATA_JSON = build_ui_artifact.DATA_PATH  # ui/data/anvesh_demo_data.json -- untouched, read-only here
DEMO_VIDEO = REPO_ROOT / "demo.mp4"
OUTPUT_HTML = REPO_ROOT / "ui" / "live" / "anvesh_dashboard_live.html"


def main() -> int:
    if not DATA_JSON.is_file():
        print(f"[rebuild] ERROR: {DATA_JSON} not found.", file=sys.stderr)
        return 1
    if not DEMO_VIDEO.is_file():
        print(f"[rebuild] ERROR: {DEMO_VIDEO} not found.", file=sys.stderr)
        return 1

    analyze_video._ensure_vendor_assets(OUTPUT_HTML.parent)
    build_ui_artifact.build_live_dashboard_artifact(
        build_ui_artifact.TEMPLATE_PATH,
        data_url="../data/anvesh_demo_data.json",
        video_url="../../demo.mp4",
        output_path=OUTPUT_HTML,
    )
    print(f"[rebuild] Wrote {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size} bytes) from existing anvesh_demo_data.json + demo.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
