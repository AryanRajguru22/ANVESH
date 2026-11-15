"""Rebuild ui/live/anvesh_live_dashboard.html from the ALREADY-COMPUTED
ui/live/anvesh_live_data.json + ui/live/anvesh_live_analyzed_clip.mp4 --
i.e. re-run only the lightweight HTML-shell build, without re-running the
real (GPU, multi-minute) ANVESH pipeline via scripts/analyze_video.py.

Use this after editing ui/anvesh_dashboard.template.html or re-running
scripts/build_offline_assets.py, when the underlying analysis output
hasn't changed.

Usage (from the repository root):

    python scripts/rebuild_live_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_ui_artifact  # noqa: E402
import analyze_video  # noqa: E402

DATA_JSON = analyze_video.DEFAULT_OUTPUT_JSON
CLIP_MP4 = DATA_JSON.parent / "anvesh_live_analyzed_clip.mp4"
OUTPUT_HTML = analyze_video.DEFAULT_OUTPUT_HTML


def main() -> int:
    if not DATA_JSON.is_file():
        print(f"[rebuild] ERROR: {DATA_JSON} not found -- run scripts/analyze_video.py first.", file=sys.stderr)
        return 1
    if not CLIP_MP4.is_file():
        print(f"[rebuild] ERROR: {CLIP_MP4} not found -- run scripts/analyze_video.py first.", file=sys.stderr)
        return 1

    analyze_video._ensure_vendor_assets(OUTPUT_HTML.parent)
    build_ui_artifact.build_live_dashboard_artifact(
        build_ui_artifact.TEMPLATE_PATH,
        data_url=DATA_JSON.name,
        video_url=CLIP_MP4.name,
        output_path=OUTPUT_HTML,
    )
    print(f"[rebuild] Wrote {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size} bytes) from existing {DATA_JSON.name} + {CLIP_MP4.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
