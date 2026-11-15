"""Build ui/anvesh_dashboard.html from the template + exported data.

Run this after re-running scripts/export_ui_data.py (e.g. after a real
ANVESH engine change) to refresh the dashboard with new real output.

Usage (from the repository root):

    python scripts/export_ui_data.py
    python scripts/build_ui_artifact.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "ui" / "anvesh_dashboard.template.html"
DATA_PATH = REPO_ROOT / "ui" / "data" / "anvesh_demo_data.json"
OUTPUT_PATH = REPO_ROOT / "ui" / "anvesh_dashboard.html"
DATA_PLACEHOLDER = "__ANVESH_DATA_JSON__"
VIDEO_PLACEHOLDER = "__ANVESH_VIDEO_DATA_URI__"

# The V1 Dataset Showcase's own artifacts (produced by
# scripts/run_s02_c006_showcase.py and scripts/export_s02_c006_lane_
# geometry.py) -- a SEPARATE dataset from `DATA_PATH`/demo.mp4, never
# merged into it. `build_live_dashboard_artifact` below points a live
# dashboard's `window.ANVESH_LIVE_CONFIG` at whichever of these already
# exist on disk, so the showcase "just works" for any live build without
# every caller having to know these paths.
SHOWCASE_DATA_JSON = REPO_ROOT / "ui" / "data" / "anvesh_s02_c006_data.json"
SHOWCASE_LANE_JSON = REPO_ROOT / "ui" / "data" / "anvesh_s02_c006_lane_geometry.json"
SHOWCASE_REFERENCE_IMAGE = REPO_ROOT / "docs" / "demo_reference" / "s02_c006_c008_inspection" / "c006_start_f0.jpg"


def build_dashboard_artifact(template_path: Path, data: dict, video_path: Path, output_path: Path) -> int:
    """Substitute `data` and `video_path`'s bytes into the template at
    `template_path`, writing the result to `output_path`. Returns the
    number of video bytes embedded.

    Pure function, no hardcoded paths -- this is the exact substitution
    `main()` below has always performed for the frozen demo artifact,
    extracted so a second caller (scripts/analyze_video.py, for a live
    judge-demo video) can produce a SEPARATE artifact the same way,
    without duplicating this logic or touching the demo's own paths.
    """
    import base64
    import json

    template = template_path.read_text(encoding="utf-8")
    compact_json = json.dumps(data, separators=(",", ":"))
    # Escape any literal "</script" so the embedded JSON can never
    # terminate the surrounding <script> tag early.
    safe_json = compact_json.replace("</script", "<\\/script")
    if DATA_PLACEHOLDER not in template:
        raise ValueError(f"template is missing the {DATA_PLACEHOLDER} placeholder")
    if VIDEO_PLACEHOLDER not in template:
        raise ValueError(f"template is missing the {VIDEO_PLACEHOLDER} placeholder")

    # Embed the real video as a data: URI -- the ORIGINAL file is already
    # H.264-encoded (confirmed via cv2 fourcc probe), so no re-encoding is
    # needed or attempted; this sidesteps this system's broken OpenH264
    # encoder entirely (see the final report's "known limitations").
    video_bytes = video_path.read_bytes()
    video_data_uri = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("ascii")

    final = template.replace(DATA_PLACEHOLDER, safe_json).replace(VIDEO_PLACEHOLDER, video_data_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final, encoding="utf-8")
    return len(video_bytes)


LIVE_HEAD_START_MARKER = '<link rel="preconnect" href="https://fonts.googleapis.com">'
LIVE_HEAD_END_MARKER = "</script>\n<style>"


def build_live_dashboard_artifact(
    template_path: Path, data_url: str, video_url: str, output_path: Path, vendor_dir_name: str = "vendor",
    include_showcase: bool = True,
) -> None:
    """Build a LIGHTWEIGHT live-presentation HTML shell from the same
    template used for the protected static demo, but WITHOUT embedding the
    JSON/video payload and WITHOUT any CDN dependency.

    Unlike `build_dashboard_artifact` (which inlines everything into one
    huge self-contained file for the frozen, protected demo), this leaves
    `ANVESH_DATA`/`VIDEO_SRC` undefined in the shipped HTML so the
    template's own external-source fallback (the last <script> block in
    ui/anvesh_dashboard.template.html) fetches `data_url` and references
    `video_url` as plain external files -- the browser's native
    fetch()/<video src> loading paths, not a multi-hundred-MB inline JS
    string. This is the fix for the live dashboard's 30-60s navigation
    stalls: nothing large is embedded, so there is nothing large to
    re-parse or hold in a permanently-retained JS string for the life of
    the page (see docs/LIVE_DASHBOARD_PERFORMANCE.md for the full
    root-cause writeup).

    Also swaps the CDN Tailwind/Google Fonts <head> block for the locally
    vendored equivalent built by scripts/build_offline_assets.py, so the
    live artifact needs no internet access at presentation time.

    `data_url`/`video_url` are resolved relative to `output_path`'s own
    directory by the browser (e.g. "anvesh_live_data.json",
    "anvesh_live_analyzed_clip.mp4" when both sit alongside `output_path`).

    `include_showcase` (default True): when the V1 Dataset Showcase's own
    artifacts (`SHOWCASE_DATA_JSON`/`SHOWCASE_LANE_JSON`/
    `SHOWCASE_REFERENCE_IMAGE` above) exist on disk, their paths -- each
    resolved relative to `output_path`'s own directory, exactly like
    `data_url`/`video_url` -- are added to `window.ANVESH_LIVE_CONFIG` as
    `showcaseDataUrl`/`showcaseLaneUrl`/`showcaseImageUrl`, so the V1
    Showcase mode (`ui/anvesh_dashboard.template.html`) can fetch them.
    A missing artifact is simply omitted (never fabricated), and the
    showcase's own UI reports "unavailable" for whichever key is absent.
    Set False to build a shell with no showcase config at all.
    """
    import json as json_mod
    import os

    template = template_path.read_text(encoding="utf-8")
    start = template.index(LIVE_HEAD_START_MARKER)
    end = template.index(LIVE_HEAD_END_MARKER, start) + len("</script>")

    config = {"dataUrl": data_url, "videoUrl": video_url}
    if include_showcase:
        out_dir = output_path.parent

        def _rel(target: Path) -> str:
            return os.path.relpath(target, start=out_dir).replace(os.sep, "/")

        if SHOWCASE_DATA_JSON.is_file():
            config["showcaseDataUrl"] = _rel(SHOWCASE_DATA_JSON)
        if SHOWCASE_LANE_JSON.is_file():
            config["showcaseLaneUrl"] = _rel(SHOWCASE_LANE_JSON)
        if SHOWCASE_REFERENCE_IMAGE.is_file():
            config["showcaseImageUrl"] = _rel(SHOWCASE_REFERENCE_IMAGE)
    config_json = json_mod.dumps(config)
    local_head = (
        f'<link rel="stylesheet" href="{vendor_dir_name}/fonts.css">\n'
        f'<link rel="stylesheet" href="{vendor_dir_name}/tailwind.css">\n'
        f"<script>window.ANVESH_LIVE_CONFIG = {config_json};</script>"
    )
    result = template[:start] + local_head + template[end:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")


def main() -> int:
    import json

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    video_path = REPO_ROOT / data["real_video"]["video_path"]
    video_bytes_len = build_dashboard_artifact(TEMPLATE_PATH, data, video_path, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes, video={video_bytes_len} bytes embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
