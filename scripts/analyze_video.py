"""Reusable CLI: run the real, existing ANVESH pipeline on an ARBITRARY
video and build a separate, self-contained live dashboard -- for a judge
demonstration -- WITHOUT touching the existing, verified demo files
(ui/data/anvesh_demo_data.json, ui/anvesh_dashboard.html).

This does not introduce a new pipeline. It reuses, unmodified in
behavior for their existing callers:

  - `export_ui_data.real_video_limited_chain` (scripts/export_ui_data.py)
    -- the only existing real-video full chain (video -> M1 detection/
    tracking -> windowing -> evidence -> Dempster-Shafer fusion -> cause
    ranking -> safety events). It was parameterized (video/camera/model/
    sample-length, all defaulting to its original hardcoded constants) so
    this script can point it at a different file; scripts/export_ui_data.
    py's own `main()` still calls it with zero arguments and is therefore
    byte-for-byte unaffected by this file's existence.
  - `export_ui_data.scenario_cross_camera_agreement` /
    `scenario_camera_disagreement` / `scenario_reliability_degradation`
    -- unmodified, so the live dashboard's Synthetic Demonstration mode
    keeps working exactly like the existing demo's.
  - `build_ui_artifact.build_dashboard_artifact` -- the existing
    template-substitution logic (extracted, not duplicated), used to
    build a SEPARATE html artifact from the SAME template file the
    verified demo uses.

Usage (from the repository root, with the venv active):

    python scripts/analyze_video.py --video path/to/traffic_video.mp4
    python scripts/analyze_video.py --video path/to/traffic_video.mp4 --camera-id judge-cam --sample-seconds 60

Writes (never touches the existing demo files):

    ui/live/anvesh_live_data.json
    ui/live/anvesh_live_dashboard.html

For a single, uncalibrated camera (true for any arbitrary judge video
with no calibration/topology supplied), corridor state, propagation, and
feedback are honestly reported UNAVAILABLE by the underlying engine --
exactly as they already are for demo.mp4 in the existing demo. This
script does not, and cannot, change that; it only runs the same real
engine on a different file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`
sys.path.insert(0, str(_SCRIPTS_DIR))  # `scripts/` is not a package -- sibling imports below

import build_ui_artifact  # noqa: E402  (scripts/build_ui_artifact.py)
import export_ui_data  # noqa: E402  (scripts/export_ui_data.py)

DEFAULT_OUTPUT_JSON = REPO_ROOT / "ui" / "live" / "anvesh_live_data.json"
DEFAULT_OUTPUT_HTML = REPO_ROOT / "ui" / "live" / "anvesh_live_dashboard.html"
DEFAULT_CAMERA_ID = "live-cam"
DEFAULT_SAMPLE_SECONDS = 60.0

# The two files this script must NEVER write to -- checked defensively at
# the top of main(), even though the CLI defaults already point elsewhere.
_PROTECTED_PATHS = {
    (REPO_ROOT / "ui" / "data" / "anvesh_demo_data.json").resolve(),
    (REPO_ROOT / "ui" / "anvesh_dashboard.html").resolve(),
}


def _print_progress(message: str) -> None:
    print(f"[ANVESH] {message}", flush=True)


def analyze_video(
    video_path: Path,
    camera_id: str = DEFAULT_CAMERA_ID,
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    model_name: str = export_ui_data.DEMO_MODEL_NAME,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_html: Path = DEFAULT_OUTPUT_HTML,
) -> dict:
    """Run the real ANVESH chain on `video_path` and write a separate
    live JSON + dashboard HTML pair. Returns the same dict written to JSON.

    Reuses `export_ui_data.real_video_limited_chain` with
    `pre_trim_to_sample=True` so an arbitrary-length input video is
    bounded to `sample_seconds` of actual detection/tracking work
    (see that function's own docstring for why this matters -- without
    it, an unbounded `run_m1_pipeline` call would process the ENTIRE
    input file regardless of `sample_seconds`).

    The dashboard embeds only the analyzed clip (via
    `trimmed_output_path`), not the full original source video -- a
    judge-supplied file could be arbitrarily large, and embedding
    everything beyond what was actually analyzed would risk an
    unusably large/slow HTML artifact for no benefit. If the source is
    already no longer than `sample_seconds`, nothing is trimmed and the
    original file is embedded as-is (same behavior as the existing demo).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_json = Path(output_json).resolve()
    output_html = Path(output_html).resolve()
    trimmed_clip_path = output_html.parent / "anvesh_live_analyzed_clip.mp4"

    t0 = time.perf_counter()
    real_video = export_ui_data.real_video_limited_chain(
        video_path=video_path,
        camera_id=camera_id,
        model_name=model_name,
        sample_seconds=sample_seconds,
        pre_trim_to_sample=True,
        trimmed_output_path=trimmed_clip_path,
        progress=_print_progress,
    )
    elapsed = time.perf_counter() - t0
    _print_progress(f"Real-video analysis complete in {elapsed:.1f}s.")
    embed_video_path = trimmed_clip_path if trimmed_clip_path.is_file() else video_path

    _print_progress("Building synthetic-scenario fixtures (unmodified; keeps Synthetic Demonstration mode working)...")
    data = {
        "synthetic": {
            "scenarios": [
                export_ui_data.scenario_cross_camera_agreement(),
                export_ui_data.scenario_camera_disagreement(),
                export_ui_data.scenario_reliability_degradation(),
            ]
        },
        "real_video": real_video,
    }

    if output_json in _PROTECTED_PATHS or output_html in _PROTECTED_PATHS:
        raise ValueError(
            "Refusing to write over a protected demo file. "
            f"output_json={output_json}, output_html={output_html}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2))
    _print_progress(f"Wrote JSON: {output_json} ({output_json.stat().st_size} bytes)")

    # Build the LIGHTWEIGHT live dashboard shell: references output_json and
    # embed_video_path as plain external files (the browser's own fetch()/
    # <video src> loading), never embedding either as a multi-hundred-MB
    # inline JS string -- see build_ui_artifact.build_live_dashboard_artifact
    # for why (this is what actually fixed the 30-60s navigation stalls).
    _ensure_vendor_assets(output_html.parent)
    build_ui_artifact.build_live_dashboard_artifact(
        build_ui_artifact.TEMPLATE_PATH,
        data_url=output_json.name,
        video_url=embed_video_path.name,
        output_path=output_html,
    )
    _print_progress(
        f"Wrote dashboard: {output_html} ({output_html.stat().st_size} bytes; "
        f"video stays external at {embed_video_path.name}, not embedded)"
    )

    return data


def _ensure_vendor_assets(output_dir: Path) -> None:
    """Make sure the locally-vendored offline CSS/fonts (built once by
    scripts/build_offline_assets.py into ui/live/vendor/) are available next
    to `output_dir`'s dashboard HTML, copying them there if this run's
    output directory isn't the default ui/live/."""
    import shutil

    canonical_vendor = DEFAULT_OUTPUT_HTML.parent / "vendor"
    target_vendor = output_dir / "vendor"
    if target_vendor.resolve() == canonical_vendor.resolve():
        if not canonical_vendor.is_dir():
            raise FileNotFoundError(
                f"Offline vendor assets not found at {canonical_vendor}. "
                "Run `python scripts/build_offline_assets.py` once (with internet access) first."
            )
        return
    if not canonical_vendor.is_dir():
        raise FileNotFoundError(
            f"Offline vendor assets not found at {canonical_vendor}. "
            "Run `python scripts/build_offline_assets.py` once (with internet access) first."
        )
    if not target_vendor.is_dir():
        shutil.copytree(canonical_vendor, target_vendor)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the real ANVESH pipeline on an arbitrary traffic video and build a separate live dashboard.",
    )
    parser.add_argument("--video", required=True, type=str, help="Path to the traffic video to analyze.")
    parser.add_argument(
        "--camera-id", type=str, default=DEFAULT_CAMERA_ID,
        help=f"Camera id label for this run (default: {DEFAULT_CAMERA_ID!r}).",
    )
    parser.add_argument(
        "--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS,
        help=f"How much of the video (from t=0) to analyze, in seconds (default: {DEFAULT_SAMPLE_SECONDS}).",
    )
    parser.add_argument(
        "--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON),
        help=f"Output path for the live JSON (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--output-html", type=str, default=str(DEFAULT_OUTPUT_HTML),
        help=f"Output path for the live dashboard HTML (default: {DEFAULT_OUTPUT_HTML}).",
    )
    parser.add_argument(
        "--model", type=str, default=export_ui_data.DEMO_MODEL_NAME,
        help=f"YOLO model weights path, relative to the repo root unless absolute (default: {export_ui_data.DEMO_MODEL_NAME}).",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    _print_progress("Loading video...")
    if not video_path.is_file():
        print(f"[ANVESH] ERROR: video file not found: {video_path}", file=sys.stderr)
        return 1

    try:
        analyze_video(
            video_path=video_path,
            camera_id=args.camera_id,
            sample_seconds=args.sample_seconds,
            model_name=args.model,
            output_json=Path(args.output_json),
            output_html=Path(args.output_html),
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the operator, not swallowed
        print(f"[ANVESH] ERROR: {exc}", file=sys.stderr)
        return 1

    output_html_path = Path(args.output_html).resolve()
    print()
    print("[ANVESH] Analysis complete.")
    print()
    print("Output:")
    print(f"  JSON:      {Path(args.output_json).resolve()}")
    print(f"  Dashboard: {output_html_path}")
    print(f"  Analyzed clip (embedded in the dashboard, also kept standalone): {output_html_path.parent / 'anvesh_live_analyzed_clip.mp4'}")
    print()
    print("The existing verified demo files were NOT modified:")
    print(f"  {REPO_ROOT / 'ui' / 'data' / 'anvesh_demo_data.json'}")
    print(f"  {REPO_ROOT / 'ui' / 'anvesh_dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
