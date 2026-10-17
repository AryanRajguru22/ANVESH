"""Single-camera S02/c006 showcase (Commit 9A).

This is the ONLY file in the repository that names `s02_c006`, `c006`,
or the `E:\\traffic dataset` CityFlowV2 path. Every generic ANVESH module
it calls (`anvesh.perception.pipeline`, `anvesh.pipeline`,
`scripts/export_ui_data.py`) takes camera_id/lane-regions/calibration
-profile as plain, caller-supplied arguments -- none of them hard-codes
this or any other camera.

Composes, in order:
  - `anvesh.world.lane_config.load_lane_config` (Commit 3) -- c006's lane
    geometry, from `configs/lanes/s02_c006.toml`.
  - `anvesh.world.calibration.load_cityflow_calibration` (Commit 5) --
    c006's homography, from the CityFlowV2 `calibration.txt` that sits
    next to `vdo.avi` in the same directory (CityFlowV2's own on-disk
    layout; see `_calibration_path_for` below).
  - `scripts/export_ui_data.real_video_limited_chain` (Commit 9A) -- the
    generic real-video chain, now given a real calibration profile and
    lane regions for the first time.

This is a SINGLE-CAMERA integration: no fusion, no corridor topology, no
propagation, no two-camera c006+c009 corridor (that is a materially
larger, separately-scoped extension -- see `ANVESH_COMMIT9_INTEGRATION_
AUDIT.md` §8, Commit 9b). `run_corridor_window` already reports
corridor/propagation/feedback as honestly UNAVAILABLE for a single
camera; this script does not work around that.

Writes a SEPARATE output file, never `ui/data/anvesh_demo_data.json`
(demo.mp4's own existing export) and never the trimmed dataset video
back into the repo.

Usage (from the repository root, with the venv active):

    python scripts/run_s02_c006_showcase.py
    python scripts/run_s02_c006_showcase.py --sample-seconds 30

Writes: ui/data/anvesh_s02_c006_data.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`
sys.path.insert(0, str(_SCRIPTS_DIR))  # `scripts/` is not a package -- sibling import below

import export_ui_data  # noqa: E402  (scripts/export_ui_data.py)

from anvesh.world.calibration import load_cityflow_calibration  # noqa: E402
from anvesh.world.lane_config import default_lane_config_path, load_lane_config  # noqa: E402

CAMERA_ID = "s02_c006"
CALIBRATION_PROFILE_ID = "calib-s02-c006"
DEFAULT_SAMPLE_SECONDS = 60.0  # a small, bounded representative slice -- not the full ~10-minute source video
DEFAULT_OUTPUT_JSON = REPO_ROOT / "ui" / "data" / "anvesh_s02_c006_data.json"


def _calibration_path_for(lane_config) -> Path:
    """The sibling `calibration.txt` CityFlowV2 places next to `vdo.avi`
    for the same camera -- pure path arithmetic, no I/O, so it is testable
    against an arbitrary `lane_config.source_video` (e.g. under `tmp_path`)
    without the real 16GB dataset present.
    """
    return Path(lane_config.source_video).parent / "calibration.txt"


def _print_progress(message: str) -> None:
    print(f"[s02_c006] {message}", flush=True)


def _lane_status_counts(frames: list) -> dict:
    counts = Counter(
        box.get("lane_status", "not_computed")
        for frame in frames
        for box in frame["boxes"]
    )
    return dict(counts)


def _speed_confidence_counts(frames: list) -> dict:
    """One count per distinct track_id (speed is a per-track quantity --
    counting per box would double-count every frame a track appears in)."""
    seen: dict = {}
    for frame in frames:
        for box in frame["boxes"]:
            if box["track_id"] not in seen:
                seen[box["track_id"]] = box.get("speed_confidence", "not_computed")
    return dict(Counter(seen.values()))


def _candidate_status_counts(windows: list) -> dict:
    return dict(Counter(w["candidate_vehicle"]["status"] for w in windows))


def run_s02_c006_showcase(
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    output_path: Path = DEFAULT_OUTPUT_JSON,
    progress=None,
) -> dict:
    """Run the real single-camera c006 chain over a bounded sample and
    write the resulting dashboard JSON to `output_path`.

    Returns the same dict that is written to JSON, plus a `"report"` key
    summarizing frames/tracks/lane/speed/candidate statistics (never
    fabricated -- computed by scanning the real, already-produced result;
    see requirement 9 of the Commit 9A plan).
    """
    _p = progress if progress is not None else _print_progress

    lane_config = load_lane_config(default_lane_config_path(CAMERA_ID))
    calibration_path = _calibration_path_for(lane_config)
    calibration = load_cityflow_calibration(CALIBRATION_PROFILE_ID, CAMERA_ID, calibration_path)
    _p(f"Loaded lane config ({len(lane_config.regions)} regions) and calibration for {CAMERA_ID}.")

    t0 = time.perf_counter()
    real_video = export_ui_data.real_video_limited_chain(
        video_path=lane_config.source_video,
        camera_id=CAMERA_ID,
        sample_seconds=sample_seconds,
        pre_trim_to_sample=True,
        progress=_p,
        calibration_profile=calibration,
        lane_regions=lane_config.regions,
    )
    elapsed = time.perf_counter() - t0
    _p(f"Real-video analysis complete in {elapsed:.1f}s.")

    report = {
        "frames_processed": len(real_video["frames"]),
        "tracks": real_video["total_tracks"],
        "lane_status_counts": _lane_status_counts(real_video["frames"]),
        "speed_confidence_counts": _speed_confidence_counts(real_video["frames"]),
        "candidate_status_counts": _candidate_status_counts(real_video["windows"]),
    }
    _p(f"Report: {report}")

    data = {"real_video": real_video, "report": report}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # encoding="utf-8" is explicit, not the default, because Windows'
    # locale-default text encoding (often cp1252) silently mis-encodes
    # non-ASCII output -- discovered by this pass's own OUTLIER speed
    # label ("UNRELIABLE — OUTLIER"), which contains an em dash.
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _p(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-seconds", type=float, default=DEFAULT_SAMPLE_SECONDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    run_s02_c006_showcase(sample_seconds=args.sample_seconds, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
