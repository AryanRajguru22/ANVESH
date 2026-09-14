"""Focused tests for scripts/run_s02_c006_showcase.py's config-resolution
and reporting logic ONLY. Does NOT touch the real 16GB CityFlowV2 dataset
or run YOLO/ByteTrack -- `_calibration_path_for` is pure path arithmetic
(tested against a fake lane-config-shaped object under `tmp_path`), and
the `_*_counts` helpers are pure aggregation over hand-authored fixtures.
A real, dataset-backed run of this script is a separate, explicit smoke
test (see the Commit 9A report), not part of the ordinary unit suite.

`scripts/` is not a package (see tests/test_export_ui_data.py for the
same reasoning), so this module is loaded via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_s02_c006_showcase", REPO_ROOT / "scripts" / "run_s02_c006_showcase.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


showcase = _load_module()


@dataclass
class _FakeLaneConfig:
    source_video: str


def test_calibration_path_is_sibling_of_source_video(tmp_path):
    video_path = tmp_path / "S02" / "c006" / "vdo.avi"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"")
    lane_config = _FakeLaneConfig(source_video=str(video_path))

    calibration_path = showcase._calibration_path_for(lane_config)

    assert calibration_path == video_path.parent / "calibration.txt"


def test_calibration_path_never_touches_the_filesystem(tmp_path):
    """Pure path arithmetic -- must not raise even when nothing on disk
    exists at the resolved path (no dataset present, e.g. on CI)."""
    lane_config = _FakeLaneConfig(source_video=str(tmp_path / "nonexistent" / "vdo.avi"))
    calibration_path = showcase._calibration_path_for(lane_config)
    assert calibration_path.name == "calibration.txt"
    assert not calibration_path.exists()


def test_camera_id_and_output_path_are_c006_specific():
    assert showcase.CAMERA_ID == "s02_c006"
    assert showcase.DEFAULT_OUTPUT_JSON.name == "anvesh_s02_c006_data.json"
    # Must never collide with the existing demo.mp4 export.
    assert showcase.DEFAULT_OUTPUT_JSON != REPO_ROOT / "ui" / "data" / "anvesh_demo_data.json"


def test_lane_status_counts_aggregates_across_frames():
    frames = [
        {"boxes": [{"track_id": "1", "lane_status": "assigned"}, {"track_id": "2", "lane_status": "unknown_outside_regions"}]},
        {"boxes": [{"track_id": "1", "lane_status": "assigned"}]},
    ]
    counts = showcase._lane_status_counts(frames)
    assert counts == {"assigned": 2, "unknown_outside_regions": 1}


def test_lane_status_counts_reports_not_computed_when_key_absent():
    frames = [{"boxes": [{"track_id": "1"}]}]
    counts = showcase._lane_status_counts(frames)
    assert counts == {"not_computed": 1}


def test_speed_confidence_counts_deduplicates_by_track_not_by_frame():
    frames = [
        {"boxes": [{"track_id": "1", "speed_confidence": "approximate"}]},
        {"boxes": [{"track_id": "1", "speed_confidence": "approximate"}]},  # same track, later frame
        {"boxes": [{"track_id": "2", "speed_confidence": "unavailable"}]},
    ]
    counts = showcase._speed_confidence_counts(frames)
    assert counts == {"approximate": 1, "unavailable": 1}


def test_candidate_status_counts_reads_window_level_field():
    windows = [
        {"candidate_vehicle": {"status": "resolved"}},
        {"candidate_vehicle": {"status": "not_ranked"}},
        {"candidate_vehicle": {"status": "resolved"}},
    ]
    counts = showcase._candidate_status_counts(windows)
    assert counts == {"resolved": 2, "not_ranked": 1}
