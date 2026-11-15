"""Per-camera tracker settings: validated, repo-relative, fail closed, and wired into the tracker factory without touching the library defaults."""

from pathlib import Path

import pytest

from anvesh.app.cameras import ConfigError, parse_camera_config
from anvesh.app.runner import _repo_file, default_tracker_factory


def test_absent_tracker_block_keeps_the_library_defaults():
    assert parse_camera_config({}).tracker == {}


def test_valid_tracker_block_is_parsed():
    cfg = parse_camera_config({"tracker": {"conf": 0.1, "config": "demo/tools/trackers/anvesh_bytetrack_m09.yaml"}})
    assert cfg.tracker == {"conf": 0.1, "config": "demo/tools/trackers/anvesh_bytetrack_m09.yaml"}


@pytest.mark.parametrize("bad", [
    {"conf": 0}, {"conf": 1}, {"conf": True}, {"conf": "0.1"}, {"conf": float("nan")},
    {"config": "/etc/x.yaml"}, {"config": "C:/x.yaml"}, {"config": "..\\x.yaml"}, {"config": "a/../../x.yaml"}, {"config": "tracker.txt"},
    {"config": 5}, {"other": 1}, "bytetrack.yaml", [],
])
def test_bad_tracker_blocks_are_config_errors(bad):
    with pytest.raises(ConfigError):
        parse_camera_config({"tracker": bad})


def test_a_repo_tracker_yaml_resolves_inside_the_repository_and_other_paths_fail_closed():
    assert Path(_repo_file("demo/tools/trackers/anvesh_bytetrack_m09.yaml")).is_file()
    with pytest.raises(FileNotFoundError):
        _repo_file("demo/trackers/does_not_exist.yaml")


def test_the_factory_passes_conf_and_config_to_the_tracker(monkeypatch, tmp_path):
    seen = {}

    class Fake:
        def __init__(self, model_path, **kw):
            seen["model"], seen["kw"] = model_path, kw

    monkeypatch.setattr("anvesh.perception.tracking.ByteTrackYoloTracker", Fake)
    monkeypatch.setattr("anvesh.perception.weights.require_weights_file", lambda p: None)
    factory = default_tracker_factory("models/yolov8n.pt")
    cam = {"source_type": "file"}
    factory(cam, parse_camera_config({}))
    assert seen["kw"] == {}  # no block: the constructor defaults (conf 0.25, bytetrack.yaml) are untouched
    factory(cam, parse_camera_config({"tracker": {"conf": 0.1, "config": "demo/tools/trackers/anvesh_bytetrack_m09.yaml"}}))
    assert seen["kw"]["confidence_threshold"] == 0.1 and seen["kw"]["tracker_config"].endswith("anvesh_bytetrack_m09.yaml")
