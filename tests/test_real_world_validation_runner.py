"""M6.5 Phase 2: tests for the optional real-camera threshold plumbing in
scripts/run_real_world_validation.py.

`scripts/` is not a package (see tests/test_evaluation_runner.py's use of
`runpy` for the same reason), so this module is loaded directly via
`importlib` to reach its private helper functions.

These tests deliberately fake `run_m1_pipeline` and `VideoFileFrameSource`
rather than running real YOLO/ByteTrack inference: M6.5 Phase 1 already
established that tracking/detection are OUT OF SCOPE for this phase, so
these tests isolate exactly what Phase 2 changed -- threshold plumbing --
from the (unmodified, already-covered-elsewhere) perception pipeline.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.evidence.evidence_builder import EvidenceThresholds
from anvesh.perception.pipeline import M1PipelineResult
from anvesh.perception.traffic_state import CongestionThresholds
from anvesh.storage.schemas import Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack


def _load_rwv_module():
    spec = importlib.util.spec_from_file_location(
        "run_real_world_validation", REPO_ROOT / "scripts" / "run_real_world_validation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rwv = _load_rwv_module()


def _fake_track(track_id, first_seen, last_seen, speed_value=5.0):
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-fake",
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=0.0),
        motion_space=MotionSpace.IMAGE,
    )


# 12 tracks spread over a 20s video -> nonzero occupancy in both 10s windows,
# large enough to be sensitive to a small max_capacity_vehicles override.
_FAKE_TRACKS = [_fake_track(str(i), 0.0, 20.0) for i in range(12)]
_FAKE_M1_RESULT = M1PipelineResult(camera_observations=[], vehicle_tracks=_FAKE_TRACKS, frame_count=240)

_VALIDATION_CFG = {
    "window_seconds": 10.0,
    "model_name": "models/yolov8n.pt",
    "confidence_threshold": 0.25,
    "tracker_config": "bytetrack.yaml",
    "max_missed_frames": 5,
    "partial_after_missed_frames": 1,
    "max_capacity_vehicles": 20,
}
_SEQ_CFG = {
    "sequence_id": "fake_seq",
    "video_path": "datasets/recorded/dev/car-detection.mp4",  # path only used for reporting/fps-mock, never opened
    "camera_id": "cam-fake",
    "camera_role": "upstream",
}


class _FakeTracker:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(rwv, "run_m1_pipeline", lambda **kwargs: _FAKE_M1_RESULT)
    monkeypatch.setattr(rwv, "ByteTrackYoloTracker", _FakeTracker)

    class _FakeFrameSource:
        def __init__(self, *args, **kwargs):
            pass

        def fps(self):
            return 12.0

    monkeypatch.setattr(rwv, "VideoFileFrameSource", _FakeFrameSource)


def _windows_summary(result):
    return [(agg.traffic_state.vehicle_count, agg.traffic_state.congestion_level.value) for agg in result.timeline]


def _outcomes_summary(result):
    return [(w.index, rk.outcome.value) for w, rk in result.per_window_rankings]


# ---------------------------------------------------------------------------
# 1. No override reproduces original M6 default exactly.
# ---------------------------------------------------------------------------


def test_no_override_call_matches_bare_default_call():
    """Calling with the new optional kwargs all left at their defaults must
    produce output identical to the original (pre-Phase-2) call signature."""
    original_style = rwv._run_sequence(_SEQ_CFG, _VALIDATION_CFG)
    via_new_plumbing = rwv._run_sequence(
        _SEQ_CFG, _VALIDATION_CFG, max_capacity_vehicles=None, evidence_thresholds=None, congestion_thresholds=None,
    )
    assert _windows_summary(original_style) == _windows_summary(via_new_plumbing)
    assert _outcomes_summary(original_style) == _outcomes_summary(via_new_plumbing)
    assert original_style.max_capacity_vehicles_used == via_new_plumbing.max_capacity_vehicles_used == 20
    assert original_style.whole_video_evidence == via_new_plumbing.whole_video_evidence


def test_run_all_sequences_with_none_thresholds_matches_default_run_sequence():
    config = {"sequence": [_SEQ_CFG]}
    via_helper = rwv._run_all_sequences(config, _VALIDATION_CFG, real_camera_thresholds=None, config_label="default")[0]
    direct = rwv._run_sequence(_SEQ_CFG, _VALIDATION_CFG)
    assert _windows_summary(via_helper) == _windows_summary(direct)
    assert via_helper.config_label == "default"


# ---------------------------------------------------------------------------
# 2. Real-camera config changes ONLY the explicitly configured values.
# ---------------------------------------------------------------------------


def test_max_capacity_override_changes_occupancy_but_not_vehicle_counts():
    default_result = rwv._run_sequence(_SEQ_CFG, _VALIDATION_CFG)
    overridden_result = rwv._run_sequence(_SEQ_CFG, _VALIDATION_CFG, max_capacity_vehicles=2)

    default_counts = [agg.traffic_state.vehicle_count for agg in default_result.timeline]
    overridden_counts = [agg.traffic_state.vehicle_count for agg in overridden_result.timeline]
    assert default_counts == overridden_counts  # raw vehicle counts are observation, never affected by capacity

    default_occupancy = [agg.traffic_state.occupancy for agg in default_result.timeline]
    overridden_occupancy = [agg.traffic_state.occupancy for agg in overridden_result.timeline]
    assert default_occupancy != overridden_occupancy  # the one thing the override is supposed to change
    assert all(o == 1.0 for o in overridden_occupancy)  # 12 vehicles / capacity 2, clamped to 1.0

    assert overridden_result.max_capacity_vehicles_used == 2
    assert default_result.max_capacity_vehicles_used == 20


def test_build_evidence_thresholds_changes_only_the_named_field():
    built = rwv._build_evidence_thresholds({"elevated_vehicle_count": 3})
    default = EvidenceThresholds()
    assert built.elevated_vehicle_count == 3
    assert built.stopped_speed_m_per_s == default.stopped_speed_m_per_s
    assert built.min_dwell_seconds == default.min_dwell_seconds


def test_build_congestion_thresholds_changes_only_the_named_field():
    built = rwv._build_congestion_thresholds({"high_occupancy": 0.9})
    default = CongestionThresholds()
    assert built.high_occupancy == 0.9
    assert built.free_speed_m_per_s == default.free_speed_m_per_s
    assert built.congested_speed_m_per_s == default.congested_speed_m_per_s
    assert built.low_occupancy == default.low_occupancy


def test_real_camera_thresholds_toml_loads_only_max_capacity_vehicles():
    path = REPO_ROOT / "configs" / "experiment" / "m6_real_camera_thresholds.toml"
    thresholds = rwv._load_real_camera_thresholds(path)
    assert thresholds["seq1_car_detection"]["max_capacity_vehicles"] == 18
    assert thresholds["seq2_person_bicycle_car"]["max_capacity_vehicles"] == 15
    assert thresholds["seq1_car_detection"]["evidence_thresholds"] == {}
    assert thresholds["seq1_car_detection"]["congestion_thresholds"] == {}


# ---------------------------------------------------------------------------
# 3. scripts/run_evaluation.py (synthetic evaluation) is completely unaffected.
# ---------------------------------------------------------------------------


def test_run_evaluation_script_still_executes_and_prints_disclaimer(capsys):
    """Re-run the M6-era evaluation-runner smoke assertion here: M6.5 Phase 2
    touches only scripts/run_real_world_validation.py and its own optional
    threshold builders, never anything run_evaluation.py depends on."""
    script_path = REPO_ROOT / "scripts" / "run_evaluation.py"
    old_argv = sys.argv
    sys.argv = ["run_evaluation.py"]
    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")
        assert exc_info.value.code == 0
    finally:
        sys.argv = old_argv
    output = capsys.readouterr().out
    assert "CONTROLLED SYNTHETIC MECHANISM EVALUATION" in output


# ---------------------------------------------------------------------------
# 4. Default M1-M5 behavior unchanged.
# ---------------------------------------------------------------------------


def test_congestion_state_tracker_none_thresholds_matches_bare_constructor():
    from anvesh.perception.traffic_state import CongestionStateTracker

    a = CongestionStateTracker()
    b = CongestionStateTracker(thresholds=None)
    for occupancy in (0.1, 0.7, 0.7, 0.1, 0.1):
        assert a.update(5.0, occupancy, MotionSpace.IMAGE) == b.update(5.0, occupancy, MotionSpace.IMAGE)


def test_empty_override_dicts_yield_none_thresholds():
    """Every M1-M5 call site constructs EvidenceThresholds()/CongestionThresholds()
    with bare defaults; the builder functions must return None (not an
    empty-but-real instance) for an empty override so callers keep using
    those exact bare defaults."""
    assert rwv._build_evidence_thresholds({}) is None
    assert rwv._build_congestion_thresholds({}) is None


# ---------------------------------------------------------------------------
# 5. Invalid threshold config values are rejected cleanly.
# ---------------------------------------------------------------------------


def test_unknown_evidence_threshold_field_raises_value_error():
    with pytest.raises(ValueError, match="Unknown EvidenceThresholds field"):
        rwv._build_evidence_thresholds({"not_a_real_field": 1.0})


def test_unknown_congestion_threshold_field_raises_value_error():
    with pytest.raises(ValueError, match="Unknown CongestionThresholds field"):
        rwv._build_congestion_thresholds({"not_a_real_field": 1.0})
