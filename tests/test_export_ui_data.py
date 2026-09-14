"""Focused tests for scripts/export_ui_data.py's presentation-video
integration (demo.mp4). Do NOT run the real M1/YOLO pipeline here (that
would make the suite slow and GPU-dependent) -- these test only the pure,
GPU-free serialization/config logic: `_box_entry`, `_detect_gpu_info`,
and that the hardcoded demo.mp4 constants match the values already used
everywhere else in the project (never a new/tuned threshold).

Commit 9A adds coverage for the generic (not c006-specific)
lane/speed/candidate-vehicle integration: `_box_entry`'s optional
lane/speed keys, `_window_result`'s `candidate_vehicle` key, and
`real_video_limited_chain`'s new `calibration_profile`/`lane_regions`
parameters -- exercised with a monkeypatched `run_m1_pipeline` (a tiny,
scripted, non-GPU fake), never the real YOLO/ByteTrack pipeline.

`scripts/` is not a package (see tests/test_real_world_validation_runner.py
for the same reasoning), so this module is loaded via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_ui_data", REPO_ROOT / "scripts" / "export_ui_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_ui_data = _load_export_module()


@dataclass
class _FakeDetection:
    bbox_xyxy: tuple
    vehicle_class: object
    confidence: float


@dataclass
class _FakeTrackedDetection:
    track_id: str
    detection: _FakeDetection


def test_box_entry_serializes_real_fields_only():
    from anvesh.storage.schemas import VehicleClass

    td = _FakeTrackedDetection(
        track_id="42",
        detection=_FakeDetection(bbox_xyxy=(1.0, 2.0, 3.0, 4.0), vehicle_class=VehicleClass.CAR, confidence=0.87),
    )
    entry = export_ui_data._box_entry(td)
    assert entry == {"track_id": "42", "bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "vehicle_class": "car", "confidence": 0.87}


def test_detect_gpu_info_reflects_actual_torch_state():
    """Never fabricated -- must match whatever torch.cuda actually reports
    right now, whichever way that goes on the machine running the test."""
    import torch

    info = export_ui_data._detect_gpu_info()
    assert info["cuda_available"] == torch.cuda.is_available()
    if info["cuda_available"]:
        assert info["device_name"] == torch.cuda.get_device_name(0)
    else:
        assert info["device_name"] is None


def test_demo_video_constants_are_not_tuned_for_this_video():
    """The presentation demo.mp4 integration must reuse the exact same
    perception/window/capacity defaults used everywhere else in the
    project (M6 validation, synthetic scenarios) -- never a value chosen
    to make this specific video's results look better."""
    assert export_ui_data.DEMO_MODEL_NAME == "models/yolov8n.pt"
    assert export_ui_data.DEMO_CONFIDENCE_THRESHOLD == 0.25
    assert export_ui_data.DEMO_TRACKER_CONFIG == "bytetrack.yaml"
    assert export_ui_data.DEMO_MAX_MISSED_FRAMES == 5
    assert export_ui_data.DEMO_PARTIAL_AFTER_MISSED_FRAMES == 1
    assert export_ui_data.DEMO_MAX_CAPACITY_VEHICLES == 20
    assert export_ui_data.WINDOW_SECONDS == 10.0


def test_demo_sample_seconds_matches_evaluated_window():
    """Must match the presentation-video comparison report's evaluated
    240-second sample exactly, so the UI never shows data claims broader
    than what was actually observed/reported."""
    assert export_ui_data.DEMO_SAMPLE_SECONDS == 240.0


def test_demo_video_path_points_at_demo_mp4_not_traffic_or_car_detection():
    assert export_ui_data.DEMO_VIDEO_PATH.name == "demo.mp4"


# ---------------------------------------------------------------------------
# Commit 9A -- single-camera lane/speed/candidate-vehicle integration.
# ---------------------------------------------------------------------------


def test_box_entry_includes_lane_fields_when_lane_result_supplied():
    from anvesh.storage.schemas import VehicleClass
    from anvesh.world.lane_geometry import LaneAssignmentResult, LaneAssignmentStatus

    td = _FakeTrackedDetection(
        track_id="42",
        detection=_FakeDetection(bbox_xyxy=(1.0, 2.0, 3.0, 4.0), vehicle_class=VehicleClass.CAR, confidence=0.87),
    )
    lane_result = LaneAssignmentResult(status=LaneAssignmentStatus.ASSIGNED, lane_id="L1", candidate_lane_ids=("L1",))
    entry = export_ui_data._box_entry(td, lane_result=lane_result)
    assert entry["lane"] == "L1"
    assert entry["lane_status"] == "assigned"
    assert "speed" not in entry  # speed_estimate still omitted -> still absent


def test_box_entry_includes_speed_fields_when_speed_estimate_supplied():
    from anvesh.storage.schemas import Measurement, VehicleClass
    from anvesh.perception.motion import RobustSpeedStatus
    from anvesh.perception.speed_estimation import SpeedConfidence, VehicleSpeedEstimate

    td = _FakeTrackedDetection(
        track_id="42",
        detection=_FakeDetection(bbox_xyxy=(1.0, 2.0, 3.0, 4.0), vehicle_class=VehicleClass.CAR, confidence=0.87),
    )
    speed_estimate = VehicleSpeedEstimate(
        track_id="cam:42", camera_id="cam", status=RobustSpeedStatus.OK,
        confidence=SpeedConfidence.APPROXIMATE, measurement=Measurement(value=34.0, error=1.0),
    )
    entry = export_ui_data._box_entry(td, speed_estimate=speed_estimate)
    assert entry["speed"] == "~34 km/h (approx.)"
    assert entry["speed_status"] == "ok"
    assert entry["speed_confidence"] == "approximate"
    assert "lane" not in entry  # lane_result still omitted -> still absent


def test_window_result_includes_candidate_vehicle_and_round_trips_json():
    import json

    from anvesh.hypotheses.candidate_vehicle import CandidateVehicleResolution, CandidateVehicleStatus

    resolution = CandidateVehicleResolution(
        status=CandidateVehicleStatus.RESOLVED, hypothesis_id="H2", belief=0.8, plausibility=1.0,
        supporting_evidence_refs=("cam:H2:0.0-10.0",), candidate_track_ids=("cam:1",), track_id="cam:1",
    )

    class _FakeResult:
        window = (0.0, 10.0)
        camera_ids = ("cam",)
        evidence_by_camera = {}
        reliability_by_camera = {}
        candidate_ranking = None
        candidate_vehicle = resolution
        corridor_state = None
        corridor_state_reason = "single-camera input -- no corridor to assemble"
        propagation_prediction = None
        propagation_reason = "requires a calibrated two-camera corridor topology"
        feedback_result = None
        feedback_reason = "no propagation prediction available"
        safety_events = ()
        limitations = ()

    # _ranking(None) would crash, so patch it out -- this test targets
    # candidate_vehicle serialization only, not the whole result shape.
    orig_ranking = export_ui_data._ranking
    export_ui_data._ranking = lambda r: None
    try:
        serialized = export_ui_data._window_result(_FakeResult(), {"cam": []})
    finally:
        export_ui_data._ranking = orig_ranking

    assert serialized["candidate_vehicle"] == {
        "status": "resolved", "hypothesis_id": "H2", "belief": 0.8, "plausibility": 1.0,
        "candidate_track_ids": ["cam:1"], "track_id": "cam:1",
    }
    json.dumps(serialized)  # must be JSON-serializable


def _write_tiny_video(path, num_frames=3, fps=10.0, size=(16, 12)):
    import cv2
    import numpy as np

    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "test setup: failed to open synthetic video writer"
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=i * 10 % 255, dtype="uint8")
            writer.write(frame)
    finally:
        writer.release()


def _fake_run_m1_pipeline_factory(ground_point_trajectory):
    """Builds a scripted, non-GPU stand-in for `run_m1_pipeline` that
    emits exactly one track with a caller-supplied ground-point
    trajectory (needed to exercise `estimate_vehicle_speed`), and calls
    `on_frame_tracked` once (matching the real function's own per-frame
    contract) so `real_video_limited_chain`'s capture/attach logic runs
    unmodified."""
    from anvesh.perception.pipeline import M1PipelineResult
    from anvesh.perception.track_manager import TrackState
    from anvesh.perception.types import RawDetection, TrackedDetection
    from anvesh.storage.schemas import Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack

    def fake_run_m1_pipeline(
        video_path, camera_id, tracker, model_id, model_version,
        max_missed_frames=5, partial_after_missed_frames=1,
        visualization_output_path=None, calibration_profile=None, on_frame_tracked=None,
    ):
        detection = RawDetection(
            frame_index=0, timestamp=0.0, class_name="car", vehicle_class=VehicleClass.CAR,
            confidence=0.9, bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        )
        td = TrackedDetection(detection=detection, track_id="1")
        if on_frame_tracked is not None:
            on_frame_tracked(0, 0.0, [td])

        result = M1PipelineResult()
        result.frame_count = 1
        result.vehicle_tracks = [
            VehicleTrack(
                track_id=f"{camera_id}:1", camera_id=camera_id, first_seen=0.0, last_seen=0.4,
                vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE, position_history=[],
                speed_estimate=Measurement(value=0.0, error=0.0), motion_space=MotionSpace.IMAGE,
            )
        ]
        result.track_states = [
            TrackState(
                track_id="1", vehicle_class=VehicleClass.CAR, first_seen=0.0, last_seen=0.4,
                occlusion_state=OcclusionState.VISIBLE, ground_point_trajectory=ground_point_trajectory,
            )
        ]
        return result

    return fake_run_m1_pipeline


def test_real_video_limited_chain_omits_lane_speed_keys_by_default(monkeypatch, tmp_path):
    """Backward compatibility: the two new parameters both default to
    'no lane/no calibration', so every existing caller (demo.mp4,
    analyze_video.py) keeps exactly today's 4-key box shape."""
    tiny_video = tmp_path / "tiny.mp4"
    _write_tiny_video(tiny_video)

    monkeypatch.setattr(export_ui_data, "run_m1_pipeline", _fake_run_m1_pipeline_factory([(5.0, 10.0, 0.0)]))
    monkeypatch.setattr(export_ui_data, "ByteTrackYoloTracker", lambda **kwargs: object())

    result = export_ui_data.real_video_limited_chain(video_path=tiny_video, camera_id="cam-test", sample_seconds=1.0)

    assert len(result["frames"]) == 1
    box = result["frames"][0]["boxes"][0]
    assert set(box.keys()) == {"track_id", "bbox_xyxy", "vehicle_class", "confidence"}


def test_real_video_limited_chain_attaches_lane_and_speed_when_supplied(monkeypatch, tmp_path):
    """c006-style injection (generic: any calibration_profile/lane_regions
    pair), exercised without the real dataset or YOLO."""
    from anvesh.world.calibration import fit_planar_homography
    from anvesh.world.lane_geometry import LaneRegion
    from tests.world.support import make_reference_points

    tiny_video = tmp_path / "tiny.mp4"
    _write_tiny_video(tiny_video)

    trajectory = [(5.0 + i, 10.0, i * 0.1) for i in range(5)]  # moving, calibratable ground points
    monkeypatch.setattr(export_ui_data, "run_m1_pipeline", _fake_run_m1_pipeline_factory(trajectory))
    monkeypatch.setattr(export_ui_data, "ByteTrackYoloTracker", lambda **kwargs: object())

    calibration = fit_planar_homography("calib-test", "cam-test", make_reference_points())
    lane_regions = (LaneRegion(lane_id="L1", vertices=((0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0))),)

    result = export_ui_data.real_video_limited_chain(
        video_path=tiny_video, camera_id="cam-test", sample_seconds=1.0,
        calibration_profile=calibration, lane_regions=lane_regions,
    )

    box = result["frames"][0]["boxes"][0]
    assert box["lane"] == "L1"
    assert box["lane_status"] == "assigned"
    assert box["speed_confidence"] in ("approximate", "validated")  # calibrated -> never "unavailable" here
    assert box["speed"] is not None

    # track-level data (per-track speed) actually reaches the per-box export --
    # not silently dropped by a track_id-keying mismatch (raw vs. camera-qualified).
    assert box["track_id"] == "1"


def test_real_video_limited_chain_uses_world_motion_space_when_calibrated(monkeypatch, tmp_path):
    """The window-aggregation MotionSpace branch (§5.3 of the Commit 9
    integration audit) is WORLD only when a calibrated profile is
    supplied -- verified directly against the call `aggregate_traffic_state`
    actually receives, not inferred from a fake track's own hardcoded
    motion_space."""
    from anvesh.storage.schemas import MotionSpace
    from anvesh.world.calibration import fit_planar_homography
    from tests.world.support import make_reference_points

    tiny_video = tmp_path / "tiny.mp4"
    _write_tiny_video(tiny_video)

    monkeypatch.setattr(export_ui_data, "run_m1_pipeline", _fake_run_m1_pipeline_factory([(5.0, 10.0, 0.0)]))
    monkeypatch.setattr(export_ui_data, "ByteTrackYoloTracker", lambda **kwargs: object())

    captured_motion_spaces = []
    orig_aggregate = export_ui_data.aggregate_traffic_state

    def _spy_aggregate(camera_id, start, end, tracks, motion_space, tracker, **kwargs):
        captured_motion_spaces.append(motion_space)
        return orig_aggregate(camera_id, start, end, tracks, motion_space, tracker, **kwargs)

    monkeypatch.setattr(export_ui_data, "aggregate_traffic_state", _spy_aggregate)

    calibration = fit_planar_homography("calib-test", "cam-test", make_reference_points())
    export_ui_data.real_video_limited_chain(
        video_path=tiny_video, camera_id="cam-test", sample_seconds=1.0, calibration_profile=calibration,
    )
    assert captured_motion_spaces and all(ms == MotionSpace.WORLD for ms in captured_motion_spaces)


def test_real_video_limited_chain_uses_image_motion_space_by_default(monkeypatch, tmp_path):
    from anvesh.storage.schemas import MotionSpace

    tiny_video = tmp_path / "tiny.mp4"
    _write_tiny_video(tiny_video)

    monkeypatch.setattr(export_ui_data, "run_m1_pipeline", _fake_run_m1_pipeline_factory([(5.0, 10.0, 0.0)]))
    monkeypatch.setattr(export_ui_data, "ByteTrackYoloTracker", lambda **kwargs: object())

    captured_motion_spaces = []
    orig_aggregate = export_ui_data.aggregate_traffic_state

    def _spy_aggregate(camera_id, start, end, tracks, motion_space, tracker, **kwargs):
        captured_motion_spaces.append(motion_space)
        return orig_aggregate(camera_id, start, end, tracks, motion_space, tracker, **kwargs)

    monkeypatch.setattr(export_ui_data, "aggregate_traffic_state", _spy_aggregate)

    export_ui_data.real_video_limited_chain(video_path=tiny_video, camera_id="cam-test", sample_seconds=1.0)
    assert captured_motion_spaces and all(ms == MotionSpace.IMAGE for ms in captured_motion_spaces)


def test_generic_modules_have_no_hardcoded_c006_paths():
    """The Commit 9 integration audit's own requirement I: camera-specific
    paths/ids must exist only at the showcase invocation boundary
    (scripts/run_s02_c006_showcase.py), never as an actual default/
    hardcoded value inside the generic modules Commit 9A touches.

    A docstring naming `scripts/run_s02_c006_showcase.py` by filename
    (to say "that's where camera-specific code lives") is fine and
    expected; a real violation would be the dataset path itself, or
    "s02_c006"/"c006" used as an assigned/default value.
    """
    touched_modules = [
        REPO_ROOT / "scripts" / "export_ui_data.py",
        REPO_ROOT / "anvesh" / "pipeline.py",
        REPO_ROOT / "anvesh" / "perception" / "pipeline.py",
    ]
    dataset_markers = ["traffic dataset", "vdo.avi", "calibration.txt"]
    value_assignment_markers = [
        '"s02_c006"', "'s02_c006'", '"c006"', "'c006'",
    ]
    for module_path in touched_modules:
        text = module_path.read_text().lower()
        for needle in dataset_markers:
            assert needle not in text, f"{module_path} unexpectedly mentions dataset path marker {needle!r}"
        for needle in value_assignment_markers:
            assert needle not in text, f"{module_path} unexpectedly hardcodes camera id {needle!r}"
