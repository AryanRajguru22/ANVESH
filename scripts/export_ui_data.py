"""Export real ANVESH engine output as JSON for the UI (productization phase).

This script does NOT reimplement or reinterpret any ANVESH algorithm. It
calls `anvesh.pipeline.run_corridor_window` (the real M8 orchestrator) on:

  - three small, deterministic, hand-authored synthetic scenarios (same
    style as every M2-M8 demo script -- explicit WORLD-space tracks,
    explicit CorridorTopology, explicit segment length), and
  - the presentation demo video's (demo.mp4) honest single-camera limited
    chain -- selected over traffic.mp4 via a separate, dedicated
    evaluation pass (see that report for the full comparison); the first
    240 seconds are processed and reported, matching that evaluation
    exactly (see DEMO_SAMPLE_SECONDS below for why the full ~34-minute
    video isn't all exported),

then serializes the REAL `AnveshWindowResult` objects those calls
actually produced into one JSON file. Every number the UI displays comes
from this file, and every number in this file comes from calling the
real engine -- nothing here is authored by hand for display purposes.

`real_video_limited_chain` additionally accepts an optional
`calibration_profile`/`lane_regions` pair (Commit 9A) so a caller can
thread in one camera's real CityFlow calibration and lane geometry --
both default to "none supplied," which reproduces this file's original
(pre-Commit-9A) behavior exactly. This module itself never mentions any
specific camera_id or dataset path; a camera-specific showcase script
(e.g. `scripts/run_s02_c006_showcase.py`) is the only place that does.

Usage (from the repository root, with the venv active):

    python scripts/export_ui_data.py

Writes: ui/data/anvesh_demo_data.json
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.evidence.evidence_builder import EvidenceThresholds
from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.perception.lane_assignment import assign_detection_lane
from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.speed_estimation import estimate_vehicle_speed
from anvesh.perception.tracking import ByteTrackYoloTracker
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.perception.visualize import VideoWriter
from anvesh.perception.windowing import generate_windows, tracks_in_window
from anvesh.rendering import format_lane_text, format_speed_text
from anvesh.storage.schemas import (
    CameraRole,
    DataCompleteness,
    Measurement,
    MotionSpace,
    OcclusionState,
    PropagationObservation,
    VehicleClass,
    VehicleTrack,
    WorldPosition,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology

OUTPUT_PATH = REPO_ROOT / "ui" / "data" / "anvesh_demo_data.json"

WINDOW_SECONDS = 10.0
SEGMENT_LENGTH_M = 100.0  # the scenario corridor's own inter-camera distance -- not a second, arbitrary number

# These SYNTHETIC corridors have a stipulated capacity, the way a synthetic
# fixture stipulates its own geometry. 1.6 veh/s is exactly the ramp
# denominator these hand-authored scenarios were originally written against
# (16 vehicles per 10s window), restated in the rate units H5 now requires --
# so every scenario keeps the numbers it was verified with. It is NOT a
# default and is NOT applied to real video: an arbitrary real camera has no
# known capacity, so H5 there is honestly reported unavailable (see
# evidence_builder's module docstring).
SYNTHETIC_EVIDENCE_THRESHOLDS = EvidenceThresholds(capacity_flow_veh_per_s=1.6)


# ---------------------------------------------------------------------------
# Serialization -- explicit to-dict functions, same style as storage/db.py.
# Read-only: never mutates, never recomputes, only reads existing fields.
# ---------------------------------------------------------------------------


def _measurement(m) -> dict:
    return {"value": m.value, "error": m.error}


def _track_summary(track: VehicleTrack) -> dict:
    return {
        "track_id": track.track_id,
        "vehicle_class": track.vehicle_class.value,
        "occlusion_state": track.occlusion_state.value,
        "first_seen": track.first_seen,
        "last_seen": track.last_seen,
        "motion_space": track.motion_space.value,
        "speed_estimate": _measurement(track.speed_estimate),
        "observation_count": len(track.position_history),
    }


def _traffic_state(ts) -> dict:
    return {
        "camera_id": ts.camera_id,
        "window": [ts.window_start, ts.window_end],
        "occupancy": ts.occupancy,
        "mean_speed": _measurement(ts.mean_speed),
        "vehicle_count": ts.vehicle_count,
        "flow_rate": ts.flow_rate,
        "density": ts.density,
        "congestion_level": ts.congestion_level.value,
        "motion_space": ts.motion_space.value,
    }


def _reliability(r) -> dict:
    return {
        "camera_id": r.camera_id,
        "score": r.score,
        "factors": dict(r.factors),
        "method": r.method,
    }


def _evidence_item(item) -> dict:
    return {
        "hypothesis_id": item.evidence.hypothesis_id,
        "signature_match_score": item.evidence.signature_match_score,
        "evidence_completeness": item.evidence.evidence_completeness.value,
        "contradicting": item.evidence.contradicting,
        "direction": item.direction.value,
        "reliability": item.reliability,
        "evidence_type": item.evidence_type,
        "supporting_track_refs": list(item.evidence.supporting_track_refs),
    }


def _ranking(ranking) -> dict:
    return {
        "ranking_id": ranking.ranking_id,
        "outcome": ranking.outcome.value,
        "ranked_list": [
            {
                "hypothesis_id": e.hypothesis_id,
                "belief": e.belief,
                "plausibility": e.plausibility,
                "confidence_tier": e.confidence_tier.value,
                "supporting_evidence_refs": list(e.supporting_evidence_refs),
                "contradicting_evidence_refs": list(e.contradicting_evidence_refs),
            }
            for e in ranking.ranked_list
        ],
    }


def _corridor_state(cs) -> dict:
    if cs is None:
        return None
    return {
        "corridor_state_id": cs.corridor_state_id,
        "window": list(cs.window),
        "fused_congestion_level": cs.fused_congestion_level.value,
        "active_ranking_id": cs.active_ranking_id,
        "segment_states": [_traffic_state(s) for s in cs.segment_states],
    }


def _propagation(p) -> dict:
    if p is None:
        return None
    return {
        "prediction_id": p.prediction_id,
        "predicted_shockwave_speed": _measurement(p.predicted_shockwave_speed),
        "predicted_arrival_camera": p.predicted_arrival_camera,
        "predicted_arrival_time": p.predicted_arrival_time,
        "predicted_queue_growth_rate": p.predicted_queue_growth_rate,
        "method": p.method,
    }


def _feedback(f) -> dict:
    if f is None:
        return None
    discrepancy = f.update.discrepancy
    return {
        "outcome": f.update.outcome.value,
        "reason": f.reason,
        "discrepancy": (
            {
                "location_error": discrepancy.location_error,
                "time_error": discrepancy.time_error,
                "rate_error": discrepancy.rate_error,
            }
            if discrepancy is not None
            else None
        ),
        "prior_top": _top_entry(f.prior_ranking),
        "revised_top": _top_entry(f.revised_ranking),
    }


def _top_entry(ranking) -> dict:
    if not ranking.ranked_list:
        return None
    top = ranking.ranked_list[0]
    return {"hypothesis_id": top.hypothesis_id, "belief": top.belief, "confidence_tier": top.confidence_tier.value}


def _safety_event(e) -> dict:
    return {
        "event_id": e.event_id,
        "camera_id": e.camera_id,
        "track_id": e.track_id,
        "window": list(e.window),
        "confidence_tier": e.confidence_tier.value,
    }


def _candidate_vehicle(resolution) -> dict:
    """Serializes `hypotheses.candidate_vehicle.CandidateVehicleResolution`
    honestly -- `status` is always one of RESOLVED/AMBIGUOUS/
    NO_VEHICLE_FOR_HYPOTHESIS/NOT_RANKED (never collapsed), and `track_id`
    is non-null if and only if `status == "resolved"`."""
    return {
        "status": resolution.status.value,
        "hypothesis_id": resolution.hypothesis_id,
        "belief": resolution.belief,
        "plausibility": resolution.plausibility,
        "candidate_track_ids": list(resolution.candidate_track_ids),
        "track_id": resolution.track_id,
    }


def _window_result(result, tracks_by_camera: dict) -> dict:
    return {
        "window": list(result.window),
        "camera_ids": list(result.camera_ids),
        "tracks_by_camera": {cid: [_track_summary(t) for t in tracks] for cid, tracks in tracks_by_camera.items()},
        "evidence_by_camera": {
            cid: [_evidence_item(item) for item in items] for cid, items in result.evidence_by_camera.items()
        },
        "reliability_by_camera": {cid: _reliability(r) for cid, r in result.reliability_by_camera.items()},
        "candidate_ranking": _ranking(result.candidate_ranking),
        "candidate_vehicle": _candidate_vehicle(result.candidate_vehicle),
        "corridor_state": _corridor_state(result.corridor_state),
        "corridor_state_reason": result.corridor_state_reason,
        "propagation_prediction": _propagation(result.propagation_prediction),
        "propagation_reason": result.propagation_reason,
        "feedback_result": _feedback(result.feedback_result),
        "feedback_reason": result.feedback_reason,
        "safety_events": [_safety_event(e) for e in result.safety_events],
        "limitations": list(result.limitations),
    }


# ---------------------------------------------------------------------------
# Synthetic fixtures -- small, hand-authored, explicitly labeled synthetic.
# ---------------------------------------------------------------------------


def _tracks(camera_id, count, speed, first_seen=0.0, last_seen=10.0, offset=0):
    return [
        VehicleTrack(
            track_id=f"{camera_id}:{offset + i}", camera_id=camera_id, first_seen=first_seen, last_seen=last_seen,
            vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE, position_history=[],
            speed_estimate=Measurement(value=speed, error=0.0), motion_space=MotionSpace.WORLD,
        )
        for i in range(count)
    ]


def _partial_tracks(camera_id, count, speed, first_seen=0.0, last_seen=10.0):
    return [
        VehicleTrack(
            track_id=f"{camera_id}:{i}", camera_id=camera_id, first_seen=first_seen, last_seen=last_seen,
            vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.PARTIAL, position_history=[],
            speed_estimate=Measurement(value=speed, error=0.0), motion_space=MotionSpace.WORLD,
        )
        for i in range(count)
    ]


def _stopped_track(camera_id, suffix="stopped"):
    """~0.1 m/s over 4.0s -- qualifies as STOPPED for both M7's own
    per-step recomputation AND evidence_builder's H2/H3 (which, unlike
    M7, reads `speed_estimate` directly) -- so `speed_estimate` here must
    genuinely match the trajectory, not a decoy value."""
    positions = [(10.0 + 0.05 * i, 0.0, i * 0.5) for i in range(9)]
    return VehicleTrack(
        track_id=f"{camera_id}:{suffix}", camera_id=camera_id, first_seen=positions[0][2], last_seen=positions[-1][2],
        vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE,
        position_history=[WorldPosition(world_x=x, world_y=y, timestamp=t) for (x, y, t) in positions],
        speed_estimate=Measurement(value=0.1, error=0.02), motion_space=MotionSpace.WORLD,
    )


def _camera_input(camera_id, role, tracks, window, tracker, max_capacity_vehicles=20):
    aggregation = aggregate_traffic_state(
        camera_id, window[0], window[1], tracks, MotionSpace.WORLD, tracker, max_capacity_vehicles=max_capacity_vehicles
    )
    return CameraWindowInput(camera_id=camera_id, role=role, aggregation=aggregation, tracks=tracks)


def _topology(corridor_id, a_id, b_id):
    return CorridorTopology(
        corridor_id=corridor_id,
        placements=(
            CameraPlacement(camera_id=a_id, role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id=b_id, role=CameraRole.DOWNSTREAM, distance_from_upstream_m=SEGMENT_LENGTH_M),
        ),
    )


def scenario_cross_camera_agreement() -> dict:
    """Three sequential windows: upstream traffic escalates (free -> building
    -> congested-ish) while downstream stays light -- exercises the
    congestion-state timeline, real two-camera fusion, propagation, an
    explicit SYNTHETIC observation confirming the prediction, and a
    stopped-vehicle safety event, all through the real engine."""
    a_id, b_id = "cam-A-upstream", "cam-B-downstream"
    topology = _topology("agreement-corridor", a_id, b_id)
    tracker_a, tracker_b = CongestionStateTracker(), CongestionStateTracker()
    counts_a = [8, 15, 17]  # escalating upstream demand across 3 windows
    windows_out = []
    last_result = None
    for i, count_a in enumerate(counts_a):
        window = (i * WINDOW_SECONDS, (i + 1) * WINDOW_SECONDS)
        a_tracks = _tracks(a_id, count_a, 2.0, window[0], window[1])
        if i == 2:
            a_tracks = a_tracks + [_stopped_track(a_id)]
        b_tracks = _tracks(b_id, 3, 8.0, window[0], window[1])
        a_input = _camera_input(a_id, CameraRole.UPSTREAM, a_tracks, window, tracker_a)
        b_input = _camera_input(b_id, CameraRole.DOWNSTREAM, b_tracks, window, tracker_b)
        result = run_corridor_window(window, (a_input, b_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
                                     evidence_thresholds=SYNTHETIC_EVIDENCE_THRESHOLDS)
        windows_out.append(_window_result(result, {a_id: a_tracks, b_id: b_tracks}))
        last_result = result

    # A second pass on the final window, now WITH an explicit SYNTHETIC
    # observation confirming the prediction, to show feedback.
    final_window = windows_out[-1]["window"]
    prediction = last_result.propagation_prediction
    feedback_windows_out = windows_out
    if prediction is not None:
        observation = PropagationObservation(
            observation_id="obs-agreement-synthetic", prediction_id=prediction.prediction_id,
            actual_arrival_camera=prediction.predicted_arrival_camera,
            actual_arrival_time=prediction.predicted_arrival_time,
            actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
            data_completeness=DataCompleteness.FULL,
        )
        a_tracks_final = _tracks(a_id, counts_a[-1], 2.0, (2 * WINDOW_SECONDS), (3 * WINDOW_SECONDS)) + [_stopped_track(a_id)]
        b_tracks_final = _tracks(b_id, 3, 8.0, (2 * WINDOW_SECONDS), (3 * WINDOW_SECONDS))
        a_input = CameraWindowInput(camera_id=a_id, role=CameraRole.UPSTREAM,
                                     aggregation=aggregate_traffic_state(a_id, final_window[0], final_window[1], a_tracks_final, MotionSpace.WORLD, CongestionStateTracker()),
                                     tracks=a_tracks_final)
        b_input = CameraWindowInput(camera_id=b_id, role=CameraRole.DOWNSTREAM,
                                     aggregation=aggregate_traffic_state(b_id, final_window[0], final_window[1], b_tracks_final, MotionSpace.WORLD, CongestionStateTracker()),
                                     tracks=b_tracks_final)
        result_with_feedback = run_corridor_window(
            tuple(final_window), (a_input, b_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
            propagation_observation=observation, evidence_thresholds=SYNTHETIC_EVIDENCE_THRESHOLDS,
        )
        feedback_windows_out = windows_out[:-1] + [_window_result(result_with_feedback, {a_id: a_tracks_final, b_id: b_tracks_final})]

    return {
        "scenario_id": "cross_camera_agreement",
        "title": "Cross-camera agreement",
        "description": "Upstream demand escalates across three windows while downstream stays clear; both cameras' evidence agrees, propagation predicts a downstream arrival, and a synthetic observation confirms it.",
        "windows": feedback_windows_out,
    }


def scenario_camera_disagreement() -> dict:
    """Two cameras whose evidence points at different causes -- exercises
    Dempster-Shafer conflict handling and a CONTRADICTED feedback branch."""
    a_id, b_id = "cam-A", "cam-B"
    topology = _topology("disagreement-corridor", a_id, b_id)
    window = (0.0, WINDOW_SECONDS)
    # cam-A: 7 vehicles against a capacity of 10 (occupancy 0.7, "congested")
    # -- kept below the elevated-flow gate (0.8 veh/s == 8 per 10s window,
    # half of this corridor's stipulated 1.6 veh/s capacity) so ONLY H1 fires, isolating a
    # clean single-hypothesis signal. cam-B: light traffic but with two
    # simultaneously-stopped vehicles -- supports H3 instead, contradicting H1.
    a_tracks = _tracks(a_id, 7, 2.0, *window)
    b_tracks = [_stopped_track(b_id, "s1"), _stopped_track(b_id, "s2")] + _tracks(b_id, 2, 8.0, *window)
    a_input = _camera_input(a_id, CameraRole.UPSTREAM, a_tracks, window, CongestionStateTracker(), max_capacity_vehicles=10)
    b_input = _camera_input(b_id, CameraRole.DOWNSTREAM, b_tracks, window, CongestionStateTracker())
    result = run_corridor_window(window, (a_input, b_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
                                 evidence_thresholds=SYNTHETIC_EVIDENCE_THRESHOLDS)

    feedback_windows = [_window_result(result, {a_id: a_tracks, b_id: b_tracks})]
    prediction = result.propagation_prediction
    if prediction is not None:
        # SYNTHETIC observation that materially disagrees with the prediction -> CONTRADICTED.
        wrong_camera = b_id if prediction.predicted_arrival_camera == a_id else a_id
        observation = PropagationObservation(
            observation_id="obs-disagreement-synthetic", prediction_id=prediction.prediction_id,
            actual_arrival_camera=wrong_camera, actual_arrival_time=prediction.predicted_arrival_time,
            actual_queue_growth_rate=prediction.predicted_queue_growth_rate, data_completeness=DataCompleteness.FULL,
        )
        result_with_feedback = run_corridor_window(
            window, (a_input, b_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
            propagation_observation=observation, evidence_thresholds=SYNTHETIC_EVIDENCE_THRESHOLDS,
        )
        feedback_windows = [_window_result(result_with_feedback, {a_id: a_tracks, b_id: b_tracks})]

    return {
        "scenario_id": "camera_disagreement",
        "title": "Camera disagreement",
        "description": "Camera A's evidence points toward demand/congestion causes while Camera B's evidence points toward a multi-vehicle stoppage; fusion must keep the conflict visible rather than average it away, and a synthetic observation contradicts the propagation prediction.",
        "windows": feedback_windows,
    }


def scenario_reliability_degradation() -> dict:
    """Same evidence pattern on both cameras, but Camera B's tracks are
    PARTIAL (poor continuity) -- exercises reliability discounting reducing
    a camera's influence on the fused ranking."""
    a_id, b_id = "cam-A", "cam-B"
    topology = _topology("reliability-corridor", a_id, b_id)
    window = (0.0, WINDOW_SECONDS)
    a_tracks = _tracks(a_id, 15, 2.0, *window)
    b_tracks = _partial_tracks(b_id, 15, 2.0, *window)  # same traffic pattern, but degraded tracking continuity
    a_input = _camera_input(a_id, CameraRole.UPSTREAM, a_tracks, window, CongestionStateTracker())
    b_input = _camera_input(b_id, CameraRole.DOWNSTREAM, b_tracks, window, CongestionStateTracker())
    result = run_corridor_window(window, (a_input, b_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
                                 evidence_thresholds=SYNTHETIC_EVIDENCE_THRESHOLDS)
    return {
        "scenario_id": "reliability_degradation",
        "title": "Reliability degradation",
        "description": "Both cameras observe the same traffic pattern, but Camera B's tracking continuity is degraded (partial occlusion throughout); reliability scoring discounts its evidence before fusion.",
        "windows": [_window_result(result, {a_id: a_tracks, b_id: b_tracks})],
    }


# ---------------------------------------------------------------------------
# Real video -- limited chain (reuses scripts/run_end_to_end_demo.py's own approach)
# ---------------------------------------------------------------------------


def _detect_gpu_info() -> dict:
    """Honest GPU/CUDA status for display -- never fabricated, never
    assumed. Torch/CUDA are already project dependencies (via ultralytics);
    this only reads their already-available status, no new dependency."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        cuda_available, device_name = False, None
    return {"cuda_available": cuda_available, "device_name": device_name}


def _box_entry(td, lane_result=None, speed_estimate=None) -> dict:
    """Serialize one tracked detection. `lane_result`
    (`world.lane_geometry.LaneAssignmentResult`) and `speed_estimate`
    (`perception.speed_estimation.VehicleSpeedEstimate`) are optional and
    additive: omitted (the default), the returned dict keeps its original
    4-key shape exactly -- a camera with no lane config/calibration (every
    camera before this commit) is completely unaffected. Display strings
    reuse `anvesh.rendering.format_lane_text`/`format_speed_text` (Commit
    8, already tested) rather than re-deciding them here.
    """
    entry = {
        "track_id": td.track_id,
        "bbox_xyxy": list(td.detection.bbox_xyxy),
        "vehicle_class": td.detection.vehicle_class.value,
        "confidence": td.detection.confidence,
    }
    if lane_result is not None:
        entry["lane"] = format_lane_text(lane_result)
        entry["lane_status"] = lane_result.status.value
    if speed_estimate is not None:
        entry["speed"] = format_speed_text(speed_estimate)
        entry["speed_status"] = speed_estimate.status.value
        entry["speed_confidence"] = speed_estimate.confidence.value
    return entry



# Presentation demo video (evaluated and selected over traffic.mp4 -- see
# the standalone presentation-video comparison report, which remains the
# source of truth for the 240s sample statistics quoted there). This is
# NOT part of the M6 validation harness/config -- m6_validation.toml is
# left untouched; these are independent, hardcoded constants reusing the
# EXACT SAME perception/window/capacity defaults used everywhere else in
# this project (never tuned for this video).
DEMO_VIDEO_PATH = REPO_ROOT / "demo.mp4"
DEMO_CAMERA_ID = "cam-demo"
DEMO_MODEL_NAME = "models/yolov8n.pt"
DEMO_CONFIDENCE_THRESHOLD = 0.25
DEMO_TRACKER_CONFIG = "bytetrack.yaml"
DEMO_MAX_MISSED_FRAMES = 5
DEMO_PARTIAL_AFTER_MISSED_FRAMES = 1
DEMO_MAX_CAPACITY_VEHICLES = 20  # same default as M6 -- see the comparison report's honest capacity-artifact note
DEMO_SAMPLE_SECONDS = 240.0  # matches the presentation-video comparison report's evaluated window exactly


def _write_trimmed_copy(source_path: Path, sample_seconds: float, fps: float, width: int, height: int):
    """Write a temporary copy of `source_path` containing only its first
    `sample_seconds` of frames, reusing the SAME `VideoWriter`
    (anvesh.perception.visualize) already used for annotated debug output
    elsewhere in this project -- no new video-encoding code. This exists
    only so `run_m1_pipeline` (which always runs to the end of whatever
    file it's given -- see its own module docstring) is bounded to a
    known-small amount of work for an arbitrary, possibly long, caller-
    supplied video; it never changes what run_m1_pipeline itself does.

    Returns (trimmed_path, frame_count_written). Caller must delete
    trimmed_path.parent when done (it is a dedicated temp directory).
    """
    import cv2
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video for trimming: {source_path}")
    max_frames = int(round(sample_seconds * fps)) if fps else None

    tmp_dir = Path(tempfile.mkdtemp(prefix="anvesh_live_trim_"))
    trimmed_path = tmp_dir / f"trimmed_{source_path.stem}.mp4"
    writer = VideoWriter(trimmed_path, fps=fps, frame_size=(width, height))
    written = 0
    try:
        while max_frames is None or written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"No frames could be read from {source_path} while trimming")
    return trimmed_path, written


def real_video_limited_chain(
    video_path=DEMO_VIDEO_PATH,
    camera_id: str = DEMO_CAMERA_ID,
    model_name: str = DEMO_MODEL_NAME,
    confidence_threshold: float = DEMO_CONFIDENCE_THRESHOLD,
    tracker_config: str = DEMO_TRACKER_CONFIG,
    max_missed_frames: int = DEMO_MAX_MISSED_FRAMES,
    partial_after_missed_frames: int = DEMO_PARTIAL_AFTER_MISSED_FRAMES,
    max_capacity_vehicles: int = DEMO_MAX_CAPACITY_VEHICLES,
    sample_seconds: float = DEMO_SAMPLE_SECONDS,
    pre_trim_to_sample: bool = False,
    trimmed_output_path=None,
    progress=None,
    calibration_profile=None,
    lane_regions: tuple = (),
) -> dict:
    """Run the real single-camera chain (M1 -> windowing -> M4-M8) on
    `video_path` and serialize it exactly like every other window result
    in this file.

    Every parameter defaults to the exact original DEMO_* constant it
    replaces, so the zero-argument call this module's own `main()` makes
    is byte-for-byte behavior-identical to before this function was
    parameterized -- this refactor is purely additive.

    `pre_trim_to_sample` (default False, preserving the original
    behavior): when True, the input is first trimmed to `sample_seconds`
    (see `_write_trimmed_copy`) before `run_m1_pipeline` ever sees it,
    instead of letting `run_m1_pipeline` run over the entire source file
    and discarding everything past `sample_seconds` afterward. Needed for
    an arbitrary, possibly long, caller-supplied video (scripts/
    analyze_video.py sets this); left False here so demo.mp4's own
    existing export is untouched.

    `trimmed_output_path` (default None): only meaningful when trimming
    actually happens. If given, the trimmed clip is additionally copied
    to this path (parent directories created as needed) before the
    internal temp copy is deleted, so a caller that does NOT want to
    embed/keep the full, possibly very large, original source video (e.g.
    scripts/analyze_video.py building a live dashboard from an
    arbitrary-length judge video) has a small, honest, exactly-what-was-
    analyzed clip to embed instead. The caller owns this path afterward
    (this function never deletes it).

    `progress` (default None, meaning silent -- exactly today's
    behavior): an optional `str -> None` callable invoked at real
    execution checkpoints only (video probe, start/end of detection+
    tracking, one line per completed window) -- never speculative, never
    printed before the thing it describes has actually happened.

    `calibration_profile` (default None, preserving today's behavior):
    an optional `world.calibration.CalibrationProfile` for this camera.
    Threaded into `run_m1_pipeline` (already-supported parameter, never
    previously used here) so `vehicle_tracks`/`window_tracks` come out in
    WORLD space (metres) when calibrated, and used to compute one
    `perception.speed_estimation.estimate_vehicle_speed` call per track
    AFTER the M1 pass completes (never inside `on_frame_tracked` --
    per-track speed needs the track's full, finished trajectory). When
    None, no speed is computed and every box's `speed`/`speed_status`/
    `speed_confidence` keys stay absent, exactly as before this
    parameter existed.

    `lane_regions` (default `()`, preserving today's behavior): an
    optional tuple of `world.lane_geometry.LaneRegion` for this camera
    (typically `world.lane_config.LaneCameraConfig.regions`). When
    non-empty, each captured box additionally gets a per-detection lane
    assignment (`perception.lane_assignment.assign_detection_lane`,
    Commit 4 -- recomputed per observation, never cached per track). When
    empty, no lane assignment is attempted and every box's `lane`/
    `lane_status` keys stay absent, exactly as before this parameter
    existed.
    """
    video_path = Path(video_path)
    _p = progress if progress is not None else (lambda msg: None)

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    full_video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    full_video_duration_s = full_video_frame_count / video_fps if video_fps else 0.0
    _p(f"Video detected: {video_width}x{video_height} @ {video_fps:.2f} FPS ({full_video_duration_s:.1f}s total, {full_video_frame_count} frames)")

    # Never analyze more than the video actually contains.
    effective_sample_seconds = min(sample_seconds, full_video_duration_s) if full_video_duration_s else sample_seconds
    if effective_sample_seconds < sample_seconds:
        _p(f"Requested sample ({sample_seconds:.1f}s) exceeds video length -- clamped to {effective_sample_seconds:.1f}s")

    m1_video_path = video_path
    trimmed_tmp_dir = None
    if pre_trim_to_sample and full_video_duration_s > effective_sample_seconds + 0.5:
        _p(f"Trimming to the first {effective_sample_seconds:.1f}s for analysis (source is {full_video_duration_s:.1f}s) ...")
        m1_video_path, _written = _write_trimmed_copy(
            video_path, effective_sample_seconds, video_fps, video_width, video_height
        )
        trimmed_tmp_dir = m1_video_path.parent
        if trimmed_output_path is not None:
            trimmed_output_path = Path(trimmed_output_path)
            trimmed_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(m1_video_path, trimmed_output_path)
            _p(f"Saved analyzed clip to {trimmed_output_path}")

    resolved_model_path = Path(model_name)
    if not resolved_model_path.is_absolute():
        resolved_model_path = REPO_ROOT / resolved_model_path
    tracker = ByteTrackYoloTracker(
        model_name=str(resolved_model_path),
        confidence_threshold=confidence_threshold,
        tracker_config=tracker_config,
    )

    # Captured via the additive `on_frame_tracked` hook (anvesh/perception/
    # pipeline.py) -- the SAME tracked_detections the pipeline itself
    # already computes per frame for track_manager.update(); never
    # recomputed or duplicated here. Frame data is only RETAINED for the
    # first effective_sample_seconds (matching the evaluated/reported
    # sample). Only the raw TrackedDetection objects are kept at capture
    # time -- lane assignment (cheap, per-observation) and speed lookup
    # (needs each track's FINISHED trajectory, only available after
    # run_m1_pipeline returns) both happen in a second, data-only pass
    # below, never during this single YOLO+ByteTrack pass.
    frames_captured = []
    frame_progress_every = max(1, int(video_fps * 2)) if video_fps else 50  # ~ every 2s of source video

    def _capture_frame(frame_index, timestamp, tracked_detections):
        if timestamp < effective_sample_seconds:
            frames_captured.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "tracked": list(tracked_detections),
            })
        if progress is not None and frame_index % frame_progress_every == 0:
            pct = min(100.0, 100.0 * timestamp / effective_sample_seconds) if effective_sample_seconds else 0.0
            _p(f"  ... frame {frame_index} (t={timestamp:.1f}s, {pct:.0f}%), {len(tracked_detections)} tracked detections")

    _p("Stage 1/2: Running vehicle detection + tracking (YOLOv8 + ByteTrack)...")
    try:
        m1_result = run_m1_pipeline(
            video_path=m1_video_path, camera_id=camera_id, tracker=tracker,
            model_id=model_name, model_version="pretrained",
            max_missed_frames=max_missed_frames,
            partial_after_missed_frames=partial_after_missed_frames,
            calibration_profile=calibration_profile,
            on_frame_tracked=_capture_frame,
        )
    finally:
        if trimmed_tmp_dir is not None:
            shutil.rmtree(trimmed_tmp_dir, ignore_errors=True)
    _p(f"Stage 1/2 complete: {len(m1_result.vehicle_tracks)} tracks from {m1_result.frame_count} frames.")

    # Per-track approximate speed -- ONE estimate_vehicle_speed call per
    # finished track (never per frame, never per box: see
    # perception.speed_estimation's own module docstring for why this is
    # a whole-trajectory, compute-once quantity), keyed by the tracker's
    # raw track_id (matching TrackedDetection.track_id, NOT the
    # camera-qualified id VehicleSpeedEstimate.track_id carries). Only
    # attempted when a calibration_profile was actually supplied --
    # otherwise this dict stays empty and every box's speed keys stay
    # absent, exactly today's behavior for every camera without one.
    speed_by_track_id = {}
    if calibration_profile is not None:
        for state in m1_result.track_states:
            speed_by_track_id[state.track_id] = estimate_vehicle_speed(camera_id, state, calibration_profile)

    # PASS 2 (data-only, no re-inference): attach lane assignment and the
    # already-computed speed estimate to each already-captured box.
    frames_out = []
    for frame in frames_captured:
        boxes = []
        for td in frame["tracked"]:
            lane_result = assign_detection_lane(td.detection, lane_regions) if lane_regions else None
            speed_estimate = speed_by_track_id.get(td.track_id)
            boxes.append(_box_entry(td, lane_result=lane_result, speed_estimate=speed_estimate))
        frames_out.append({
            "frame_index": frame["frame_index"],
            "timestamp": frame["timestamp"],
            "boxes": boxes,
        })

    # Windows are bounded to the evaluated/reported sample explicitly
    # (not `video_duration_from_tracks`, which would span the full source
    # video when pre_trim_to_sample is False) -- tracks_in_window() below
    # naturally excludes any track outside [0, effective_sample_seconds)
    # from each window regardless.
    _p(f"Stage 2/2: Building {effective_sample_seconds:.0f}s of temporal windows ({WINDOW_SECONDS:.0f}s each), "
       f"then evidence -> hypothesis evaluation -> Dempster-Shafer ranking -> safety events per window "
       f"(exact order documented in anvesh/pipeline.py:run_corridor_window)...")
    windows = generate_windows(effective_sample_seconds, WINDOW_SECONDS)
    congestion_tracker = CongestionStateTracker()

    # WORLD only when this camera is actually calibrated -- mirrors
    # `schema_conversion.build_vehicle_track`'s own existing motion-space
    # branch (§1 of the Commit 9 integration audit), not new logic. A
    # camera with no calibration_profile (every camera before this
    # commit) keeps exactly today's IMAGE-space aggregation.
    motion_space = (
        MotionSpace.WORLD if calibration_profile is not None and calibration_profile.is_calibrated
        else MotionSpace.IMAGE
    )

    windows_out = []
    for window_obj in windows:
        window = (window_obj.start, window_obj.end)
        window_tracks = tracks_in_window(m1_result.vehicle_tracks, window_obj)
        aggregation = aggregate_traffic_state(
            camera_id, window[0], window[1], window_tracks, motion_space, congestion_tracker,
            max_capacity_vehicles=max_capacity_vehicles,
        )
        camera_input = CameraWindowInput(camera_id=camera_id, role=CameraRole.UPSTREAM, aggregation=aggregation, tracks=window_tracks)
        result = run_corridor_window(window, (camera_input,))
        windows_out.append(_window_result(result, {camera_id: window_tracks}))
        top = result.candidate_ranking.ranked_list[0] if result.candidate_ranking.ranked_list else None
        top_str = f"{top.hypothesis_id}({top.belief:.2f})" if top and top.belief > 0 else "none"
        _p(f"  window [{window[0]:>5.1f}-{window[1]:>5.1f}s]: vehicles={aggregation.traffic_state.vehicle_count:<3} "
           f"outcome={result.candidate_ranking.outcome.value:<20} top={top_str:<14} safety_events={len(result.safety_events)}")

    sampled_frame_count = int(effective_sample_seconds * video_fps) if video_fps else len(frames_out)

    return {
        "video_name": video_path.name,
        "video_path": video_path.name,
        "camera_id": camera_id,
        "frame_count": sampled_frame_count,
        "full_video_frame_count": full_video_frame_count,
        "full_video_duration_s": full_video_duration_s,
        "sampled_duration_s": effective_sample_seconds,
        "total_tracks": len({t.track_id for t in m1_result.vehicle_tracks if t.first_seen < effective_sample_seconds}),
        "video_fps": video_fps,
        "video_width": video_width,
        "video_height": video_height,
        "gpu": _detect_gpu_info(),
        "perception_mode": "precomputed_replay",
        "perception_method": "YOLOv8n + ByteTrack",
        "frames": frames_out,
        "windows": windows_out,
    }


def main() -> int:
    data = {
        "synthetic": {
            "scenarios": [
                scenario_cross_camera_agreement(),
                scenario_camera_disagreement(),
                scenario_reliability_degradation(),
            ]
        },
        "real_video": real_video_limited_chain(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
