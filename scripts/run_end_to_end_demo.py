"""M8 demo: exercise the anvesh/pipeline.py orchestrator two ways.

Usage (from the repository root, with the venv active):

    python scripts/run_end_to_end_demo.py

Section 1 (SYNTHETIC END-TO-END) is a small, deterministic, hand-authored
two-camera WORLD-space scenario -- same style as the M2-M5 demo scripts
(explicit tracks, explicit CorridorTopology, explicit segment length) --
run through every stage the orchestrator supports, including a clearly
labeled SYNTHETIC PropagationObservation to exercise feedback. THIS IS
SYNTHETIC DATA. THIS IS NOT REAL-WORLD VALIDATION.

Section 2 (REAL VIDEO -- LIMITED CHAIN) runs the real M1 pipeline on one
already-used M6 video and pushes its real tracks through the SAME
orchestrator in single-camera mode, showing exactly where the honest
chain stops: evidence, single-camera ranking, and stopped-vehicle safety
detection are real; corridor state, propagation, and feedback are
reported UNAVAILABLE with their exact reasons, never fabricated. This is
NOT a real end-to-end corridor validation -- see
`scripts/run_real_world_validation.py`'s own module docstring for the
same honesty rules this section follows.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.tracking import ByteTrackYoloTracker
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.perception.windowing import generate_windows, tracks_in_window, video_duration_from_tracks
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

M6_CONFIG_PATH = REPO_ROOT / "configs" / "experiment" / "m6_validation.toml"


def _line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))


# ===========================================================================
# SECTION 1: SYNTHETIC END-TO-END
# ===========================================================================

WINDOW = (0.0, 10.0)
SEGMENT_LENGTH_M = 100.0  # the corridor's own inter-camera distance below -- not an arbitrary second number


def _synthetic_tracks(camera_id, count, speed, first_seen=0.0, last_seen=10.0):
    return [
        VehicleTrack(
            track_id=f"{camera_id}:{i}", camera_id=camera_id, first_seen=first_seen, last_seen=last_seen,
            vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE, position_history=[],
            speed_estimate=Measurement(value=speed, error=0.0), motion_space=MotionSpace.WORLD,
        )
        for i in range(count)
    ]


def _synthetic_stopped_track(camera_id):
    """One vehicle that barely moves for 4s -- enough to qualify as
    STOPPED under the default StoppedVehicleThresholds (dwell_seconds=3.0,
    world_speed_threshold_m_per_s=0.5)."""
    positions = [(10.0 + 0.05 * i, 0.0, i * 0.5) for i in range(9)]  # ~0.1 m/s over 4.0s
    return VehicleTrack(
        track_id=f"{camera_id}:stopped", camera_id=camera_id, first_seen=positions[0][2], last_seen=positions[-1][2],
        vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE,
        position_history=[WorldPosition(world_x=x, world_y=y, timestamp=t) for (x, y, t) in positions],
        speed_estimate=Measurement(value=99.0, error=0.0),  # deliberately unused
        motion_space=MotionSpace.WORLD,
    )


def _synthetic_topology():
    return CorridorTopology(
        corridor_id="demo-corridor",
        placements=(
            CameraPlacement(camera_id="cam-A-upstream", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(
                camera_id="cam-B-downstream", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=SEGMENT_LENGTH_M
            ),
        ),
    )


def _camera_input(camera_id, role, tracks):
    tracker = CongestionStateTracker()
    aggregation = aggregate_traffic_state(camera_id, WINDOW[0], WINDOW[1], tracks, MotionSpace.WORLD, tracker)
    return CameraWindowInput(camera_id=camera_id, role=role, aggregation=aggregation, tracks=tracks)


def _print_result(result) -> None:
    print(f"  camera_ids={result.camera_ids}")
    print(f"  candidate_ranking: outcome={result.candidate_ranking.outcome.value}", end="")
    if result.candidate_ranking.ranked_list:
        top = result.candidate_ranking.ranked_list[0]
        print(f"  top-1={top.hypothesis_id} belief={top.belief:.3f} tier={top.confidence_tier.value}")
    else:
        print()
    print(f"  corridor_state: {'present' if result.corridor_state else 'None'}"
          f"{'' if result.corridor_state else f'  reason={result.corridor_state_reason!r}'}")
    if result.corridor_state:
        print(f"    active_ranking_id={result.corridor_state.active_ranking_id}  "
              f"fused_congestion_level={result.corridor_state.fused_congestion_level.value}")
    print(f"  propagation_prediction: {'present' if result.propagation_prediction else 'None'}"
          f"{'' if result.propagation_prediction else f'  reason={result.propagation_reason!r}'}")
    if result.propagation_prediction:
        p = result.propagation_prediction
        print(f"    shockwave_speed={p.predicted_shockwave_speed.value:.2f} m/s  "
              f"arrival_camera={p.predicted_arrival_camera}  arrival_time={p.predicted_arrival_time:.1f}s")
    print(f"  feedback_result: {'present' if result.feedback_result else 'None'}"
          f"{'' if result.feedback_result else f'  reason={result.feedback_reason!r}'}")
    if result.feedback_result:
        print(f"    outcome={result.feedback_result.update.outcome.value}  reason={result.feedback_result.reason}")
    print(f"  safety_events: {len(result.safety_events)}")
    for event in result.safety_events:
        print(f"    track_id={event.track_id}  window={event.window}  confidence_tier={event.confidence_tier.value}")
    print(f"  limitations: {list(result.limitations)}")


def run_synthetic_section() -> None:
    print("=" * 78)
    print("SYNTHETIC END-TO-END")
    print("=" * 78)
    print("THIS IS SYNTHETIC DATA. THIS IS NOT REAL-WORLD VALIDATION.")
    print("Hand-authored tracks/corridor, same style as the M2-M5 demo scripts.")

    upstream_tracks = _synthetic_tracks("cam-A-upstream", count=15, speed=2.0) + [
        _synthetic_stopped_track("cam-A-upstream")
    ]
    downstream_tracks = _synthetic_tracks("cam-B-downstream", count=3, speed=8.0)
    upstream_input = _camera_input("cam-A-upstream", CameraRole.UPSTREAM, upstream_tracks)
    downstream_input = _camera_input("cam-B-downstream", CameraRole.DOWNSTREAM, downstream_tracks)
    topology = _synthetic_topology()

    _line("Pass 1: no propagation observation supplied yet")
    result = run_corridor_window(
        WINDOW, (upstream_input, downstream_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
    )
    _print_result(result)

    _line("Pass 2: an explicit SYNTHETIC PROPAGATION OBSERVATION is supplied")
    print("  SYNTHETIC PROPAGATION OBSERVATION -- hand-authored to match the prediction; NOT measured.")
    prediction = result.propagation_prediction
    observation = PropagationObservation(
        observation_id="obs-demo-synthetic",
        prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )
    result_with_feedback = run_corridor_window(
        WINDOW, (upstream_input, downstream_input), corridor_topology=topology, segment_length_m=SEGMENT_LENGTH_M,
        propagation_observation=observation,
    )
    _print_result(result_with_feedback)


# ===========================================================================
# SECTION 2: REAL VIDEO -- LIMITED CHAIN
# ===========================================================================


def run_real_video_section() -> None:
    print("\n" + "=" * 78)
    print("REAL VIDEO -- LIMITED CHAIN")
    print("=" * 78)
    print("Reuses one existing M6 video. This is NOT a real end-to-end corridor")
    print("validation -- it is a single, uncalibrated camera. No topology, no")
    print("calibration, no corridor is fabricated to make later stages 'work'.")

    with M6_CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)
    validation_cfg = config["validation"]
    seq_cfg = config["sequence"][0]  # one M6 video is enough for this demo; the second adds no genuine value here
    video_path = REPO_ROOT / seq_cfg["video_path"]
    camera_id = seq_cfg["camera_id"]

    print(f"\nvideo: {video_path.name}  camera_id={camera_id}")

    tracker = ByteTrackYoloTracker(
        model_name=str(REPO_ROOT / validation_cfg["model_name"]),
        confidence_threshold=validation_cfg["confidence_threshold"],
        tracker_config=validation_cfg["tracker_config"],
    )
    m1_result = run_m1_pipeline(
        video_path=video_path, camera_id=camera_id, tracker=tracker,
        model_id=validation_cfg["model_name"], model_version="pretrained",
        max_missed_frames=validation_cfg["max_missed_frames"],
        partial_after_missed_frames=validation_cfg["partial_after_missed_frames"],
    )
    print(f"OBSERVED: {len(m1_result.vehicle_tracks)} tracks from {m1_result.frame_count} frames "
          "(real M1 perception, uncalibrated -> IMAGE motion_space)")

    total_duration = video_duration_from_tracks(m1_result.vehicle_tracks)
    windows = generate_windows(total_duration, validation_cfg["window_seconds"])
    congestion_tracker = CongestionStateTracker()

    total_safety_events = 0
    for window_obj in windows:
        window = (window_obj.start, window_obj.end)
        window_tracks = tracks_in_window(m1_result.vehicle_tracks, window_obj)
        aggregation = aggregate_traffic_state(
            camera_id, window[0], window[1], window_tracks, MotionSpace.IMAGE, congestion_tracker,
            max_capacity_vehicles=validation_cfg["max_capacity_vehicles"],
        )
        camera_input = CameraWindowInput(
            camera_id=camera_id, role=CameraRole.UPSTREAM, aggregation=aggregation, tracks=window_tracks,
        )
        result = run_corridor_window(window, (camera_input,))
        total_safety_events += len(result.safety_events)
        print(
            f"  window=[{window[0]:>5.1f},{window[1]:>5.1f})  vehicle_count={aggregation.traffic_state.vehicle_count:>2}  "
            f"ranking_outcome={result.candidate_ranking.outcome.value:<22}  "
            f"corridor_state={'UNAVAILABLE' if result.corridor_state is None else 'present'}  "
            f"propagation={'UNAVAILABLE' if result.propagation_prediction is None else 'present'}  "
            f"feedback={'UNAVAILABLE' if result.feedback_result is None else 'present'}  "
            f"safety_events={len(result.safety_events)}"
        )

    print(f"\ncorridor_state = UNAVAILABLE  reason = no genuine two-camera corridor exists for this video")
    print(f"propagation    = UNAVAILABLE  reason = no calibrated two-camera corridor")
    print(f"feedback       = UNAVAILABLE  reason = no physical propagation observation exists")
    print(f"total safety events observed across all windows: {total_safety_events}")
    print("\nThis section demonstrates REAL VIDEO -- LIMITED CHAIN, not real end-to-end")
    print("corridor validation. Evidence/ranking/safety-event stages are real; corridor")
    print("state, propagation, and feedback are honestly unavailable for this input.")


def main() -> int:
    print("ANVESH M8 -- end-to-end orchestrator demo (anvesh/pipeline.py)")
    print("This composes existing M1-M7 functions only; no new algorithm is introduced here.\n")
    run_synthetic_section()
    run_real_video_section()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
