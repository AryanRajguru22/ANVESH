"""M7 demo: persistent stopped-vehicle detection over a handful of
deterministic, hand-constructed tracks (not real video).

Usage (from the repository root, with the venv active):

    python scripts/run_stopped_vehicle_demo.py

Demonstrates MOVING, STOPPED, and INSUFFICIENT_EVIDENCE outcomes,
SafetyEvent creation (only for STOPPED), and that repeated execution on
the same input is deterministic. This is candidate-primitive detection
only -- not accident detection, not a claim of real-world accuracy (see
`anvesh/safety/stopped_vehicle.py`'s module docstring for the exact scope).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.safety.stopped_vehicle import build_safety_event, classify_stopped_vehicle
from anvesh.storage.schemas import Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack, WorldPosition


def _image_track(track_id: str, positions: list, occlusion_state=OcclusionState.VISIBLE) -> VehicleTrack:
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-demo",
        first_seen=positions[0][2],
        last_seen=positions[-1][2],
        vehicle_class=VehicleClass.CAR,
        occlusion_state=occlusion_state,
        position_history=list(positions),
        speed_estimate=Measurement(value=0.0, error=0.0),  # deliberately unused by the detector
        motion_space=MotionSpace.IMAGE,
    )


def _world_track(track_id: str, positions: list) -> VehicleTrack:
    return VehicleTrack(
        track_id=track_id,
        camera_id="cam-demo-world",
        first_seen=positions[0][2],
        last_seen=positions[-1][2],
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[WorldPosition(world_x=x, world_y=y, timestamp=t) for (x, y, t) in positions],
        speed_estimate=Measurement(value=0.0, error=0.0),
        motion_space=MotionSpace.WORLD,
    )


def _print_result(label: str, track: VehicleTrack) -> None:
    result = classify_stopped_vehicle(track)
    event = build_safety_event(result)
    print(f"\n--- {label} ---")
    print(f"  track_id={track.track_id}  motion_space={result.motion_space.value}")
    print(f"  status={result.status.value}  heuristic_score={result.heuristic_score:.3f} (NOT a probability)"
          f"  confidence_tier={result.confidence_tier.value}")
    print(f"  dwell_window={result.dwell_window}")
    print(f"  factors: observed_duration={result.factors['observed_duration']:.2f}s "
          f"observation_count={result.factors['observation_count']} "
          f"temporal_coverage={result.factors['temporal_coverage']:.2f} "
          f"reasons={result.factors['reasons']}")
    print(f"  SafetyEvent: {event if event is None else f'event_type={event.event_type} window={event.window}'}")


def main() -> int:
    print("ANVESH M7 -- persistent stopped-vehicle detection demo")
    print("Deterministic, hand-constructed tracks only. Not real video, not accident detection.")

    moving = _image_track("cam-demo:1", [(20.0 * i, 0.0, float(i)) for i in range(6)])
    stopped_image = _image_track("cam-demo:2", [(100.0, 100.0, i * 0.5) for i in range(8)])
    insufficient = _image_track("cam-demo:3", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)])
    stopped_world = _world_track("cam-demo-world:1", [(10.0 + 0.05 * i, 20.0, i * 0.5) for i in range(8)])

    _print_result("MOVING", moving)
    _print_result("STOPPED (IMAGE-space heuristic)", stopped_image)
    _print_result("INSUFFICIENT_EVIDENCE (too few observations)", insufficient)
    _print_result("STOPPED (WORLD-space metric)", stopped_world)

    print("\n--- Determinism check ---")
    first = classify_stopped_vehicle(stopped_image)
    second = classify_stopped_vehicle(stopped_image)
    print(f"  repeated classification identical: {first == second}")

    print("\nDisclaimer: heuristic_score is a documented, non-calibrated heuristic, never a")
    print("probability. This detects a persistent-stop candidate primitive only -- it makes")
    print("no accident, wrong-way, or speeding determination, and no real-world accuracy claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
