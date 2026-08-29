"""M2 demo: single-camera observations -> common world coordinates ->
camera ordering -> temporally aligned observations.

Usage (from the repository root, with the venv active):

    python scripts/run_world_alignment_demo.py

This uses the synthetic, manually-authored two-camera fixture in
`anvesh/world/synthetic_fixture.py` -- NOT real video, NOT a measured
calibration. It exists purely to exercise calibration, corridor topology,
world-space motion, and temporal alignment end-to-end, and to print a
plain diagnostic of each stage. No dashboard, no fusion/cause reasoning.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.fusion.alignment import align_observations
from anvesh.perception.schema_conversion import build_camera_observation, build_vehicle_track
from anvesh.perception.track_manager import TrackManager
from anvesh.world.synthetic_fixture import build_fixture


def _line(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))


def build_camera_track_and_observations(camera_id, calibration, tracked_detections, model_id):
    manager = TrackManager(max_missed_frames=1, partial_after_missed_frames=1)
    observations = []
    for td in tracked_detections:
        ts = td.detection.timestamp
        manager.update(ts, [td])
        observations.append(
            build_camera_observation(
                camera_id=camera_id,
                frame_index=td.detection.frame_index,
                timestamp=ts,
                detections=[td.detection],
                model_id=model_id,
                model_version="synthetic-fixture",
            )
        )
    manager.finalize()
    (state,) = manager.all_tracks()  # exactly one synthetic vehicle per camera in this fixture
    track = build_vehicle_track(camera_id, state, calibration_profile=calibration)
    return track, observations


def main() -> int:
    print("ANVESH M2 demo -- SYNTHETIC fixture (not real video, not measured calibration)")

    fixture = build_fixture()

    _line("1. Per-camera observations (image-space) -> calibrated world-space tracks")
    track_a, observations_a = build_camera_track_and_observations(
        fixture.camera_a_calibration.camera_id,
        fixture.camera_a_calibration,
        fixture.camera_a_tracked_detections,
        model_id="synthetic",
    )
    track_b, observations_b = build_camera_track_and_observations(
        fixture.camera_b_calibration.camera_id,
        fixture.camera_b_calibration,
        fixture.camera_b_tracked_detections,
        model_id="synthetic",
    )

    for label, obs_list, track, calib in (
        ("Camera A (upstream)", observations_a, track_a, fixture.camera_a_calibration),
        ("Camera B (downstream)", observations_b, track_b, fixture.camera_b_calibration),
    ):
        print(f"\n{label}: calibration status={calib.status.value}, "
              f"reprojection_error_m={calib.reprojection_error_m:.6f}")
        for obs in obs_list:
            d = obs.detections[0]
            print(
                f"  frame={d['frame_index']} t={d['timestamp']:.2f}s "
                f"image_bbox={tuple(round(v, 1) for v in d['bbox_xyxy'])}"
            )
        print(f"  -> VehicleTrack.motion_space = {track.motion_space.value}")
        world_positions = ", ".join(f"({p.world_x:.2f}, {p.world_y:.2f})m @t={p.timestamp:.2f}s" for p in track.position_history)
        print(f"  -> world positions: {world_positions}")
        print(
            f"  -> speed_estimate = {track.speed_estimate.value:.2f} m/s "
            f"(+/- {track.speed_estimate.error:.2f}), NEVER a pixels/second value"
        )

    _line("2. Corridor topology / camera ordering")
    corridor = fixture.corridor
    print(f"corridor_id={corridor.corridor_id}")
    print(f"ordered_camera_ids (upstream -> downstream) = {corridor.ordered_camera_ids()}")
    print(
        f"is_upstream_of(camera_a, camera_b) = "
        f"{corridor.is_upstream_of(fixture.camera_a_calibration.camera_id, fixture.camera_b_calibration.camera_id)}"
    )
    print(
        f"distance_between_m(camera_a, camera_b) = "
        f"{corridor.distance_between_m(fixture.camera_a_calibration.camera_id, fixture.camera_b_calibration.camera_id):.1f} m"
    )

    _line("3. Temporal alignment on the fixture's own observation streams")
    print(
        "Camera A's 3 observations occur in [0.0s, 1.0s]; camera B's occur "
        f"in [{observations_b[0].frame_timestamp:.1f}s, {observations_b[-1].frame_timestamp:.1f}s] "
        "because the vehicle takes real travel time to reach camera B. These "
        "are genuinely different wall-clock instants, so a tight tolerance "
        "correctly reports them as unmatched -- alignment synchronizes camera "
        "CLOCKS, it does not claim the same vehicle is 'the same event' at both cameras."
    )
    pairs = align_observations(observations_a, observations_b, max_offset_seconds=0.1)
    for pair in pairs:
        a_ts = pair.camera_a_observation.frame_timestamp if pair.camera_a_observation else None
        b_ts = pair.camera_b_observation.frame_timestamp if pair.camera_b_observation else None
        print(f"  ref_t={pair.reference_timestamp:.2f}s  camera_a_t={a_ts}  camera_b_t={b_ts}  complete={pair.is_complete}")

    _line("4. Temporal alignment mechanics on an independent, overlapping synthetic pair")
    print(
        "A separate, small illustrative pair of streams (not tied to the "
        "vehicle scenario above) that DO overlap in time, showing equal "
        "timestamps, a small clock offset, and a missing frame all handled."
    )
    a_stream, b_stream = _overlapping_demo_streams()
    overlap_pairs = align_observations(a_stream, b_stream, max_offset_seconds=0.05)
    for pair in overlap_pairs:
        a_ts = pair.camera_a_observation.frame_timestamp if pair.camera_a_observation else None
        b_ts = pair.camera_b_observation.frame_timestamp if pair.camera_b_observation else None
        print(
            f"  ref_t={pair.reference_timestamp:.2f}s  camera_a_t={a_ts}  camera_b_t={b_ts}  "
            f"offset={pair.time_offset_seconds}  complete={pair.is_complete}"
        )

    return 0


@dataclass
class _DemoObservation:
    frame_timestamp: float


def _overlapping_demo_streams():
    # Equal timestamps at t=0.0; a 0.02s clock-offset step at t=1.0/1.02;
    # a missing camera-B frame at t=2.0.
    camera_a = [_DemoObservation(0.0), _DemoObservation(1.0), _DemoObservation(2.0)]
    camera_b = [_DemoObservation(0.0), _DemoObservation(1.02)]
    return camera_a, camera_b


if __name__ == "__main__":
    raise SystemExit(main())
