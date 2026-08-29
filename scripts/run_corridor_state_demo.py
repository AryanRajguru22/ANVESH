"""M3 demo: two cameras' VehicleTracks -> per-window TrafficState (with
hysteresis) -> reliability -> CorridorState, persisted to SQLite, printed
as a timeline table.

Usage (from the repository root, with the venv active):

    python scripts/run_corridor_state_demo.py

Uses a SYNTHETIC, scripted 5-window traffic timeline (not real video) --
camera A (upstream) cycles free -> building -> congested -> dissipating
-> free while camera B (downstream) stays light throughout, so the
timeline shows the full FREE_FLOW/BUILDING/CONGESTED/DISSIPATING cycle and
how the corridor-level state takes the worse of the two cameras. This is
a diagnostic CLI table, not a dashboard, and no cause/propagation claims
are made anywhere in this output.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.evidence.reliability import compute_reliability
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.db import CorridorStateStore
from anvesh.storage.schemas import Measurement, MotionSpace, OcclusionState, VehicleClass, VehicleTrack
from anvesh.world.corridor_state import assemble_corridor_state
from anvesh.world.synthetic_fixture import build_corridor_topology

CORRIDOR_ID = "dev-corridor-01"
WINDOW_SECONDS = 10.0


def _track(track_id, camera_id, window_start, window_end, speed_value):
    return VehicleTrack(
        track_id=track_id,
        camera_id=camera_id,
        first_seen=window_start,
        last_seen=window_end,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=0.0),
        motion_space=MotionSpace.WORLD,
    )


def _cam_a_tracks(window_start, window_end, window_index):
    # free(0) -> building(1) -> congested(2) -> dissipating(3) -> free(4)
    scripted = [
        (2, 10.0),  # light, fast
        (14, 1.5),  # sudden heavy, slow
        (14, 1.5),  # sustained -> confirms CONGESTED
        (3, 9.0),  # sudden clear signal
        (3, 9.0),  # sustained -> confirms FREE_FLOW
    ]
    count, speed = scripted[window_index]
    return [_track(f"a{i}", "cam-A-upstream", window_start, window_end, speed) for i in range(count)]


def _cam_b_tracks(window_start, window_end, window_index):
    return [_track("b0", "cam-B-downstream", window_start, window_end, 9.0)]


def main() -> int:
    topology = build_corridor_topology()
    tracker_a = CongestionStateTracker()
    tracker_b = CongestionStateTracker()

    db_path = REPO_ROOT / "outputs" / "corridor_state_demo.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # fresh demo DB each run
    store = CorridorStateStore(db_path)

    header = (
        f"{'time (s)':>12} | {'cam A state':<12} | {'cam B state':<12} | {'corridor state':<14} | "
        f"{'A n/v(m/s)':<12} | {'B n/v(m/s)':<12} | {'A rel':>6} | {'B rel':>6}"
    )
    print("ANVESH M3 demo -- SYNTHETIC scripted timeline (not real video)")
    print(header)
    print("-" * len(header))

    try:
        for i in range(5):
            ws, we = i * WINDOW_SECONDS, (i + 1) * WINDOW_SECONDS
            a_tracks = _cam_a_tracks(ws, we, i)
            b_tracks = _cam_b_tracks(ws, we, i)

            agg_a = aggregate_traffic_state("cam-A-upstream", ws, we, a_tracks, MotionSpace.WORLD, tracker_a)
            agg_b = aggregate_traffic_state("cam-B-downstream", ws, we, b_tracks, MotionSpace.WORLD, tracker_b)
            rel_a = compute_reliability("cam-A-upstream", ws, we, a_tracks, agg_a)
            rel_b = compute_reliability("cam-B-downstream", ws, we, b_tracks, agg_b)

            assembly = assemble_corridor_state(
                f"cs-window-{i}",
                topology,
                (ws, we),
                {"cam-A-upstream": agg_a.traffic_state, "cam-B-downstream": agg_b.traffic_state},
                {"cam-A-upstream": rel_a, "cam-B-downstream": rel_b},
            )
            store.save(CORRIDOR_ID, assembly)

            a_ts, b_ts = agg_a.traffic_state, agg_b.traffic_state
            row = (
                f"{ws:>5.0f}-{we:<5.0f} | {a_ts.congestion_level.value:<12} | {b_ts.congestion_level.value:<12} | "
                f"{assembly.corridor_state.fused_congestion_level.value:<14} | "
                f"{a_ts.vehicle_count:>2}/{a_ts.mean_speed.value:<7.1f} | "
                f"{b_ts.vehicle_count:>2}/{b_ts.mean_speed.value:<7.1f} | "
                f"{rel_a.score:>6.2f} | {rel_b.score:>6.2f}"
            )
            print(row)

        print()
        print(f"Persisted {5} CorridorState records to {db_path}")
        print("Reload check (store.list_for_corridor):")
        for record in store.list_for_corridor(CORRIDOR_ID):
            cs = record.corridor_state
            print(
                f"  {cs.corridor_state_id}: window={cs.window}, "
                f"fused={cs.fused_congestion_level.value}, "
                f"cameras_with_data={record.cameras_with_data}, "
                f"active_ranking_id={cs.active_ranking_id!r} (placeholder -- no cause ranking until M4)"
            )
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
