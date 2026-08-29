"""M3 -> M4 integration: two cameras' VehicleTracks -> TrafficState (M3) ->
CorridorState assembly with UNRANKED_PLACEHOLDER (M3) -> Evidence -> fused
belief -> ranking (M4) -> UNRANKED_PLACEHOLDER replaced with the real
ranking_id -> persisted -> reloaded.
"""

from pathlib import Path

from anvesh.evidence.evidence_builder import build_evidence_for_camera
from anvesh.evidence.reliability import compute_reliability
from anvesh.fusion.ds_fusion import fuse_two_cameras
from anvesh.hypotheses.ranking import attach_ranking_to_corridor_state, build_ranking
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.db import CorridorStateStore, RankingStore
from anvesh.storage.schemas import (
    CameraRole,
    CandidateOutcome,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import UNRANKED_PLACEHOLDER, assemble_corridor_state

WINDOW = (0.0, 10.0)


def _track(track_id, camera_id, first_seen, last_seen, speed_value):
    return VehicleTrack(
        track_id=track_id,
        camera_id=camera_id,
        first_seen=first_seen,
        last_seen=last_seen,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[],
        speed_estimate=Measurement(value=speed_value, error=0.0),
        motion_space=MotionSpace.WORLD,
    )


def _topology():
    return CorridorTopology(
        corridor_id="corridor-m4",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
        ),
    )


def test_m3_to_m4_pipeline_replaces_placeholder_and_round_trips(tmp_path: Path):
    topology = _topology()

    # Camera A (upstream): a stopped vehicle -> strong H2 evidence
    a_tracks = [_track("a-stopped", "cam-A", 0.0, 10.0, speed_value=0.3)]
    # Camera B (downstream): light, free-flowing traffic -> weak/neutral evidence
    # everywhere (near-pure ignorance), compatible with (not contradicting) H2
    b_tracks = [_track(f"b{i}", "cam-B", 0.0, 10.0, speed_value=10.0) for i in range(2)]

    tracker_a, tracker_b = CongestionStateTracker(), CongestionStateTracker()
    agg_a = aggregate_traffic_state("cam-A", *WINDOW, a_tracks, MotionSpace.WORLD, tracker_a)
    agg_b = aggregate_traffic_state("cam-B", *WINDOW, b_tracks, MotionSpace.WORLD, tracker_b)

    rel_a = compute_reliability("cam-A", *WINDOW, a_tracks, agg_a)
    rel_b = compute_reliability("cam-B", *WINDOW, b_tracks, agg_b)

    # --- M3: corridor state assembly, still carrying the M3 placeholder ---
    assembly = assemble_corridor_state(
        "cs-m4-1",
        topology,
        WINDOW,
        {"cam-A": agg_a.traffic_state, "cam-B": agg_b.traffic_state},
        {"cam-A": rel_a, "cam-B": rel_b},
    )
    assert assembly.corridor_state.active_ranking_id == UNRANKED_PLACEHOLDER

    # --- M4: evidence -> fusion -> ranking ---
    evidence_a = build_evidence_for_camera("cam-A", WINDOW, CameraRole.UPSTREAM, agg_a, a_tracks, rel_a.score)
    evidence_b = build_evidence_for_camera("cam-B", WINDOW, CameraRole.DOWNSTREAM, agg_b, b_tracks, rel_b.score)

    fusion_result = fuse_two_cameras("cam-A", evidence_a, rel_a.score, "cam-B", evidence_b, rel_b.score, WINDOW)
    ranking = build_ranking(
        "rank-cs-m4-1", assembly.corridor_state.corridor_state_id, fusion_result,
        {"cam-A": evidence_a, "cam-B": evidence_b},
    )

    assert ranking.outcome in (CandidateOutcome.RANKED, CandidateOutcome.INSUFFICIENT_EVIDENCE, CandidateOutcome.HIGH_CONFLICT)
    top = ranking.ranked_list[0]
    assert top.hypothesis_id == "H2"  # the single stopped vehicle is the strongest signal here

    # --- placeholder replacement (the exact M4 requirement) ---
    updated_corridor_state = attach_ranking_to_corridor_state(assembly.corridor_state, ranking)
    assert updated_corridor_state.active_ranking_id == ranking.ranking_id
    assert updated_corridor_state.active_ranking_id != UNRANKED_PLACEHOLDER

    # --- persistence round-trip of both the corridor state and the ranking ---
    corridor_store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    ranking_store = RankingStore(tmp_path / "anvesh.sqlite3")
    try:
        from anvesh.world.corridor_state import CorridorStateAssembly

        updated_assembly = CorridorStateAssembly(
            corridor_state=updated_corridor_state,
            reliability_by_camera=assembly.reliability_by_camera,
            cameras_with_data=assembly.cameras_with_data,
            cameras_missing=assembly.cameras_missing,
        )
        corridor_store.save("corridor-m4", updated_assembly)
        ranking_store.save(ranking)

        reloaded_corridor = corridor_store.get("cs-m4-1")
        reloaded_ranking = ranking_store.get(ranking.ranking_id)

        assert reloaded_corridor.corridor_state.active_ranking_id == ranking.ranking_id
        assert reloaded_ranking.ranked_list[0].hypothesis_id == "H2"
        assert reloaded_ranking.outcome == ranking.outcome
    finally:
        corridor_store.close()
        ranking_store.close()


def test_m3_traffic_state_aggregation_behavior_is_unchanged_by_m4():
    """M3 behavior must remain intact (design rule #15): plain aggregation
    with no evidence/fusion involved still works exactly as in M3."""
    tracks = [_track("1", "cam-A", 0.0, 10.0, speed_value=10.0)]
    result = aggregate_traffic_state("cam-A", 0.0, 10.0, tracks, MotionSpace.WORLD, CongestionStateTracker())
    assert result.traffic_state.vehicle_count == 1
    assert result.traffic_state.mean_speed.value == 10.0
