"""M4 -> M5 integration: two cameras' Evidence -> reliability-aware fusion
(M4) -> ranking -> propagation prediction (M5) -> observed propagation ->
feedback -> revised ranking -> persisted across all three SQLite stores
-> reloaded.
"""

from pathlib import Path

from anvesh.evaluation.baselines import compare_anvesh_vs_baseline_c
from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.feedback.update_engine import apply_feedback
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.propagation.shockwave import predict_propagation
from anvesh.storage.db import CorridorStateStore, FeedbackRecordStore, RankingStore
from anvesh.storage.schemas import (
    CameraRole,
    CongestionLevel,
    DataCompleteness,
    Evidence,
    EvidenceCompleteness,
    HypothesisUpdateOutcome,
    Measurement,
    MotionSpace,
    PropagationObservation,
    TrafficState,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import assemble_corridor_state

WINDOW = (0.0, 10.0)
SEGMENT_LENGTH_M = 100.0


def _evidence_item(camera_id, hid, score, direction):
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hid}:0-10",
        camera_id=camera_id,
        window=WINDOW,
        hypothesis_id=hid,
        signature_match_score=score,
        supporting_track_refs=[],
        contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(evidence=evidence, evidence_type="test", measured_value=None, direction=direction, reliability=1.0, source="test", metadata={})


def _neutral_except(camera_id, hid, score):
    return tuple(
        _evidence_item(camera_id, h, score if h == hid else 0.0, EvidenceDirection.SUPPORTS if h == hid else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )


def _topology():
    return CorridorTopology(
        corridor_id="corridor-m5",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=SEGMENT_LENGTH_M),
        ),
    )


def _traffic_state(camera_id, vehicle_count, flow_rate):
    return TrafficState(
        camera_id=camera_id, window_start=0.0, window_end=10.0, occupancy=min(1.0, vehicle_count / 20.0),
        mean_speed=Measurement(value=5.0, error=0.0), vehicle_count=vehicle_count, flow_rate=flow_rate,
        density=float(vehicle_count), congestion_level=CongestionLevel.BUILDING, motion_space=MotionSpace.WORLD,
    )


def test_full_m4_to_m5_pipeline_with_persistence(tmp_path: Path):
    topology = _topology()

    # --- M4: evidence -> fusion -> ranking ---
    evidence_by_camera = {
        "cam-A": _neutral_except("cam-A", "H1", 0.8),
        "cam-B": _neutral_except("cam-B", "H1", 0.7),
    }
    reliability_by_camera = {"cam-A": 0.9, "cam-B": 0.9}

    # --- M5: propagation prediction from the corridor's traffic states ---
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)
    prediction = predict_propagation(
        "pred-m5-1", "rank-baseline-c", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, reference_timestamp=10.0
    )
    assert prediction.predicted_arrival_camera == "cam-B"

    # --- Observation: consistent with the prediction ---
    observation = PropagationObservation(
        observation_id="obs-m5-1",
        prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )

    # --- ANVESH vs Baseline C, isolating feedback's value ---
    comparison = compare_anvesh_vs_baseline_c(
        "rank-baseline-c", "cs-m5-1", WINDOW, evidence_by_camera, reliability_by_camera,
        prediction, observation, "upd-m5-1", "rank-anvesh-1", corridor_topology=topology,
    )
    assert comparison.anvesh_feedback_result.update.outcome == HypothesisUpdateOutcome.CONFIRMED
    assert comparison.anvesh_ranking.ranked_list[0].belief > comparison.baseline_c_ranking.ranked_list[0].belief

    # --- Persist everything across the three stores ---
    corridor_store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    ranking_store = RankingStore(tmp_path / "anvesh.sqlite3")
    feedback_store = FeedbackRecordStore(tmp_path / "anvesh.sqlite3")
    try:
        assembly = assemble_corridor_state(
            "cs-m5-1", topology, WINDOW, {"cam-A": upstream_ts, "cam-B": downstream_ts}, {}
        )
        corridor_store.save("corridor-m5", assembly)
        ranking_store.save(comparison.baseline_c_ranking)
        ranking_store.save(comparison.anvesh_ranking)
        feedback_store.save(comparison.anvesh_feedback_result)

        reloaded_corridor = corridor_store.get("cs-m5-1")
        reloaded_baseline_c = ranking_store.get("rank-baseline-c")
        reloaded_anvesh = ranking_store.get("rank-anvesh-1")
        reloaded_feedback = feedback_store.get("upd-m5-1")

        assert reloaded_corridor.corridor_state.corridor_state_id == "cs-m5-1"
        assert reloaded_baseline_c.ranked_list[0].belief == comparison.baseline_c_ranking.ranked_list[0].belief
        assert reloaded_anvesh.ranked_list[0].belief == comparison.anvesh_ranking.ranked_list[0].belief
        assert reloaded_feedback.update.outcome == HypothesisUpdateOutcome.CONFIRMED
        assert reloaded_feedback.prediction.prediction_id == "pred-m5-1"
        assert reloaded_feedback.observation.observation_id == "obs-m5-1"
    finally:
        corridor_store.close()
        ranking_store.close()
        feedback_store.close()


def test_m4_ranking_and_fusion_behavior_is_unchanged_by_m5():
    """M4 behavior must remain intact: Baseline A/B/C/ANVESH still all
    execute independently and produce comparable rankings without any
    M5 involvement."""
    from anvesh.evaluation.baselines import run_baseline
    from anvesh.storage.schemas import BaselineType

    evidence_by_camera = {
        "cam-A": _neutral_except("cam-A", "H1", 0.8),
        "cam-B": _neutral_except("cam-B", "H1", 0.7),
    }
    a = run_baseline(BaselineType.A, "r-a", "cs-1", WINDOW, evidence_by_camera, single_camera_id="cam-A")
    b = run_baseline(BaselineType.B, "r-b", "cs-1", WINDOW, evidence_by_camera)
    c = run_baseline(BaselineType.C, "r-c", "cs-1", WINDOW, evidence_by_camera, reliability_by_camera={"cam-A": 0.9, "cam-B": 0.9})
    assert a.ranked_list[0].hypothesis_id == "H1"
    assert b.ranked_list[0].hypothesis_id == "H1"
    assert c.ranked_list[0].hypothesis_id == "H1"


def test_feedback_evidence_missing_leaves_history_and_ranking_fully_intact():
    """If evidence goes missing at the M5 stage, ANVESH's ranking must
    equal Baseline C's exactly -- feedback must not force a change."""
    topology = _topology()
    evidence_by_camera = {
        "cam-A": _neutral_except("cam-A", "H1", 0.8),
        "cam-B": _neutral_except("cam-B", "H1", 0.7),
    }
    reliability_by_camera = {"cam-A": 0.9, "cam-B": 0.9}
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)
    prediction = predict_propagation("pred-2", "rank-c2", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, 0.0)
    missing_observation = PropagationObservation(
        observation_id="obs-2", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.MISSING,
    )

    comparison = compare_anvesh_vs_baseline_c(
        "rank-c2", "cs-2", WINDOW, evidence_by_camera, reliability_by_camera,
        prediction, missing_observation, "upd-2", "rank-anvesh-2",
    )

    assert comparison.anvesh_feedback_result.update.outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING
    assert comparison.anvesh_ranking is comparison.baseline_c_ranking
