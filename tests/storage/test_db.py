from pathlib import Path

from anvesh.evidence.reliability import ReliabilityScore
from anvesh.feedback.update_engine import apply_feedback
from anvesh.storage.db import CorridorStateStore, FeedbackRecordStore, RankingStore
from anvesh.storage.schemas import (
    CameraRole,
    CandidateCauseHypothesis,
    CandidateOutcome,
    ConfidenceTier,
    CongestionLevel,
    DataCompleteness,
    HypothesisUpdateOutcome,
    Measurement,
    MotionSpace,
    PropagationObservation,
    PropagationPrediction,
    RankedHypothesisEntry,
    TrafficState,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import assemble_corridor_state


def _traffic_state(camera_id, congestion_level=CongestionLevel.FREE_FLOW):
    return TrafficState(
        camera_id=camera_id,
        window_start=0.0,
        window_end=10.0,
        occupancy=0.3,
        mean_speed=Measurement(value=8.0, error=0.5),
        vehicle_count=3,
        flow_rate=0.3,
        density=3.0,
        congestion_level=congestion_level,
        motion_space=MotionSpace.WORLD,
    )


def _topology():
    return CorridorTopology(
        corridor_id="corridor-01",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=40.0),
        ),
    )


def _assembly(corridor_state_id="cs-1", window=(0.0, 10.0)):
    topology = _topology()
    traffic_states = {
        "cam-A": _traffic_state("cam-A", CongestionLevel.FREE_FLOW),
        "cam-B": _traffic_state("cam-B", CongestionLevel.BUILDING),
    }
    reliability = {
        "cam-A": ReliabilityScore(camera_id="cam-A", window_start=0.0, window_end=10.0, score=0.9, factors={"x": 0.9}),
        "cam-B": ReliabilityScore(camera_id="cam-B", window_start=0.0, window_end=10.0, score=0.7, factors={"x": 0.7}),
    }
    return assemble_corridor_state(corridor_state_id, topology, window, traffic_states, reliability)


def test_save_and_get_round_trip(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        original = _assembly()
        store.save("corridor-01", original)

        loaded = store.get("cs-1")
        assert loaded is not None
        assert loaded.corridor_state.corridor_state_id == "cs-1"
        assert loaded.corridor_state.fused_congestion_level == CongestionLevel.BUILDING
        assert len(loaded.corridor_state.segment_states) == 2
        assert loaded.cameras_with_data == ("cam-A", "cam-B")
        assert loaded.cameras_missing == ()
        assert loaded.reliability_by_camera["cam-A"].score == 0.9
        assert loaded.reliability_by_camera["cam-B"].factors == {"x": 0.7}
    finally:
        store.close()


def test_get_missing_id_returns_none(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        assert store.get("does-not-exist") is None
    finally:
        store.close()


def test_list_for_corridor_orders_by_window_start(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        store.save("corridor-01", _assembly("cs-2", window=(10.0, 20.0)))
        store.save("corridor-01", _assembly("cs-1", window=(0.0, 10.0)))  # inserted second, earlier window

        results = store.list_for_corridor("corridor-01")
        assert [r.corridor_state.corridor_state_id for r in results] == ["cs-1", "cs-2"]
    finally:
        store.close()


def test_save_upserts_on_same_id(tmp_path: Path):
    store = CorridorStateStore(tmp_path / "anvesh.sqlite3")
    try:
        store.save("corridor-01", _assembly("cs-1"))
        updated = _assembly("cs-1")
        store.save("corridor-01", updated)

        results = store.list_for_corridor("corridor-01")
        assert len(results) == 1  # upsert, not a duplicate row
    finally:
        store.close()


def test_store_persists_across_reconnect(tmp_path: Path):
    db_path = tmp_path / "anvesh.sqlite3"
    store1 = CorridorStateStore(db_path)
    store1.save("corridor-01", _assembly())
    store1.close()

    store2 = CorridorStateStore(db_path)
    try:
        loaded = store2.get("cs-1")
        assert loaded is not None
        assert loaded.corridor_state.corridor_state_id == "cs-1"
    finally:
        store2.close()


def test_context_manager_closes_connection(tmp_path: Path):
    with CorridorStateStore(tmp_path / "anvesh.sqlite3") as store:
        store.save("corridor-01", _assembly())
        assert store.get("cs-1") is not None


def _ranking(ranking_id="rank-1", corridor_state_id="cs-1", outcome=CandidateOutcome.RANKED):
    return CandidateCauseHypothesis(
        ranking_id=ranking_id,
        corridor_state_id=corridor_state_id,
        outcome=outcome,
        ranked_list=[
            RankedHypothesisEntry(
                hypothesis_id="H1",
                belief=0.6,
                plausibility=0.8,
                confidence_tier=ConfidenceTier.MEDIUM,
                supporting_evidence_refs=["cam-A:H1:0-10"],
                contradicting_evidence_refs=[],
            )
        ],
        engine_model_id="ds_fusion",
        engine_model_version="v1",
    )


def test_ranking_store_save_and_get_round_trip(tmp_path: Path):
    with RankingStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(_ranking())
        loaded = store.get("rank-1")

        assert loaded is not None
        assert loaded.ranking_id == "rank-1"
        assert loaded.outcome == CandidateOutcome.RANKED
        assert len(loaded.ranked_list) == 1
        assert loaded.ranked_list[0].hypothesis_id == "H1"
        assert loaded.ranked_list[0].confidence_tier == ConfidenceTier.MEDIUM
        assert loaded.ranked_list[0].supporting_evidence_refs == ["cam-A:H1:0-10"]


def test_ranking_store_get_missing_returns_none(tmp_path: Path):
    with RankingStore(tmp_path / "anvesh.sqlite3") as store:
        assert store.get("does-not-exist") is None


def test_ranking_store_list_for_corridor_state(tmp_path: Path):
    with RankingStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(_ranking("rank-1", "cs-1"))
        store.save(_ranking("rank-2", "cs-1"))
        store.save(_ranking("rank-3", "cs-OTHER"))

        results = store.list_for_corridor_state("cs-1")
        assert {r.ranking_id for r in results} == {"rank-1", "rank-2"}


def test_ranking_store_upserts_on_same_id(tmp_path: Path):
    with RankingStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(_ranking("rank-1", outcome=CandidateOutcome.RANKED))
        store.save(_ranking("rank-1", outcome=CandidateOutcome.HIGH_CONFLICT))

        results = store.list_for_corridor_state("cs-1")
        assert len(results) == 1
        assert results[0].outcome == CandidateOutcome.HIGH_CONFLICT


def test_ranking_store_persists_high_conflict_outcome(tmp_path: Path):
    with RankingStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(_ranking("rank-1", outcome=CandidateOutcome.HIGH_CONFLICT))
        loaded = store.get("rank-1")
        assert loaded.outcome == CandidateOutcome.HIGH_CONFLICT


# ---------------------------------------------------------------------------
# M5: FeedbackRecordStore
# ---------------------------------------------------------------------------


def _prediction():
    return PropagationPrediction(
        prediction_id="pred-1",
        based_on_ranking_id="rank-1",
        predicted_shockwave_speed=Measurement(value=2.0, error=0.0),
        predicted_arrival_camera="cam-B",
        predicted_arrival_time=100.0,
        predicted_queue_growth_rate=5.0,
        method="rankine_hugoniot_kinematic_wave_v1",
    )


def _observation(completeness=DataCompleteness.FULL, camera="cam-B"):
    return PropagationObservation(
        observation_id="obs-1",
        prediction_id="pred-1",
        actual_arrival_camera=camera,
        actual_arrival_time=100.0,
        actual_queue_growth_rate=5.0,
        data_completeness=completeness,
    )


def test_feedback_record_store_confirmed_round_trip(tmp_path: Path):
    prior = _ranking("rank-1")
    result = apply_feedback("upd-1", "rank-1-revised", prior, _prediction(), _observation())

    with FeedbackRecordStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(result)
        loaded = store.get("upd-1")

        assert loaded is not None
        assert loaded.update.outcome == HypothesisUpdateOutcome.CONFIRMED
        assert loaded.prior_ranking.ranking_id == "rank-1"
        assert loaded.revised_ranking.ranking_id == "rank-1-revised"
        assert loaded.prediction.predicted_arrival_camera == "cam-B"
        assert loaded.observation.actual_arrival_camera == "cam-B"
        assert loaded.reason == result.reason


def test_feedback_record_store_preserves_none_discrepancy_for_evidence_missing(tmp_path: Path):
    prior = _ranking("rank-1")
    result = apply_feedback("upd-2", "rank-1-revised", prior, _prediction(), _observation(completeness=DataCompleteness.MISSING))

    with FeedbackRecordStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(result)
        loaded = store.get("upd-2")

        assert loaded.update.outcome == HypothesisUpdateOutcome.EVIDENCE_MISSING
        assert loaded.update.discrepancy is None
        # the revised ranking must round-trip as identical in content to the prior one
        assert [e.belief for e in loaded.revised_ranking.ranked_list] == [e.belief for e in loaded.prior_ranking.ranked_list]


def test_feedback_record_store_get_missing_returns_none(tmp_path: Path):
    with FeedbackRecordStore(tmp_path / "anvesh.sqlite3") as store:
        assert store.get("does-not-exist") is None


def test_feedback_record_store_list_for_prior_ranking(tmp_path: Path):
    prior = _ranking("rank-1")
    r1 = apply_feedback("upd-a", "rank-a", prior, _prediction(), _observation())
    r2 = apply_feedback("upd-b", "rank-b", prior, _prediction(), _observation(camera="cam-A"))  # contradicted

    with FeedbackRecordStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(r1)
        store.save(r2)
        results = store.list_for_prior_ranking("rank-1")
        assert {r.update.update_id for r in results} == {"upd-a", "upd-b"}


def test_feedback_record_store_upserts_on_same_update_id(tmp_path: Path):
    prior = _ranking("rank-1")
    confirmed = apply_feedback("upd-1", "rank-r1", prior, _prediction(), _observation())
    contradicted = apply_feedback("upd-1", "rank-r2", prior, _prediction(), _observation(camera="cam-A"))

    with FeedbackRecordStore(tmp_path / "anvesh.sqlite3") as store:
        store.save(confirmed)
        store.save(contradicted)
        results = store.list_for_prior_ranking("rank-1")
        assert len(results) == 1
        assert results[0].update.outcome == HypothesisUpdateOutcome.CONTRADICTED
