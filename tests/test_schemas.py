import dataclasses

import pytest

from anvesh.storage import schemas
from anvesh.storage.schemas import (
    BaselineType,
    CameraRole,
    CandidateCauseHypothesis,
    CandidateOutcome,
    CauseHypothesis,
    Camera,
    CameraObservation,
    CongestionLevel,
    ConfidenceTier,
    CorridorState,
    DataCompleteness,
    Discrepancy,
    Evidence,
    EvidenceCompleteness,
    EvaluationMetrics,
    EvaluationRecord,
    HypothesisUpdate,
    HypothesisUpdateOutcome,
    Measurement,
    OcclusionState,
    PropagationObservation,
    PropagationPrediction,
    RankedHypothesisEntry,
    TrafficState,
    VehicleClass,
    VehicleTrack,
    WorldPosition,
)


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_module_imports():
    assert schemas is not None


# ---------------------------------------------------------------------------
# 2. Required fields exist
# ---------------------------------------------------------------------------


def test_camera_required_fields():
    field_names = {f.name for f in dataclasses.fields(Camera)}
    assert field_names == {
        "camera_id",
        "position_description",
        "role",
        "calibration_profile_id",
        "schema_version",
    }


def test_traffic_state_required_fields():
    field_names = {f.name for f in dataclasses.fields(TrafficState)}
    assert {
        "camera_id",
        "window_start",
        "window_end",
        "occupancy",
        "mean_speed",
        "vehicle_count",
        "flow_rate",
        "density",
        "congestion_level",
    } <= field_names


def test_every_top_level_entity_has_schema_version():
    top_level_entities = [
        Camera,
        CameraObservation,
        VehicleTrack,
        TrafficState,
        Evidence,
        CauseHypothesis,
        CandidateCauseHypothesis,
        PropagationPrediction,
        PropagationObservation,
        HypothesisUpdate,
        CorridorState,
        EvaluationRecord,
    ]
    for entity in top_level_entities:
        field_names = {f.name for f in dataclasses.fields(entity)}
        assert "schema_version" in field_names, entity


# ---------------------------------------------------------------------------
# 3. Valid instances can be created
# ---------------------------------------------------------------------------


def test_camera_valid_instance():
    camera = Camera(
        camera_id="cam-upstream",
        position_description="North entry, pole 3",
        role="upstream",
        calibration_profile_id="calib-v1",
    )
    assert camera.role is CameraRole.UPSTREAM
    assert camera.schema_version == "1.0.0"


def test_vehicle_track_valid_instance():
    track = VehicleTrack(
        track_id="cam-upstream:12",
        camera_id="cam-upstream",
        first_seen=10.0,
        last_seen=12.5,
        vehicle_class="car",
        occlusion_state="visible",
        position_history=[WorldPosition(world_x=1.0, world_y=2.0, timestamp=10.0)],
        speed_estimate=Measurement(value=8.3, error=0.4),
    )
    assert track.vehicle_class is VehicleClass.CAR
    assert track.occlusion_state is OcclusionState.VISIBLE


def test_traffic_state_valid_instance():
    state = TrafficState(
        camera_id="cam-downstream",
        window_start=0.0,
        window_end=30.0,
        occupancy=0.42,
        mean_speed=Measurement(value=6.1, error=0.5),
        vehicle_count=14,
        flow_rate=1.8,
        density=0.12,
        congestion_level="building",
    )
    assert state.congestion_level is CongestionLevel.BUILDING


def test_evidence_valid_instance():
    evidence = Evidence(
        evidence_id="ev-1",
        camera_id="cam-downstream",
        window=(0.0, 30.0),
        hypothesis_id="H1",
        signature_match_score=0.75,
        supporting_track_refs=["cam-downstream:12"],
        contradicting=False,
        evidence_completeness="full",
    )
    assert evidence.evidence_completeness is EvidenceCompleteness.FULL


def test_candidate_cause_hypothesis_valid_instance():
    ranking = CandidateCauseHypothesis(
        ranking_id="rank-1",
        corridor_state_id="cs-1",
        outcome="ranked",
        ranked_list=[
            RankedHypothesisEntry(
                hypothesis_id="H1",
                belief=0.6,
                plausibility=0.8,
                confidence_tier="medium",
                supporting_evidence_refs=["ev-1"],
                contradicting_evidence_refs=[],
            )
        ],
        engine_model_id="ds-fusion",
        engine_model_version="0.1.0",
    )
    assert ranking.outcome is CandidateOutcome.RANKED
    assert ranking.ranked_list[0].confidence_tier is ConfidenceTier.MEDIUM


def test_propagation_prediction_and_observation_valid_instances():
    prediction = PropagationPrediction(
        prediction_id="pred-1",
        based_on_ranking_id="rank-1",
        predicted_shockwave_speed=Measurement(value=-4.2, error=0.3),
        predicted_arrival_camera="cam-upstream",
        predicted_arrival_time=120.0,
        predicted_queue_growth_rate=2.5,
        method="kinematic_wave_v1",
    )
    observation = PropagationObservation(
        observation_id="obs-1",
        prediction_id=prediction.prediction_id,
        actual_arrival_camera="cam-upstream",
        actual_arrival_time=125.0,
        actual_queue_growth_rate=2.1,
        data_completeness="full",
    )
    assert observation.data_completeness is DataCompleteness.FULL


def test_hypothesis_update_valid_instance_with_and_without_discrepancy():
    with_discrepancy = HypothesisUpdate(
        update_id="upd-1",
        prior_ranking_id="rank-1",
        propagation_observation_id="obs-1",
        outcome="confirmed",
        revised_ranking_id="rank-2",
        discrepancy=Discrepancy(location_error=0.0, time_error=5.0, rate_error=0.4),
    )
    assert with_discrepancy.outcome is HypothesisUpdateOutcome.CONFIRMED

    missing_evidence = HypothesisUpdate(
        update_id="upd-2",
        prior_ranking_id="rank-1",
        propagation_observation_id="obs-2",
        outcome="evidence_missing",
        revised_ranking_id="rank-1",
        discrepancy=None,
    )
    assert missing_evidence.discrepancy is None


def test_corridor_state_valid_instance():
    segment = TrafficState(
        camera_id="cam-downstream",
        window_start=0.0,
        window_end=30.0,
        occupancy=0.42,
        mean_speed=Measurement(value=6.1, error=0.5),
        vehicle_count=14,
        flow_rate=1.8,
        density=0.12,
        congestion_level="building",
    )
    corridor = CorridorState(
        corridor_state_id="cs-1",
        window=(0.0, 30.0),
        segment_states=[segment],
        fused_congestion_level="building",
        active_ranking_id="rank-1",
    )
    assert corridor.fused_congestion_level is CongestionLevel.BUILDING


def test_evaluation_record_valid_instance():
    metrics = EvaluationMetrics(
        top1_accuracy=0.8,
        topk_accuracy=0.9,
        brier_score=0.1,
        ece=0.05,
        false_confidence_rate=0.02,
        propagation_location_error=0.0,
        propagation_time_error=3.0,
        hypothesis_revision_accuracy=0.7,
        congestion_p=0.9,
        congestion_r=0.85,
        congestion_f1=0.87,
    )
    record = EvaluationRecord(
        record_id="eval-1",
        experiment_run_id="run-1",
        baseline="ANVESH",
        scenario_id="sumo-h1-01",
        metrics=metrics,
        ground_truth_ref="sumo-h1-01:H1",
    )
    assert record.baseline is BaselineType.ANVESH


# ---------------------------------------------------------------------------
# 4. Invalid required data is rejected
# ---------------------------------------------------------------------------


def test_missing_required_field_raises_type_error():
    with pytest.raises(TypeError):
        Camera(camera_id="cam-1", position_description="desc", role="upstream")  # type: ignore[call-arg]


def test_invalid_enum_value_raises_value_error():
    with pytest.raises(ValueError):
        Camera(
            camera_id="cam-1",
            position_description="desc",
            role="sideways",
            calibration_profile_id="calib-v1",
        )


def test_empty_required_string_raises_value_error():
    with pytest.raises(ValueError):
        Camera(
            camera_id="",
            position_description="desc",
            role="upstream",
            calibration_profile_id="calib-v1",
        )


def test_occupancy_out_of_range_raises_value_error():
    with pytest.raises(ValueError):
        TrafficState(
            camera_id="cam-downstream",
            window_start=0.0,
            window_end=30.0,
            occupancy=1.5,
            mean_speed=Measurement(value=6.1, error=0.5),
            vehicle_count=14,
            flow_rate=1.8,
            density=0.12,
            congestion_level="building",
        )


def test_negative_vehicle_count_raises_value_error():
    with pytest.raises(ValueError):
        TrafficState(
            camera_id="cam-downstream",
            window_start=0.0,
            window_end=30.0,
            occupancy=0.5,
            mean_speed=Measurement(value=6.1, error=0.5),
            vehicle_count=-1,
            flow_rate=1.8,
            density=0.12,
            congestion_level="building",
        )


def test_window_end_before_start_raises_value_error():
    with pytest.raises(ValueError):
        Evidence(
            evidence_id="ev-1",
            camera_id="cam-downstream",
            window=(30.0, 0.0),
            hypothesis_id="H1",
            signature_match_score=0.5,
            supporting_track_refs=[],
            contradicting=False,
            evidence_completeness="full",
        )


def test_wrong_type_for_nested_value_object_raises_type_error():
    with pytest.raises(TypeError):
        TrafficState(
            camera_id="cam-downstream",
            window_start=0.0,
            window_end=30.0,
            occupancy=0.5,
            mean_speed=(6.1, 0.5),  # not a Measurement instance
            vehicle_count=14,
            flow_rate=1.8,
            density=0.12,
            congestion_level="building",
        )


def test_negative_measurement_error_raises_value_error():
    with pytest.raises(ValueError):
        Measurement(value=5.0, error=-0.1)
