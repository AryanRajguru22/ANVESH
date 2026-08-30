"""M8: tests for the anvesh/pipeline.py orchestrator.

Every fixture below is deterministic and synthetic -- no real video
dependency (per M8 instructions, a real-video smoke test would make this
suite slow/YOLO-dependent; scripts/run_end_to_end_demo.py covers that
separately). Traffic-state aggregation is produced by actually calling
`aggregate_traffic_state` (not hand-faked), so these tests exercise the
real M3 math the orchestrator composes, not a stand-in.
"""

from __future__ import annotations

import pytest

from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.storage.schemas import (
    CameraRole,
    CandidateOutcome,
    DataCompleteness,
    Measurement,
    MotionSpace,
    OcclusionState,
    PropagationObservation,
    VehicleClass,
    VehicleTrack,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology
from anvesh.world.corridor_state import UNRANKED_PLACEHOLDER

WINDOW = (0.0, 10.0)


def _tracks(camera_id, count, speed, motion_space=MotionSpace.WORLD, first_seen=0.0, last_seen=10.0):
    return [
        VehicleTrack(
            track_id=f"{camera_id}:{i}",
            camera_id=camera_id,
            first_seen=first_seen,
            last_seen=last_seen,
            vehicle_class=VehicleClass.CAR,
            occlusion_state=OcclusionState.VISIBLE,
            position_history=[],
            speed_estimate=Measurement(value=speed, error=0.0),
            motion_space=motion_space,
        )
        for i in range(count)
    ]


def _camera_input(camera_id, role, tracks, motion_space=MotionSpace.WORLD, window=WINDOW, max_capacity_vehicles=20):
    tracker = CongestionStateTracker()
    aggregation = aggregate_traffic_state(
        camera_id, window[0], window[1], tracks, motion_space, tracker, max_capacity_vehicles=max_capacity_vehicles
    )
    return CameraWindowInput(camera_id=camera_id, role=role, aggregation=aggregation, tracks=tracks)


def _upstream_input(camera_id="cam-A", motion_space=MotionSpace.WORLD):
    # 15 tracks / 20 capacity, 2.0 m/s -- congested (occupancy>=0.6 and speed<=3.0), not "stopped" (>1.0 m/s)
    return _camera_input(camera_id, CameraRole.UPSTREAM, _tracks(camera_id, 15, 2.0, motion_space), motion_space)


def _downstream_input(camera_id="cam-B", motion_space=MotionSpace.WORLD):
    # 3 tracks / 20 capacity, 8.0 m/s -- free-flowing, genuinely different density from upstream
    return _camera_input(camera_id, CameraRole.DOWNSTREAM, _tracks(camera_id, 3, 8.0, motion_space), motion_space)


def _topology(a_id="cam-A", b_id="cam-B"):
    return CorridorTopology(
        corridor_id="test-corridor",
        placements=(
            CameraPlacement(camera_id=a_id, role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id=b_id, role=CameraRole.DOWNSTREAM, distance_from_upstream_m=100.0),
        ),
    )


# ---------------------------------------------------------------------------
# 1-5. single-camera behavior
# ---------------------------------------------------------------------------


def test_single_camera_pipeline_produces_evidence():
    result = run_corridor_window(WINDOW, (_upstream_input(),))
    assert len(result.evidence_by_camera["cam-A"]) == 6  # H1-H6, always all six


def test_single_camera_pipeline_produces_ranking():
    result = run_corridor_window(WINDOW, (_upstream_input(),))
    assert result.candidate_ranking is not None
    assert result.candidate_ranking.ranked_list  # not empty


def test_single_camera_pipeline_has_no_corridor_state():
    result = run_corridor_window(WINDOW, (_upstream_input(),))
    assert result.corridor_state is None
    assert "single-camera" in result.corridor_state_reason


def test_single_camera_pipeline_has_no_propagation():
    result = run_corridor_window(WINDOW, (_upstream_input(),))
    assert result.propagation_prediction is None
    assert result.propagation_reason


def test_single_camera_pipeline_has_no_feedback():
    result = run_corridor_window(WINDOW, (_upstream_input(),))
    assert result.feedback_result is None
    assert result.feedback_reason


# ---------------------------------------------------------------------------
# 6-8. two-camera synthetic corridor behavior
# ---------------------------------------------------------------------------


def test_two_camera_synthetic_pipeline_produces_corridor_state():
    result = run_corridor_window(WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology())
    assert result.corridor_state is not None
    assert result.corridor_state_reason is None


def test_two_camera_synthetic_pipeline_produces_fused_ranking():
    result = run_corridor_window(WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology())
    assert result.candidate_ranking.outcome == CandidateOutcome.RANKED
    assert result.candidate_ranking.ranked_list[0].hypothesis_id.startswith("H")


def test_ranking_id_replaces_unranked_placeholder():
    result = run_corridor_window(WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology())
    assert result.corridor_state.active_ranking_id == result.candidate_ranking.ranking_id
    assert result.corridor_state.active_ranking_id != UNRANKED_PLACEHOLDER


# ---------------------------------------------------------------------------
# 9. WORLD-space valid input can produce propagation
# ---------------------------------------------------------------------------


def test_world_space_valid_input_produces_propagation():
    result = run_corridor_window(
        WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology(), segment_length_m=50.0,
    )
    assert result.propagation_prediction is not None
    assert result.propagation_reason is None


# ---------------------------------------------------------------------------
# 10-11. feedback only when an observation is explicitly supplied
# ---------------------------------------------------------------------------


def test_supplied_propagation_observation_produces_feedback():
    baseline = run_corridor_window(
        WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology(), segment_length_m=50.0,
    )
    prediction = baseline.propagation_prediction
    observation = PropagationObservation(
        observation_id="obs-test",
        prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )
    result = run_corridor_window(
        WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology(), segment_length_m=50.0,
        propagation_observation=observation,
    )
    assert result.feedback_result is not None
    assert result.feedback_reason is None


def test_no_observation_supplied_means_no_feedback_with_reason():
    result = run_corridor_window(
        WINDOW, (_upstream_input(), _downstream_input()), corridor_topology=_topology(), segment_length_m=50.0,
    )
    assert result.feedback_result is None
    assert "no propagation observation" in result.feedback_reason


# ---------------------------------------------------------------------------
# 12-13. propagation refusal cases
# ---------------------------------------------------------------------------


def test_image_space_input_refuses_propagation_cleanly():
    upstream = _upstream_input(motion_space=MotionSpace.IMAGE)
    downstream = _downstream_input(motion_space=MotionSpace.IMAGE)
    result = run_corridor_window(
        WINDOW, (upstream, downstream), corridor_topology=_topology(), segment_length_m=50.0,
    )
    assert result.propagation_prediction is None
    assert "not calibrated" in result.propagation_reason


def test_degenerate_propagation_is_surfaced_as_unavailable():
    # Equal vehicle_count on both cameras -> equal density -> degenerate shockwave input.
    a = _camera_input("cam-A", CameraRole.UPSTREAM, _tracks("cam-A", 5, 2.0))
    b = _camera_input("cam-B", CameraRole.DOWNSTREAM, _tracks("cam-B", 5, 2.0))
    result = run_corridor_window(WINDOW, (a, b), corridor_topology=_topology(), segment_length_m=50.0)
    assert result.propagation_prediction is None
    assert "degenerate" in result.propagation_reason


# ---------------------------------------------------------------------------
# 14-15. safety-event integration
# ---------------------------------------------------------------------------


def test_stopped_track_produces_safety_event_in_unified_output():
    stopped_track = VehicleTrack(
        track_id="cam-A:stopped",
        camera_id="cam-A",
        first_seen=0.0,
        last_seen=3.5,
        vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE,
        position_history=[(100.0, 100.0, i * 0.5) for i in range(8)],
        speed_estimate=Measurement(value=0.0, error=0.0),
        motion_space=MotionSpace.IMAGE,
    )
    tracks = _tracks("cam-A", 5, 20.0, motion_space=MotionSpace.IMAGE) + [stopped_track]
    camera_input = _camera_input("cam-A", CameraRole.UPSTREAM, tracks, motion_space=MotionSpace.IMAGE)
    result = run_corridor_window(WINDOW, (camera_input,))
    assert len(result.safety_events) == 1
    assert result.safety_events[0].track_id == "cam-A:stopped"
    assert result.safety_events[0].event_type == "stopped_vehicle"


def test_moving_tracks_produce_no_safety_events():
    result = run_corridor_window(WINDOW, (_upstream_input(),))  # 15 tracks at 2.0 m/s, none stopped long enough
    assert result.safety_events == ()


# ---------------------------------------------------------------------------
# 16. missing/weak evidence -> abstention, not a fabricated cause
# ---------------------------------------------------------------------------


def test_missing_evidence_produces_abstention_not_fabricated_cause():
    empty_input = _camera_input("cam-A", CameraRole.UPSTREAM, [])
    result = run_corridor_window(WINDOW, (empty_input,))
    assert result.candidate_ranking.outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# 17. deterministic repeated run
# ---------------------------------------------------------------------------


def test_deterministic_repeated_run():
    camera_inputs = (_upstream_input(), _downstream_input())
    topology = _topology()
    first = run_corridor_window(WINDOW, camera_inputs, corridor_topology=topology, segment_length_m=50.0)
    second = run_corridor_window(WINDOW, camera_inputs, corridor_topology=topology, segment_length_m=50.0)
    assert first == second


# ---------------------------------------------------------------------------
# 18. unexpected programming error is NOT swallowed
# ---------------------------------------------------------------------------


def test_unexpected_error_propagates_uncaught(monkeypatch):
    import anvesh.pipeline as pipeline_module

    def _boom(*args, **kwargs):
        raise RuntimeError("boom -- simulated programming error")

    monkeypatch.setattr(pipeline_module, "build_evidence_for_camera", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        run_corridor_window(WINDOW, (_upstream_input(),))


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_wrong_number_of_camera_inputs_rejected():
    with pytest.raises(ValueError):
        run_corridor_window(WINDOW, ())
    with pytest.raises(ValueError):
        run_corridor_window(WINDOW, (_upstream_input(), _downstream_input(), _upstream_input("cam-C")))


def test_duplicate_camera_id_rejected():
    with pytest.raises(ValueError):
        run_corridor_window(WINDOW, (_upstream_input("cam-A"), _downstream_input("cam-A")))
