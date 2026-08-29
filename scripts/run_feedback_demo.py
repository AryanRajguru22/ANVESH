"""M5 demo: propagation prediction -> observed downstream change ->
feedback -> revised candidate-cause ranking, four deterministic scenarios.

Usage (from the repository root, with the venv active):

    python scripts/run_feedback_demo.py

Uses hand-constructed, clearly-synthetic evidence and traffic states (not
real video) so each scenario's expected behavior is exact and auditable.
This demonstrates the FEEDBACK LAYER specifically (prediction -> observed
propagation -> revision) -- see scripts/run_fusion_demo.py for the M4
fusion-only demo this one builds on.

Terminology note (per M5 scope): this is candidate-cause reasoning under
uncertainty, revised by propagation evidence -- not causal inference, not
accident detection, not a claim of universal propagation accuracy or
calibrated probability.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from anvesh.evaluation.baselines import compare_anvesh_vs_baseline_c
from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceItem
from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
from anvesh.propagation.shockwave import predict_propagation
from anvesh.storage.schemas import (
    CameraRole,
    CongestionLevel,
    DataCompleteness,
    Evidence,
    EvidenceCompleteness,
    Measurement,
    MotionSpace,
    PropagationObservation,
    TrafficState,
)
from anvesh.world.corridor import CameraPlacement, CorridorTopology

WINDOW = (0.0, 10.0)
SEGMENT_LENGTH_M = 100.0


def _topology():
    return CorridorTopology(
        corridor_id="corridor-demo",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=SEGMENT_LENGTH_M),
        ),
    )


def _item(camera_id, hid, score, direction):
    evidence = Evidence(
        evidence_id=f"{camera_id}:{hid}:{WINDOW[0]}-{WINDOW[1]}",
        camera_id=camera_id, window=WINDOW, hypothesis_id=hid, signature_match_score=score,
        supporting_track_refs=[], contradicting=(direction == EvidenceDirection.CONTRADICTS),
        evidence_completeness=EvidenceCompleteness.FULL,
    )
    return EvidenceItem(evidence=evidence, evidence_type="demo", measured_value=None, direction=direction, reliability=1.0, source="demo_script", metadata={})


def _evidence_by_camera():
    """Initial ranking: H1 > H2 -- cam-A strongly supports H1, cam-B
    somewhat supports H1 and mildly supports H2, so H2 is a visible
    second-place candidate for the CONTRADICTED scenario to promote."""
    ev_a = tuple(
        _item("cam-A", h, 0.7 if h == "H1" else 0.0, EvidenceDirection.SUPPORTS if h == "H1" else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )
    ev_b = tuple(
        _item("cam-B", h, {"H1": 0.4, "H2": 0.5}.get(h, 0.0), EvidenceDirection.SUPPORTS if h in ("H1", "H2") else EvidenceDirection.NEUTRAL)
        for h in HYPOTHESIS_IDS
    )
    return {"cam-A": ev_a, "cam-B": ev_b}


def _traffic_states():
    upstream = TrafficState(
        camera_id="cam-A", window_start=0.0, window_end=10.0, occupancy=1.0,
        mean_speed=Measurement(value=1.0, error=0.0), vehicle_count=20, flow_rate=2.0,
        density=20.0, congestion_level=CongestionLevel.CONGESTED, motion_space=MotionSpace.WORLD,
    )
    downstream = TrafficState(
        camera_id="cam-B", window_start=0.0, window_end=10.0, occupancy=0.25,
        mean_speed=Measurement(value=9.0, error=0.0), vehicle_count=5, flow_rate=1.0,
        density=5.0, congestion_level=CongestionLevel.FREE_FLOW, motion_space=MotionSpace.WORLD,
    )
    return upstream, downstream


def _print_ranking(label, ranking):
    print(f"  {label} (outcome={ranking.outcome.value}):")
    for entry in ranking.ranked_list[:3]:
        print(f"    {entry.hypothesis_id}: belief={entry.belief:.3f} plausibility={entry.plausibility:.3f} tier={entry.confidence_tier.value}")


def _print_prediction(prediction):
    print(
        f"  PREDICTED propagation: shockwave_speed={prediction.predicted_shockwave_speed.value:.2f} m/s, "
        f"arrival_camera={prediction.predicted_arrival_camera}, arrival_time={prediction.predicted_arrival_time:.1f}s, "
        f"queue_growth_rate={prediction.predicted_queue_growth_rate:.2f} veh/min  [method={prediction.method}]"
    )


def _print_observation(observation):
    print(
        f"  OBSERVED propagation (data_completeness={observation.data_completeness.value}): "
        f"arrival_camera={observation.actual_arrival_camera}, arrival_time={observation.actual_arrival_time:.1f}s, "
        f"queue_growth_rate={observation.actual_queue_growth_rate:.2f} veh/min"
    )


def _run_scenario(title, observation, prediction, evidence_by_camera, topology, ranking_id, revised_ranking_id, update_id):
    print(f"\n=== {title} ===")
    comparison = compare_anvesh_vs_baseline_c(
        ranking_id, "cs-demo", WINDOW, evidence_by_camera, {"cam-A": 0.9, "cam-B": 0.9},
        prediction, observation, update_id, revised_ranking_id, corridor_topology=topology,
    )
    feedback = comparison.anvesh_feedback_result

    _print_ranking("BEFORE feedback (= Baseline C)", comparison.baseline_c_ranking)
    _print_prediction(prediction)
    _print_observation(observation)
    print(f"  FEEDBACK outcome: {feedback.update.outcome.value}")
    print(f"  REVISION reason: {feedback.reason}")
    if feedback.update.discrepancy is not None:
        d = feedback.update.discrepancy
        print(f"  discrepancy: location_error={d.location_error:.1f} time_error={d.time_error:.1f}s rate_error={d.rate_error:.2f} veh/min")
    else:
        print("  discrepancy: None (no comparison was possible)")
    _print_ranking("AFTER feedback (= ANVESH)", comparison.anvesh_ranking)
    print(
        f"  PROVENANCE: prior_ranking_id={feedback.update.prior_ranking_id} -> "
        f"revised_ranking_id={feedback.update.revised_ranking_id}  "
        f"(prediction_id={prediction.prediction_id}, observation_id={observation.observation_id}, update_id={feedback.update.update_id})"
    )
    return comparison


def main() -> int:
    print("ANVESH M5 demo -- propagation prediction + feedback revision (SYNTHETIC evidence/traffic states)")
    topology = _topology()
    evidence_by_camera = _evidence_by_camera()
    upstream_ts, downstream_ts = _traffic_states()

    prediction = predict_propagation(
        "pred-demo", "rank-baseline-c-demo", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, reference_timestamp=10.0
    )

    # Scenario 1: CONFIRMED -- observation matches the prediction closely
    confirmed_observation = PropagationObservation(
        observation_id="obs-confirmed", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )
    _run_scenario(
        "Scenario 1: CONFIRMED -- consistent downstream propagation observed",
        confirmed_observation, prediction, evidence_by_camera, topology,
        "rank-baseline-c-1", "rank-anvesh-1", "upd-1",
    )

    # Scenario 2: PARTIAL -- right camera, but only partial/weak evidence
    partial_observation = PropagationObservation(
        observation_id="obs-partial", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,
        actual_arrival_time=prediction.predicted_arrival_time + 5.0,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate * 2.5,  # weak/divergent rate match
        data_completeness=DataCompleteness.PARTIAL,
    )
    _run_scenario(
        "Scenario 2: PARTIAL -- weak/incomplete downstream evidence observed",
        partial_observation, prediction, evidence_by_camera, topology,
        "rank-baseline-c-2", "rank-anvesh-2", "upd-2",
    )

    # Scenario 3: CONTRADICTED -- reliable observation, wrong camera entirely
    contradicted_observation = PropagationObservation(
        observation_id="obs-contradicted", prediction_id=prediction.prediction_id,
        actual_arrival_camera="cam-A",  # predicted cam-B; effect showed up upstream instead
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.FULL,
    )
    contradicted_comparison = _run_scenario(
        "Scenario 3: CONTRADICTED -- reliable observation materially disagrees",
        contradicted_observation, prediction, evidence_by_camera, topology,
        "rank-baseline-c-3", "rank-anvesh-3", "upd-3",
    )
    print(
        f"  CHECK: top candidate changed from "
        f"'{contradicted_comparison.baseline_c_ranking.ranked_list[0].hypothesis_id}' to "
        f"'{contradicted_comparison.anvesh_ranking.ranked_list[0].hypothesis_id}'"
    )

    # Scenario 4: EVIDENCE_MISSING -- prediction exists, no later observation available
    missing_observation = PropagationObservation(
        observation_id="obs-missing", prediction_id=prediction.prediction_id,
        actual_arrival_camera=prediction.predicted_arrival_camera,  # documented placeholder -- never read when MISSING
        actual_arrival_time=prediction.predicted_arrival_time,
        actual_queue_growth_rate=prediction.predicted_queue_growth_rate,
        data_completeness=DataCompleteness.MISSING,
    )
    missing_comparison = _run_scenario(
        "Scenario 4: EVIDENCE_MISSING -- no observation available",
        missing_observation, prediction, evidence_by_camera, topology,
        "rank-baseline-c-4", "rank-anvesh-4", "upd-4",
    )
    print(
        f"  CHECK: ranking is the SAME object before/after feedback: "
        f"{missing_comparison.anvesh_ranking is missing_comparison.baseline_c_ranking}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
