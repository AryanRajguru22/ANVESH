"""M8 end-to-end orchestrator (integration only -- no new algorithms).

Pure composition: this module sequences calls to already-existing M1-M7
functions and bundles their outputs into one `AnveshWindowResult`. It
reimplements none of reliability scoring, evidence rules, Dempster-Shafer
fusion, ranking, congestion classification, shockwave propagation,
feedback adjustment, or stopped-vehicle detection -- every number in the
result comes from the module that already owns that computation.

Scope, one window at a time (mirrors `world.corridor_state.
assemble_corridor_state`'s own scope, not a multi-window streaming
engine): the caller has already run M1 (tracking) and M3 (windowing/
aggregation) for one window on one or two cameras and hands this module
the results; this module does reliability -> evidence -> fusion/ranking
-> corridor state -> propagation -> feedback -> safety events, in that
order, calling the SAME functions every prior milestone's own demo/script
already calls.

WHAT THIS MODULE NEVER DOES (M8 Phase 2 explicit constraints):
  - Never constructs a `CorridorTopology`, `CalibrationProfile`, or
    `segment_length_m` itself -- these are always caller-supplied, real or
    explicitly-synthetic, exactly like every prior milestone.
  - Never infers or fabricates a `PropagationObservation`. Feedback runs
    ONLY when the caller supplies one explicitly; there is no "observe
    propagation from later data" function here or anywhere else.
  - Never bypasses `propagation.shockwave.predict_propagation`'s own
    calibration/degeneracy checks -- it is called exactly as-is, and only
    the specific `ValueError`s it already documents are caught. Any other
    exception (a real bug) propagates uncaught.
  - Never treats a single-camera run, a missing corridor topology, or an
    uncalibrated camera as an error -- these are legitimate inputs whose
    unavailable stages are reported via an explicit `*_reason` string,
    never silently skipped without explanation and never worked around.

IDs: `ranking_id`/`corridor_state_id`/`prediction_id`/`update_id`/
`revised_ranking_id` are all deterministic strings derived from the
window and participating camera IDs (see the `_default_*` helpers below)
unless the caller overrides them -- never a random UUID, so that
identical inputs produce an identical, comparable result on repeat runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.evidence.evidence_builder import build_evidence_for_camera
from anvesh.evidence.reliability import compute_reliability
from anvesh.feedback.update_engine import apply_feedback
from anvesh.fusion.ds_fusion import fuse_two_cameras, single_camera_result
from anvesh.hypotheses.ranking import attach_ranking_to_corridor_state, build_ranking
from anvesh.propagation.shockwave import predict_propagation
from anvesh.safety.stopped_vehicle import build_safety_event, classify_stopped_vehicle
from anvesh.storage.schemas import CameraRole
from anvesh.world.corridor_state import assemble_corridor_state


@dataclass(frozen=True)
class CameraWindowInput:
    """One camera's already-computed M1/M3 output for one window.

    This module does not compute tracking, windowing, or aggregation --
    `aggregation` is a `perception.traffic_state.TrafficStateAggregation`
    the caller already produced (e.g. via `aggregate_traffic_state`), and
    `tracks` is the SAME overlapping-track list used to produce it (the
    same contract `evidence_builder.build_evidence_for_camera` and
    `evidence.reliability.compute_reliability` already require)."""

    camera_id: str
    role: CameraRole
    aggregation: object  # perception.traffic_state.TrafficStateAggregation
    tracks: list

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        object.__setattr__(self, "role", CameraRole(self.role))


@dataclass(frozen=True)
class AnveshWindowResult:
    """Plain orchestration-layer bundle -- NOT a `storage.schemas` entity
    (same treatment as `feedback.update_engine.FeedbackResult` and
    `world.corridor_state.CorridorStateAssembly`: it references existing
    frozen schema objects rather than flattening or copying their
    fields). Not persisted by this module."""

    window: tuple
    camera_ids: tuple
    evidence_by_camera: dict
    reliability_by_camera: dict
    candidate_ranking: object  # schemas.CandidateCauseHypothesis
    corridor_state: object  # schemas.CorridorState, or None
    corridor_state_reason: object  # str, or None if corridor_state is present
    propagation_prediction: object  # schemas.PropagationPrediction, or None
    propagation_reason: object  # str, or None if propagation_prediction is present
    feedback_result: object  # feedback.update_engine.FeedbackResult, or None
    feedback_reason: object  # str, or None if feedback_result is present
    safety_events: tuple  # tuple of schemas.SafetyEvent (possibly empty)
    limitations: tuple  # tuple of human-readable strings


def _default_ranking_id(camera_ids: tuple, window: tuple) -> str:
    return f"rank:{'-'.join(sorted(camera_ids))}:{window[0]}-{window[1]}"


def _default_corridor_state_id(corridor_label: str, window: tuple) -> str:
    return f"cs:{corridor_label}:{window[0]}-{window[1]}"


def _default_prediction_id(ranking_id: str) -> str:
    return f"pred:{ranking_id}"


def _default_update_id(ranking_id: str) -> str:
    return f"upd:{ranking_id}"


def _default_revised_ranking_id(ranking_id: str) -> str:
    return f"{ranking_id}:revised"


def run_corridor_window(
    window: tuple,
    camera_inputs: tuple,
    corridor_topology=None,
    segment_length_m: float = None,
    propagation_reference_timestamp: float = None,
    propagation_observation=None,
    ranking_id: str = None,
    corridor_state_id: str = None,
    prediction_id: str = None,
    update_id: str = None,
    revised_ranking_id: str = None,
    reliability_weights=None,
    saturating_observation_count: int = 3,
    evidence_thresholds=None,
    insufficient_evidence_theta_threshold: float = None,
    high_conflict_threshold: float = None,
    feedback_thresholds=None,
    feedback_adjustment_config=None,
    stopped_vehicle_thresholds=None,
    engine_model_id: str = "ds_fusion",
    engine_model_version: str = "v1",
) -> AnveshWindowResult:
    """Run one window through reliability -> evidence -> fusion/ranking ->
    corridor state -> propagation -> feedback -> safety events, composing
    existing M1-M7 functions only.

    `camera_inputs` must have exactly 1 (single-camera) or 2 (candidate
    two-camera corridor) `CameraWindowInput` entries. `corridor_topology`,
    `segment_length_m`, and `propagation_observation` are all optional and
    caller-supplied (real or explicitly-synthetic) -- this function never
    constructs any of them itself. See the module docstring for the exact
    rules governing when corridor state, propagation, and feedback are
    attempted versus reported unavailable.
    """
    camera_inputs = tuple(camera_inputs)
    if len(camera_inputs) not in (1, 2):
        raise ValueError(f"run_corridor_window requires 1 or 2 CameraWindowInput entries, got {len(camera_inputs)}")
    camera_ids = tuple(ci.camera_id for ci in camera_inputs)
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"duplicate camera_id in camera_inputs: {camera_ids}")

    limitations = []

    # --- per-camera reliability + evidence (evidence.reliability, evidence.evidence_builder) ---
    reliability_by_camera = {}
    evidence_by_camera = {}
    for ci in camera_inputs:
        reliability = compute_reliability(
            ci.camera_id, window[0], window[1], ci.tracks, ci.aggregation,
            weights=reliability_weights, saturating_observation_count=saturating_observation_count,
        )
        reliability_by_camera[ci.camera_id] = reliability
        evidence_by_camera[ci.camera_id] = build_evidence_for_camera(
            ci.camera_id, window, ci.role, ci.aggregation, ci.tracks, reliability.score,
            thresholds=evidence_thresholds,
        )

    corridor_label = corridor_topology.corridor_id if corridor_topology is not None else f"single-{camera_ids[0]}"
    ranking_id = ranking_id or _default_ranking_id(camera_ids, window)
    corridor_state_id = corridor_state_id or _default_corridor_state_id(corridor_label, window)

    # --- fusion (single-camera or two-camera) + ranking (fusion.ds_fusion, hypotheses.ranking) ---
    if len(camera_inputs) == 1:
        ci = camera_inputs[0]
        fusion_result = single_camera_result(ci.camera_id, evidence_by_camera[ci.camera_id], window)
    else:
        a, b = camera_inputs
        fusion_result = fuse_two_cameras(
            a.camera_id, evidence_by_camera[a.camera_id], reliability_by_camera[a.camera_id].score,
            b.camera_id, evidence_by_camera[b.camera_id], reliability_by_camera[b.camera_id].score,
            window,
        )

    ranking_kwargs = {}
    if insufficient_evidence_theta_threshold is not None:
        ranking_kwargs["insufficient_evidence_theta_threshold"] = insufficient_evidence_theta_threshold
    if high_conflict_threshold is not None:
        ranking_kwargs["high_conflict_threshold"] = high_conflict_threshold
    candidate_ranking = build_ranking(
        ranking_id, corridor_state_id, fusion_result, evidence_by_camera,
        engine_model_id=engine_model_id, engine_model_version=engine_model_version, **ranking_kwargs,
    )
    if candidate_ranking.outcome.value != "ranked":
        limitations.append(f"candidate ranking outcome is '{candidate_ranking.outcome.value}', not a ranked cause")

    # --- corridor state (world.corridor_state) -- only when a real topology was supplied ---
    corridor_state = None
    corridor_state_reason = None
    if corridor_topology is None:
        corridor_state_reason = (
            "single-camera input -- no corridor to assemble"
            if len(camera_inputs) == 1
            else "two-camera evidence fusion only -- no corridor topology supplied"
        )
    else:
        traffic_states_by_camera = {ci.camera_id: ci.aggregation.traffic_state for ci in camera_inputs}
        assembly = assemble_corridor_state(
            corridor_state_id, corridor_topology, window, traffic_states_by_camera, reliability_by_camera,
        )
        corridor_state = attach_ranking_to_corridor_state(assembly.corridor_state, candidate_ranking)
        if assembly.cameras_missing:
            limitations.append(f"corridor cameras with no data this window: {assembly.cameras_missing}")

    # --- propagation (propagation.shockwave) -- only when its own requirements can be met ---
    propagation_prediction = None
    propagation_reason = None
    if corridor_topology is None:
        propagation_reason = "requires a calibrated two-camera corridor topology"
    elif len(camera_inputs) != 2:
        propagation_reason = "requires exactly two camera inputs (upstream and downstream)"
    elif segment_length_m is None:
        propagation_reason = "no segment_length_m supplied -- no real corridor geometry available"
    else:
        upstream_ci = next((ci for ci in camera_inputs if ci.role == CameraRole.UPSTREAM), None)
        downstream_ci = next((ci for ci in camera_inputs if ci.role == CameraRole.DOWNSTREAM), None)
        if upstream_ci is None or downstream_ci is None:
            propagation_reason = "requires one upstream and one downstream camera input"
        else:
            effective_prediction_id = prediction_id or _default_prediction_id(ranking_id)
            reference_timestamp = (
                propagation_reference_timestamp if propagation_reference_timestamp is not None else window[1]
            )
            try:
                propagation_prediction = predict_propagation(
                    effective_prediction_id, ranking_id, corridor_topology,
                    upstream_ci.aggregation.traffic_state, downstream_ci.aggregation.traffic_state,
                    segment_length_m, reference_timestamp,
                )
            except ValueError as exc:
                # The documented, expected "cannot proceed" signal from
                # flow_density_from_traffic_state/predict_shockwave_speed
                # (uncalibrated camera, degenerate density, negligible
                # speed) -- never a broad `except Exception`, so a real
                # bug elsewhere still raises.
                propagation_reason = str(exc)

    # --- feedback (feedback.update_engine) -- only when BOTH a prediction and an explicit observation exist ---
    feedback_result = None
    feedback_reason = None
    if propagation_prediction is None:
        feedback_reason = "no propagation prediction available"
    elif propagation_observation is None:
        feedback_reason = "no propagation observation supplied"
    else:
        effective_update_id = update_id or _default_update_id(ranking_id)
        effective_revised_ranking_id = revised_ranking_id or _default_revised_ranking_id(ranking_id)
        feedback_result = apply_feedback(
            effective_update_id, effective_revised_ranking_id, candidate_ranking, propagation_prediction,
            propagation_observation, thresholds=feedback_thresholds,
            adjustment_config=feedback_adjustment_config, corridor_topology=corridor_topology,
        )

    # --- safety events (safety.stopped_vehicle) -- independent of every stage above ---
    safety_events = []
    for ci in camera_inputs:
        for track in ci.tracks:
            classification = classify_stopped_vehicle(track, stopped_vehicle_thresholds)
            event = build_safety_event(classification)
            if event is not None:
                safety_events.append(event)

    return AnveshWindowResult(
        window=window,
        camera_ids=camera_ids,
        evidence_by_camera=evidence_by_camera,
        reliability_by_camera=reliability_by_camera,
        candidate_ranking=candidate_ranking,
        corridor_state=corridor_state,
        corridor_state_reason=corridor_state_reason,
        propagation_prediction=propagation_prediction,
        propagation_reason=propagation_reason,
        feedback_result=feedback_result,
        feedback_reason=feedback_reason,
        safety_events=tuple(safety_events),
        limitations=tuple(limitations),
    )
