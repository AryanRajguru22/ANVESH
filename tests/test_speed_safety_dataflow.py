"""M-A speed-safety invariant, tested as DATA FLOW (not imports):

    UNVALIDATED calibration
        -> cannot produce a metric speed
        -> cannot reach m/s consumers
        -> cannot support metric-speed evidence

The pre-existing `tests/perception/test_speed_evidence_safety.py` AST-checks
that five reasoning modules do not import `speed_estimation`. It passed while
71-83% of c006 tracks carried >200 km/h "m/s" values into evidence, because
that data arrived through `VehicleTrack.speed_estimate`, not through an
import (ANVESH_PRODUCT_GAP_AUDIT.md finding 2 / SS18 gap 2).

This test drives the REAL chain end to end --
    frames -> StreamingPerceptionEngine (tracker, identity, store, windows)
           -> WindowReasoner (aggregate_traffic_state -> run_corridor_window:
              reliability -> evidence -> fusion -> ranking -> candidate vehicle
              -> safety events)
-- with calibration profiles whose homography scale would, under the old
`is_calibrated == metric` logic, turn pixel jitter into either absurdly FAST
("m/s" in the hundreds) or deceptively STOPPED (~0 "m/s") vehicles, and
checks every hop. A positive control proves the same chain DOES deliver
metric evidence for a validated profile, and a mutation test proves the
assertions go red if the old coupling is reintroduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from anvesh.hypotheses.candidate_vehicle import CandidateVehicleStatus
from anvesh.perception.speed_estimation import SpeedConfidence, estimate_vehicle_speed
from anvesh.perception.schema_conversion import motion_space_for
from anvesh.rendering import format_speed_text
from anvesh.storage.schemas import EvidenceCompleteness, GeometryValidity, MotionSpace
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.reasoning import WindowReasoner
from anvesh.streaming.windows import WindowSpec
from anvesh.world.calibration import CalibrationProfile, from_cityflow_homography
from anvesh.world.synthetic_fixture import fit_synthetic_validated_homography
from tests.streaming.support import ScriptedTracker, make_frames
from tests.world.support import make_reference_points

CAMERA = "cam-flow"
FPS = 10.0
N_FRAMES = 140  # 14 s: window [0,10) closes inside the stream


def _script(frame_index: int):
    """Six vehicles that are genuinely stationary (1 px detector jitter) for
    the whole stream, plus one that really moves. Image positions sit in the
    region where the survey homography is well conditioned."""
    out = []
    jitter = 1.0 if frame_index % 2 else 0.0
    for k in range(6):
        out.append((f"s{k}", (100.0 + 60.0 * k + jitter, 300.0)))
    out.append(("mover", (100.0 + 4.0 * frame_index, 120.0)))
    return out


def run_chain(profile):
    tracker = ScriptedTracker(_script)
    engine = StreamingPerceptionEngine(CAMERA, tracker, FPS, windows=WindowSpec(10.0), calibration_profile=profile)
    reasoner = WindowReasoner(CAMERA, calibration_profile=profile)
    analyses = []
    for frame in make_frames(N_FRAMES, FPS):
        for snap in engine.process_frame(frame):
            analyses.append(reasoner.process(snap))
    for snap in engine.finish():
        analyses.append(reasoner.process(snap))
    return engine, analyses


def _evidence_by_hypothesis(analysis):
    return {item.evidence.hypothesis_id: item for item in analysis.result.evidence_by_camera[CAMERA]}


def assert_no_metric_speed_reaches_consumers(engine, analyses, profile):
    """Every hop of the data flow, for a camera whose geometry is NOT
    validated metric."""
    assert analyses, "chain produced no windows"
    for a in analyses:
        # hop 1: the tracks the window hands to the reasoning layer
        assert a.snapshot.tracks, "scenario must put vehicles in every window"
        for t in a.snapshot.tracks:
            assert t.motion_space == MotionSpace.IMAGE
            assert t.geometry_validity != GeometryValidity.WORLD_VALIDATED
            assert all(isinstance(p, tuple) for p in t.position_history)  # pixels, never WorldPosition
        # hop 2: the aggregated state any m/s threshold would read
        assert a.aggregation.traffic_state.motion_space == MotionSpace.IMAGE
        # hop 3: evidence that needs a metric speed threshold cannot be formed
        ev = _evidence_by_hypothesis(a)
        for hid in ("H2", "H3", "H5"):
            assert ev[hid].evidence.evidence_completeness == EvidenceCompleteness.NONE, hid
            assert ev[hid].evidence.supporting_track_refs == [], hid
        # hop 4: reliability does not credit metric calibration
        assert a.result.reliability_by_camera[CAMERA].factors["calibration"] == 0.5
        # hop 5: no vehicle is named as a stopped-vehicle candidate from speed evidence
        cv = a.result.candidate_vehicle
        assert cv.status != CandidateVehicleStatus.RESOLVED and cv.track_id is None
    # and the per-track speed estimator never emits a km/h number either
    for state in engine.store.manager.all_tracks():
        est = estimate_vehicle_speed(CAMERA, state, profile)
        assert est.measurement is None and est.metric_speed_kmh is None
        text = format_speed_text(est)
        assert text is None or "km/h" not in text


@pytest.mark.parametrize("scale", [0.01, 40.0], ids=["tiny-scale-looks-stopped", "huge-scale-looks-supersonic"])
def test_unvalidated_cityflow_calibration_cannot_reach_metric_consumers(scale):
    profile = from_cityflow_homography("calib-cf", CAMERA, np.diag([scale, scale, 1.0]), reprojection_error=10.25)
    assert profile.is_calibrated  # the old code's only gate
    assert profile.metric_validation.value == "unvalidated"
    assert motion_space_for(profile) == MotionSpace.IMAGE

    engine, analyses = run_chain(profile)

    assert_no_metric_speed_reaches_consumers(engine, analyses, profile)


def test_no_calibration_is_also_safe():
    engine, analyses = run_chain(None)
    assert_no_metric_speed_reaches_consumers(engine, analyses, None)


def test_positive_control_validated_calibration_does_reach_metric_consumers():
    """The same chain with a VALIDATED survey profile DOES deliver metric
    tracks and metric-speed evidence -- the safe tests above are not passing
    merely because nothing can ever flow."""
    profile = fit_synthetic_validated_homography("calib-survey", CAMERA, make_reference_points())
    assert profile.supports_metric_speed

    engine, analyses = run_chain(profile)

    first = analyses[0]
    assert first.aggregation.traffic_state.motion_space == MotionSpace.WORLD
    stationary = [t for t in first.snapshot.tracks if t.speed_available and t.speed_estimate.value <= 1.0]
    assert len(stationary) >= 5
    ev = _evidence_by_hypothesis(first)
    assert ev["H2"].evidence.evidence_completeness != EvidenceCompleteness.NONE
    assert first.result.reliability_by_camera[CAMERA].factors["calibration"] == 1.0


def test_mutation_reintroducing_is_calibrated_equals_metric_is_detected(monkeypatch):
    """Sensitivity check: if `geometry_validity` is rewired so that merely
    being calibrated counts as validated (the old bug), the data-flow
    assertions must FAIL -- proving they actually observe the flow."""
    monkeypatch.setattr(
        CalibrationProfile, "geometry_validity",
        property(lambda self: GeometryValidity.WORLD_VALIDATED if self.is_calibrated else GeometryValidity.IMAGE),
    )
    monkeypatch.setattr(CalibrationProfile, "supports_metric_speed", property(lambda self: self.is_calibrated))
    profile = from_cityflow_homography("calib-cf", CAMERA, np.diag([0.01, 0.01, 1.0]), reprojection_error=10.25)

    engine, analyses = run_chain(profile)

    with pytest.raises(AssertionError):
        assert_no_metric_speed_reaches_consumers(engine, analyses, profile)


# ---------------------------------------------------------------------------
# Structural guards: the invariant cannot be violated by constructing objects
# ---------------------------------------------------------------------------


def test_a_world_track_cannot_be_declared_without_validated_geometry():
    from anvesh.storage.schemas import Measurement, OcclusionState, VehicleClass, VehicleTrack

    kwargs = dict(
        track_id="c:1", camera_id="c", first_seen=0.0, last_seen=1.0, vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE, position_history=[], speed_estimate=Measurement(5.0, 0.0),
    )
    with pytest.raises(ValueError):
        VehicleTrack(motion_space=MotionSpace.WORLD, geometry_validity=GeometryValidity.WORLD_UNVALIDATED, **kwargs)
    with pytest.raises(ValueError):
        VehicleTrack(motion_space=MotionSpace.WORLD, geometry_validity=GeometryValidity.IMAGE, **kwargs)
    with pytest.raises(ValueError):
        VehicleTrack(motion_space=MotionSpace.IMAGE, geometry_validity=GeometryValidity.WORLD_VALIDATED, **kwargs)
    # M-A.1: validity is never derived for a metric track -- omitting it is an error ...
    with pytest.raises(ValueError, match="explicitly"):
        VehicleTrack(motion_space=MotionSpace.WORLD, **kwargs)
    # ... and the only accepted metric declaration is an explicit WORLD_VALIDATED.
    ok = VehicleTrack(motion_space=MotionSpace.WORLD, geometry_validity=GeometryValidity.WORLD_VALIDATED, **kwargs)
    assert ok.geometry_validity == GeometryValidity.WORLD_VALIDATED


def test_a_validated_profile_must_have_metre_units():
    with pytest.raises(ValueError):
        CalibrationProfile(
            calibration_profile_id="p", camera_id="c", status="calibrated", reference_points=(),
            homography=np.eye(3), inverse_homography=np.eye(3), reprojection_error_m=0.0,
            units="unknown", provenance="cityflow_supplied", metric_validation="validated",
        )


def test_aggregation_refuses_a_motion_space_that_disagrees_with_its_tracks():
    from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state

    profile = from_cityflow_homography("calib-cf", CAMERA, np.diag([40.0, 40.0, 1.0]), reprojection_error=1.0)
    engine, analyses = run_chain(profile)
    image_tracks = list(analyses[0].snapshot.tracks)
    with pytest.raises(ValueError, match="exactly one motion space"):
        aggregate_traffic_state(CAMERA, 0.0, 10.0, image_tracks, MotionSpace.WORLD, CongestionStateTracker())


def test_metric_stopped_filter_ignores_tracks_without_a_real_speed():
    """A validated-geometry track whose speed could not be measured carries a
    placeholder 0.0; it must never be counted as "stopped"."""
    from anvesh.evidence.evidence_builder import EvidenceThresholds, _stopped_tracks
    from anvesh.storage.schemas import Measurement, OcclusionState, VehicleClass, VehicleTrack

    def trk(tid, available):
        return VehicleTrack(
            track_id=tid, camera_id="c", first_seen=0.0, last_seen=10.0, vehicle_class=VehicleClass.CAR,
            occlusion_state=OcclusionState.VISIBLE, position_history=[], speed_estimate=Measurement(0.0, 0.0),
            motion_space=MotionSpace.WORLD, geometry_validity=GeometryValidity.WORLD_VALIDATED, speed_available=available,
        )

    stopped = _stopped_tracks([trk("c:a", True), trk("c:b", False)], MotionSpace.WORLD, EvidenceThresholds())
    assert [t.track_id for t in stopped] == ["c:a"]
