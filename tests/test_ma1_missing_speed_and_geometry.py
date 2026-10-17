"""M-A.1 regression tests: missing-speed evidence safety (Fix 1), explicit geometry
validity (Fix 2) and evidence-derived hypothesis eligibility (Fix 6).

Every scenario here was reproduced against the pre-M-A.1 code first (the validated
camera whose speeds were all unavailable ranked "H1, belief 0.946, tier high").
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from anvesh.evidence.evidence_builder import EvidenceDirection, EvidenceThresholds, build_evidence_for_camera
from anvesh.evidence.reliability import compute_reliability
from anvesh.fusion.ds_fusion import single_camera_discounted_result, single_camera_result
from anvesh.hypotheses.ranking import build_ranking, evaluable_hypothesis_ids
from anvesh.perception.speed_estimation import SpeedConfidence, VehicleSpeedEstimate
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.storage import db as storage_db
from anvesh.storage.schemas import (
    CameraRole,
    CandidateOutcome,
    ConfidenceTier,
    EvidenceCompleteness,
    GeometryValidity,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
    WorldPosition,
)

CAM = "cam-v"
WINDOW = (0.0, 10.0)


def _world_track(i, speed=5.0, available=True, first=0.0, last=10.0, occlusion=OcclusionState.VISIBLE):
    """A SYNTHETIC validated-metric track (validity declared by the fixture)."""
    return VehicleTrack(
        track_id=f"{CAM}:s000-t{i:05d}", camera_id=CAM, first_seen=first, last_seen=last,
        vehicle_class=VehicleClass.CAR, occlusion_state=occlusion, position_history=[],
        speed_estimate=Measurement(value=speed, error=0.1), motion_space=MotionSpace.WORLD,
        geometry_validity=GeometryValidity.WORLD_VALIDATED, speed_available=available,
    )


def _window_inputs(tracks, congested=True):
    tracker = CongestionStateTracker()
    last = None
    for _ in range(3):  # settle the congestion hysteresis deterministically
        last = aggregate_traffic_state(CAM, *WINDOW, tracks, MotionSpace.WORLD, tracker, max_capacity_vehicles=len(tracks) + 1 if congested else 1000)
    return last


def _run(tracks):
    aggregation = _window_inputs(tracks)
    ci = CameraWindowInput(camera_id=CAM, role=CameraRole.UPSTREAM, aggregation=aggregation, tracks=tracks)
    return aggregation, run_corridor_window(WINDOW, (ci,))


def _evidence(tracks):
    aggregation = _window_inputs(tracks)
    rel = compute_reliability(CAM, *WINDOW, tracks, aggregation).score
    items = build_evidence_for_camera(CAM, WINDOW, CameraRole.UPSTREAM, aggregation, tracks, rel)
    return {i.evidence.hypothesis_id: i for i in items}, aggregation


# ---------------------------------------------------------------- Fix 1 ----

def test_validated_geometry_with_all_speeds_unavailable_cannot_rank_or_claim_high_confidence():
    tracks = [_world_track(i, speed=0.0, available=False) for i in range(18)]
    aggregation, result = _run(tracks)

    assert aggregation.traffic_state.congestion_level.value in ("building", "congested")  # genuinely crowded
    assert aggregation.traffic_state.mean_speed_available is False
    assert aggregation.speed_quality.value == "unavailable"

    ranking = result.candidate_ranking
    assert ranking.outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE
    assert all(e.confidence_tier == ConfidenceTier.LOW for e in ranking.ranked_list)
    ev = result.evidence_by_camera[CAM]
    for item in ev:
        if item.evidence.hypothesis_id in ("H1", "H2", "H3"):
            assert item.evidence.evidence_completeness == EvidenceCompleteness.NONE
            assert item.direction == EvidenceDirection.INSUFFICIENT
            assert item.evidence.signature_match_score == 0.0
    assert result.candidate_vehicle.status.value != "resolved"


def test_missing_speed_is_not_zero_speed_and_not_a_stopped_vehicle():
    tracks = [_world_track(i, speed=0.0, available=False, first=0.0, last=10.0) for i in range(3)]
    items, _ = _evidence(tracks)
    assert items["H2"].evidence.supporting_track_refs == [] and items["H3"].evidence.supporting_track_refs == []
    assert items["H2"].direction != EvidenceDirection.SUPPORTS and items["H3"].direction != EvidenceDirection.SUPPORTS


def test_missing_speed_is_not_evidence_against_a_hypothesis():
    """No measurement => NONE/INSUFFICIENT (zero mass), never CONTRADICTS."""
    items, _ = _evidence([_world_track(i, available=False) for i in range(10)])
    for hid in ("H1", "H2", "H3"):
        assert items[hid].direction == EvidenceDirection.INSUFFICIENT
        assert items[hid].evidence.contradicting is False


def test_mixed_available_and_unavailable_with_adequate_coverage_is_partial_and_discounted():
    tracks = [_world_track(i, speed=6.0, available=(i < 7)) for i in range(10)]  # 7/10 measured, none stopped
    items, aggregation = _evidence(tracks)
    assert items["H1"].evidence.evidence_completeness == EvidenceCompleteness.PARTIAL
    assert items["H2"].evidence.evidence_completeness == EvidenceCompleteness.PARTIAL
    assert items["H1"].metadata["speed_measured_tracks"] == 7 and items["H1"].metadata["tracks"] == 10
    # the traffic-state mean only uses the measured vehicles
    assert aggregation.traffic_state.mean_speed_available is True
    assert aggregation.traffic_state.mean_speed.value == pytest.approx(6.0)


def test_insufficient_speed_coverage_makes_obstruction_hypotheses_unavailable():
    tracks = [_world_track(i, speed=6.0, available=(i < 4)) for i in range(10)]  # 40% < 50% policy minimum
    items, _ = _evidence(tracks)
    for hid in ("H1", "H2", "H3"):
        assert items[hid].evidence.evidence_completeness == EvidenceCompleteness.NONE
    assert "4/10" in items["H1"].metadata["reason"]


def test_coverage_threshold_is_a_parameter_not_a_constant():
    tracks = [_world_track(i, speed=6.0, available=(i < 4)) for i in range(10)]
    agg = _window_inputs(tracks)
    items = build_evidence_for_camera(CAM, WINDOW, CameraRole.UPSTREAM, agg, tracks, 0.8, EvidenceThresholds(min_speed_coverage=0.3))
    assert items[0].evidence.evidence_completeness == EvidenceCompleteness.PARTIAL


def test_a_genuinely_measured_zero_speed_is_a_stopped_vehicle_and_distinct_from_unavailable():
    measured_stop = _world_track(0, speed=0.0, available=True, first=0.0, last=8.0)
    unmeasured = _world_track(1, speed=0.0, available=False, first=0.0, last=8.0)
    moving = _world_track(2, speed=7.0, available=True, first=0.0, last=8.0)
    items, _ = _evidence([measured_stop, unmeasured, moving])
    assert items["H2"].direction == EvidenceDirection.SUPPORTS
    assert items["H2"].evidence.supporting_track_refs == [measured_stop.track_id]  # only the MEASURED zero
    assert unmeasured.track_id not in items["H2"].evidence.supporting_track_refs


def test_full_coverage_full_evidence_still_ranks_and_is_discounted_by_reliability():
    tracks = [_world_track(i, speed=6.0, occlusion=OcclusionState.PARTIAL if i % 3 == 0 else OcclusionState.VISIBLE) for i in range(18)]
    aggregation, result = _run(tracks)
    ev = {i.evidence.hypothesis_id: i for i in result.evidence_by_camera[CAM]}
    assert result.reliability_by_camera[CAM].score < 1.0
    assert ev["H1"].evidence.evidence_completeness == EvidenceCompleteness.FULL
    ranking = result.candidate_ranking
    assert ranking.outcome == CandidateOutcome.RANKED and ranking.ranked_list[0].hypothesis_id == "H1"
    # belief is the raw H1 score DISCOUNTED by reliability, never the raw score itself
    raw = ev["H1"].evidence.signature_match_score
    assert ranking.ranked_list[0].belief == pytest.approx(raw * result.reliability_by_camera[CAM].score)
    assert ranking.ranked_list[0].belief < raw


def test_traffic_state_availability_round_trips_through_storage():
    _, aggregation = _evidence([_world_track(i, available=False) for i in range(4)])
    ts = aggregation.traffic_state
    restored = storage_db._traffic_state_from_dict(storage_db._traffic_state_to_dict(ts))
    assert restored.mean_speed_available is False and ts.mean_speed_available is False
    legacy = storage_db._traffic_state_to_dict(ts)
    legacy.pop("mean_speed_available")  # pre-1.1.0 rows keep their meaning
    assert storage_db._traffic_state_from_dict(legacy).mean_speed_available is True


def test_reliability_calibration_factor_reflects_speed_coverage():
    full = [_world_track(i) for i in range(4)]
    none = [_world_track(i, available=False) for i in range(4)]
    agg_f, agg_n = _window_inputs(full), _window_inputs(none)
    f = compute_reliability(CAM, *WINDOW, full, agg_f).factors["calibration"]
    n = compute_reliability(CAM, *WINDOW, none, agg_n).factors["calibration"]
    assert f == 1.0 and n == 0.5


def test_exporter_never_exports_a_placeholder_speed_as_a_number():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("export_ui_data_ma1", pathlib.Path(__file__).parents[1] / "scripts" / "export_ui_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod._track_summary(_world_track(0, speed=0.0, available=False))["speed_estimate"] is None
    assert mod._track_summary(_world_track(1, speed=0.0, available=True))["speed_estimate"] == {"value": 0.0, "error": 0.1}
    _, agg = _evidence([_world_track(i, available=False) for i in range(3)])
    assert mod._traffic_state(agg.traffic_state)["mean_speed"] is None


# ---------------------------------------------------------------- Fix 2 ----

_KW = dict(track_id="c:1", camera_id="c", first_seen=0.0, last_seen=1.0, vehicle_class=VehicleClass.CAR,
           occlusion_state=OcclusionState.VISIBLE, speed_estimate=Measurement(5.0, 0.0))
_WP = [WorldPosition(1.0, 2.0, 0.0)]


def test_world_track_without_explicit_validity_is_rejected():
    with pytest.raises(ValueError, match="explicitly"):
        VehicleTrack(position_history=_WP, motion_space=MotionSpace.WORLD, **_KW)
    with pytest.raises(ValueError, match="explicitly"):
        VehicleTrack(position_history=[], motion_space="world", **_KW)


@pytest.mark.parametrize("validity", [GeometryValidity.IMAGE, GeometryValidity.WORLD_UNVALIDATED])
def test_world_track_with_contradictory_validity_is_rejected(validity):
    with pytest.raises(ValueError):
        VehicleTrack(position_history=_WP, motion_space=MotionSpace.WORLD, geometry_validity=validity, **_KW)


def test_image_track_claiming_validated_geometry_is_rejected():
    with pytest.raises(ValueError):
        VehicleTrack(position_history=[(1.0, 2.0, 0.0)], motion_space=MotionSpace.IMAGE,
                     geometry_validity=GeometryValidity.WORLD_VALIDATED, **_KW)


def test_image_track_may_omit_validity_and_is_conservatively_image():
    t = VehicleTrack(position_history=[(1.0, 2.0, 0.0)], motion_space=MotionSpace.IMAGE, **_KW)
    assert t.geometry_validity == GeometryValidity.IMAGE


def test_unit_inconsistent_positions_are_rejected():
    with pytest.raises(ValueError, match="WorldPosition"):  # metric track with pixel tuples
        VehicleTrack(position_history=[(1.0, 2.0, 0.0)], motion_space=MotionSpace.WORLD,
                     geometry_validity=GeometryValidity.WORLD_VALIDATED, **_KW)
    with pytest.raises(ValueError, match="WorldPosition"):  # pixel track with metre positions
        VehicleTrack(position_history=_WP, motion_space=MotionSpace.IMAGE, **_KW)
    with pytest.raises(ValueError):  # mixed
        VehicleTrack(position_history=_WP + [(1.0, 2.0, 1.0)], motion_space=MotionSpace.WORLD,
                     geometry_validity=GeometryValidity.WORLD_VALIDATED, **_KW)


def test_explicit_valid_declaration_is_accepted():
    t = VehicleTrack(position_history=_WP, motion_space=MotionSpace.WORLD,
                     geometry_validity=GeometryValidity.WORLD_VALIDATED, **_KW)
    assert t.geometry_validity == GeometryValidity.WORLD_VALIDATED


def test_a_kmh_speed_estimate_never_implies_its_own_validity():
    with pytest.raises(ValueError, match="explicitly"):
        VehicleSpeedEstimate(track_id="c:1", camera_id="c", status="ok", confidence=SpeedConfidence.VALIDATED,
                             measurement=Measurement(30.0, 1.0))
    ok = VehicleSpeedEstimate(track_id="c:1", camera_id="c", status="ok", confidence=SpeedConfidence.VALIDATED,
                              measurement=Measurement(30.0, 1.0), geometry_validity=GeometryValidity.WORLD_VALIDATED)
    assert ok.metric_speed_kmh.value == 30.0


def test_production_code_declares_validity_at_every_vehicle_track_construction():
    """Audit: every `VehicleTrack(...)` call under anvesh/ passes geometry_validity explicitly."""
    root = pathlib.Path(__file__).parents[1] / "anvesh"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "VehicleTrack":
                if "geometry_validity" not in {k.arg for k in node.keywords}:
                    offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert offenders == []


# ---------------------------------------------------------------- Fix 6 ----

def _ev(items_spec):
    """items_spec: {hid: (completeness, direction, score)} -> evidence tuple for one camera."""
    from anvesh.evidence.evidence_builder import EvidenceItem
    from anvesh.hypotheses.cause_model import HYPOTHESIS_IDS
    from anvesh.storage.schemas import Evidence

    out = []
    for hid in HYPOTHESIS_IDS:
        comp, direction, score = items_spec.get(hid, (EvidenceCompleteness.NONE, EvidenceDirection.INSUFFICIENT, 0.0))
        out.append(EvidenceItem(
            evidence=Evidence(evidence_id=f"x:{hid}", camera_id="x", window=WINDOW, hypothesis_id=hid,
                              signature_match_score=score, supporting_track_refs=[],
                              contradicting=direction == EvidenceDirection.CONTRADICTS, evidence_completeness=comp),
            evidence_type="t", measured_value=None, direction=direction, reliability=1.0, source="t", metadata={},
        ))
    return tuple(out)


F, P_, N = EvidenceCompleteness.FULL, EvidenceCompleteness.PARTIAL, EvidenceCompleteness.NONE
S, C, NE, I = EvidenceDirection.SUPPORTS, EvidenceDirection.CONTRADICTS, EvidenceDirection.NEUTRAL, EvidenceDirection.INSUFFICIENT


def test_h1_is_not_a_competitor_without_a_possible_obstruction_test():
    only_h1 = _ev({"H1": (F, S, 0.9), "H5": (F, NE, 0.1)})
    assert evaluable_hypothesis_ids({"x": only_h1}) == {"H5"}  # H1 excluded; H5 alone is < 2
    ranking = build_ranking("r", "c", single_camera_result("x", only_h1, WINDOW), {"x": only_h1})
    assert ranking.outcome == CandidateOutcome.INSUFFICIENT_EVIDENCE


def test_h1_with_an_evaluated_obstruction_test_is_a_competitor():
    ev = _ev({"H1": (F, S, 0.9), "H2": (F, C, 0.15), "H3": (F, NE, 0.05)})
    assert evaluable_hypothesis_ids({"x": ev}) == {"H1", "H2", "H3"}
    assert build_ranking("r", "c", single_camera_result("x", ev, WINDOW), {"x": ev}).outcome == CandidateOutcome.RANKED


def test_evidence_that_reports_insufficient_is_never_eligible_whatever_its_completeness_flag():
    ev = _ev({"H2": (F, I, 0.0), "H3": (F, I, 0.0), "H1": (F, S, 0.9)})
    assert evaluable_hypothesis_ids({"x": ev}) == set()


def test_evaluated_but_unsupported_hypotheses_remain_eligible_competitors():
    ev = _ev({"H2": (F, S, 0.8), "H3": (F, C, 0.2)})
    assert evaluable_hypothesis_ids({"x": ev}) == {"H2", "H3"}


def test_missing_evidence_contributes_no_mass_and_no_contradiction():
    ev = _ev({"H2": (F, S, 0.8), "H3": (F, C, 0.2)})
    bpa = single_camera_result("x", ev, WINDOW).final_bpa
    assert bpa.masses["H4"] == bpa.masses["H6"] == 0.0 and bpa.masses["H3"] == 0.0


def test_single_camera_discounting_moves_unreliable_mass_to_ignorance():
    ev = _ev({"H2": (F, S, 0.8), "H3": (F, C, 0.2)})
    raw = single_camera_result("x", ev, WINDOW)
    disc = single_camera_discounted_result("x", ev, WINDOW, 0.5)
    assert disc.method == "single_camera_discounted_v1" and raw.method == "single_camera_v1"  # Baseline A untouched
    assert disc.final_bpa.masses["H2"] == pytest.approx(0.4)
    assert disc.final_bpa.theta_mass == pytest.approx(1.0 - 0.4)
    assert raw.final_bpa.masses["H2"] == pytest.approx(0.8)


def test_partial_speed_coverage_scales_the_absence_evidence_and_never_reaches_high_confidence():
    """H1 is an absence claim: 'no obstruction found' is only as strong as the search behind it.
    Before the scaling a half-searched window ranked H1 'high' (belief 0.913)."""
    beliefs = {}
    for measured in (18, 13, 11, 9):  # 100%, 72%, 61%, 50% of 18 vehicles measured
        _, result = _run([_world_track(i, speed=6.0, available=(i < measured)) for i in range(18)])
        top = result.candidate_ranking.ranked_list[0]
        assert result.candidate_ranking.outcome == CandidateOutcome.RANKED and top.hypothesis_id == "H1"
        beliefs[measured] = top.belief
        if measured < 18:
            assert top.confidence_tier != ConfidenceTier.HIGH
        else:
            assert top.confidence_tier == ConfidenceTier.HIGH  # a fully searched window is unaffected
    assert beliefs[18] > beliefs[13] > beliefs[11] > beliefs[9]  # strictly weaker as less of the road was searched
