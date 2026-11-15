"""Hardening A2: TrafficState carries explicit geometry validity; legacy/unverified states are never metric."""

import numpy as np
import pytest

from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.propagation.shockwave import flow_density_from_traffic_state
from anvesh.storage import db as storage_db
from anvesh.storage.schemas import (
    CongestionLevel,
    GeometryValidity,
    Measurement,
    MotionSpace,
    OcclusionState,
    TrafficState,
    VehicleClass,
    VehicleTrack,
)
from anvesh.streaming.reasoning import WindowReasoner
from anvesh.streaming.windows import WindowSnapshot
from anvesh.world.calibration import fit_planar_homography, from_cityflow_homography
from anvesh.world.synthetic_fixture import fit_synthetic_validated_homography
from tests.world.support import make_reference_points

CAM = "cam-a2"


def _state(**over):
    base = dict(camera_id=CAM, window_start=0.0, window_end=10.0, occupancy=0.3,
                mean_speed=Measurement(value=6.0, error=0.5), vehicle_count=4, flow_rate=0.4, density=4.0,
                congestion_level=CongestionLevel.BUILDING, motion_space=MotionSpace.WORLD,
                geometry_validity=GeometryValidity.WORLD_VALIDATED)
    base.update(over)
    return TrafficState(**base)


def _track(i, space, validity, speed=5.0):
    from anvesh.storage.schemas import WorldPosition
    return VehicleTrack(
        track_id=f"{CAM}:{i}", camera_id=CAM, first_seen=0.0, last_seen=9.0, vehicle_class=VehicleClass.CAR,
        occlusion_state=OcclusionState.VISIBLE, position_history=[], speed_estimate=Measurement(speed, 0.1),
        motion_space=space, geometry_validity=validity,
    )


# ---- storage ---------------------------------------------------------------------------------


def test_geometry_validity_round_trips_through_storage():
    for s in (_state(), _state(motion_space=MotionSpace.IMAGE, geometry_validity=GeometryValidity.WORLD_UNVALIDATED)):
        d = storage_db._traffic_state_to_dict(s)
        assert d["geometry_validity"] == s.geometry_validity.value and d["schema_version"] == "1.2.0"
        r = storage_db._traffic_state_from_dict(d)
        assert r.geometry_validity == s.geometry_validity and r.supports_metric == s.supports_metric


def _legacy_world_row(version, **extra):
    row = {"camera_id": CAM, "window_start": 0.0, "window_end": 10.0, "occupancy": 0.3,
           "mean_speed": {"value": 8.0, "error": 0.5}, "vehicle_count": 4, "flow_rate": 0.4, "density": 4.0,
           "congestion_level": "building", "motion_space": "world", "schema_version": version}
    row.update(extra)
    return row


@pytest.mark.parametrize("row", [
    _legacy_world_row("1.0.0"),                                   # pre-flag row
    _legacy_world_row("1.1.0", mean_speed_available=True),        # flagged "measured", still no geometry record
])
def test_a_legacy_world_row_loads_as_unverified_and_is_never_metric(row):
    state = storage_db._traffic_state_from_dict(row)
    assert state.motion_space == MotionSpace.WORLD and state.geometry_validity is None
    assert not state.supports_metric
    assert state.measured_mean_speed() is None
    with pytest.raises(ValueError, match="WORLD_VALIDATED"):
        flow_density_from_traffic_state(state, 100.0)  # no metric estimate from an unverified state


def test_a_row_claiming_the_current_schema_but_missing_validity_is_rejected_not_defaulted():
    with pytest.raises(ValueError, match="explicitly"):
        storage_db._traffic_state_from_dict(_legacy_world_row("1.2.0", mean_speed_available=True))


# ---- shockwave gate ---------------------------------------------------------------------------


def test_shockwave_accepts_only_world_validated_states():
    assert flow_density_from_traffic_state(_state(), 100.0).density_veh_per_m == pytest.approx(0.04)
    with pytest.raises(ValueError, match="not calibrated"):
        flow_density_from_traffic_state(_state(motion_space=MotionSpace.IMAGE, geometry_validity=GeometryValidity.IMAGE), 100.0)
    with pytest.raises(ValueError, match="WORLD_VALIDATED"):
        flow_density_from_traffic_state(_state(schema_version="1.1.0", geometry_validity=None), 100.0)


# ---- aggregate_traffic_state ------------------------------------------------------------------


def test_aggregate_derives_validity_from_tracks_but_never_for_an_empty_world_window():
    tr = CongestionStateTracker
    world = [_track(i, MotionSpace.WORLD, GeometryValidity.WORLD_VALIDATED) for i in range(2)]
    assert aggregate_traffic_state(CAM, 0, 10, world, MotionSpace.WORLD, tr()).traffic_state.geometry_validity == GeometryValidity.WORLD_VALIDATED

    unval = [_track(i, MotionSpace.IMAGE, GeometryValidity.WORLD_UNVALIDATED) for i in range(2)]
    assert aggregate_traffic_state(CAM, 0, 10, unval, MotionSpace.IMAGE, tr()).traffic_state.geometry_validity == GeometryValidity.WORLD_UNVALIDATED
    mixed = unval + [_track(9, MotionSpace.IMAGE, GeometryValidity.IMAGE)]
    assert aggregate_traffic_state(CAM, 0, 10, mixed, MotionSpace.IMAGE, tr()).traffic_state.geometry_validity == GeometryValidity.IMAGE  # weaker claim wins

    assert aggregate_traffic_state(CAM, 0, 10, [], MotionSpace.IMAGE, tr()).traffic_state.geometry_validity == GeometryValidity.IMAGE
    with pytest.raises(ValueError, match="explicit geometry_validity"):
        aggregate_traffic_state(CAM, 0, 10, [], MotionSpace.WORLD, tr())
    ok = aggregate_traffic_state(CAM, 0, 10, [], MotionSpace.WORLD, tr(), geometry_validity=GeometryValidity.WORLD_VALIDATED)
    assert ok.traffic_state.supports_metric and ok.traffic_state.measured_mean_speed() is None  # empty: no speed


def test_aggregate_rejects_a_declared_validity_that_contradicts_the_space():
    tr = CongestionStateTracker
    with pytest.raises(ValueError):
        aggregate_traffic_state(CAM, 0, 10, [], MotionSpace.WORLD, tr(), geometry_validity=GeometryValidity.WORLD_UNVALIDATED)
    with pytest.raises(ValueError):
        aggregate_traffic_state(CAM, 0, 10, [], MotionSpace.IMAGE, tr(), geometry_validity=GeometryValidity.WORLD_VALIDATED)


# ---- WindowReasoner passes the CAMERA's validity (so an EMPTY window is classified correctly) ----


def _empty_snapshot():
    return WindowSnapshot(window_index=0, window_id=f"{CAM}:w000000", camera_id=CAM, start=0.0, end=10.0, tracks=(),
                          frames_observed=100, dropped_frames=0, sessions=(), closed_at_stream_time=10.0)


@pytest.mark.parametrize("profile_factory, space, validity", [
    (lambda: fit_synthetic_validated_homography("p", CAM, make_reference_points()), MotionSpace.WORLD, GeometryValidity.WORLD_VALIDATED),
    (lambda: fit_planar_homography("p", CAM, make_reference_points()), MotionSpace.IMAGE, GeometryValidity.WORLD_UNVALIDATED),
    (lambda: from_cityflow_homography("p", CAM, np.eye(3), reprojection_error=1.0), MotionSpace.IMAGE, GeometryValidity.WORLD_UNVALIDATED),
    (lambda: None, MotionSpace.IMAGE, GeometryValidity.IMAGE),
])
def test_window_reasoner_states_carry_the_cameras_geometry_validity(profile_factory, space, validity):
    analysis = WindowReasoner(CAM, calibration_profile=profile_factory()).process(_empty_snapshot())
    ts = analysis.aggregation.traffic_state
    assert (ts.motion_space, ts.geometry_validity) == (space, validity)
    assert ts.mean_speed_available is False and ts.measured_mean_speed() is None
