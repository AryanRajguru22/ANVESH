import pytest

from anvesh.propagation.shockwave import (
    FlowDensityState,
    flow_density_from_traffic_state,
    predict_propagation,
    predict_shockwave_speed,
)
from anvesh.storage.schemas import CameraRole, CongestionLevel, Measurement, MotionSpace, TrafficState
from anvesh.world.corridor import CameraPlacement, CorridorTopology

SEGMENT_LENGTH_M = 100.0


def _traffic_state(camera_id, vehicle_count, flow_rate, motion_space=MotionSpace.WORLD, congestion=CongestionLevel.BUILDING):
    return TrafficState(
        camera_id=camera_id,
        window_start=0.0,
        window_end=10.0,
        occupancy=min(1.0, vehicle_count / 20.0),
        mean_speed=Measurement(value=5.0, error=0.0),
        vehicle_count=vehicle_count,
        flow_rate=flow_rate,
        density=float(vehicle_count),
        congestion_level=congestion,
        motion_space=motion_space,
    )


def _topology():
    return CorridorTopology(
        corridor_id="corridor-1",
        placements=(
            CameraPlacement(camera_id="cam-A", role=CameraRole.UPSTREAM, distance_from_upstream_m=0.0),
            CameraPlacement(camera_id="cam-B", role=CameraRole.DOWNSTREAM, distance_from_upstream_m=SEGMENT_LENGTH_M),
        ),
    )


# ---------------------------------------------------------------------------
# FlowDensityState / flow_density_from_traffic_state
# ---------------------------------------------------------------------------


def test_flow_density_from_traffic_state_computes_metric_density():
    ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    fd = flow_density_from_traffic_state(ts, SEGMENT_LENGTH_M)
    assert fd.density_veh_per_m == pytest.approx(0.2)
    assert fd.flow_veh_per_s == 2.0
    assert fd.camera_id == "cam-A"


def test_flow_density_rejects_uncalibrated_camera():
    ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0, motion_space=MotionSpace.IMAGE)
    with pytest.raises(ValueError):
        flow_density_from_traffic_state(ts, SEGMENT_LENGTH_M)


def test_flow_density_rejects_invalid_segment_length():
    ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    with pytest.raises(ValueError):
        flow_density_from_traffic_state(ts, 0.0)
    with pytest.raises(ValueError):
        flow_density_from_traffic_state(ts, -10.0)


def test_flow_density_rejects_missing_state():
    with pytest.raises(ValueError):
        flow_density_from_traffic_state(None, SEGMENT_LENGTH_M)


def test_flow_density_state_rejects_negative_values():
    with pytest.raises(ValueError):
        FlowDensityState(camera_id="cam-A", flow_veh_per_s=-1.0, density_veh_per_m=0.1)
    with pytest.raises(ValueError):
        FlowDensityState(camera_id="cam-A", flow_veh_per_s=1.0, density_veh_per_m=-0.1)
    with pytest.raises(ValueError):
        FlowDensityState(camera_id="", flow_veh_per_s=1.0, density_veh_per_m=0.1)


# ---------------------------------------------------------------------------
# predict_shockwave_speed -- hand-computed reference values
# ---------------------------------------------------------------------------


def test_shockwave_speed_downstream_dissipation_hand_computed():
    # k1=0.2, k2=0.05 -> delta_k=-0.15; q1=2.0, q2=1.0 -> delta_q=-1.0
    # v_shock = -1.0 / -0.15 = 6.666... > 0 (downstream / dissipating)
    upstream = FlowDensityState(camera_id="cam-A", flow_veh_per_s=2.0, density_veh_per_m=0.2)
    downstream = FlowDensityState(camera_id="cam-B", flow_veh_per_s=1.0, density_veh_per_m=0.05)
    speed = predict_shockwave_speed(upstream, downstream)
    assert speed.value == pytest.approx(-1.0 / -0.15)
    assert speed.value > 0


def test_shockwave_speed_upstream_growth_hand_computed():
    # k1=0.05, k2=0.2 -> delta_k=0.15; q1=2.0, q2=1.0 -> delta_q=-1.0
    # v_shock = -1.0 / 0.15 = -6.666... < 0 (upstream / growing)
    upstream = FlowDensityState(camera_id="cam-A", flow_veh_per_s=2.0, density_veh_per_m=0.05)
    downstream = FlowDensityState(camera_id="cam-B", flow_veh_per_s=1.0, density_veh_per_m=0.2)
    speed = predict_shockwave_speed(upstream, downstream)
    assert speed.value == pytest.approx(-1.0 / 0.15)
    assert speed.value < 0


def test_shockwave_speed_zero_density_difference_rejected():
    upstream = FlowDensityState(camera_id="cam-A", flow_veh_per_s=2.0, density_veh_per_m=0.1)
    downstream = FlowDensityState(camera_id="cam-B", flow_veh_per_s=1.0, density_veh_per_m=0.1)
    with pytest.raises(ValueError):
        predict_shockwave_speed(upstream, downstream)


def test_shockwave_speed_missing_state_rejected():
    upstream = FlowDensityState(camera_id="cam-A", flow_veh_per_s=2.0, density_veh_per_m=0.1)
    with pytest.raises(ValueError):
        predict_shockwave_speed(None, upstream)
    with pytest.raises(ValueError):
        predict_shockwave_speed(upstream, None)


def test_shockwave_speed_zero_flow_difference_is_zero_but_not_an_error():
    upstream = FlowDensityState(camera_id="cam-A", flow_veh_per_s=1.0, density_veh_per_m=0.05)
    downstream = FlowDensityState(camera_id="cam-B", flow_veh_per_s=1.0, density_veh_per_m=0.2)
    speed = predict_shockwave_speed(upstream, downstream)
    assert speed.value == 0.0


# ---------------------------------------------------------------------------
# predict_propagation -- full pipeline, direction/unit handling
# ---------------------------------------------------------------------------


def test_predict_propagation_downstream_direction_and_units():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)

    prediction = predict_propagation(
        "pred-1", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, reference_timestamp=100.0
    )

    assert prediction.predicted_shockwave_speed.value > 0
    assert prediction.predicted_arrival_camera == "cam-B"  # downstream -- dissipation propagates toward it
    assert prediction.predicted_arrival_time > 100.0
    expected_growth = abs(1.0 - 2.0) * 60.0
    assert prediction.predicted_queue_growth_rate == pytest.approx(expected_growth)
    assert prediction.method == "rankine_hugoniot_kinematic_wave_v1"
    assert isinstance(prediction.predicted_shockwave_speed, Measurement)


def test_predict_propagation_upstream_direction():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=5, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=20, flow_rate=1.0)

    prediction = predict_propagation(
        "pred-2", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, reference_timestamp=0.0
    )

    assert prediction.predicted_shockwave_speed.value < 0
    assert prediction.predicted_arrival_camera == "cam-A"  # growing -- propagates toward upstream


def test_predict_propagation_arrival_time_matches_distance_over_speed():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)

    prediction = predict_propagation(
        "pred-3", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, reference_timestamp=50.0
    )
    expected_speed = abs(-1.0 / -0.15)
    expected_arrival = 50.0 + SEGMENT_LENGTH_M / expected_speed
    assert prediction.predicted_arrival_time == pytest.approx(expected_arrival)


def test_predict_propagation_rejects_missing_state():
    topology = _topology()
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)
    with pytest.raises(ValueError):
        predict_propagation("pred-4", "rank-1", topology, None, downstream_ts, SEGMENT_LENGTH_M, 0.0)


def test_predict_propagation_rejects_uncalibrated_camera():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0, motion_space=MotionSpace.IMAGE)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)
    with pytest.raises(ValueError):
        predict_propagation("pred-5", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, 0.0)


def test_predict_propagation_rejects_degenerate_equal_density():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=10, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=10, flow_rate=1.0)
    with pytest.raises(ValueError):
        predict_propagation("pred-6", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, 0.0)


def test_predict_propagation_rejects_too_small_shockwave_speed():
    topology = _topology()
    # equal flow rates -> delta_q = 0 exactly -> v_shock = 0, too small for an ETA
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=1.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)
    with pytest.raises(ValueError):
        predict_propagation("pred-7", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, 0.0)


def test_predict_propagation_rejects_invalid_segment_length():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)
    with pytest.raises(ValueError):
        predict_propagation("pred-8", "rank-1", topology, upstream_ts, downstream_ts, -5.0, 0.0)


def test_predict_propagation_is_deterministic():
    topology = _topology()
    upstream_ts = _traffic_state("cam-A", vehicle_count=20, flow_rate=2.0)
    downstream_ts = _traffic_state("cam-B", vehicle_count=5, flow_rate=1.0)

    p1 = predict_propagation("pred-9", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, 10.0)
    p2 = predict_propagation("pred-9", "rank-1", topology, upstream_ts, downstream_ts, SEGMENT_LENGTH_M, 10.0)

    assert p1.predicted_shockwave_speed.value == p2.predicted_shockwave_speed.value
    assert p1.predicted_arrival_time == p2.predicted_arrival_time
    assert p1.predicted_arrival_camera == p2.predicted_arrival_camera
