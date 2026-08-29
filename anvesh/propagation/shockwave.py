"""Propagation prediction (blueprint Part 3, module 13; Part 8): classical
kinematic-wave / Rankine-Hugoniot shockwave-speed estimation.

    v_shock = (q2 - q1) / (k2 - k1)

This is a well-established, decades-old result from LWR/kinematic-wave
traffic-flow theory (blueprint Part 8) -- nothing in this module claims
the equation itself is novel, and no ML/learned propagation model is used
(blueprint Part 2: explicitly LATER, not V1).

Sign convention (blueprint Part 5's own field comment on
`PropagationPrediction.predicted_shockwave_speed`): negative = moving
upstream (queue growing), positive = moving downstream (queue
dissipating).

V1's 2-camera constraint means the corridor's two per-camera
`TrafficState`s stand in for "just upstream of the queue front" / "just
downstream of it" (blueprint Part 8's q1,k1 / q2,k2) -- a documented
approximation, not a claim of measuring the shockwave front's exact
location. See `FlowDensityState`/`flow_density_from_traffic_state` for
why density is recomputed fresh from `vehicle_count` rather than trusted
from `TrafficState.density` directly.

PREDICTED vs OBSERVED: this module only ever produces a
`schemas.PropagationPrediction` (a forecast). Comparing it against what
was later actually observed (`schemas.PropagationObservation`) is
`feedback/update_engine.py`'s job, not this module's -- the two must
never be conflated.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.storage.schemas import Measurement, MotionSpace, PropagationPrediction, TrafficState
from anvesh.world.corridor import CorridorTopology

_MIN_DENSITY_DIFFERENCE_VEH_PER_M = 1e-6
_MIN_SHOCKWAVE_SPEED_M_PER_S = 1e-6  # below this, an arrival-time estimate is not meaningful


@dataclass(frozen=True)
class FlowDensityState:
    """The two physical quantities Rankine-Hugoniot needs, in real
    (metric) units: vehicles/second and vehicles/metre.

    Deliberately NOT read from `TrafficState.density` directly: that
    field's units depend on whether a `segment_length_m` was supplied
    during M3 aggregation (see `perception/traffic_state.py`'s module
    docstring for the full explanation) -- information the frozen
    `TrafficState` schema has no field to record. `from_traffic_state`
    below recomputes density fresh and unambiguously from `vehicle_count`
    and a segment length the caller supplies explicitly here, so this
    module never has to guess what `TrafficState.density` means.
    """

    camera_id: str
    flow_veh_per_s: float
    density_veh_per_m: float

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if self.flow_veh_per_s < 0:
            raise ValueError(f"flow_veh_per_s must be >= 0, got {self.flow_veh_per_s}")
        if self.density_veh_per_m < 0:
            raise ValueError(f"density_veh_per_m must be >= 0, got {self.density_veh_per_m}")


def flow_density_from_traffic_state(traffic_state: TrafficState, segment_length_m: float) -> FlowDensityState:
    """Build a `FlowDensityState` from a `TrafficState`, requiring a
    calibrated (WORLD) camera and an explicit, manually-known segment
    length -- raises rather than silently using an ambiguous or
    non-metric density (design rule: never manufacture a propagation
    estimate from missing/invalid data)."""
    if traffic_state is None:
        raise ValueError("missing TrafficState -- cannot derive a flow/density state from missing data")
    if traffic_state.motion_space != MotionSpace.WORLD:
        raise ValueError(
            f"camera '{traffic_state.camera_id}' is not calibrated "
            f"(motion_space={traffic_state.motion_space.value}) -- cannot compute a physical shockwave speed"
        )
    if segment_length_m <= 0:
        raise ValueError(f"segment_length_m must be > 0, got {segment_length_m}")

    return FlowDensityState(
        camera_id=traffic_state.camera_id,
        flow_veh_per_s=traffic_state.flow_rate,
        density_veh_per_m=traffic_state.vehicle_count / segment_length_m,
    )


def predict_shockwave_speed(upstream: FlowDensityState, downstream: FlowDensityState) -> Measurement:
    """v_shock = (q2 - q1) / (k2 - k1).

    Rejects the degenerate case (upstream and downstream density
    effectively equal) rather than dividing by ~0 and producing NaN/Inf.
    """
    if upstream is None or downstream is None:
        raise ValueError("missing flow/density state -- cannot compute a shockwave speed from missing data")

    delta_k = downstream.density_veh_per_m - upstream.density_veh_per_m
    if abs(delta_k) < _MIN_DENSITY_DIFFERENCE_VEH_PER_M:
        raise ValueError(
            "degenerate shockwave input: upstream and downstream density are "
            "effectively equal -- no shockwave front to estimate a speed for"
        )

    delta_q = downstream.flow_veh_per_s - upstream.flow_veh_per_s
    v_shock = delta_q / delta_k
    return Measurement(value=v_shock, error=0.0)


def predict_propagation(
    prediction_id: str,
    based_on_ranking_id: str,
    corridor_topology: CorridorTopology,
    upstream_state: TrafficState,
    downstream_state: TrafficState,
    segment_length_m: float,
    reference_timestamp: float,
) -> PropagationPrediction:
    """The full Part 8 pipeline for one corridor: flow/density -> shockwave
    speed -> predicted arrival camera/time/queue-growth-rate.

    `predicted_arrival_camera` follows the sign convention directly:
    negative speed (queue growing upstream) -> the upstream camera is
    predicted to next observe the effect; positive (dissipating
    downstream) -> the downstream camera is. `predicted_arrival_time` is
    `reference_timestamp + inter_camera_distance / |v_shock|` -- V1's
    documented simplification given only two fixed observation points
    (there is no way to know the shockwave front's exact current position
    between them, so the full inter-camera distance is used as the
    remaining travel distance).
    """
    if upstream_state is None or downstream_state is None:
        raise ValueError("missing corridor TrafficState -- cannot predict propagation from missing data")

    upstream_fd = flow_density_from_traffic_state(upstream_state, segment_length_m)
    downstream_fd = flow_density_from_traffic_state(downstream_state, segment_length_m)
    v_shock = predict_shockwave_speed(upstream_fd, downstream_fd)

    speed_magnitude = abs(v_shock.value)
    if speed_magnitude < _MIN_SHOCKWAVE_SPEED_M_PER_S:
        raise ValueError(
            f"shockwave speed ({v_shock.value} m/s) is too small to produce a "
            "meaningful arrival-time estimate"
        )

    arrival_camera = upstream_state.camera_id if v_shock.value < 0 else downstream_state.camera_id
    distance_m = corridor_topology.distance_between_m(upstream_state.camera_id, downstream_state.camera_id)
    arrival_time = reference_timestamp + distance_m / speed_magnitude

    # Conservation of vehicles: the flow discontinuity across the shockwave
    # front approximates the rate vehicles accumulate in (or drain from)
    # the queue -- converted here to vehicles/minute to match Part 5's
    # documented unit for this specific field.
    queue_growth_rate_veh_per_min = abs(downstream_fd.flow_veh_per_s - upstream_fd.flow_veh_per_s) * 60.0

    return PropagationPrediction(
        prediction_id=prediction_id,
        based_on_ranking_id=based_on_ranking_id,
        predicted_shockwave_speed=v_shock,
        predicted_arrival_camera=arrival_camera,
        predicted_arrival_time=arrival_time,
        predicted_queue_growth_rate=queue_growth_rate_veh_per_min,
        method="rankine_hugoniot_kinematic_wave_v1",
    )
