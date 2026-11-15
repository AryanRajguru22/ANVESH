"""Event-logic regressions: stopped vs queued traffic, duplicate events, incident lifecycle, direction boundaries, and the
classification of every event category.

Scenarios are SIMULATED (scripted tracks), never real footage. They establish that the rules behave as designed, not how often they
are right on a real road: severity labels such as 'low because queued' are heuristics whose thresholds are untuned.
"""

import math

import pytest

from anvesh.app.capabilities import CAPABILITIES, IMPLEMENTATION_CLASSES, category_status, implementation_class
from anvesh.app.detectors import DetectorContext, TrackHistory, WrongWayDetector
from anvesh.app.cameras import ConfigError, parse_camera_config

from .conftest import seed_incident
from .test_detectors import ZONE, drive, make, obs
from .test_session_synthetic import run_scenario


# -- stopped vehicle vs ordinary queued traffic -----------------------------------------------------------------------


def _stop_incidents(services, runner):
    rows, _ = services.incidents.list({"session_id": runner.session_id}, limit=200)
    return [r for r in rows if r["event_type"] in ("stopped_vehicle", "long_duration_stop")]


def test_a_lone_stationary_vehicle_among_moving_traffic_is_the_obstruction_candidate(services):
    _, runner = run_scenario(services, "stopped_car", {"stop_criterion": "displacement"})
    stops = _stop_incidents(services, runner)
    assert len(stops) == 1
    assert stops[0]["severity"] == "medium" and stops[0]["indicators"]["queue_signal"] is None
    assert stops[0]["indicators"]["stationary_vehicles_in_window"] == 1


def test_several_vehicles_stopped_together_are_treated_as_a_queue_not_as_eight_obstructions(services):
    """8 cars stationary for 40 s on a camera whose congestion proxy still reads free-flow (default capacity 20). Before the
    stationary-group rule this produced eight HIGH 'long stop outside congested traffic' incidents."""
    _, runner = run_scenario(services, "queue")
    stops = _stop_incidents(services, runner)
    assert len(stops) == 8
    for inc in stops:
        assert inc["severity"] == "low", inc["severity_basis"]
        assert inc["indicators"]["queue_signal"] == "stationary_group" and inc["indicators"]["stationary_vehicles_in_window"] >= 3
        assert any("queue" in u.lower() for u in inc["uncertainty"])
        assert "check the clip" in " ".join(inc["uncertainty"])  # the head-of-queue obstruction caveat is stated, not hidden
    # still raised (a human must be able to see them) and still one incident per vehicle - no duplicates from window overlap
    assert len({tuple(i["track_ids"]) for i in stops}) == 8
    assert len({i["natural_key"] for i in stops}) == 8


def test_the_group_size_is_configurable_and_a_higher_value_restores_the_obstruction_reading(services):
    _, runner = run_scenario(services, "queue", {"thresholds": {"queue_peer_stops": 9}})
    stops = _stop_incidents(services, runner)
    assert stops and all(i["severity"] == "high" and i["indicators"]["queue_signal"] is None for i in stops)


def test_queue_peer_stops_is_validated_like_every_other_threshold():
    assert parse_camera_config({"thresholds": {"queue_peer_stops": 4}}).threshold("queue_peer_stops", 3) == 4
    for bad in (0, -1, True, "3", None):
        with pytest.raises(ConfigError):
            parse_camera_config({"thresholds": {"queue_peer_stops": bad}})


# -- direction boundaries (wrong way) ---------------------------------------------------------------------------------


@pytest.mark.parametrize("angle_deg,fires", [(0, False), (90, False), (120, False), (134, False), (140, True), (180, True), (-180, True)])
def test_wrong_way_direction_boundary_in_degrees_from_the_expected_direction(angle_deg, fires):
    """ZONE expects travel along +x. The rule fires only for travel more than ~135 degrees against it (cos < -0.7)."""
    det, ctx = make(WrongWayDetector, ZONE)
    a = math.radians(angle_deg)
    # 30 px/s from the middle of the zone: long enough (>= min_displacement and min_duration inside the zone) for the rule to be evaluated
    frames = [(i / 10, [obs("v", 320 + 30 * math.cos(a) * (i / 10), 220 + 30 * math.sin(a) * (i / 10))]) for i in range(60)]
    events = drive(det, ctx, frames)
    assert (len(events) == 1) == fires, [e.indicators for e in events]


def test_a_vehicle_leaving_the_zone_and_a_second_one_with_a_new_id_nearby_do_not_double_report():
    det, ctx = make(WrongWayDetector, ZONE)
    frames = []
    for i in range(120):
        t = i / 10
        tid = "a" if i < 50 else ("b" if i < 90 else "c")  # two identity switches on the same vehicle
        frames.append((t, [obs(tid, 650 - 45 * t, 260)]))
    assert len(drive(det, ctx, frames)) == 1


# -- incident lifecycle -------------------------------------------------------------------------------------------------


def test_incident_status_machine_rejects_illegal_moves_and_never_reopens_a_closed_incident(services, users):
    from anvesh.app.incidents import STATUS_TRANSITIONS

    inc = seed_incident(services)
    iid = inc["incident_id"]
    with pytest.raises(ValueError):
        services.incidents.change_status(iid, "closed", "inv1", "investigator")  # detected -> closed is not allowed
    with pytest.raises(ValueError):
        services.incidents.change_status(iid, "banana", "inv1", "investigator")
    services.incidents.change_status(iid, "under_investigation", "inv1", "investigator")
    services.incidents.change_status(iid, "closed", "inv1", "investigator", "resolved")
    for target in STATUS_TRANSITIONS["closed"] | {"detected", "under_investigation", "confirmed"}:
        with pytest.raises(ValueError):
            services.incidents.change_status(iid, target, "inv1", "investigator")
    assert STATUS_TRANSITIONS["closed"] == set()
    kinds = [t["kind"] for t in services.incidents.get(iid, with_timeline=True)["timeline"]]
    assert kinds.count("detected") == 1 and "status" in kinds


def test_a_dismissed_incident_cannot_seed_a_case_but_can_be_reopened_for_investigation(services, users):
    from anvesh.app.cases import CaseError

    inc = seed_incident(services)
    iid = inc["incident_id"]
    services.incidents.change_status(iid, "dismissed", "inv1", "investigator", "false alarm")
    with pytest.raises(CaseError) as e:
        services.cases.create(iid, "STOP-OBS", users["inv1"])
    assert e.value.status == 409
    services.incidents.change_status(iid, "under_investigation", "inv1", "investigator", "reopened")
    assert services.cases.create(iid, "STOP-OBS", users["inv1"])["state"] == "detected"


# -- classification of every event category -----------------------------------------------------------------------------------


def test_every_event_category_has_exactly_one_honest_implementation_class():
    classes = {t: implementation_class(t) for t in CAPABILITIES}
    assert set(classes.values()) <= set(IMPLEMENTATION_CLASSES)
    # no event category is decided by a trained event model - the only learned component is the vehicle detector
    assert "trained_model" not in classes.values()
    assert classes["helmet_violation"] == "not_implemented" and classes["seatbelt_violation"] == "not_implemented"
    assert classes["overspeed"] == classes["sudden_braking"] == classes["red_light_violation"] == "integration_dependent"
    for t in ("stopped_vehicle", "long_duration_stop", "congestion", "possible_collision", "wrong_way", "wrong_lane", "illegal_turn", "rash_driving"):
        assert classes[t] == "implemented_heuristic", t
    assert all(category_status(t)["implementation_class"] == classes[t] for t in CAPABILITIES)
    # the class and the availability must agree: nothing 'not implemented' can ever report itself available
    for t, c in classes.items():
        if c == "not_implemented":
            assert category_status(t)["available"] is False


def test_nothing_speed_based_is_available_without_validated_geometry():
    for t in ("overspeed", "sudden_braking"):
        st = category_status(t, parse_camera_config({"speed_limit_kmh": 50}), geometry_validated=False)
        assert st["available"] is False and any("validated metric geometry" in m for m in st["missing"])
        assert st["validated_on_real_footage"] is False
