"""Honest capability registry for every event category and every external integration.

A category appearing in an enum or a dropdown does not make it operational. Each entry
here states: which detector implements it (or None), what inputs it REQUIRES, whether it
has been exercised on real footage, and what the code can and cannot claim. The API
serves this to the dashboard, which shows the state instead of hiding gaps.

STATES (`status`)
  operational        -- detector exists, needs only video; exercised in the test suite.
  experimental       -- detector exists and runs on video, but its false-positive behaviour
                        is not characterised; every output is a candidate for human review.
  needs_input        -- detector exists but cannot run until the camera is configured with
                        something ANVESH cannot invent (direction zones, lanes, a signal
                        feed, turn rules).
  needs_calibration  -- needs VALIDATED metric geometry (`GeometryValidity.WORLD_VALIDATED`).
                        A fitted homography alone does not qualify (M-A safety design).
  needs_model        -- needs a trained classifier that is not shipped/installed. No detector.

`validated_on_real_footage` is True only where a detection of that category was actually
produced from real inference on real footage in this repository, and `real_footage_note`
says which footage and what was and was not established.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class EventCapability:
    event_type: str
    label: str
    detector: str  # dotted name of the implementing detector, or None
    status: str
    requires: tuple  # human-readable input requirements
    output: str  # what an incident of this type carries
    default_severity: str = "low"
    validated_on_real_footage: bool = False
    real_footage_note: str = ""
    enforcement_candidate: bool = False  # may seed an e-challan case
    description: str = ""


CAPABILITIES = {c.event_type: c for c in (
    EventCapability(
        "stopped_vehicle", "Stopped / stalled vehicle", "anvesh.app.detectors.StoppedVehicleDetector", "operational",
        ("video", "optional: validated metric geometry for m/s thresholds"),
        "episode window, dwell, track, image- or metric-space basis", "medium",
        True, "demo.mp4: stop episodes confirmed by the real tracker (displacement criterion). Without a calibration the judgement is "
              "image-space: 'stopped' means the box barely moved on screen. With an APPROXIMATE calibration (the demo camera) the 0.75 m radius is read "
              "on that approximate scale across the whole frame, including regions outside the validated near field.",
        description="A vehicle that stays within a small radius for longer than the dwell threshold. Queued vehicles also stop; "
                    "this is a candidate, not an established incident.",
    ),
    EventCapability(
        "long_duration_stop", "Long-duration stop / obstruction", "anvesh.app.detectors.StoppedVehicleDetector", "operational",
        ("video",), "same episode, escalated once it exceeds the configured long-stop threshold", "high",
        False, "Escalation of a stop episode; requires an episode to last beyond the threshold.",
        description="Escalation of a stopped_vehicle incident once the stop exceeds `long_stop_s`.",
    ),
    EventCapability(
        "congestion", "Congestion / queue formation", "anvesh.app.detectors.CongestionDetector", "operational",
        ("video",), "level history, vehicle counts, flow, trend, candidate-cause ranking when available", "medium",
        True, "demo.mp4: FREE_FLOW -> BUILDING -> CONGESTED -> DISSIPATING on the real tracker. The level is a count/occupancy "
              "proxy (mean concurrency against an ASSUMED capacity), not a measured density; dense but still-flowing traffic can read "
              "'congested', and the incident says so when the window mean speed was high. Free-flow gaps shorter than the close grace are "
              "merged into one stop-and-go episode. Cause ranking is structurally limited without calibration.",
        description="Hysteresis-damped congestion state from the existing traffic-state module.",
    ),
    EventCapability(
        "wrong_way", "Wrong-way movement", "anvesh.app.detectors.WrongWayDetector", "needs_input",
        ("per-zone expected travel direction configured on the camera",), "zone, expected vs observed heading, displacement",
        "high", False,
        "Implemented and unit-tested on synthetic trajectories. Not demonstrated on real footage: no wrong-way vehicle in the "
        "available footage. Direction is image-space and perspective-dependent; zones must be configured per camera.",
        enforcement_candidate=False,
    ),
    EventCapability(
        "wrong_lane", "Wrong-lane movement", "anvesh.app.detectors.WrongLaneDetector", "needs_input",
        ("lane polygons with allowed vehicle classes configured on the camera",), "lane, vehicle class, dwell",
        "medium", False, "Implemented and unit-tested on synthetic trajectories; not demonstrated on real footage.",
        enforcement_candidate=True,
    ),
    EventCapability(
        "overspeed", "Overspeeding", "anvesh.app.detectors.OverspeedDetector", "needs_calibration",
        ("validated metric geometry (GeometryValidity.WORLD_VALIDATED)", "speed limit configured on the camera"),
        "metric speed, limit, margin", "high", False,
        "Gated on validated geometry. A fitted-but-unvalidated homography (e.g. CityFlow c006) yields UNAVAILABLE, never a speed.",
        enforcement_candidate=True,
    ),
    EventCapability(
        "sudden_braking", "Sudden braking / abnormal deceleration", "anvesh.app.detectors.BrakingDetector", "needs_calibration",
        ("validated metric geometry",), "deceleration in m/s^2", "medium", False,
        "Gated on validated geometry; image-space pixel deceleration is perspective-dependent and is not used.",
    ),
    EventCapability(
        "rash_driving", "Aggressive-driving candidate (repeated lane changes)", "anvesh.app.detectors.RapidLaneChangeDetector", "needs_input",
        ("lane polygons configured on the camera",), "lane sequence, number of lane changes, interval", "medium", False,
        "Candidate from repeated lane changes in a short interval; requires configured lanes. Unit-tested on synthetic tracks only.",
    ),
    EventCapability(
        "possible_collision", "Possible collision / accident candidate", "anvesh.app.detectors.CollisionCandidateDetector",
        "experimental", ("video",), "involved track pair, contact interval, pre-contact motion, post-contact stop",
        "high", False,
        "Candidate heuristic (approach, sustained box contact, both vehicles stop). No collision exists in the available "
        "footage, so it is NOT validated; false-positive rate in dense traffic is reported by the demo run. Every output "
        "requires human review and states that contact is inferred from 2-D boxes only.",
        description="Two vehicles moving, boxes overlapping, then both stopped. Proximity is not causation.",
    ),
    EventCapability(
        "post_incident_disruption", "Post-incident traffic disruption", "anvesh.app.incidents.IncidentService", "operational",
        ("an incident and a congestion incident on the same camera",), "temporal association link between incidents", "medium",
        False, "A temporal association only; it never asserts that the first incident caused the congestion.",
    ),
    EventCapability(
        "red_light_violation", "Red-light violation", "anvesh.app.detectors.RedLightDetector", "needs_input",
        ("stop line configured on the camera", "a signal-state feed (external or configured schedule)"),
        "stop line, signal state at crossing, crossing time", "high", False,
        "Implemented and unit-tested with a fixture signal. ANVESH does not read the signal colour from video, so without a "
        "signal feed the detector cannot run. A configured schedule is an ASSUMPTION about the junction, not an observation.",
        enforcement_candidate=True,
    ),
    EventCapability(
        "illegal_turn", "Illegal turn", "anvesh.app.detectors.IllegalTurnDetector", "needs_input",
        ("entry/exit zones and the prohibited movements configured on the camera",), "entry zone, exit zone, track",
        "medium", False, "Implemented and unit-tested on synthetic trajectories; not demonstrated on real footage.",
        enforcement_candidate=True,
    ),
    EventCapability(
        "helmet_violation", "Helmet non-compliance", None, "needs_model",
        ("a trained rider/helmet classifier (not shipped)", "sufficient image resolution on the rider"), "-", "low",
        False, "No detector. COCO YOLOv8n does not distinguish helmets. Do not treat any output as available.",
        enforcement_candidate=True,
    ),
    EventCapability(
        "seatbelt_violation", "Seatbelt non-compliance", None, "needs_model",
        ("a trained seatbelt classifier (not shipped)", "windscreen-level camera view"), "-", "low",
        False, "No detector. Not available.", enforcement_candidate=True,
    ),
)}


@dataclasses.dataclass(frozen=True)
class IntegrationStatus:
    key: str
    label: str
    state: str  # connected | sandbox | not_connected | unavailable
    detail: str
    setup: str = ""


# How each category is actually produced. NOTHING here is decided by a trained event model: the only learned component in the
# pipeline is the vehicle detector (YOLO) that supplies boxes; every event is a rule over tracked boxes.
IMPLEMENTATION_CLASSES = ("implemented_heuristic", "trained_model", "integration_dependent", "not_implemented")
_INTEGRATION_DEPENDENT = frozenset({"overspeed", "sudden_braking", "red_light_violation"})  # need survey calibration / an external signal feed


def implementation_class(event_type: str) -> str:
    """implemented_heuristic | trained_model | integration_dependent | not_implemented."""
    cap = CAPABILITIES[event_type]
    if cap.detector is None:
        return "not_implemented"
    if event_type in _INTEGRATION_DEPENDENT:
        return "integration_dependent"
    return "implemented_heuristic"


def category_status(event_type: str, camera_config=None, geometry_validated: bool = False) -> dict:
    """Whether `event_type` can run on THIS camera right now, and what is missing."""
    cap = CAPABILITIES[event_type]
    missing = []
    available = cap.status in ("operational", "experimental")
    cfg = camera_config
    if cap.status == "needs_model":
        available = False
        missing = list(cap.requires)
    elif cap.status == "needs_calibration":
        if not geometry_validated:
            missing.append("validated metric geometry (a fitted homography alone is not enough)")
        if event_type == "overspeed" and (cfg is None or cfg.speed_limit_kmh is None):
            missing.append("speed_limit_kmh")
        available = not missing
    elif cap.status == "needs_input":
        if cfg is None:
            missing = list(cap.requires)
        elif event_type == "wrong_way" and not cfg.direction_zones:
            missing.append("direction_zones")
        elif event_type in ("wrong_lane", "rash_driving") and not cfg.lanes:
            missing.append("lanes")
        elif event_type == "wrong_lane" and not any(l.get("allowed_classes") for l in cfg.lanes):
            missing.append("lane allowed_classes")
        elif event_type == "red_light_violation" and not (cfg.signal and cfg.signal.get("stop_line")):
            missing.append("signal.stop_line + signal state feed")
        elif event_type == "illegal_turn" and not cfg.turn_rules:
            missing.append("turn_rules")
        available = not missing
    return {
        "event_type": event_type, "label": cap.label, "status": cap.status, "available": available,
        "missing": missing, "requires": list(cap.requires), "output": cap.output,
        "validated_on_real_footage": cap.validated_on_real_footage, "real_footage_note": cap.real_footage_note,
        "detector": cap.detector, "enforcement_candidate": cap.enforcement_candidate,
        "description": cap.description, "default_severity": cap.default_severity,
        "implementation_class": implementation_class(event_type),
    }


def all_category_status(camera_config=None, geometry_validated: bool = False) -> list:
    return [category_status(t, camera_config, geometry_validated) for t in CAPABILITIES]
