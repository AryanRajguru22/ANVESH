"""Video-overlay integration: lane geometry, approximate speed, and
evidence-backed candidate-vehicle highlighting (forensic audit SS6.5/SS8.3,
plan SS11 Commit 8).

VISUALIZATION/OUTPUT INTEGRATION ONLY. This module draws structured
results that already exist elsewhere -- `world.lane_geometry.
LaneAssignmentResult` (Commit 2/4), `perception.speed_estimation.
VehicleSpeedEstimate` (Commit 6), and `hypotheses.candidate_vehicle.
CandidateVehicleResolution` (Commit 7) -- onto video frames. It computes
NONE of evidence, ranking, candidate resolution, lane assignment, or
speed math itself: every value drawn here is read directly off an
already-computed result object, never recomputed or second-guessed. This
is what keeps "reasoning" and "rendering" strictly separate (design
requirement 10) -- a bug in this module can only misdraw a result, never
change which vehicle is the highest-ranked candidate.

`anvesh.perception.visualize.draw_frame` (the pre-existing plain
bounding-box/track-ID/trail overlay used by `anvesh.perception.pipeline`'s
per-frame M1 tracking loop) is completely untouched by this module --
every existing caller keeps its exact current behavior (design
requirement 9/"existing rendering behavior remains intact"). This module
adds a second, richer overlay, `render_annotated_frame`, for callers that
have the additional per-track lane/speed data and a window-level candidate
resolution available -- which the frame-by-frame M1 loop does not (lane
assignment, speed, and candidate resolution all need track-lifetime or
cross-camera/window information). `render_annotated_frame` is NOT wired
into `anvesh.perception.pipeline`, `anvesh.pipeline`, or any script by
this commit -- see `docs/LANE_SPEED_IMPLEMENTATION_NOTES.md`'s Commit 8
entry for the deliberate stopping point.

Never hard-codes a track_id: the only track ever colored as the
"highest-ranked candidate" is whichever one `CandidateVehicleResolution.
track_id` already names for the given `CandidateOutcome.RANKED` window --
render this module for a different window's resolution and a different
vehicle (or none) is highlighted, automatically.
"""

from __future__ import annotations

import cv2
import numpy as np

from anvesh.hypotheses.candidate_vehicle import CandidateVehicleResolution, CandidateVehicleStatus
from anvesh.perception.speed_estimation import SpeedConfidence, VehicleSpeedEstimate
from anvesh.world.lane_geometry import LaneAssignmentResult, LaneAssignmentStatus

# BGR. Same green tone `perception.visualize.draw_frame` already uses for
# an un-flagged vehicle box -- kept independent (not imported) so this
# module never has to reach into that module's private constants, but
# deliberately matched for visual consistency between the two overlays.
_BOX_COLOR = (60, 200, 60)
_TEXT_COLOR = (255, 255, 255)
_LANE_COLOR = (255, 200, 0)

# BGR translation of the dashboard's EXISTING red-highlight convention
# (`#ef4444`, `ui/live/anvesh_live_dashboard.html`'s `flagged`/safety-event
# styling -- RGB (239, 68, 68)) -- the SAME red, reused here rather than a
# second, unrelated red being invented for the video overlay (design
# requirement 9).
_CANDIDATE_HIGHLIGHT_COLOR = (68, 68, 239)

# BGR amber -- explicit "uncertain", deliberately never red (red is
# reserved for a RESOLVED single vehicle, never for "these vehicles are
# under consideration").
_AMBIGUOUS_BANNER_COLOR = (0, 165, 255)

_LANE_STATUS_TEXT = {
    LaneAssignmentStatus.AMBIGUOUS: "Lane ?",
    LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS: "Lane —",
    LaneAssignmentStatus.UNKNOWN_NO_GEOMETRY: "Lane —",
    LaneAssignmentStatus.INVALID_GEOMETRY: "Lane ?",
}


def format_lane_text(lane_result: LaneAssignmentResult) -> object:
    """`None` (omit the field entirely) only when no lane result was
    computed for this vehicle at all -- distinct from every
    `LaneAssignmentStatus`, each of which always renders something
    explicit (design requirement 8: never silently guess a lane, but also
    never silently drop an uncertain result). `ASSIGNED` shows the bare
    `lane_id` (e.g. `"L1"`); every other status shows its own explicit
    `"Lane ?"`/`"Lane —"` marker, never a guessed lane name.
    """
    if lane_result is None:
        return None
    status = LaneAssignmentStatus(lane_result.status)
    if status == LaneAssignmentStatus.ASSIGNED:
        return lane_result.lane_id
    return _LANE_STATUS_TEXT[status]


def format_speed_text(speed_estimate: VehicleSpeedEstimate) -> object:
    """`None` (omit the field entirely) only when no speed estimate was
    computed for this vehicle at all. When one WAS computed but its
    confidence is `UNAVAILABLE`, that is reported explicitly (`"speed
    —"`) rather than vanishing indistinguishably from "never
    attempted" (design requirement 7's "explicit unavailable state").
    When confidence is `OUTLIER` (a mathematically `OK` speed that fails
    `speed_estimation.MAX_PLAUSIBLE_SPEED_KMH`'s physical-plausibility
    check -- e.g. a c006 ground-point measurement amplified by that
    homography's unvalidated, oversized world-scale into a physically
    impossible value), the raw number is likewise NEVER shown: silently
    printing "~2084 km/h" would present a rejected value as an ordinary
    observation, exactly what this module exists to prevent.

    M-A speed-safety: a km/h figure is rendered ONLY for a `VALIDATED`
    (validated-metric, plausible) estimate, via `metric_speed_kmh`. An
    `APPROXIMATE` estimate comes from an UNVALIDATED world transform (every
    CityFlow-sourced camera): its number is in unknown world-frame units, so
    it is rendered as `"speed unvalidated"`, never as km/h -- not even with an
    "(approx.)" caveat, which would still be a km/h claim. Whole km/h only
    (never `"34.28 km/h"`).
    """
    if speed_estimate is None:
        return None
    if speed_estimate.confidence == SpeedConfidence.UNAVAILABLE:
        return "speed —"
    if speed_estimate.confidence == SpeedConfidence.OUTLIER:
        return "UNRELIABLE — OUTLIER"
    metric = speed_estimate.metric_speed_kmh
    if metric is None:
        return "speed unvalidated"
    return f"~{round(metric.value)} km/h"


def format_vehicle_label(
    track_id: str,
    vehicle_class_value: str,
    lane_text: object = None,
    speed_text: object = None,
    is_candidate: bool = False,
) -> str:
    """Build one vehicle's overlay label, e.g. `"ID 3363 | Car | L1 | ~34
    km/h (approx.)"`. `lane_text`/`speed_text` are whatever
    `format_lane_text`/`format_speed_text` returned -- `None` omits that
    segment entirely (requirement 1: "where information is available").

    `is_candidate` appends an explicit, non-committal marker -- NEVER
    "culprit", "responsible vehicle", or "confirmed vehicle" (requirement
    3) -- and is the ONLY thing this function does with candidate status;
    it does not decide `is_candidate` itself (see `is_resolved_candidate`).
    """
    parts = [f"ID {track_id}", vehicle_class_value.capitalize()]
    if lane_text is not None:
        parts.append(lane_text)
    if speed_text is not None:
        parts.append(speed_text)
    label = " | ".join(parts)
    if is_candidate:
        label += " (Highest-ranked candidate)"
    return label


def is_resolved_candidate(camera_id: str, track_id: str, resolution: CandidateVehicleResolution) -> bool:
    """True only when `resolution` -- the unmodified output of
    `hypotheses.candidate_vehicle.resolve_candidate_vehicle` -- is
    `RESOLVED` and names exactly this vehicle. This is a single string
    comparison against an already-computed result; it never inspects the
    vehicle's own properties (speed, position, class, track age, ...) to
    decide anything (design requirement 10). Never hard-codes a track_id:
    the comparison is always against whatever `resolution.track_id`
    currently holds, so a different window's resolution highlights a
    different vehicle (or none) automatically.
    """
    if resolution is None or resolution.status != CandidateVehicleStatus.RESOLVED:
        return False
    return resolution.track_id == f"{camera_id}:{track_id}"


def ambiguity_banner_text(resolution: CandidateVehicleResolution) -> object:
    """`None` unless `resolution.status == AMBIGUOUS`. Represents the
    ambiguity explicitly (requirement 4) as a frame-level banner rather
    than ever picking one of the candidate vehicles to color red -- no
    per-vehicle box is colored for an `AMBIGUOUS` result anywhere in this
    module.
    """
    if resolution is None or resolution.status != CandidateVehicleStatus.AMBIGUOUS:
        return None
    ids = ", ".join(resolution.candidate_track_ids)
    return f"Candidate ambiguous ({resolution.hypothesis_id}): {ids} -- no single vehicle highlighted"


def render_annotated_frame(
    image: np.ndarray,
    frame_index: int,
    timestamp: float,
    tracked_detections: list,
    camera_id: str,
    lane_regions: tuple = (),
    lane_results_by_track_id: dict = None,
    speed_estimates_by_track_id: dict = None,
    candidate_resolution: CandidateVehicleResolution = None,
) -> np.ndarray:
    """Draw lane regions, per-vehicle ID/class/lane/speed labels, and the
    resolved candidate-vehicle highlight (if any) onto one frame.

    `lane_regions`: exactly the camera's configured `LaneRegion`s (e.g.
    `world.lane_config.load_lane_config(...).regions`) -- ONLY these are
    ever drawn, so an excluded region (a turn pocket, the far-side
    carriageway, ...) that was never included in the camera's lane config
    in the first place is never drawn or labeled either (requirement 2);
    this function has no independent notion of "the road" to draw beyond
    what it is explicitly given.

    `lane_results_by_track_id`/`speed_estimates_by_track_id`: `track_id ->
    LaneAssignmentResult`/`VehicleSpeedEstimate`, keyed by the SAME raw
    (not camera-qualified) `track_id` `tracked_detections` uses. A vehicle
    missing from one of these dicts simply omits that label segment
    (requirement 1); this is independent of whether it is missing from
    the other, and independent of candidate highlighting (requirement 6).

    `candidate_resolution`: the unmodified `CandidateVehicleResolution`
    for this window (or `None`/`NOT_RANKED`/`NO_VEHICLE_FOR_HYPOTHESIS`,
    all of which color every vehicle the same un-flagged green). Never
    recomputed here -- see the module docstring.
    """
    lane_results_by_track_id = lane_results_by_track_id or {}
    speed_estimates_by_track_id = speed_estimates_by_track_id or {}

    annotated = image.copy()

    for region in lane_regions:
        points = np.array([[int(x), int(y)] for x, y in region.vertices], dtype=np.int32)
        cv2.polylines(annotated, [points], isClosed=True, color=_LANE_COLOR, thickness=2)
        label_x, label_y = region.vertices[0]
        cv2.putText(
            annotated, region.lane_id, (int(label_x), int(label_y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, _LANE_COLOR, 2, cv2.LINE_AA,
        )

    for td in tracked_detections:
        x1, y1, x2, y2 = (int(v) for v in td.detection.bbox_xyxy)
        candidate = is_resolved_candidate(camera_id, td.track_id, candidate_resolution)
        color = _CANDIDATE_HIGHLIGHT_COLOR if candidate else _BOX_COLOR
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = format_vehicle_label(
            td.track_id,
            td.detection.vehicle_class.value,
            lane_text=format_lane_text(lane_results_by_track_id.get(td.track_id)),
            speed_text=format_speed_text(speed_estimates_by_track_id.get(td.track_id)),
            is_candidate=candidate,
        )
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    header = f"frame={frame_index} t={timestamp:.2f}s"
    cv2.putText(annotated, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 2, cv2.LINE_AA)

    banner = ambiguity_banner_text(candidate_resolution)
    if banner is not None:
        cv2.putText(
            annotated, banner, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _AMBIGUOUS_BANNER_COLOR, 2, cv2.LINE_AA
        )

    return annotated
