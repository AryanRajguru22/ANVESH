import numpy as np
import pytest

from anvesh.evidence.evidence_builder import build_evidence_for_camera
from anvesh.fusion.ds_fusion import single_camera_result
from anvesh.hypotheses.candidate_vehicle import (
    CandidateVehicleResolution,
    CandidateVehicleStatus,
    resolve_candidate_vehicle,
)
from anvesh.hypotheses.ranking import build_ranking
from anvesh.perception.speed_estimation import SpeedConfidence, VehicleSpeedEstimate, estimate_vehicle_speed
from anvesh.perception.track_manager import TrackState
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.perception.types import RawDetection, TrackedDetection
from anvesh.rendering import (
    ambiguity_banner_text,
    format_lane_text,
    format_speed_text,
    format_vehicle_label,
    is_resolved_candidate,
    render_annotated_frame,
)
from anvesh.storage.schemas import (
    CameraRole,
    CandidateOutcome,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
)
from anvesh.world.calibration import from_cityflow_homography
from anvesh.world.lane_geometry import LaneAssignmentResult, LaneAssignmentStatus, LaneRegion, assign_lane

WINDOW = (0.0, 10.0)
CAMERA_ID = "cam-A"


def _td(track_id, bbox):
    detection = RawDetection(
        frame_index=0, timestamp=0.0, class_name="car", vehicle_class=VehicleClass.CAR,
        confidence=0.9, bbox_xyxy=bbox,
    )
    return TrackedDetection(detection=detection, track_id=track_id)


def _resolution(status, track_id=None, candidate_track_ids=(), hypothesis_id="H2"):
    return CandidateVehicleResolution(
        status=status,
        hypothesis_id=hypothesis_id,
        belief=0.8,
        plausibility=0.9,
        supporting_evidence_refs=(f"{CAMERA_ID}:{hypothesis_id}:0-10",),
        candidate_track_ids=candidate_track_ids,
        track_id=track_id,
    )


RED = (68, 68, 239)
GREEN = (60, 200, 60)


def _box_edge_pixel(annotated, bbox):
    x1, _y1, x2, y2 = bbox
    # bottom edge -- clear of the label text drawn above the box
    return tuple(int(c) for c in annotated[y2, (x1 + x2) // 2])


# ---------------------------------------------------------------------------
# a: normal vehicle receives ID + lane + approximate speed
# ---------------------------------------------------------------------------


def test_normal_vehicle_label_has_id_lane_and_speed():
    lane_result = LaneAssignmentResult(status=LaneAssignmentStatus.ASSIGNED, lane_id="L1", candidate_lane_ids=("L1",))
    speed = VehicleSpeedEstimate(
        track_id=f"{CAMERA_ID}:1", camera_id=CAMERA_ID, status="ok", confidence=SpeedConfidence.APPROXIMATE,
        measurement=Measurement(value=34.28, error=1.0),
    )
    label = format_vehicle_label(
        "3363", "car", lane_text=format_lane_text(lane_result), speed_text=format_speed_text(speed)
    )
    assert label == "ID 3363 | Car | L1 | ~34 km/h (approx.)"


# ---------------------------------------------------------------------------
# b: resolved candidate is red
# ---------------------------------------------------------------------------


def test_resolved_candidate_is_highlighted_red():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td = _td("1", (10, 10, 30, 30))
    resolution = _resolution(CandidateVehicleStatus.RESOLVED, track_id=f"{CAMERA_ID}:1", candidate_track_ids=(f"{CAMERA_ID}:1",))

    annotated = render_annotated_frame(image, 0, 0.0, [td], CAMERA_ID, candidate_resolution=resolution)

    assert _box_edge_pixel(annotated, (10, 10, 30, 30)) == RED


def test_non_candidate_vehicle_stays_green_even_when_a_candidate_is_resolved():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td1 = _td("1", (10, 10, 30, 30))
    td2 = _td("2", (50, 10, 70, 30))
    resolution = _resolution(CandidateVehicleStatus.RESOLVED, track_id=f"{CAMERA_ID}:1", candidate_track_ids=(f"{CAMERA_ID}:1",))

    annotated = render_annotated_frame(image, 0, 0.0, [td1, td2], CAMERA_ID, candidate_resolution=resolution)

    assert _box_edge_pixel(annotated, (10, 10, 30, 30)) == RED
    assert _box_edge_pixel(annotated, (50, 10, 70, 30)) == GREEN


# ---------------------------------------------------------------------------
# c: candidate changes between windows -> red highlight changes
# ---------------------------------------------------------------------------


def test_red_highlight_moves_when_resolution_changes():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td1 = _td("1", (10, 10, 30, 30))
    td2 = _td("2", (50, 10, 70, 30))

    resolution_window_1 = _resolution(CandidateVehicleStatus.RESOLVED, track_id=f"{CAMERA_ID}:1", candidate_track_ids=(f"{CAMERA_ID}:1",))
    resolution_window_2 = _resolution(CandidateVehicleStatus.RESOLVED, track_id=f"{CAMERA_ID}:2", candidate_track_ids=(f"{CAMERA_ID}:2",))

    frame1 = render_annotated_frame(image, 0, 0.0, [td1, td2], CAMERA_ID, candidate_resolution=resolution_window_1)
    frame2 = render_annotated_frame(image, 1, 1.0, [td1, td2], CAMERA_ID, candidate_resolution=resolution_window_2)

    assert _box_edge_pixel(frame1, (10, 10, 30, 30)) == RED
    assert _box_edge_pixel(frame1, (50, 10, 70, 30)) == GREEN

    assert _box_edge_pixel(frame2, (10, 10, 30, 30)) == GREEN
    assert _box_edge_pixel(frame2, (50, 10, 70, 30)) == RED


# ---------------------------------------------------------------------------
# d: ambiguous candidate -> no arbitrary red selection
# ---------------------------------------------------------------------------


def test_ambiguous_candidate_never_arbitrarily_highlights_one_vehicle():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td1 = _td("1", (10, 10, 30, 30))
    td2 = _td("2", (50, 10, 70, 30))
    resolution = _resolution(
        CandidateVehicleStatus.AMBIGUOUS, track_id=None,
        candidate_track_ids=(f"{CAMERA_ID}:1", f"{CAMERA_ID}:2"),
    )

    annotated = render_annotated_frame(image, 0, 0.0, [td1, td2], CAMERA_ID, candidate_resolution=resolution)

    assert _box_edge_pixel(annotated, (10, 10, 30, 30)) == GREEN
    assert _box_edge_pixel(annotated, (50, 10, 70, 30)) == GREEN
    banner = ambiguity_banner_text(resolution)
    assert banner is not None
    assert f"{CAMERA_ID}:1" in banner and f"{CAMERA_ID}:2" in banner


# ---------------------------------------------------------------------------
# e/f: no vehicle for hypothesis / not ranked -> no red
# ---------------------------------------------------------------------------


def test_no_vehicle_for_hypothesis_never_highlights_anything():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td = _td("1", (10, 10, 30, 30))
    resolution = _resolution(CandidateVehicleStatus.NO_VEHICLE_FOR_HYPOTHESIS, hypothesis_id="H1")

    annotated = render_annotated_frame(image, 0, 0.0, [td], CAMERA_ID, candidate_resolution=resolution)

    assert _box_edge_pixel(annotated, (10, 10, 30, 30)) == GREEN
    assert ambiguity_banner_text(resolution) is None


def test_not_ranked_never_highlights_anything():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td = _td("1", (10, 10, 30, 30))
    resolution = CandidateVehicleResolution(
        status=CandidateVehicleStatus.NOT_RANKED, hypothesis_id=None, belief=None, plausibility=None,
        supporting_evidence_refs=(), candidate_track_ids=(), track_id=None,
    )

    annotated = render_annotated_frame(image, 0, 0.0, [td], CAMERA_ID, candidate_resolution=resolution)

    assert _box_edge_pixel(annotated, (10, 10, 30, 30)) == GREEN


def test_no_resolution_at_all_never_highlights_anything():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    td = _td("1", (10, 10, 30, 30))
    annotated = render_annotated_frame(image, 0, 0.0, [td], CAMERA_ID, candidate_resolution=None)
    assert _box_edge_pixel(annotated, (10, 10, 30, 30)) == GREEN
    assert is_resolved_candidate(CAMERA_ID, "1", None) is False


# ---------------------------------------------------------------------------
# g: lane ambiguity / uncertainty displayed explicitly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (LaneAssignmentStatus.AMBIGUOUS, "Lane ?"),
        (LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS, "Lane —"),
        (LaneAssignmentStatus.UNKNOWN_NO_GEOMETRY, "Lane —"),
        (LaneAssignmentStatus.INVALID_GEOMETRY, "Lane ?"),
    ],
)
def test_lane_uncertainty_is_shown_explicitly_never_guessed(status, expected):
    candidate_ids = ("L1", "L2") if status in (LaneAssignmentStatus.AMBIGUOUS,) else ()
    result = LaneAssignmentResult(status=status, lane_id=None, candidate_lane_ids=candidate_ids)
    assert format_lane_text(result) == expected


def test_lane_assigned_shows_bare_lane_id():
    result = LaneAssignmentResult(status=LaneAssignmentStatus.ASSIGNED, lane_id="L1", candidate_lane_ids=("L1",))
    assert format_lane_text(result) == "L1"


def test_lane_omitted_entirely_when_no_result_supplied():
    assert format_lane_text(None) is None


# ---------------------------------------------------------------------------
# h: speed unavailable is handled explicitly
# ---------------------------------------------------------------------------


def test_speed_unavailable_is_shown_explicitly_not_silently_dropped():
    estimate = VehicleSpeedEstimate(
        track_id=f"{CAMERA_ID}:1", camera_id=CAMERA_ID, status="calibration_unavailable",
        confidence=SpeedConfidence.UNAVAILABLE, measurement=None,
    )
    assert format_speed_text(estimate) == "speed —"


def test_speed_omitted_entirely_when_no_estimate_supplied_at_all():
    assert format_speed_text(None) is None


# ---------------------------------------------------------------------------
# i: approximate speed never gets excessive precision
# ---------------------------------------------------------------------------


def test_speed_text_never_shows_excessive_decimal_precision():
    estimate = VehicleSpeedEstimate(
        track_id=f"{CAMERA_ID}:1", camera_id=CAMERA_ID, status="ok", confidence=SpeedConfidence.APPROXIMATE,
        measurement=Measurement(value=34.284719, error=2.1),
    )
    text = format_speed_text(estimate)
    assert text == "~34 km/h (approx.)"
    assert "." not in text.split("km/h")[0]


def test_validated_speed_has_no_approx_suffix():
    estimate = VehicleSpeedEstimate(
        track_id=f"{CAMERA_ID}:1", camera_id=CAMERA_ID, status="ok", confidence=SpeedConfidence.VALIDATED,
        measurement=Measurement(value=50.0, error=0.5),
    )
    assert format_speed_text(estimate) == "~50 km/h"


# ---------------------------------------------------------------------------
# j: excluded regions do not receive lane labels
# ---------------------------------------------------------------------------


def test_only_configured_lane_regions_are_drawn():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    l1 = LaneRegion(lane_id="L1", vertices=((10, 10), (50, 10), (50, 50), (10, 50)))
    # a location that would be inside an EXCLUDED turn-pocket region, were
    # one configured -- since it is never passed in `lane_regions`, no
    # lane color should ever appear there.
    annotated = render_annotated_frame(image, 0, 0.0, [], CAMERA_ID, lane_regions=(l1,))

    lane_color = (255, 200, 0)
    assert tuple(annotated[10, 30]) == lane_color  # on L1's drawn top edge
    assert tuple(annotated[150, 150]) == (0, 0, 0)  # far outside any configured region -- untouched


# ---------------------------------------------------------------------------
# lane/speed independent of candidate highlighting (requirement 6)
# ---------------------------------------------------------------------------


def test_lane_and_speed_labels_are_independent_of_candidate_status():
    lane_result = LaneAssignmentResult(status=LaneAssignmentStatus.ASSIGNED, lane_id="L1", candidate_lane_ids=("L1",))
    speed = VehicleSpeedEstimate(
        track_id=f"{CAMERA_ID}:1", camera_id=CAMERA_ID, status="ok", confidence=SpeedConfidence.APPROXIMATE,
        measurement=Measurement(value=35.0, error=1.0),
    )
    label_not_candidate = format_vehicle_label(
        "1", "car", lane_text=format_lane_text(lane_result), speed_text=format_speed_text(speed), is_candidate=False
    )
    label_candidate = format_vehicle_label(
        "1", "car", lane_text=format_lane_text(lane_result), speed_text=format_speed_text(speed), is_candidate=True
    )
    assert label_not_candidate == "ID 1 | Car | L1 | ~35 km/h (approx.)"
    assert label_candidate == "ID 1 | Car | L1 | ~35 km/h (approx.) (Highest-ranked candidate)"


def test_candidate_label_never_uses_culprit_or_confirmed_wording():
    label = format_vehicle_label("1", "car", is_candidate=True)
    for forbidden in ("culprit", "responsible", "confirmed"):
        assert forbidden not in label.lower()
    assert "highest-ranked candidate" in label.lower()


# ---------------------------------------------------------------------------
# 12: end-to-end fixture -- tracked vehicle -> lane -> speed -> evidence/
# ranking -> candidate resolution -> rendered output
# ---------------------------------------------------------------------------


def test_end_to_end_stopped_vehicle_is_rendered_as_resolved_candidate():
    raw_track_id = "1"
    qualified_track_id = f"{CAMERA_ID}:{raw_track_id}"

    # -- lane geometry: one lane region containing the vehicle's ground point --
    lane_region = LaneRegion(lane_id="L1", vertices=((0, 400), (50, 400), (50, 600), (0, 600)))
    ground_point = (10.0, 500.0)
    lane_result = assign_lane(ground_point, (lane_region,))
    assert lane_result.status == LaneAssignmentStatus.ASSIGNED
    assert lane_result.lane_id == "L1"

    # -- approximate speed: a stationary ground-point trajectory through an
    # unvalidated (CityFlow-style) calibration --
    calibration_profile = from_cityflow_homography(
        "calib-cf", CAMERA_ID, np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]), reprojection_error=5.0
    )
    track_state = TrackState(
        track_id=raw_track_id, vehicle_class=VehicleClass.CAR, first_seen=0.0, last_seen=0.4,
        occlusion_state=OcclusionState.VISIBLE,
        trajectory=[(*ground_point, t) for t in (0.0, 0.1, 0.2, 0.3, 0.4)],
        ground_point_trajectory=[(*ground_point, t) for t in (0.0, 0.1, 0.2, 0.3, 0.4)],
    )
    speed_estimate = estimate_vehicle_speed(CAMERA_ID, track_state, calibration_profile)
    assert speed_estimate.confidence == SpeedConfidence.APPROXIMATE
    assert speed_estimate.measurement.value == pytest.approx(0.0, abs=1e-6)

    # -- evidence + ranking: one sustained-stopped vehicle should support H2 --
    schema_track = VehicleTrack(
        track_id=qualified_track_id, camera_id=CAMERA_ID, first_seen=0.0, last_seen=4.0,
        vehicle_class=VehicleClass.CAR, occlusion_state=OcclusionState.VISIBLE, position_history=[],
        speed_estimate=Measurement(value=0.0, error=0.0), motion_space=MotionSpace.WORLD,
    )
    aggregation = aggregate_traffic_state(CAMERA_ID, *WINDOW, [schema_track], MotionSpace.WORLD, CongestionStateTracker())
    evidence = build_evidence_for_camera(CAMERA_ID, WINDOW, CameraRole.DOWNSTREAM, aggregation, [schema_track], reliability=0.9)
    evidence_by_camera = {CAMERA_ID: evidence}
    fusion = single_camera_result(CAMERA_ID, evidence, WINDOW)
    ranking = build_ranking("rank-1", "cs-1", fusion, evidence_by_camera)
    assert ranking.outcome == CandidateOutcome.RANKED
    assert ranking.ranked_list[0].hypothesis_id == "H2"

    # -- candidate resolution: consumed as-is, never recomputed by the renderer --
    resolution = resolve_candidate_vehicle(ranking, evidence_by_camera)
    assert resolution.status == CandidateVehicleStatus.RESOLVED
    assert resolution.track_id == qualified_track_id

    # -- render --
    image = np.zeros((700, 100, 3), dtype=np.uint8)
    bbox = (0.0, 480.0, 20.0, 500.0)  # bottom-center == ground_point (10, 500)
    detection = RawDetection(
        frame_index=0, timestamp=0.4, class_name="car", vehicle_class=VehicleClass.CAR,
        confidence=0.95, bbox_xyxy=bbox,
    )
    td = TrackedDetection(detection=detection, track_id=raw_track_id)

    annotated = render_annotated_frame(
        image, 0, 0.4, [td], CAMERA_ID,
        lane_regions=(lane_region,),
        lane_results_by_track_id={raw_track_id: lane_result},
        speed_estimates_by_track_id={raw_track_id: speed_estimate},
        candidate_resolution=resolution,
    )

    assert _box_edge_pixel(annotated, (0, 480, 20, 500)) == RED

    label = format_vehicle_label(
        raw_track_id, detection.vehicle_class.value,
        lane_text=format_lane_text(lane_result),
        speed_text=format_speed_text(speed_estimate),
        is_candidate=is_resolved_candidate(CAMERA_ID, raw_track_id, resolution),
    )
    assert label == "ID 1 | Car | L1 | ~0 km/h (approx.) (Highest-ranked candidate)"
