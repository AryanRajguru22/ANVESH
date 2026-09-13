"""Lane-region geometry primitives (blueprint-adjacent; see
`ANVESH_LANE_SPEED_FORENSIC_AUDIT.md` SS8.1/SS11 Commit 2): a deterministic,
auditable representation of a camera's lane regions and point-in-lane
membership testing.

V1 scope: polygonal regions in image (pixel) coordinates, authored per
camera (by config, eventually -- see the audit's Commit 3). This module
does not assign lanes to vehicles, does not know about `RawDetection` or
`ground_point`, does not touch calibration/world-space, and is not wired
into the pipeline. It is a pure, standalone geometry primitive: `(point,
regions) -> LaneAssignmentResult`.

A camera with no lane regions, or a point outside every region, is an
explicit state (`UNKNOWN_NO_GEOMETRY` / `UNKNOWN_OUTSIDE_REGIONS`), never a
guessed lane ID -- following the same explicit-state convention as
`anvesh.world.calibration.CalibrationStatus.UNCALIBRATED` and
`anvesh.safety.stopped_vehicle.StoppedVehicleStatus`. A point that matches
more than one region (overlapping regions, or exactly/nearly on a boundary
shared by two adjacent lanes) is `AMBIGUOUS`, never silently resolved to
whichever region happened to be tested first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# Distance (in the same coordinate units as region vertices -- pixels for
# image-space regions) within which a point is treated as lying ON a lane
# boundary rather than strictly inside/outside it. Deliberately not exactly
# 0.0: floating-point vertex/point coordinates make exact equality an
# unreliable test for "on the boundary", and callers with looser-precision
# geometry (e.g. hand-surveyed polygons) may want a larger value -- pass
# `boundary_tolerance` explicitly to `assign_lane` to override.
DEFAULT_BOUNDARY_TOLERANCE = 1e-6

# Polygon area (shoelace formula, same coordinate units squared) below which
# a polygon is treated as degenerate (collinear/zero-area vertices) and
# rejected at construction.
_MIN_POLYGON_AREA = 1e-9


class LaneAssignmentStatus(str, Enum):
    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    UNKNOWN_NO_GEOMETRY = "unknown_no_geometry"
    UNKNOWN_OUTSIDE_REGIONS = "unknown_outside_regions"
    INVALID_GEOMETRY = "invalid_geometry"


@dataclass(frozen=True)
class LaneRegion:
    """One lane's polygonal region, in image (pixel) coordinates.

    `lane_id` is an explicit, caller-supplied, stable identifier -- never
    inferred from list position (that is exactly V0's `LANE_1`/`LANE_2`
    linear-scan-order defect; see the audit SS1.2). `vertices` supports any
    polygon with >= 3 vertices, not just quadrilaterals.
    """

    lane_id: str
    vertices: tuple

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("lane_id must be a non-empty string")
        if len(self.vertices) < 3:
            raise ValueError(f"a lane region requires at least 3 vertices, got {len(self.vertices)}")
        for vertex in self.vertices:
            if len(vertex) != 2:
                raise ValueError(f"each vertex must be an (x, y) pair, got {vertex!r}")
            x, y = vertex
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"vertex coordinates must be finite, got {vertex!r}")

        area = _polygon_area(self.vertices)
        if area < _MIN_POLYGON_AREA:
            raise ValueError(
                f"lane region '{self.lane_id}' is degenerate (zero-area or collinear vertices): "
                f"{self.vertices!r}"
            )


@dataclass(frozen=True)
class LaneAssignmentResult:
    """Outcome of testing one point against a set of lane regions.

    `lane_id` is set only for `ASSIGNED`. `candidate_lane_ids` carries the
    full sorted set of lane IDs the point matched (`ASSIGNED`: exactly one,
    equal to `lane_id`; `AMBIGUOUS`: two or more; the two `UNKNOWN_*`
    statuses: empty; `INVALID_GEOMETRY`: the sorted duplicate lane_id(s)
    that made the region collection invalid).
    """

    status: LaneAssignmentStatus
    lane_id: object
    candidate_lane_ids: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", LaneAssignmentStatus(self.status))

        if self.status == LaneAssignmentStatus.ASSIGNED:
            if self.lane_id is None or self.candidate_lane_ids != (self.lane_id,):
                raise ValueError("an ASSIGNED result must carry exactly one lane_id, matching candidate_lane_ids")
        elif self.status == LaneAssignmentStatus.AMBIGUOUS:
            if self.lane_id is not None:
                raise ValueError("an AMBIGUOUS result must not carry a single lane_id")
            if len(self.candidate_lane_ids) < 2:
                raise ValueError("an AMBIGUOUS result must carry two or more candidate_lane_ids")
        else:
            if self.lane_id is not None:
                raise ValueError(f"a {self.status.value} result must not carry a lane_id")


def assign_lane(
    point: tuple,
    regions,
    boundary_tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
) -> LaneAssignmentResult:
    """Test `point` (an (x, y) pair, same coordinate space as `regions`'
    vertices) against `regions` and return an explicit, deterministic
    `LaneAssignmentResult`.

    Every region is tested independently (no first-match-wins linear scan,
    unlike V0's `LaneManager.detect_lane` -- see the audit SS1.2); the result
    does not depend on the order `regions` is given in. A point within
    `boundary_tolerance` of a region's edge counts as matching that region.
    """
    regions = tuple(regions)

    if not regions:
        return LaneAssignmentResult(
            status=LaneAssignmentStatus.UNKNOWN_NO_GEOMETRY,
            lane_id=None,
            candidate_lane_ids=(),
        )

    lane_ids = [region.lane_id for region in regions]
    if len(lane_ids) != len(set(lane_ids)):
        duplicates = sorted({lane_id for lane_id in lane_ids if lane_ids.count(lane_id) > 1})
        return LaneAssignmentResult(
            status=LaneAssignmentStatus.INVALID_GEOMETRY,
            lane_id=None,
            candidate_lane_ids=tuple(duplicates),
        )

    matches = tuple(
        sorted(region.lane_id for region in regions if _point_in_region(point, region, boundary_tolerance))
    )

    if len(matches) == 0:
        return LaneAssignmentResult(
            status=LaneAssignmentStatus.UNKNOWN_OUTSIDE_REGIONS,
            lane_id=None,
            candidate_lane_ids=(),
        )
    if len(matches) == 1:
        return LaneAssignmentResult(
            status=LaneAssignmentStatus.ASSIGNED,
            lane_id=matches[0],
            candidate_lane_ids=matches,
        )
    return LaneAssignmentResult(
        status=LaneAssignmentStatus.AMBIGUOUS,
        lane_id=None,
        candidate_lane_ids=matches,
    )


def _point_in_region(point: tuple, region: LaneRegion, boundary_tolerance: float) -> bool:
    if _point_on_boundary(point, region.vertices, boundary_tolerance):
        return True
    return _point_strictly_inside(point, region.vertices)


def _point_on_boundary(point: tuple, vertices: tuple, tolerance: float) -> bool:
    px, py = point
    n = len(vertices)
    for i in range(n):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % n]
        if _distance_point_to_segment(px, py, ax, ay, bx, by) <= tolerance:
            return True
    return False


def _distance_point_to_segment(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _point_strictly_inside(point: tuple, vertices: tuple) -> bool:
    """Even-odd ray-casting test. Deterministic per-polygon; does not by
    itself decide boundary membership -- callers check `_point_on_boundary`
    first, since ray casting's own boundary behaviour is inconsistent
    (a point exactly on a horizontal edge, or exactly at a vertex, can be
    classified inside or outside depending on which edges are scanned)."""
    px, py = point
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > py) != (yj > py):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def _polygon_area(vertices: tuple) -> float:
    n = len(vertices)
    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0
