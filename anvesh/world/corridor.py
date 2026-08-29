"""Corridor topology (blueprint Part 3, module 7): the corridor's segment
ordering and known inter-camera distance.

V1 scope is exactly the blueprint's 2-camera corridor: one upstream, one
downstream camera, a known distance between them, on a static (operator-
defined, not learned) topology. This deliberately does not generalize to
a multi-corridor or city-scale road graph (blueprint Part 2, OUT OF SCOPE).
"""

from __future__ import annotations

from dataclasses import dataclass

from anvesh.storage.schemas import CameraRole


@dataclass(frozen=True)
class CameraPlacement:
    """One camera's position along the corridor's centerline."""

    camera_id: str
    role: CameraRole
    distance_from_upstream_m: float

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be a non-empty string")
        object.__setattr__(self, "role", CameraRole(self.role))
        if self.distance_from_upstream_m < 0:
            raise ValueError("distance_from_upstream_m must be >= 0")


@dataclass(frozen=True)
class CorridorTopology:
    """A single corridor's camera ordering -- V1 requires exactly one
    upstream and one downstream camera (the blueprint's 2-camera scope)."""

    corridor_id: str
    placements: tuple

    def __post_init__(self) -> None:
        if not self.corridor_id:
            raise ValueError("corridor_id must be a non-empty string")
        if len(self.placements) != 2:
            raise ValueError("V1 CorridorTopology requires exactly 2 camera placements")

        camera_ids = [p.camera_id for p in self.placements]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("duplicate camera_id in CorridorTopology placements")

        roles = [p.role for p in self.placements]
        if roles.count(CameraRole.UPSTREAM) != 1 or roles.count(CameraRole.DOWNSTREAM) != 1:
            raise ValueError("CorridorTopology requires exactly one upstream and one downstream camera")

        upstream = self._placement_by_role(CameraRole.UPSTREAM)
        downstream = self._placement_by_role(CameraRole.DOWNSTREAM)
        if downstream.distance_from_upstream_m <= upstream.distance_from_upstream_m:
            raise ValueError(
                "the downstream camera must have a strictly greater "
                "distance_from_upstream_m than the upstream camera"
            )

    def _placement_by_role(self, role: CameraRole) -> CameraPlacement:
        return next(p for p in self.placements if p.role == role)

    def placement_for(self, camera_id: str) -> CameraPlacement:
        for p in self.placements:
            if p.camera_id == camera_id:
                return p
        raise KeyError(f"Unknown camera_id in this corridor: {camera_id}")

    def is_upstream_of(self, camera_a: str, camera_b: str) -> bool:
        """True if camera_a sits upstream (a smaller distance-from-upstream) of camera_b."""
        return self.placement_for(camera_a).distance_from_upstream_m < self.placement_for(
            camera_b
        ).distance_from_upstream_m

    def ordered_camera_ids(self) -> list:
        """Camera IDs ordered upstream -> downstream."""
        return [p.camera_id for p in sorted(self.placements, key=lambda p: p.distance_from_upstream_m)]

    def distance_between_m(self, camera_a: str, camera_b: str) -> float:
        return abs(
            self.placement_for(camera_b).distance_from_upstream_m
            - self.placement_for(camera_a).distance_from_upstream_m
        )
