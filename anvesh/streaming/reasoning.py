"""Bridge from a streaming `WindowSnapshot` to the EXISTING reasoning layer
(`aggregate_traffic_state` -> `run_corridor_window`). No reasoning logic lives
here: the same functions the batch path called per fixed window are called
per streamed window, so a window means the same thing for a recorded file and
a live stream.

The reasoner owns the camera's `CongestionStateTracker` (hysteresis needs one
persistent instance across consecutive windows) and therefore refuses
overlapping windows -- see `anvesh.streaming.windows` for why.

OBSERVABILITY (M-A.1). The reasoner trusts a window only as far as the camera
delivered it:
  - UNAVAILABLE (the window was effectively not observed, see
    `WindowAvailability`): NOTHING is reasoned. No traffic state, no ranking, no
    safety events -- in particular no "empty road / free flow" -- and the
    congestion hysteresis memory is reset, because the level before an outage says
    nothing about the road after it. The analysis carries `status=UNAVAILABLE`.
  - DEGRADED (some frames/seconds lost, enough observed): reasoned, but the result
    is flagged and says its counts are lower bounds. The hysteresis state machine
    is advanced only if at least `min_update_coverage` of the window was observed;
    otherwise the window is analysed against a COPY of the tracker so a thin window
    cannot, for instance, step CONGESTED to DISSIPATING on its own.
  - OBSERVED, with zero vehicles: a healthy empty scene (`update_empty`), which is
    legitimately free flow.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
from dataclasses import dataclass

from anvesh.perception.schema_conversion import motion_space_for
from anvesh.perception.traffic_state import CongestionStateTracker, aggregate_traffic_state
from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.safety.stop_episodes import build_stop_event
from anvesh.storage.schemas import CameraRole


class AnalysisStatus(str, enum.Enum):
    OBSERVED = "observed"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class WindowAnalysis:
    snapshot: object  # WindowSnapshot
    aggregation: object  # perception.traffic_state.TrafficStateAggregation; None if UNAVAILABLE
    result: object  # pipeline.AnveshWindowResult; None if UNAVAILABLE
    degraded: bool  # the window is not fully observed (frames lost / silence / absent)
    status: AnalysisStatus = AnalysisStatus.OBSERVED
    reason: str = ""  # why the window is degraded / unavailable
    congestion_advanced: bool = True  # False if the hysteresis state was NOT updated by this window
    stop_episodes: tuple = ()  # stop episodes touching the window (see safety.stop_episodes)
    stop_events: tuple = ()  # SafetyEvents for episodes OPENED in this window (also when UNAVAILABLE)

    @property
    def unavailable(self) -> bool:
        return self.status == AnalysisStatus.UNAVAILABLE


class WindowReasoner:
    def __init__(
        self,
        camera_id: str,
        role: CameraRole = CameraRole.UPSTREAM,
        calibration_profile=None,
        max_capacity_vehicles: int = 20,
        congestion_tracker: CongestionStateTracker = None,
        segment_length_m: float = None,
        min_update_coverage: float = 0.9,
        **run_corridor_window_kwargs,
    ) -> None:
        if not (0.0 < min_update_coverage <= 1.0):
            raise ValueError("min_update_coverage must be within (0, 1]")
        self._min_update_coverage = min_update_coverage
        self.camera_id = camera_id
        self._role = role
        self._profile = calibration_profile
        self._max_capacity = max_capacity_vehicles
        self._tracker = congestion_tracker or CongestionStateTracker()
        self._segment_length_m = segment_length_m
        self._kwargs = run_corridor_window_kwargs
        self._last_end = None

    @property
    def congestion_state(self):
        """The hysteresis tracker's current level (read-only)."""
        return self._tracker.state

    def process(self, snapshot) -> WindowAnalysis:
        if snapshot.camera_id != self.camera_id:
            raise ValueError(f"snapshot is for camera {snapshot.camera_id!r}, reasoner for {self.camera_id!r}")
        if self._last_end is not None and snapshot.start < self._last_end - 1e-9:
            raise ValueError(
                "WindowReasoner needs consecutive non-overlapping windows (its CongestionStateTracker "
                f"hysteresis advances once per window); got start {snapshot.start} < previous end {self._last_end}"
            )
        self._last_end = snapshot.end

        views = tuple(snapshot.stop_episodes) if snapshot.stop_episodes is not None else ()
        stop_events = tuple(e for e in (build_stop_event(v) for v in views) if e is not None)

        if snapshot.unavailable or (snapshot.expected_frames and snapshot.frames_observed == 0):
            self._tracker.reset()
            # A stop confirmed from frames that WERE observed is still a real observation,
            # so its event is not lost just because the rest of the window was dark.
            return WindowAnalysis(
                snapshot=snapshot, aggregation=None, result=None, degraded=True,
                status=AnalysisStatus.UNAVAILABLE, congestion_advanced=False,
                stop_episodes=views, stop_events=stop_events,
                reason=(f"camera unavailable: {snapshot.frames_observed} of ~{snapshot.expected_frames} frames, "
                        f"{snapshot.silent_s:.1f}s silent, coverage {snapshot.coverage:.2f}"),
            )

        advance = (not snapshot.degraded) or snapshot.coverage >= self._min_update_coverage
        tracker = self._tracker if advance else copy.deepcopy(self._tracker)
        tracks = list(snapshot.tracks)
        # The space is a property of the CAMERA'S GEOMETRY, identical to what
        # `build_vehicle_track` used; `aggregate_traffic_state` re-verifies it
        # against the tracks and raises on any mismatch.
        motion_space = motion_space_for(self._profile)
        aggregation = aggregate_traffic_state(
            self.camera_id, snapshot.start, snapshot.end, tracks, motion_space, tracker,
            segment_length_m=self._segment_length_m, max_capacity_vehicles=self._max_capacity,
        )
        camera_input = CameraWindowInput(
            camera_id=self.camera_id, role=self._role, aggregation=aggregation, tracks=tracks
        )
        result = run_corridor_window(
            snapshot.window, (camera_input,),
            stop_episodes=views if snapshot.stop_episodes is not None else None, **self._kwargs,
        )
        if not snapshot.degraded:
            return WindowAnalysis(
                snapshot=snapshot, aggregation=aggregation, result=result, degraded=False,
                stop_episodes=views, stop_events=tuple(result.safety_events),
            )
        reason = (f"{snapshot.dropped_frames} frames lost, {snapshot.silent_s:.1f}s silent, "
                  f"coverage {snapshot.coverage:.2f}")
        note = f"window degraded ({reason}): vehicle counts, flow and stopped-vehicle evidence are lower bounds"
        if not advance:
            note += "; congestion hysteresis was NOT advanced by this window"
        result = dataclasses.replace(result, limitations=tuple(result.limitations) + (note,))
        return WindowAnalysis(
            snapshot=snapshot, aggregation=aggregation, result=result, degraded=True,
            status=AnalysisStatus.DEGRADED, reason=reason, congestion_advanced=advance,
            stop_episodes=views, stop_events=tuple(result.safety_events),
        )
