"""Event detectors: turn the perception core's outputs into candidate incidents.

TWO KINDS.
  window-level  -- consume a closed window (`streaming.reasoning.WindowAnalysis`): stopped
                   vehicle (stop episodes), congestion (traffic state), overspeed and
                   sudden braking (metric tracks).
  frame-level   -- consume each frame's identified detections plus a bounded per-track
                   history: wrong-way, wrong-lane, rapid lane changes, red-light, illegal
                   turn, collision candidate.

HONESTY RULES (each directly tested):
  - Detectors that need inputs ANVESH cannot invent (direction zones, lanes, a signal feed,
    turn rules) do nothing unless the camera is configured with them, and say so through
    `anvesh.app.capabilities`.
  - Overspeed and braking read ONLY tracks the existing M-A safety layer marks
    `motion_space=WORLD` + `geometry_validity=WORLD_VALIDATED` + `speed_available`. A fitted but
    unvalidated homography yields no events, never a guessed speed.
  - Image-space quantities (stop displacement in pixels, heading) are labelled as such in the
    event's `speed_basis` / `indicators`.
  - A detector emits a CANDIDATE with stated uncertainty. It never names a cause. "Involved"
    vehicles carry the role `observed` or `potentially_involved`, never `contributing`/cause:
    proximity and correlation are not causation.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import threading

from anvesh.app.livespeed import LiveSpeed
from anvesh.storage.schemas import CongestionLevel, GeometryValidity, MotionSpace
from anvesh.world.lane_geometry import LaneAssignmentStatus, LaneRegion, assign_lane

DETECTOR_VERSION = "1.0.0"


# -- shared types ------------------------------------------------------------


@dataclasses.dataclass
class DetectedEvent:
    event_type: str
    natural_key: str  # unique within a session; the incident service prefixes the session
    start_offset_s: float
    detected_offset_s: float
    detector: str
    track_ids: list = dataclasses.field(default_factory=list)
    involved: list = dataclasses.field(default_factory=list)  # {track_id, role, vehicle_class, note}
    indicators: dict = dataclasses.field(default_factory=dict)
    end_offset_s: float = None
    severity: str = "low"
    severity_basis: str = ""
    confidence_tier: str = "low"
    confidence_basis: str = ""
    uncertainty: list = dataclasses.field(default_factory=list)
    candidate_causes: list = dataclasses.field(default_factory=list)
    detector_config: dict = dataclasses.field(default_factory=dict)
    motion_space: str = "image"
    geometry_validity: str = "image"
    speed_basis: str = None
    key_frames: list = dataclasses.field(default_factory=list)  # [(offset_s, label, track_id|None)]
    clip_window: tuple = None  # (t0, t1) stream seconds
    summary: str = ""
    escalate_to: str = None  # event_type to switch an existing incident to (e.g. long_duration_stop)
    closed: bool = False


@dataclasses.dataclass(frozen=True)
class Observation:
    track_id: str
    vehicle_class: str
    bbox: tuple
    ground: tuple
    confidence: float


@dataclasses.dataclass(frozen=True)
class Sample:
    t: float
    frame_index: int
    bbox: tuple
    ground: tuple
    vehicle_class: str


class TrackHistory:
    """Bounded per-track sample history shared by frame-level detectors and evidence rendering.
    Written by the worker thread and read by the window-consumer thread, hence the lock."""

    def __init__(self, history_s: float = 60.0, prune_after_s: float = 15.0) -> None:
        self._history_s, self._prune_s = history_s, prune_after_s
        self._tracks: dict = {}
        self._last_seen: dict = {}
        self._lock = threading.RLock()

    def update(self, t: float, frame_index: int, observations: list) -> None:
        with self._lock:
            for o in observations:
                dq = self._tracks.setdefault(o.track_id, collections.deque())
                dq.append(Sample(t, frame_index, o.bbox, o.ground, o.vehicle_class))
                self._last_seen[o.track_id] = t
                while dq and t - dq[0].t > self._history_s:
                    dq.popleft()
            for tid in [k for k, seen in self._last_seen.items() if t - seen > self._prune_s]:
                self._tracks.pop(tid, None)
                self._last_seen.pop(tid, None)

    def samples(self, track_id: str) -> list:
        with self._lock:
            return list(self._tracks.get(track_id, ()))

    def active(self, t: float, within_s: float = 0.5) -> list:
        with self._lock:
            return [k for k, seen in self._last_seen.items() if t - seen <= within_s]

    def sample_at(self, track_id: str, t: float, tol_s: float = 0.4):
        with self._lock:
            best = None
            for s in self._tracks.get(track_id, ()):
                d = abs(s.t - t)
                if d <= tol_s and (best is None or d < abs(best.t - t)):
                    best = s
            return best

    def boxes_at(self, t: float, tol_s: float = 0.4) -> list:
        with self._lock:
            out = []
            for tid in list(self._tracks):
                s = self.sample_at(tid, t, tol_s)
                if s is not None:
                    out.append((tid, s.vehicle_class, s.bbox))
            return out

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._last_seen.clear()

    def tracks_count(self) -> int:
        with self._lock:
            return len(self._tracks)


@dataclasses.dataclass
class DetectorContext:
    camera_id: str
    fps: float
    config: object  # cameras.CameraConfig
    history: TrackHistory
    profile: object = None  # world.calibration.CalibrationProfile or None
    geometry_validity: GeometryValidity = GeometryValidity.IMAGE
    motion_space: MotionSpace = MotionSpace.IMAGE
    signal: object = None  # SignalState for red-light detection
    frame_size: tuple = None  # (width, height) of the processed frames, set by the runner on the first frame


class Detector:
    name = "detector"
    event_types = ()

    def __init__(self, ctx: DetectorContext) -> None:
        self.ctx = ctx

    def on_frame(self, t: float, frame_index: int, observations: list) -> list:
        return []

    def on_window(self, analysis) -> list:
        return []

    def finish(self, t: float) -> list:
        return []


# -- geometry helpers -------------------------------------------------------


def point_in_polygon(pt: tuple, poly: tuple) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def iou(a: tuple, b: tuple) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _involved(track_id: str, role: str, vehicle_class: str, note: str) -> dict:
    return {"track_id": track_id, "role": role, "vehicle_class": vehicle_class, "note": note}


def _strip_camera(track_id: str) -> str:
    """Engine stop episodes use '<camera>:<identity>'; history uses the bare identity."""
    return track_id.split(":", 1)[-1]


# -- window-level detectors --------------------------------------------------


class StoppedVehicleDetector(Detector):
    """Stop episodes (engine, across window boundaries) -> stopped_vehicle incidents, escalated to
    long_duration_stop once the stop outlasts `long_stop_s`.

    A vehicle that stops in a queue is not an incident; the event carries the camera's congestion
    level at the time and says so in `uncertainty`, and its severity drops accordingly.

    QUEUE vs OBSTRUCTION. Two independent signals mark a stop as probably queued traffic (both only LOWER the severity and add
    an uncertainty note; the incident is still raised, because a human must be able to see it):
      1. the camera's congestion level is BUILDING/CONGESTED (a count/occupancy proxy relative to `max_capacity_vehicles`, so it
         under-reports on a camera whose capacity was left at the default and whose queue is short);
      2. a STATIONARY GROUP: at least `queue_peer_stops` vehicles (this one included, default 3) are stationary in the same
         window. A lone stationary vehicle among moving traffic is the obstruction candidate; several vehicles stopped together
         usually are not.
    `queue_peer_stops` is an untuned policy value, not a measured one: it needs checking on the real cameras.

    The queue reading is LATCHED per stop episode. The incident is re-emitted every window and the last emission is the one that is
    stored; before the latch, a stop that began inside a queue was re-rated "medium, outside congested traffic" in the window where
    the congestion state had just fallen to DISSIPATING/FREE_FLOW, i.e. a queued vehicle was reported as an isolated obstruction at
    the moment its queue cleared. DISSIPATING (queue still clearing) counts as queue context. Every queue-context stop carries a
    `queue_group_id` (stops within `QUEUE_GROUP_GAP_S` of each other share one) so the list can show them as members of one queue
    instead of as unrelated obstructions; the incidents themselves are still one per vehicle, so none is hidden."""

    name = "stopped_vehicle"
    QUEUE_GROUP_GAP_S = 30.0
    event_types = ("stopped_vehicle", "long_duration_stop")

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.long_stop_s = ctx.config.threshold("long_stop_s", 30.0)
        self.queue_peer_stops = ctx.config.threshold("queue_peer_stops", 3)
        self._escalated: set = set()
        self._latch: dict = {}  # episode_id -> {"signal", "level", "group"} once the stop has been judged part of a queue
        self._group_seq = 0
        self._group = None  # {"id", "last_t", "peak"}

    def _queue_group(self, now: float, stationary_now: int) -> dict:
        """The queue group this window's queue-context stops belong to: a new group when the previous one was last active more than
        QUEUE_GROUP_GAP_S ago. The dict is shared by reference with every member, so `peak` is the group's largest simultaneous count."""
        g = self._group
        if g is None or now - g["last_t"] > self.QUEUE_GROUP_GAP_S:
            self._group_seq += 1
            g = self._group = {"id": f"queue:{self.ctx.camera_id}:{self._group_seq}", "last_t": now, "peak": 0}
        g["last_t"] = now
        g["peak"] = max(g["peak"], stationary_now)
        return g

    def on_window(self, analysis) -> list:
        from anvesh.safety.stop_episodes import EpisodePhase

        views = tuple(analysis.stop_episodes or ())
        level = None
        if analysis.aggregation is not None:
            level = analysis.aggregation.traffic_state.congestion_level
        out = []
        # Every episode that overlaps this window was stationary at some point in it (an episode that ENDS in the window is a
        # vehicle that was stopped until then), so the group size is the number of views, whatever their phase.
        stationary_now = len(views)
        stationary_group = stationary_now >= self.queue_peer_stops
        for v in views:
            ep = v.episode
            tid = _strip_camera(ep.track_id)
            sample = self.ctx.history.samples(tid)
            cls = sample[-1].vehicle_class if sample else "unknown"
            image_space = ep.motion_space == MotionSpace.IMAGE
            latched = self._latch.get(ep.episode_id)
            congestion_queue = level in (CongestionLevel.CONGESTED, CongestionLevel.BUILDING, CongestionLevel.DISSIPATING)
            group_queue = stationary_group
            queue_signal = "congestion_level" if congestion_queue else ("stationary_group" if group_queue else None)
            level_text = level.value if level is not None else None
            if queue_signal is None and latched is not None:  # judged part of a queue earlier in this stop: stay judged that way
                queue_signal, level_text = latched["signal"], latched["level"]
                congestion_queue, group_queue = queue_signal == "congestion_level", queue_signal == "stationary_group"
            queue_context = queue_signal is not None
            if queue_context:
                group = self._queue_group(analysis.snapshot.end, stationary_now)  # also keeps the group alive and tracks its peak
                if latched is None:
                    latched = self._latch[ep.episode_id] = {"signal": queue_signal, "level": level_text, "group": group}
            duration = (ep.end if ep.end is not None else ep.last_low_speed) - ep.start
            escalated = duration >= self.long_stop_s
            etype = "long_duration_stop" if escalated else "stopped_vehicle"
            if congestion_queue:
                severity, basis = "low", f"stopped while the camera is {level_text}: most likely queued traffic, not an obstruction"
            elif group_queue:
                severity, basis = "low", (f"{stationary_now} vehicles are stationary together (>= {self.queue_peer_stops}): "
                                          "most likely queued traffic, not a single obstruction")
            elif escalated:
                severity, basis = "high", f"stopped {duration:.0f}s (>= {self.long_stop_s:.0f}s long-stop threshold) outside congested traffic"
            else:
                severity, basis = "medium", "stationary vehicle outside congested traffic"
            unc = []
            if image_space:
                unc.append("Stop judged in IMAGE space (the box barely moved on screen); no real-world speed was measured.")
            elif getattr(self.ctx.profile, "is_marking_standard_validated", False):
                unc.append("Stop judged on an APPROXIMATE metric scale (lane-marking standard, site not surveyed) over the whole frame; outside the "
                           "validated near-field zone the scale is extrapolated and pixel noise is several decimetres.")
            if congestion_queue:
                unc.append(f"Camera congestion level is {level_text}; a vehicle stopped in a queue is expected behaviour.")
            if group_queue:
                unc.append(f"{stationary_now} vehicles were stationary in the same window; that pattern is typical of a queue or a signal "
                           "stop. A genuine obstruction at the head of a queue cannot be told apart from this alone - check the clip.")
            unc.append("Same physical vehicle after a tracker break is not established; track identity is session-scoped.")
            ev = DetectedEvent(
                event_type=etype, natural_key=f"{ep.episode_id}", start_offset_s=ep.start, detected_offset_s=ep.confirmed_at,
                end_offset_s=ep.end, detector=self.name, track_ids=[tid],
                involved=[_involved(tid, "observed", cls, "the stopped vehicle itself; no cause is implied")],
                indicators={"dwell_s": round(duration, 2), "phase": v.phase.value, "criterion": ep.criterion,
                            "speed_basis": ep.speed_basis, "speed_threshold": ep.speed_threshold,
                            "bridged_gaps": ep.bridged_gaps, "episode_open": ep.is_open,
                            "closed_reason": ep.closed_reason, "congestion_level_in_window": level.value if level else None,
                            "stationary_vehicles_in_window": stationary_now, "queue_signal": queue_signal,
                            "queue_member": bool(latched), "queue_group_id": latched["group"]["id"] if latched else None,
                            "queue_group_peak_stationary": latched["group"]["peak"] if latched else None},
                severity=severity, severity_basis=basis, confidence_tier=ep.confidence_tier.value,
                confidence_basis=f"dwell {duration:.1f}s vs {ep.dwell_required:.1f}s required (score {ep.score:.2f}); a heuristic tier, not a probability",
                uncertainty=unc, detector_config={"long_stop_s": self.long_stop_s, "criterion": ep.criterion},
                motion_space=ep.motion_space.value, geometry_validity=self.ctx.geometry_validity.value,
                speed_basis=ep.speed_basis,
                key_frames=[(ep.start, "stop begins", tid), (ep.confirmed_at, "stop confirmed", tid)],
                clip_window=(max(0.0, ep.start - 5.0), ep.confirmed_at + 5.0),
                summary=f"Vehicle {tid} ({cls}) stationary for {duration:.0f}s",
                closed=v.phase == EpisodePhase.CLOSED, escalate_to="long_duration_stop" if escalated else None,
            )
            out.append(ev)
        return out


class CongestionDetector(Detector):
    """One incident per congestion episode on a camera: opens when the hysteresis state leaves
    FREE_FLOW (BUILDING = queue formation, low severity), escalates at CONGESTED, closes once the
    camera has STAYED in FREE_FLOW for `congestion_close_grace_s`. Each window adds a point to the trend;
    counts are lower bounds on a degraded window (the trend point says so).

    CLOSE GRACE. The traffic-state machine can fall to FREE_FLOW for a single 10 s window in the middle of stop-and-go traffic (the
    occupancy proxy hovers around its threshold). Closing on that window produced a string of short incidents (a close and a reopen
    in the same window; 13 episodes in 20 minutes of one continuous stop-and-go regime). The free-flow windows inside the grace
    period stay in the trend (nothing is hidden), the episode continues if congestion returns, and if it does not the incident is
    closed with its end BACK-DATED to the first free-flow window. Severity counts only the time the camera was not free-flowing
    (`non_free_s`), never the span, so a long episode with long gaps is not made to look sustained."""

    name = "congestion"
    event_types = ("congestion",)

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._open = None  # dict state of the open episode
        self._seq = 0
        self.close_grace_s = ctx.config.threshold("congestion_close_grace_s", 30.0)
        self.capacity = ctx.config.threshold("max_capacity_vehicles", 20)

    @staticmethod
    def _point(analysis) -> dict:
        s = analysis.snapshot
        ts = analysis.aggregation.traffic_state
        mean = ts.measured_mean_speed()
        return {
            "window_id": s.window_id, "start": s.start, "end": s.end, "level": ts.congestion_level.value,
            "vehicle_count": ts.vehicle_count, "mean_concurrency": round(analysis.aggregation.mean_concurrency, 2),
            "occupancy": round(ts.occupancy, 3), "flow_rate": round(ts.flow_rate, 3),
            "mean_speed": None if mean is None else round(mean.value, 2),
            "speed_unit": None if mean is None else ("m/s" if ts.motion_space == MotionSpace.WORLD else "px/s"),
            "class_counts": {getattr(k, "value", str(k)): v for k, v in analysis.aggregation.class_counts.items()},
            "degraded": bool(analysis.degraded),
        }

    def on_window(self, analysis) -> list:
        if analysis.unavailable or analysis.aggregation is None:
            return self._close(analysis.snapshot.end, "camera unavailable; episode ended without a closing observation") if self._open else []
        level = analysis.aggregation.traffic_state.congestion_level
        point = self._point(analysis)
        s = analysis.snapshot
        if self._open is None:
            if level == CongestionLevel.FREE_FLOW:
                return []
            self._seq += 1
            self._open = {"key": f"congestion:{self.ctx.camera_id}:{s.start:.0f}:{self._seq}", "start": s.start,
                          "trend": [], "peak": level, "causes": [], "quiet_since": None, "interruptions": 0}
        ep = self._open
        ep["trend"].append(point)
        order = [CongestionLevel.FREE_FLOW, CongestionLevel.DISSIPATING, CongestionLevel.BUILDING, CongestionLevel.CONGESTED]
        if order.index(level) > order.index(ep["peak"]):
            ep["peak"] = level
        if analysis.result is not None:
            ep["causes"] = _candidate_causes(analysis.result)
        if level == CongestionLevel.FREE_FLOW:
            if ep["quiet_since"] is None:
                ep["quiet_since"] = s.start
            if s.end - ep["quiet_since"] >= self.close_grace_s:
                return self._close(ep["quiet_since"], f"returned to free flow and stayed there for {self.close_grace_s:.0f} s")
            return [self._event(ep, s.end, closed=False)]
        if ep["quiet_since"] is not None:  # congestion came back inside the grace period: the same episode continues
            ep["interruptions"] += 1
            ep["quiet_since"] = None
        return [self._event(ep, s.end, closed=False)]

    @staticmethod
    def _non_free_s(ep) -> float:
        return sum(p["end"] - p["start"] for p in ep["trend"] if p["level"] != CongestionLevel.FREE_FLOW.value)

    def _occupancy_driven_note(self, ep):
        """Say so when the congested/building label was reached on the occupancy proxy while traffic was still moving fast."""
        flagged = [p for p in ep["trend"] if p["level"] in ("congested", "building") and p["mean_speed"] is not None and p["speed_unit"] == "m/s"]
        if not flagged:
            return None
        fast = [p for p in flagged if p["mean_speed"] >= 8.0]  # CongestionThresholds.free_speed_m_per_s
        if len(fast) * 2 < len(flagged):
            return None
        kmh = sorted(p["mean_speed"] * 3.6 for p in fast)
        return (f"In {len(fast)} of {len(flagged)} congested/building windows the mean speed of all tracked vehicles was still {kmh[0]:.0f}-{kmh[-1]:.0f} km/h "
                f"(approximate scale; every carriageway in view together). The level came from the occupancy proxy (mean concurrency against an assumed "
                f"capacity of {self.capacity} vehicles), i.e. dense traffic, not necessarily slow traffic. A queue on one carriageway can be hidden by free "
                f"flow on the other.")

    def _event(self, ep, now, closed: bool, end_override: float = None) -> DetectedEvent:
        peak = ep["peak"]
        quiet = ep["quiet_since"]
        last_active = quiet if quiet is not None else now
        duration = last_active - ep["start"]
        non_free = self._non_free_s(ep)
        if peak == CongestionLevel.CONGESTED:
            severity = "high" if non_free >= 300 else "medium"
            basis = f"peak level congested, {non_free:.0f}s not free-flowing" + (" (>= 300 s sustained)" if non_free >= 300 else "")
        else:
            severity, basis = "low", f"queue forming (peak level {peak.value})"
        causes = ep["causes"]
        image_space = self.ctx.motion_space == MotionSpace.IMAGE
        unc = ["Congestion level is a count/occupancy proxy in image space; no speed or density was measured in metres."
               if image_space else ("Metric geometry is APPROXIMATE for this camera (scaled from a published lane-marking module; site not surveyed)."
                                    if getattr(self.ctx.profile, "is_marking_standard_validated", False) else "Metric geometry validated for this camera.")]
        note = self._occupancy_driven_note(ep)
        if note:
            unc.append(note)
        if ep["interruptions"]:
            unc.append(f"The camera returned to free flow {ep['interruptions']} time(s) inside this episode, each for less than {self.close_grace_s:.0f} s; "
                       "they are merged as one stop-and-go episode and the free-flow windows are kept in the trend.")
        if not causes or causes[0].get("outcome") != "ranked":
            unc.append("No candidate cause could be ranked from the available evidence.")
        else:
            unc.append("Candidate causes are ranked hypotheses with belief/plausibility, not probabilities and not established causes.")
        if image_space:
            unc.append("Without validated metric geometry only the load-based hypothesis can be supported; cause ranking is structurally limited.")
        unc.append("Affected lanes: not available (no lane geometry configured)." if not self.ctx.config.lanes
                   else "Affected lanes are reported per lane polygon.")
        return DetectedEvent(
            event_type="congestion", natural_key=ep["key"], start_offset_s=ep["start"], detected_offset_s=ep["start"],
            end_offset_s=end_override if closed else None, detector=self.name,
            indicators={"peak_level": peak.value, "current_level": ep["trend"][-1]["level"], "duration_s": round(duration, 1),
                        "non_free_s": round(non_free, 1), "interruptions": ep["interruptions"], "clearing_since_s": quiet,
                        "close_grace_s": self.close_grace_s,
                        "trend": list(ep["trend"]), "windows": len(ep["trend"]),
                        "affected_lanes": None if not self.ctx.config.lanes else "see lane polygons"},
            severity=severity, severity_basis=basis, confidence_tier="medium",
            confidence_basis="hysteresis-damped state from count/occupancy over observed windows",
            uncertainty=unc, candidate_causes=causes, detector_config={"window_s": 10.0, "close_grace_s": self.close_grace_s},
            motion_space=self.ctx.motion_space.value, geometry_validity=self.ctx.geometry_validity.value,
            key_frames=[(ep["start"], "congestion begins", None), (last_active, "latest", None)],
            clip_window=(max(0.0, ep["start"] - 3.0), min(last_active, ep["start"] + 30.0)),
            summary=f"Congestion on {self.ctx.camera_id}: peak {peak.value}, {duration:.0f}s" + (" (clearing)" if quiet is not None and not closed else ""),
            closed=closed,
        )

    def _close(self, end, reason) -> list:
        ep, self._open = self._open, None
        if ep is None:
            return []
        ev = self._event(ep, end, closed=True, end_override=end)
        ev.indicators["closed_reason"] = reason
        return [ev]

    def finish(self, t: float) -> list:
        if not self._open:
            return []
        quiet = self._open["quiet_since"]
        if quiet is not None:
            return self._close(quiet, "returned to free flow; the stream ended before the close grace period elapsed")
        return self._close(t, "stream ended while congestion was still open")


def _candidate_causes(result) -> list:
    """Serialise the existing ranking WITHOUT upgrading it: belief/plausibility stay as given, the
    outcome (ranked / insufficient_evidence / high_conflict) is carried, and no cause is 'established'."""
    ranking = getattr(result, "candidate_ranking", None)
    if ranking is None:
        return []
    entries = []
    for e in ranking.ranked_list:
        entries.append({
            "outcome": ranking.outcome.value, "hypothesis_id": e.hypothesis_id, "belief": round(e.belief, 4),
            "plausibility": round(e.plausibility, 4), "confidence_tier": e.confidence_tier.value,
            "supporting_evidence_refs": list(e.supporting_evidence_refs), "contradicting_evidence_refs": list(e.contradicting_evidence_refs),
            "status": "candidate - not an established cause",
        })
    if not entries:
        entries.append({"outcome": ranking.outcome.value, "status": "no ranked candidate"})
    return entries


class OverspeedDetector(Detector):
    """Sustained speed above the configured limit, judged on the SMOOTHED metric speed of a track inside the speed zone.

    Speed is the least-squares slope of the track's calibrated ground-point positions over 1.6 s (`LiveSpeed`), which does not
    accumulate box jitter the way a summed path length does. An event needs the estimate to exceed `limit + MARGIN_KMH` AND the
    lower edge of its own error band to exceed the limit, continuously for `DURATION_S`. It is never produced without a validated
    calibration, and with a marking-standard calibration (site not surveyed) it is a low-confidence CANDIDATE."""

    name = "overspeed"
    event_types = ("overspeed",)
    MAX_PLAUSIBLE_KMH = 300.0
    MARGIN_KMH = 5.0
    DURATION_S = 2.0
    STEP_S = 0.25

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._speed = LiveSpeed(ctx.profile, ctx.config.speed_zone, window_s=1.6, min_span_s=0.8, min_samples=6, ema=0.5,
                                frame_size=lambda: ctx.frame_size)
        self._over: dict = {}  # tid -> {"since": t, "peak": v, "err": e, "first": first_seen}
        self._next: dict = {}
        self._seen: set = set()
        self.skipped_non_metric = 0
        self.skipped_outside_zone = 0

    def on_frame(self, t, frame_index, observations) -> list:
        limit = self.ctx.config.speed_limit_kmh
        if limit is None:
            return []
        out = []
        for o in observations:
            tid = o.track_id
            if tid in self._seen or t < self._next.get(tid, 0.0):
                continue
            self._next[tid] = t + self.STEP_S
            est = self._speed.estimate(tid, self.ctx.history.samples(tid))
            if est["status"] == "no_metric_calibration":
                self.skipped_non_metric += 1
                self._over.pop(tid, None)
                continue
            if est["status"] == "outside_speed_zone":
                self.skipped_outside_zone += 1
            if est["status"] != "ok":
                self._over.pop(tid, None)
                continue
            v, err = est["speed_kmh"], est["error_kmh"]
            if v > self.MAX_PLAUSIBLE_KMH or v <= limit + self.MARGIN_KMH or v - err <= limit:
                self._over.pop(tid, None)
                continue
            st = self._over.setdefault(tid, {"since": t, "peak": v, "err": err})
            st["peak"], st["err"] = max(st["peak"], v), err
            if t - st["since"] < self.DURATION_S:
                continue
            self._seen.add(tid)
            out.append(self._event(o, t, st, v, limit))
        return out

    def _event(self, o, t, st, kmh, limit):
        tid = o.track_id
        synthetic = bool(getattr(self.ctx.profile, "is_synthetic_validated", False))
        approx = bool(getattr(self.ctx.profile, "is_marking_standard_validated", False))
        return DetectedEvent(
            event_type="overspeed", natural_key=f"overspeed:{tid}", start_offset_s=st["since"], detected_offset_s=t,
            end_offset_s=t, detector=self.name, track_ids=[tid],
            involved=[_involved(tid, "observed", o.vehicle_class, f"{kmh:.0f} km/h for {t - st['since']:.1f} s vs limit {limit:.0f}")],
            indicators={"speed_kmh": round(kmh, 1), "peak_kmh": round(st["peak"], 1), "speed_error_kmh": st["err"], "limit_kmh": limit,
                        "margin_kmh": self.MARGIN_KMH, "sustained_s": round(t - st["since"], 2), "min_duration_s": self.DURATION_S,
                        "speed_kind": "smoothed_regression_1.6s", "synthetic_calibration": synthetic, "approximate_calibration": approx},
            severity="medium" if kmh < limit + 20 else "high", severity_basis=f"{kmh - limit:.0f} km/h over the configured limit",
            confidence_tier="low" if (synthetic or approx) else "medium",
            confidence_basis=("SYNTHETIC calibration fixture: not a real camera calibration" if synthetic
                              else "scale taken from a published lane-marking module, site not surveyed: speed is approximate" if approx
                              else "metric geometry validated by recorded evidence; speed is a smoothed estimate with stated error"),
            uncertainty=["Speed is a smoothed estimate over ~1.6 s of tracked box positions, not a speed-gun reading.",
                         "Accuracy depends on the calibration; the configured limit is a demonstration threshold unless stated otherwise."],
            detector_config={"limit_kmh": limit, "margin_kmh": self.MARGIN_KMH, "min_duration_s": self.DURATION_S},
            motion_space="world", geometry_validity="world_validated", speed_basis="metric_m_per_s",
            key_frames=[(st["since"], "exceeds limit", tid), (t, "still exceeding", tid)],
            clip_window=(max(0.0, st["since"] - 3.0), t + 3.0), summary=f"{o.vehicle_class} {tid} at {kmh:.0f} km/h (limit {limit:.0f})",
        )


class BrakingDetector(Detector):
    """Sudden braking: a vehicle inside the speed zone whose SMOOTHED metric speed falls by a large, statistically
    significant amount, quickly and monotonically.

    Speeds are least-squares slopes over 1.2 s windows (not frame-to-frame differences, which are dominated by box jitter),
    sampled every 0.25 s from the track's image ground points through the validated calibration. An event needs ALL of:
      - a starting speed of at least MIN_PRE_KMH (an ordinary stop from walking pace is not "sudden braking"),
      - a drop of at least DROP_KMH that exceeds the combined error bands of the two estimates,
      - an average deceleration of at least MIN_DECEL_MPS2 over MIN_SPAN_S..MAX_SPAN_S,
      - a roughly monotonic fall in between (a noisy dip that recovers is rejected).
    It reads no pixel-space quantity and never runs without a validated calibration. Thresholds are policy defaults, not
    tuned on labelled braking data."""

    name = "sudden_braking"
    event_types = ("sudden_braking",)
    DROP_KMH = 25.0
    MIN_PRE_KMH = 40.0
    MIN_DECEL_MPS2 = 3.5
    MIN_SPAN_S, MAX_SPAN_S = 1.0, 2.5
    STEP_S = 0.25
    DEBOUNCE_S = 30.0

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._speed = LiveSpeed(ctx.profile, ctx.config.speed_zone, window_s=1.2, min_span_s=0.8, min_samples=8, ema=1.0,
                                frame_size=lambda: ctx.frame_size)
        self._series: dict = {}
        self._next: dict = {}
        self._last_fired: dict = {}
        self.rejected = {"no_speed": 0}

    def on_frame(self, t, frame_index, observations) -> list:
        out = []
        for o in observations:
            if t < self._next.get(o.track_id, 0.0):
                continue
            self._next[o.track_id] = t + self.STEP_S
            est = self._speed.estimate(o.track_id, self.ctx.history.samples(o.track_id))
            ser = self._series.setdefault(o.track_id, collections.deque(maxlen=20))
            if est["status"] != "ok":
                ser.clear()
                self.rejected["no_speed"] += 1
                continue
            v, err = est["speed_kmh"], est["error_kmh"]
            ev = self._check(o, t, v, err, ser)
            ser.append((t, v, err))
            if ev is not None:
                out.append(ev)
        return out

    def _check(self, o, t, v, err, ser):
        if t - self._last_fired.get(o.track_id, -1e9) < self.DEBOUNCE_S:
            return None
        best = None
        for k, (t0, v0, e0) in enumerate(ser):
            span = t - t0
            if not (self.MIN_SPAN_S <= span <= self.MAX_SPAN_S) or v0 < self.MIN_PRE_KMH:
                continue
            drop = v0 - v
            decel = drop / 3.6 / span
            if drop < self.DROP_KMH or drop <= (e0 + err) or decel < self.MIN_DECEL_MPS2:
                continue
            between = [x[1] for x in list(ser)[k:]] + [v]
            if any(b > a + 5.0 for a, b in zip(between, between[1:])):  # a rebound: noise, not a braking manoeuvre
                continue
            if best is None or decel > best[0]:
                best = (decel, t0, v0, e0, drop, span)
        if best is None:
            return None
        decel, t0, v0, e0, drop, span = best
        self._last_fired[o.track_id] = t
        tid = o.track_id
        approx = bool(getattr(self.ctx.profile, "is_marking_standard_validated", False))
        return DetectedEvent(
            event_type="sudden_braking", natural_key=f"braking:{tid}:{t0:.0f}", start_offset_s=t0, detected_offset_s=t,
            end_offset_s=t, detector=self.name, track_ids=[tid],
            involved=[_involved(tid, "observed", o.vehicle_class, f"speed fell {v0:.0f} -> {v:.0f} km/h in {span:.1f} s")],
            indicators={"decel_mps2": round(decel, 2), "from_kmh": round(v0, 1), "to_kmh": round(v, 1), "drop_kmh": round(drop, 1),
                        "span_s": round(span, 2), "error_band_kmh": round(e0 + err, 1), "threshold_mps2": self.MIN_DECEL_MPS2,
                        "min_drop_kmh": self.DROP_KMH, "min_pre_speed_kmh": self.MIN_PRE_KMH, "approximate_calibration": approx},
            severity="medium", severity_basis=f"{decel:.1f} m/s^2 average deceleration over {span:.1f} s ({v0:.0f} -> {v:.0f} km/h)",
            confidence_tier="low",
            confidence_basis=("smoothed speeds from a marking-standard (site not surveyed) calibration: approximate" if approx
                              else "smoothed speeds (1.2 s regression) from validated metric geometry"),
            uncertainty=["Speeds come from tracked box ground points; a tracker identity switch or box change can mimic a speed drop.",
                         "Braking is not a violation by itself: it may be a reaction to traffic ahead. The cause is not established."],
            detector_config={"drop_kmh": self.DROP_KMH, "min_pre_kmh": self.MIN_PRE_KMH, "min_decel_mps2": self.MIN_DECEL_MPS2,
                             "span_s": [self.MIN_SPAN_S, self.MAX_SPAN_S]},
            motion_space="world", geometry_validity="world_validated", speed_basis="metric_m_per_s",
            key_frames=[(t0, "before braking", tid), (t, "after braking", tid)], clip_window=(max(0.0, t0 - 3), t + 3),
            summary=f"{tid} braked {v0:.0f} -> {v:.0f} km/h ({decel:.1f} m/s^2)",
        )


# -- frame-level detectors ------------------------------------------------------


class WrongWayDetector(Detector):
    """A vehicle travelling against a zone's configured expected direction.

    Direction is IMAGE-space (net ground-point displacement over the time spent in the zone) and
    therefore perspective-dependent; it is legitimate for a configured zone with a clear travel
    axis. It is tolerant to identity switches in one way: it needs `min_duration_s` of continuous
    in-zone travel for ONE track, and a second event in the same zone close in time and space to a
    previous one is suppressed as the same vehicle under a new id."""

    name = "wrong_way"
    event_types = ("wrong_way",)
    CONTINUITY_S = 3.0
    SAME_VEHICLE_PX = 120.0

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._fired: set = set()
        self._flagged: dict = {}  # track_id -> (zone_id, last_seen_t, last_ground) of vehicles already reported

    def on_frame(self, t, frame_index, observations) -> list:
        out = []
        for o in observations:
            if o.track_id in self._flagged:
                zone_id = self._flagged[o.track_id][0]
                self._flagged[o.track_id] = (zone_id, t, o.ground)
        for zone in self.ctx.config.direction_zones:
            ex, ey = zone["expected"]
            norm = math.hypot(ex, ey)
            for o in observations:
                if not point_in_polygon(o.ground, zone["polygon"]):
                    continue
                key = (o.track_id, zone["zone_id"])
                if key in self._fired:
                    continue
                inside = [s for s in self.ctx.history.samples(o.track_id) if point_in_polygon(s.ground, zone["polygon"])]
                if len(inside) < 3 or inside[-1].t - inside[0].t < zone["min_duration_s"]:
                    continue
                dx, dy = inside[-1].ground[0] - inside[0].ground[0], inside[-1].ground[1] - inside[0].ground[1]
                disp = math.hypot(dx, dy)
                if disp < zone["min_displacement_px"]:
                    continue
                cos = (dx * ex + dy * ey) / (disp * norm)
                if cos > -0.7:  # more than ~135 degrees against the expected direction
                    continue
                if any(fid != o.track_id and z == zone["zone_id"] and t - ft <= self.CONTINUITY_S
                       and math.hypot(inside[0].ground[0] - fg[0], inside[0].ground[1] - fg[1]) < self.SAME_VEHICLE_PX
                       for fid, (z, ft, fg) in self._flagged.items()):
                    self._fired.add(key)  # picks up where an already-reported vehicle vanished: same vehicle, new id
                    self._flagged[o.track_id] = (zone["zone_id"], t, o.ground)
                    continue
                self._fired.add(key)
                self._flagged[o.track_id] = (zone["zone_id"], t, o.ground)
                duration = inside[-1].t - inside[0].t
                strong = disp >= 2 * zone["min_displacement_px"] and duration >= 2 * zone["min_duration_s"]
                out.append(DetectedEvent(
                    event_type="wrong_way", natural_key=f"wrong_way:{o.track_id}:{zone['zone_id']}",
                    start_offset_s=inside[0].t, detected_offset_s=t, detector=self.name, track_ids=[o.track_id],
                    involved=[_involved(o.track_id, "observed", o.vehicle_class, "moving against the configured direction of the zone")],
                    indicators={"zone_id": zone["zone_id"], "expected_direction": list(zone["expected"]),
                                "observed_displacement_px": [round(dx, 1), round(dy, 1)], "cos_to_expected": round(cos, 3),
                                "duration_in_zone_s": round(duration, 2), "space": "image"},
                    severity="high", severity_basis="vehicle moving against the configured travel direction",
                    confidence_tier="medium" if strong else "low",
                    confidence_basis=f"net image-space displacement {disp:.0f}px over {duration:.1f}s inside the zone",
                    uncertainty=["Direction is measured in image space and depends on the camera perspective.",
                                 "A tracker identity switch could split or merge vehicles; confirm on the clip.",
                                 "The zone's expected direction is operator-configured, not learned."],
                    detector_config={"zone": zone["zone_id"], "min_displacement_px": zone["min_displacement_px"],
                                     "min_duration_s": zone["min_duration_s"]},
                    key_frames=[(inside[0].t, "enters zone", o.track_id), (t, "detected", o.track_id)],
                    clip_window=(max(0.0, inside[0].t - 3.0), t + 3.0),
                    summary=f"{o.vehicle_class} {o.track_id} moving against zone {zone['zone_id']}",
                ))
        return out


def _lane_regions(config) -> tuple:
    return tuple(LaneRegion(lane_id=l["lane_id"], vertices=l["vertices"]) for l in config.lanes)


class WrongLaneDetector(Detector):
    """A vehicle class dwelling in a lane that does not allow it (per-lane `allowed_classes`)."""

    name = "wrong_lane"
    event_types = ("wrong_lane",)
    DWELL_S = 3.0

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._regions = _lane_regions(ctx.config)
        self._allowed = {l["lane_id"]: l["allowed_classes"] for l in ctx.config.lanes}
        self._since: dict = {}
        self._fired: set = set()

    def on_frame(self, t, frame_index, observations) -> list:
        out = []
        for o in observations:
            res = assign_lane(o.ground, self._regions)
            lane = res.lane_id if res.status == LaneAssignmentStatus.ASSIGNED else None
            allowed = self._allowed.get(lane)
            bad = lane is not None and allowed is not None and o.vehicle_class not in allowed
            key = (o.track_id, lane)
            if not bad:
                self._since.pop(o.track_id, None)
                continue
            first = self._since.setdefault(o.track_id, (lane, t))
            if first[0] != lane:
                self._since[o.track_id] = first = (lane, t)
            if t - first[1] >= self.DWELL_S and key not in self._fired:
                self._fired.add(key)
                out.append(DetectedEvent(
                    event_type="wrong_lane", natural_key=f"wrong_lane:{o.track_id}:{lane}", start_offset_s=first[1], detected_offset_s=t,
                    detector=self.name, track_ids=[o.track_id],
                    involved=[_involved(o.track_id, "observed", o.vehicle_class, f"in lane {lane}, which does not allow {o.vehicle_class}")],
                    indicators={"lane_id": lane, "allowed_classes": list(allowed), "dwell_s": round(t - first[1], 2)},
                    severity="medium", severity_basis="vehicle class dwelling in a lane that excludes it", confidence_tier="low",
                    confidence_basis="lane polygon membership of the box ground point; class from a COCO detector",
                    uncertainty=["Lane membership uses the bottom-centre of the box; occlusion biases it.",
                                 "Vehicle class comes from a generic detector and can be wrong (e.g. van vs truck)."],
                    detector_config={"dwell_s": self.DWELL_S}, key_frames=[(first[1], "enters lane", o.track_id), (t, "detected", o.track_id)],
                    clip_window=(max(0.0, first[1] - 2), t + 2), summary=f"{o.vehicle_class} {o.track_id} in lane {lane}",
                ))
        return out


class RapidLaneChangeDetector(Detector):
    """Rash-driving CANDIDATE: >= `min_changes` debounced lane changes within `window_s`."""

    name = "rash_driving"
    event_types = ("rash_driving",)
    MIN_CHANGES, WINDOW_S, STABLE_S = 3, 10.0, 0.5

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._regions = _lane_regions(ctx.config)
        self._state: dict = {}  # tid -> {"lane", "cand", "cand_t", "changes":[t...]}
        self._fired: set = set()

    def on_frame(self, t, frame_index, observations) -> list:
        out = []
        for o in observations:
            res = assign_lane(o.ground, self._regions)
            lane = res.lane_id if res.status == LaneAssignmentStatus.ASSIGNED else None
            st = self._state.setdefault(o.track_id, {"lane": None, "cand": None, "cand_t": 0.0, "changes": [], "cls": o.vehicle_class})
            if lane is None or lane == st["lane"]:
                st["cand"] = None
                continue
            if st["cand"] != lane:
                st["cand"], st["cand_t"] = lane, t
                continue
            if t - st["cand_t"] >= self.STABLE_S:
                if st["lane"] is not None:
                    st["changes"].append(t)
                st["lane"], st["cand"] = lane, None
                st["changes"] = [c for c in st["changes"] if t - c <= self.WINDOW_S]
                if len(st["changes"]) >= self.MIN_CHANGES and o.track_id not in self._fired:
                    self._fired.add(o.track_id)
                    out.append(DetectedEvent(
                        event_type="rash_driving", natural_key=f"rash:{o.track_id}", start_offset_s=st["changes"][0], detected_offset_s=t,
                        detector=self.name, track_ids=[o.track_id],
                        involved=[_involved(o.track_id, "observed", o.vehicle_class, "repeated lane changes")],
                        indicators={"lane_changes": len(st["changes"]), "within_s": round(t - st["changes"][0], 1), "to_lane": lane},
                        severity="medium", severity_basis=f"{len(st['changes'])} lane changes in {t - st['changes'][0]:.0f}s",
                        confidence_tier="low", confidence_basis="debounced lane-polygon transitions of the box ground point",
                        uncertainty=["Repeated lane changes are a CANDIDATE indicator of rash driving, not a finding of it.",
                                     "Lane polygons are operator-drawn; box jitter near a boundary can create spurious changes."],
                        detector_config={"min_changes": self.MIN_CHANGES, "window_s": self.WINDOW_S},
                        key_frames=[(st["changes"][0], "first change", o.track_id), (t, "detected", o.track_id)],
                        clip_window=(max(0.0, st["changes"][0] - 2), t + 2), summary=f"{o.track_id} changed lanes {len(st['changes'])}x in {t - st['changes'][0]:.0f}s",
                    ))
        return out


class SignalState:
    """Runtime signal state for red-light detection. ANVESH does not read the signal from video:
    the state comes from a configured schedule (an ASSUMPTION about the junction), a manual
    operator setting, or an external feed pushed through the API."""

    def __init__(self, mode: str, schedule: dict = None) -> None:
        self.mode, self.schedule = mode, schedule
        self._events: list = []  # [(t, state)] for manual/external

    @property
    def source_label(self) -> str:
        return {"schedule": "configured_schedule (assumption, not observed)", "manual": "manual_operator_setting",
                "external": "external_feed"}[self.mode]

    def set(self, state: str, t: float) -> None:
        if state not in ("red", "amber", "green"):
            raise ValueError("state must be red, amber or green")
        self._events.append((t, state))

    def at(self, t: float):
        if self.mode == "schedule" and self.schedule:
            s = self.schedule
            cycle = s["red_s"] + s["green_s"] + s["amber_s"]
            x = (t - s.get("offset_s", 0.0)) % cycle
            return "red" if x < s["red_s"] else ("green" if x < s["red_s"] + s["green_s"] else "amber")
        latest = None
        for et, st in self._events:
            if et <= t:
                latest = st
        return latest

    def red_since(self, t: float):
        """How long the signal has been continuously red at `t` (None if not red / unknown)."""
        if self.at(t) != "red":
            return None
        lo = t
        step = 0.1
        while lo > 0 and self.at(lo - step) == "red" and t - lo < 120:
            lo -= step
        return t - lo


def _side(line, p) -> float:
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)


class RedLightDetector(Detector):
    """A tracked vehicle crosses the configured stop line, in the approach direction, while the
    signal has been red for at least `GRACE_S`. With an unknown signal state it emits nothing (it
    counts the crossings it could not judge) -- it never assumes red."""

    name = "red_light"
    event_types = ("red_light_violation",)
    GRACE_S = 1.0

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._fired: set = set()
        self.unjudged_crossings = 0

    def on_frame(self, t, frame_index, observations) -> list:
        sig_cfg, signal = self.ctx.config.signal, self.ctx.signal
        if not sig_cfg or signal is None:
            return []
        line, (ax, ay) = sig_cfg["stop_line"], sig_cfg["approach"]
        out = []
        for o in observations:
            samples = self.ctx.history.samples(o.track_id)
            if len(samples) < 2 or o.track_id in self._fired:
                continue
            prev, cur = samples[-2], samples[-1]
            s0, s1 = _side(line, prev.ground), _side(line, cur.ground)
            if s0 == 0 or (s0 > 0) == (s1 > 0):
                continue
            mx, my = cur.ground[0] - prev.ground[0], cur.ground[1] - prev.ground[1]
            if mx * ax + my * ay <= 0:
                continue  # crossing against the approach direction
            state = signal.at(t)
            red_for = signal.red_since(t)
            if state is None:
                self.unjudged_crossings += 1
                continue
            if state != "red" or red_for is None or red_for < self.GRACE_S:
                continue
            self._fired.add(o.track_id)
            assumed = signal.mode != "external"
            out.append(DetectedEvent(
                event_type="red_light_violation", natural_key=f"red_light:{o.track_id}", start_offset_s=prev.t, detected_offset_s=t,
                detector=self.name, track_ids=[o.track_id],
                involved=[_involved(o.track_id, "observed", o.vehicle_class, "crossed the stop line while the signal was red")],
                indicators={"signal_state": state, "red_for_s": round(red_for, 2), "grace_s": self.GRACE_S,
                            "signal_source": signal.source_label, "stop_line": [list(p) for p in line]},
                severity="high", severity_basis="stop-line crossing during red", confidence_tier="low" if assumed else "medium",
                confidence_basis=("signal state is " + signal.source_label + "; the colour was NOT observed in the video") if assumed
                                 else "signal state from an external feed; crossing from tracked ground point",
                uncertainty=["The signal state was not read from the video.",
                             "Crossing is judged from the box ground point; confirm on the clip that the vehicle crossed after the signal turned red."]
                            + (["The signal schedule/setting is an operator assumption about the junction."] if assumed else []),
                detector_config={"grace_s": self.GRACE_S, "signal_mode": signal.mode},
                key_frames=[(prev.t, "before line", o.track_id), (t, "after line", o.track_id)],
                clip_window=(max(0.0, t - 5), t + 3), summary=f"{o.vehicle_class} {o.track_id} crossed the stop line on red",
            ))
        return out


class IllegalTurnDetector(Detector):
    """Track enters a rule's entry zone and later its exit zone (a prohibited movement)."""

    name = "illegal_turn"
    event_types = ("illegal_turn",)
    MAX_TRANSIT_S = 20.0

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._entered: dict = {}  # (tid, rule) -> t
        self._fired: set = set()

    def on_frame(self, t, frame_index, observations) -> list:
        out = []
        for o in observations:
            for rule in self.ctx.config.turn_rules:
                key = (o.track_id, rule["rule_id"])
                if key in self._fired:
                    continue
                if point_in_polygon(o.ground, rule["entry"]):
                    self._entered.setdefault(key, t)
                elif point_in_polygon(o.ground, rule["exit"]) and key in self._entered and t - self._entered[key] <= self.MAX_TRANSIT_S:
                    self._fired.add(key)
                    t0 = self._entered[key]
                    out.append(DetectedEvent(
                        event_type="illegal_turn", natural_key=f"turn:{o.track_id}:{rule['rule_id']}", start_offset_s=t0, detected_offset_s=t,
                        detector=self.name, track_ids=[o.track_id],
                        involved=[_involved(o.track_id, "observed", o.vehicle_class, f"moved from entry to exit of prohibited movement '{rule['label']}'")],
                        indicators={"rule_id": rule["rule_id"], "label": rule["label"], "transit_s": round(t - t0, 2)},
                        severity="medium", severity_basis="prohibited movement per configured turn rule", confidence_tier="low",
                        confidence_basis="entry-zone then exit-zone membership of the box ground point",
                        uncertainty=["The prohibition is operator-configured; applicable signage/time-of-day restrictions are not modelled.",
                                     "A tracker identity switch between the zones would hide or invent a movement."],
                        detector_config={"rule_id": rule["rule_id"]},
                        key_frames=[(t0, "enters entry zone", o.track_id), (t, "reaches exit zone", o.track_id)],
                        clip_window=(max(0.0, t0 - 2), t + 2), summary=f"{o.vehicle_class} {o.track_id}: prohibited movement {rule['label']}",
                    ))
        return out


class CollisionCandidateDetector(Detector):
    """Possible-collision CANDIDATE: two vehicles that were MOVING and CLOSING, whose boxes then stay
    in contact for `min_contact_s`, after which BOTH are stationary for `STOP_S`.

    Everything is inferred from 2-D boxes: box overlap can be occlusion (one vehicle behind another),
    so contact alone is never enough -- the approach and the joint stop are required, which also keeps
    ordinary queue creeping from qualifying. It has NOT been validated against any real collision
    (the available footage has none); outputs are always for human review."""

    name = "possible_collision"
    event_types = ("possible_collision",)
    STOP_S = 2.0
    MOVE_WIDTHS_PER_S = 0.8
    STOP_WIDTHS_PER_S = 0.15
    LOOKBACK_S = 1.5
    # -- gates added so that vehicles which merely stop together are NOT reported (each has a regression test) --
    SUDDEN_S = 2.5  # the pair must be stationary within this long of first contact; a gradual slowdown is not a crash
    MIN_CLOSING_WIDTHS = 0.15  # centre distance must have SHRUNK by this many box widths in the second before contact
    NEIGHBOUR_RADIUS_WIDTHS = 6.0
    STANDSTILL_FRACTION = 0.6  # >= this share of the other vehicles stationary (and >= MIN_OTHERS of them) = a standstill, not a crash
    MIN_OTHERS = 4

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.min_iou = ctx.config.threshold("collision_min_iou", 0.10)
        self.min_contact_s = ctx.config.threshold("collision_min_contact_s", 1.0)
        self.queue_peer_stops = int(ctx.config.threshold("queue_peer_stops", 3))
        self._contact: dict = {}  # (a,b) -> {"start": t, "peak_iou": x, "pre": (va, vb, closing)}
        self._fired: set = set()
        self._suppressed_keys: set = set()
        self.suppressed = {"not_closing": 0, "gradual_stop": 0, "stationary_queue": 0, "standstill": 0}

    def _centre_dist_w(self, a, b, t) -> float:
        """Distance between the two ground points at time t, in widths of the narrower box (None if unknown)."""
        sa, sb = self.ctx.history.sample_at(a, t), self.ctx.history.sample_at(b, t)
        if sa is None or sb is None:
            return None
        w = max(1.0, min(sa.bbox[2] - sa.bbox[0], sb.bbox[2] - sb.bbox[0]))
        return math.hypot(sa.ground[0] - sb.ground[0], sa.ground[1] - sb.ground[1]) / w

    def _context_gate(self, pair, c, t) -> tuple:
        """(suppress_reason | None, indicators). Looks at the OTHER vehicles around the pair: a stationary queue, or a
        whole-scene standstill, explains two vehicles stopping together without any collision."""
        a, b = pair
        sa = self.ctx.history.sample_at(a, t)
        others = [tid for tid in self.ctx.history.active(t) if tid not in pair]
        stopped_now, stopped_before, moving_near, near = 0, 0, 0, 0
        for tid in others:
            smp = self.ctx.history.samples(tid)
            v_now = self._speed_w(smp, t, 0.8)
            if v_now is None:
                continue
            v_before = self._speed_w(smp, c["start"], 0.8)
            is_stopped = v_now <= self.STOP_WIDTHS_PER_S
            stopped_now += is_stopped
            if sa is not None:
                last = smp[-1]
                w = max(1.0, sa.bbox[2] - sa.bbox[0])
                if math.hypot(last.ground[0] - sa.ground[0], last.ground[1] - sa.ground[1]) <= self.NEIGHBOUR_RADIUS_WIDTHS * w:
                    near += 1
                    moving_near += (not is_stopped)
                    stopped_before += is_stopped and v_before is not None and v_before <= self.STOP_WIDTHS_PER_S
        info = {"others_tracked": len(others), "others_stationary": stopped_now, "neighbours_within_6_widths": near,
                "neighbours_moving": moving_near, "neighbours_already_stationary_at_contact": stopped_before}
        if stopped_before >= self.queue_peer_stops:
            return "stationary_queue", info
        if len(others) >= self.MIN_OTHERS and stopped_now / len(others) >= self.STANDSTILL_FRACTION:
            return "standstill", info
        return None, info

    def _speed_w(self, samples, t, window) -> float:
        """Speed in box-widths per second over the last `window` s up to `t` (scale-free, so it does
        not depend on camera distance)."""
        pts = [s for s in samples if t - window <= s.t <= t]
        if len(pts) < 2:
            return None
        d = math.hypot(pts[-1].ground[0] - pts[0].ground[0], pts[-1].ground[1] - pts[0].ground[1])
        width = max(1.0, pts[-1].bbox[2] - pts[-1].bbox[0])
        dt = pts[-1].t - pts[0].t
        return d / width / dt if dt > 0 else None

    def on_frame(self, t, frame_index, observations) -> list:
        out = []
        boxes = {o.track_id: o for o in observations}
        ids = sorted(boxes)
        touching = set()
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                v = iou(boxes[a].bbox, boxes[b].bbox)
                if v >= self.min_iou:
                    touching.add((a, b))
                    c = self._contact.get((a, b))
                    if c is None:
                        sa, sb = self.ctx.history.samples(a), self.ctx.history.samples(b)
                        va, vb = self._speed_w(sa, t - 0.3, self.LOOKBACK_S), self._speed_w(sb, t - 0.3, self.LOOKBACK_S)
                        self._contact[(a, b)] = {"start": t, "peak_iou": v, "va": va, "vb": vb, "stopped_since": None}
                    else:
                        c["peak_iou"] = max(c["peak_iou"], v)
        for pair in list(self._contact):
            if pair not in touching and pair not in self._fired:
                # contact broke before they both stopped: ordinary passing/occlusion
                self._contact.pop(pair)
        for pair, c in list(self._contact.items()):
            if pair in self._fired or pair not in touching or t - c["start"] < self.min_contact_s:
                continue
            a, b = pair
            va, vb = c["va"], c["vb"]
            if va is None or vb is None or max(va, vb) < self.MOVE_WIDTHS_PER_S:
                continue  # neither was moving before contact: parked/queued vehicles touching in the image
            now_a = self._speed_w(self.ctx.history.samples(a), t, 0.8)
            now_b = self._speed_w(self.ctx.history.samples(b), t, 0.8)
            if now_a is None or now_b is None or max(now_a, now_b) > self.STOP_WIDTHS_PER_S:
                c["stopped_since"] = None
                continue
            c["stopped_since"] = c["stopped_since"] if c["stopped_since"] is not None else t
            if t - c["stopped_since"] < self.STOP_S:
                continue
            if pair in self._suppressed_keys:
                continue
            # --- gates: only evaluated once the pair has already looked like "approach + contact + joint stop" ---
            d_before, d_at = self._centre_dist_w(a, b, c["start"] - 1.0), self._centre_dist_w(a, b, c["start"])
            if d_before is None or d_at is None or d_before - d_at < self.MIN_CLOSING_WIDTHS:
                self._suppressed_keys.add(pair)
                self.suppressed["not_closing"] += 1
                continue
            if c["stopped_since"] - c["start"] > self.SUDDEN_S:
                self._suppressed_keys.add(pair)
                self.suppressed["gradual_stop"] += 1
                continue
            reason, ctx_info = self._context_gate(pair, c, t)
            if reason is not None:
                self._suppressed_keys.add(pair)
                self.suppressed[reason] += 1
                continue
            self._fired.add(pair)
            cls = {o: boxes[o].vehicle_class for o in pair}
            out.append(DetectedEvent(
                event_type="possible_collision", natural_key=f"collision:{a}:{b}", start_offset_s=c["start"], detected_offset_s=t,
                detector=self.name, track_ids=[a, b],
                involved=[_involved(a, "potentially_involved", cls[a], "box contact then joint stop; involvement not established"),
                          _involved(b, "potentially_involved", cls[b], "box contact then joint stop; involvement not established")],
                indicators={"contact_start_s": round(c["start"], 2), "peak_iou": round(c["peak_iou"], 3),
                            "pre_contact_speed_widths_per_s": [round(va, 2), round(vb, 2)], "joint_stop_s": round(t - c["stopped_since"], 2),
                            "closing_before_contact_widths": round(d_before - d_at, 2),
                            "time_contact_to_stop_s": round(c["stopped_since"] - c["start"], 2), "surroundings": ctx_info,
                            "gates_passed": ["moving_before_contact", "closing", "sudden_stop", "not_a_stationary_queue", "not_a_standstill"],
                            "inference": "2-D box overlap; contact in the world is not established"},
                severity="high", severity_basis="two moving vehicles stopped in box contact (candidate)", confidence_tier="low",
                confidence_basis="approach + sustained box contact + sudden joint stop outside a queue/standstill, from 2-D boxes; not validated on any real collision",
                uncertainty=["Box overlap can be occlusion or perspective, not physical contact.",
                             "No collision exists in the available validation footage: false-positive rate in dense traffic is uncharacterised.",
                             "Which vehicle (if either) caused anything is not established; proximity is not causation."],
                detector_config={"min_iou": self.min_iou, "min_contact_s": self.min_contact_s, "stop_s": self.STOP_S},
                key_frames=[(max(0.0, c["start"] - 1.0), "before contact", None), (c["start"], "contact begins", a), (t, "both stopped", a)],
                clip_window=(max(0.0, c["start"] - 5.0), t + 3.0), summary=f"Possible collision candidate: {a} and {b}",
            ))
        return out


DETECTOR_CLASSES = (StoppedVehicleDetector, CongestionDetector, OverspeedDetector, BrakingDetector, WrongWayDetector,
                    WrongLaneDetector, RapidLaneChangeDetector, RedLightDetector, IllegalTurnDetector, CollisionCandidateDetector)


def build_detectors(ctx: DetectorContext) -> list:
    """Instantiate every detector that can run on this camera. A detector whose required input is
    missing is simply not built (the capability endpoint reports why)."""
    cfg = ctx.config
    metric = ctx.geometry_validity == GeometryValidity.WORLD_VALIDATED
    out = [StoppedVehicleDetector(ctx), CongestionDetector(ctx), CollisionCandidateDetector(ctx)]
    if metric and cfg.speed_limit_kmh is not None:
        out.append(OverspeedDetector(ctx))
    if metric:
        out.append(BrakingDetector(ctx))
    if cfg.direction_zones:
        out.append(WrongWayDetector(ctx))
    if cfg.lanes:
        if any(l.get("allowed_classes") for l in cfg.lanes):
            out.append(WrongLaneDetector(ctx))
        out.append(RapidLaneChangeDetector(ctx))
    if cfg.signal and ctx.signal is not None:
        out.append(RedLightDetector(ctx))
    if cfg.turn_rules:
        out.append(IllegalTurnDetector(ctx))
    return out
