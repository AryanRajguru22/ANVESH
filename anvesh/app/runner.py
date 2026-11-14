"""Processing sessions: one camera/source -> the REAL streaming perception engine ->
detectors -> incidents + evidence.

    source (file | webcam | rtsp | rtmp | http | synthetic)
      -> ControlledSource (pause/resume/close)
      -> StreamingPipeline  (reader / worker / consumer threads, bounded queues)
           worker:   TapTracker(detector+tracker) -> StreamingPerceptionEngine -> on_frame
                       on_frame: track history, frame ring, live preview, frame-level detectors
           consumer: closed window -> WindowReasoner -> window-level detectors
      -> IncidentService.upsert -> evidence capture (key frames, clip) -> alert callbacks

IDENTITY. Every session gets its own random `run_id`, so tracker/episode ids carry the session
and two sessions over the same file never collide. A reconnect starts a new identity session
inside the engine (`Frame.epoch`), so a tracker id is never a vehicle identity across them.

NOTHING BLOCKS THE SERVER. The source is opened and the model is loaded on the session thread;
a camera that cannot connect leaves its session in `connecting`/`reconnecting`/`failed` and
affects no other session.

STATES (`runtime()["state"]`): starting, connecting, live, stalled, reconnecting, failed,
paused, stopped, completed, interrupted (the server restarted while the session was running).
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import threading
import time
import uuid

import cv2
import numpy as np

from anvesh.app import cameras as cams
from anvesh.app.detectors import (DetectorContext, Observation, SignalState, TrackHistory, build_detectors)
from anvesh.app.evidence import FrameRing, annotate_frame, decode_jpeg
from anvesh.app.incidents import SessionInfo
from anvesh.app.livespeed import LiveSpeed
from anvesh.app.officer import OfficerLayer
from anvesh.app.plate_reader import PlateReader
from anvesh.app.timeutil import add_seconds, iso, now_iso, now_utc, to_local_label
from anvesh.perception.schema_conversion import geometry_validity_for, motion_space_for
from anvesh.safety.stop_episodes import DisplacementCriterion
from anvesh.streaming.engine import StreamingPerceptionEngine, StreamHealth
from anvesh.streaming.pipeline import StreamingPipeline
from anvesh.streaming.queues import DropPolicy
from anvesh.streaming.reasoning import WindowReasoner
from anvesh.streaming.windows import WindowSpec

log = logging.getLogger(__name__)

TERMINAL_STATES = ("stopped", "completed", "failed", "interrupted")
_CLASS_COLOURS = {"car": (200, 180, 60), "bus": (60, 160, 255), "truck": (60, 200, 255), "motorcycle": (180, 90, 255), "bicycle": (120, 255, 120)}


class ControlledSource:
    """Wraps a FrameSource so a recorded source can be paused/resumed and any source closed promptly."""

    def __init__(self, inner, supports_pause: bool, max_stream_s: float = None) -> None:
        self._inner, self.supports_pause, self._max_s = inner, supports_pause, max_stream_s
        self._run = threading.Event()
        self._run.set()
        self._closed = False

    def fps(self) -> float:
        return self._inner.fps()

    def health(self):
        return self._inner.health()

    def close(self) -> None:
        self._closed = True
        self._run.set()
        self._inner.close()

    def pause(self) -> None:
        self._run.clear()

    def resume(self) -> None:
        self._run.set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def __getattr__(self, name):  # forwards `stream_time` only when the inner source has it
        if name == "stream_time":
            return getattr(self._inner, "stream_time")
        raise AttributeError(name)

    def __iter__(self):
        for frame in self._inner:
            while not self._run.wait(0.2):
                if self._closed:
                    return
            if self._closed:
                return
            if self._max_s is not None and frame.timestamp > self._max_s:
                self._inner.close()
                return
            yield frame


class TapTracker:
    """Delegates to the real tracker and keeps the frame it was just given, so the engine's
    `on_frame` callback (which gets detections but not pixels) can render evidence."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_image = None
        self.lost_buffer_frames = getattr(inner, "lost_buffer_frames", 30)
        self.simulated = bool(getattr(inner, "simulated", False))

    def reset(self) -> None:
        reset = getattr(self._inner, "reset", None)
        if reset is not None:
            reset()

    def update(self, image, frame_index, timestamp):
        self.last_image = image
        return self._inner.update(image, frame_index, timestamp)

    @property
    def model_id(self) -> str:
        return getattr(self._inner, "model_id", "scripted" if self.simulated else "unknown")


def _model_version(tracker) -> str:
    """Provenance label for evidence: a hash of the actual checkpoint file (`sha256-<12 hex>`), or `n/a`
    for a scripted/synthetic tracker that has no weights."""
    from pathlib import Path
    mid = getattr(tracker, "model_id", None)
    if mid and Path(str(mid)).is_file():
        from anvesh.perception.weights import checkpoint_version_label
        return checkpoint_version_label(mid)
    return "n/a"


def default_tracker_factory(model_path: str):
    def factory(camera: dict, config) -> object:
        if camera["source_type"] == "synthetic":
            from anvesh.app.synthetic import ScriptedTracker
            return ScriptedTracker(camera["source_uri"].split(":", 1)[-1] or "stopped_car")
        from anvesh.perception.tracking import ByteTrackYoloTracker
        from anvesh.perception.weights import require_weights_file
        # Fail closed: Ultralytics would otherwise try to DOWNLOAD a recognised weight name such as yolov8n.pt.
        require_weights_file(model_path)
        kwargs = {}
        tcfg = getattr(config, "tracker", None) or {}
        if "conf" in tcfg:
            kwargs["confidence_threshold"] = tcfg["conf"]
        if "config" in tcfg:
            kwargs["tracker_config"] = _repo_file(tcfg["config"])
        return ByteTrackYoloTracker(model_path, **kwargs)
    return factory


def _repo_file(rel: str) -> str:
    """A repo-relative tracker yaml, resolved and required to exist inside the repository (fail closed, never a download)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    p = (root / rel).resolve()
    if root not in p.parents or not p.is_file():
        raise FileNotFoundError(f"tracker config {rel!r} not found inside the repository")
    return str(p)


# A plate is roughly 0.25-0.3 of a car's body width (0.5 m of ~1.8 m). This is a HEURISTIC used only to say, honestly, that a plate in the
# current picture is far too small to read; it is not a plate measurement.
PLATE_TO_BOX_WIDTH = 0.27
PLATE_PX_MARGINAL, PLATE_PX_OK = 40, 80


def plate_readability(est_plate_px: float) -> str:
    if est_plate_px < PLATE_PX_MARGINAL:
        return "unreadable_too_small"
    return "marginal" if est_plate_px < PLATE_PX_OK else "candidate_size"


def heading_deg(samples: list, window_s: float = 1.0):
    """Image-space heading of a track in degrees (0 = right, 90 = down), from its ground-point
    displacement over the last `window_s`; None when it has barely moved. NOT a compass bearing."""
    if len(samples) < 2:
        return None
    end = samples[-1]
    start = next((s for s in samples if end.t - s.t <= window_s), samples[0])
    dx, dy = end.ground[0] - start.ground[0], end.ground[1] - start.ground[1]
    width = max(1.0, end.bbox[2] - end.bbox[0])
    if math.hypot(dx, dy) < 0.15 * width:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


class ProcessingSession:
    def __init__(self, services, camera_row: dict, session_id: str, user: str, tracker_factory, source_factory=None,
                 ring_fps: float = 10.0) -> None:
        self.s = services
        self.camera = camera_row
        self.session_id = session_id
        self.config = cams.parse_camera_config(cams.loads(camera_row["config_json"], {}))
        self.user = user
        self._tracker_factory = tracker_factory
        self._source_factory = source_factory
        self._ring_fps = ring_fps
        self._thread = None
        self._stop = threading.Event()
        self._state = "starting"
        self._state_lock = threading.Lock()
        self._error = None
        self.errors = collections.deque(maxlen=20)
        self.source = None
        self.pipeline = None
        self.engine = None
        self.signal = None
        self.profile = None
        self.started_wall = time.time()
        self.last_frame_wall = None
        self.last_jpeg = None
        self._last_preview = 0.0
        self.history = TrackHistory()
        self.ring = None
        self._frames_in = 0
        self._classes_seen: dict = {}  # track_id -> class (bounded by session length; counts only)
        self._latest_obs: list = []
        self._latest_t = 0.0
        self.windows_trend: collections.deque = collections.deque(maxlen=120)
        self.events_emitted = 0
        self.evidence_failures = 0
        self.detectors: list = []
        self._window_detectors: list = []
        self._frame_detectors: list = []
        self.reasoner = None
        self.session_info = None
        self.fps = None
        self._asset_keys_done: set = set()
        self.tracker_label = "unknown"
        self.live_speed = None
        self._frame_size = None
        self._live_tracks: list = []
        self._duration_s = None
        self.released = False
        self._final_unique = None
        self.simulated = camera_row["source_type"] == "synthetic"
        self.officer = None
        self.plate_reader = None
        self._officer_jam: dict = {}
        self._officer_tiles: list = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"anvesh-session-{self.session_id}", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 15.0) -> None:
        self._stop.set()
        if self.pipeline is not None:
            self.pipeline.stop()
        elif self.source is not None:
            self.source.close()
        if self._thread is not None:
            self._thread.join(join_timeout)

    def pause(self) -> None:
        if self.source is None or not self.source.supports_pause:
            raise ValueError("pause is only supported for recorded sources")
        self.source.pause()

    def resume(self) -> None:
        if self.source is not None:
            self.source.resume()

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state

    def _release(self) -> None:
        """Free what a finished session holds (frame ring of JPEGs, track history, preview, the detector/tracker and its GPU
        memory). Counts shown by `runtime()` are kept; the evidence already written to disk is unaffected."""
        self._final_unique = self._unique_counts()
        self._classes_seen = {}
        self._latest_obs = []
        self.last_jpeg = None
        if self.plate_reader is not None:
            self.plate_reader.stop()
        self.history.clear()
        if self.ring is not None:
            self.ring.clear()
        if getattr(self, "tracker", None) is not None:
            self.tracker.last_image = None
            self.tracker._inner = None
        if self.engine is not None:
            self.engine._tracker = None
        self.released = True

    def _unique_counts(self) -> dict:
        unique: dict = {}
        for cls in self._classes_seen.values():
            unique[cls] = unique.get(cls, 0) + 1
        return unique

    def _db_update(self, **cols) -> None:
        sets = ", ".join(f"{k} = ?" for k in cols)
        with self.s.db.write() as conn:
            conn.execute(f"UPDATE processing_sessions SET {sets} WHERE session_id = ?", list(cols.values()) + [self.session_id])

    # -- the session thread -------------------------------------------------

    def _run(self) -> None:
        final_state, error = "completed", None
        try:
            self._set_state("connecting")
            self._build()
            stats = self.pipeline.run()
            for ev in self._finish_detectors():
                self._emit(ev)
            final_state = "stopped" if self._stop.is_set() else "completed"
            final = self._final_stats(stats)
            end_health = self.source.health() if self.source is not None else None
            if final_state == "completed" and end_health is not None and end_health.state == "ended" and end_health.last_error:
                final["source_warning"] = end_health.last_error  # a completed session on a file that ended short is not a clean one
                self.errors.append("source: " + end_health.last_error)
            self._db_update(stats_json=cams.dumps(final))
        except Exception as exc:  # noqa: BLE001 - a failed session must not take the process down
            log.exception("session %s failed", self.session_id)
            final_state, error = "failed", f"{type(exc).__name__}: {exc}"
            self.errors.append(error)
        finally:
            self._error = error
            health = self.source.health() if self.source is not None else None
            if final_state == "completed" and health is not None and health.state == "failed":
                final_state, self._error = "failed", health.last_error or "source failed"
            self._set_state(final_state)
            self._db_update(state=final_state, ended_at=now_iso(), error=self._error)
            self._release()
            self.s.audit.append("system", "session.end", "session", self.session_id, {"state": final_state, "error": self._error})

    def _build(self) -> None:
        cam, cfg = self.camera, self.config
        if cfg.calibration:
            self.profile = cams.build_calibration_profile(f"{cam['camera_id']}-cal", cam["camera_id"], cfg.calibration)
        geometry = geometry_validity_for(self.profile)
        motion = motion_space_for(self.profile)

        inner = self._source_factory(cam, cfg) if self._source_factory else cams.build_frame_source(
            cam["source_type"], cam["source_uri"], cam["nominal_fps"], cfg)
        adapter = cams.ADAPTERS[cam["source_type"]]
        self.source = ControlledSource(inner, adapter.supports_pause, cfg.max_stream_s)
        tracker = TapTracker(self._tracker_factory(cam, cfg))
        self.tracker = tracker
        self.tracker_label = tracker.model_id
        self.fps = float(self.source.fps())
        self.live_speed = LiveSpeed(self.profile, cfg.speed_zone, frame_size=lambda: self._frame_size)
        self._duration_s = self._source_duration(cam, cfg)
        if not self.simulated and not cfg.threshold("plate_reader_off", 0):
            self.plate_reader = PlateReader(min_vehicle_px=int(cfg.threshold("plate_min_vehicle_px", 110)))
            self.plate_reader.start()
        self.officer = OfficerLayer(self.s.config, cam, cfg, self.history, self.simulated, self.plate_reader)

        window_s = cfg.threshold("window_s", 10.0)
        spec = WindowSpec(window_s=window_s)
        displacement = None
        if cfg.stop_criterion == "displacement":
            displacement = DisplacementCriterion(
                image_radius_px=cfg.threshold("stop_radius_px", 7.5), anchor_window_s=cfg.threshold("dwell_s", 3.0))
        self.engine = StreamingPerceptionEngine(
            camera_id=cam["camera_id"], tracker=tracker, fps=self.fps, windows=spec, calibration_profile=self.profile,
            run_id=self.session_id.replace("SES-", "").lower(), model_id=str(tracker.model_id), model_version=_model_version(tracker),
            on_frame=self._on_frame, stop_displacement=displacement,
        )
        self.reasoner = WindowReasoner(cam["camera_id"], calibration_profile=self.profile,
                                       max_capacity_vehicles=cfg.threshold("max_capacity_vehicles", 20))
        if cfg.signal:
            self.signal = SignalState(cfg.signal["mode"], cfg.signal.get("schedule"))
        ctx = DetectorContext(camera_id=cam["camera_id"], fps=self.fps, config=cfg, history=self.history, profile=self.profile,
                              geometry_validity=geometry, motion_space=motion, signal=self.signal)
        self.detectors = build_detectors(ctx)
        self._window_detectors = [d for d in self.detectors if "on_window" in type(d).__dict__]
        self._frame_detectors = [d for d in self.detectors if "on_frame" in type(d).__dict__]
        ring_s = window_s + cfg.threshold("dwell_s", 3.0) + 30.0
        self.ring = FrameRing(max_seconds=ring_s, max_fps=self._ring_fps)

        rec_start, basis = self._recording_start(adapter)
        self.session_info = SessionInfo(
            self.session_id, cam["camera_id"], cam["data_class"], rec_start, basis,
            location={"label": cam["location_label"], "latitude": cam["latitude"], "longitude": cam["longitude"],
                      "source": "configured_camera_location (approximate; not the vehicle's position)"},
            utc_offset_minutes=cam["utc_offset_minutes"])
        policy = DropPolicy.DROP_OLDEST if adapter.live else DropPolicy.BLOCK
        self.pipeline = StreamingPipeline(self.source, self.engine, on_window=self._on_window, frame_policy=policy)
        self._db_update(recording_start_utc=rec_start, time_basis=basis, state="running")
        self._set_state("connecting")

    def _recording_start(self, adapter) -> tuple:
        if adapter.live:
            return now_iso(), "live_capture"
        if self.config.recording_start_utc:
            return self.config.recording_start_utc, "declared_recording_start"
        return now_iso(), "processing_clock"

    # -- per frame (worker thread) -----------------------------------------------

    def _on_frame(self, frame_index: int, timestamp: float, identified: list) -> None:
        image = self.tap_image()
        obs = []
        for d in identified:
            det = d.detection
            obs.append(Observation(d.track_id, det.vehicle_class.value, det.bbox_xyxy, det.ground_point, det.confidence))
            self._classes_seen.setdefault(d.track_id, det.vehicle_class.value)
        self.history.update(timestamp, frame_index, obs)
        self._latest_obs, self._latest_t = obs, timestamp
        self._frames_in += 1
        self.last_frame_wall = time.time()
        if image is not None:
            if self._frame_size is None:
                self._frame_size = (image.shape[1], image.shape[0])
                for d in self.detectors:
                    d.ctx.frame_size = self._frame_size
            self.ring.add(frame_index, timestamp, image)
        for det in self._frame_detectors:
            try:
                for ev in det.on_frame(timestamp, frame_index, obs):
                    self._emit(ev)
            except Exception as exc:  # noqa: BLE001 - one bad detector must not stop perception
                self._detector_error(det, exc)
        if image is not None and time.time() - self._last_preview >= 0.1:
            self._last_preview = time.time()
            self._render_preview(image, timestamp, obs)

    def tap_image(self):
        return self.tracker.last_image

    def _detector_error(self, det, exc) -> None:
        msg = f"detector {det.name} error: {type(exc).__name__}: {exc}"
        log.exception(msg)
        self.errors.append(msg)

    def _thumb(self, width: int = 480):
        """A small JPEG of the annotated frame as it is now (for the alert cards); None when there is no frame yet."""
        jpeg = self.last_jpeg
        if jpeg is None:
            return None
        try:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            f = width / img.shape[1]
            img = cv2.resize(img, (width, int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            return buf.tobytes() if ok else None
        except Exception:  # noqa: BLE001
            return None

    def _render_preview(self, image, timestamp: float, obs: list) -> None:
        live = []
        for o in obs:
            x1, y1, x2, y2 = (int(v) for v in o.bbox)
            samples = self.history.samples(o.track_id)
            sp = self.live_speed.estimate(o.track_id, samples) if self.live_speed is not None else {"status": "no_metric_calibration", "speed_kmh": None}
            h = heading_deg(samples)
            lane = self.officer.lanes.lane_of(o.ground) if self.officer is not None else None
            live.append({"track_id": o.track_id.rsplit("-t", 1)[-1], "track_full": o.track_id, "vehicle_class": o.vehicle_class,
                         "confidence": round(float(o.confidence), 2),
                         "heading_deg": None if h is None else round(h), "speed_kmh": sp.get("speed_kmh"), "speed_error_kmh": sp.get("error_kmh"),
                         "speed_status": sp["status"], "age_s": round(samples[-1].t - samples[0].t, 1) if samples else 0.0,
                         "box_w_px": round(x2 - x1), "est_plate_px": round(PLATE_TO_BOX_WIDTH * (x2 - x1)),
                         "plate_readability": plate_readability(PLATE_TO_BOX_WIDTH * (x2 - x1)),
                         "lane": lane["id"] if lane else None, "lane_label": lane["label"] if lane else None})
        if self.live_speed is not None:
            self.live_speed.forget_absent(o.track_id for o in obs)
        self._live_tracks = [{k: v for k, v in r.items() if k != "track_full"} for r in live]
        jam = self.officer.on_frame(timestamp, image, obs, live, self._thumb)
        self._officer_jam = jam
        out = self.officer.render(image, timestamp, obs, live, jam, self.config.speed_zone)
        self._draw_signal_line(out, out.shape[1] / image.shape[1])
        approx = bool(self.profile is not None and getattr(self.profile, "is_marking_standard_validated", False))
        self._officer_tiles = self.officer.tiles(timestamp, jam, self.config.speed_limit_kmh, self.live_speed is not None and self.profile is not None,
                                                 approx, live)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            self.last_jpeg = buf.tobytes()

    def _draw_signal_line(self, out, scale: float) -> None:
        cfg = self.config
        if cfg.signal:
            (x1, y1), (x2, y2) = cfg.signal["stop_line"]
            state = self.signal.at(self._latest_t) if self.signal else None
            colour = {"red": (0, 0, 255), "green": (0, 200, 0), "amber": (0, 180, 255)}.get(state, (150, 150, 150))
            cv2.line(out, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), colour, 4)

    def officer_state(self) -> dict:
        """Everything the one-screen officer view shows, in one small payload (polled once a second)."""
        rt = self.runtime()
        o = self.officer
        t = rt["stream_time_s"]
        return {
            "session_id": self.session_id, "camera_id": self.camera["camera_id"], "camera_name": self.camera["name"], "state": rt["state"],
            "simulated": self.simulated, "data_class": self.camera["data_class"], "t": t, "duration_s": rt["duration_s"], "progress": rt["progress"],
            "vehicles": rt["concurrent_vehicles"], "vehicles_total": rt["unique_tracks_total"], "by_class": rt["concurrent_by_class"],
            "fps": round(rt["processing_fps"], 1), "frames_dropped": rt["frames_dropped"], "supports_pause": rt["supports_pause"],
            "tiles": list(self._officer_tiles), "jam": self._officer_jam, "alerts": o.alerts.recent(14) if o else [],
            "alert_counts": dict(o.alerts.counts) if o else {}, "plates": o.plates.summary() if o is not None and o.plates is not None else None,
            "geometry": rt["geometry"], "errors": rt["errors"],
        }

    # -- per window (consumer thread) ---------------------------------------------

    def _on_window(self, snapshot) -> None:
        try:
            analysis = self.reasoner.process(snapshot)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"window reasoning error: {type(exc).__name__}: {exc}")
            log.exception("window reasoning failed")
            return
        if analysis.aggregation is not None:
            ts = analysis.aggregation.traffic_state
            self.windows_trend.append({
                "window_id": snapshot.window_id, "start": snapshot.start, "end": snapshot.end, "level": ts.congestion_level.value,
                "vehicle_count": ts.vehicle_count, "flow_rate": round(ts.flow_rate, 3), "occupancy": round(ts.occupancy, 3),
                "availability": snapshot.availability.value, "degraded": bool(analysis.degraded),
            })
        else:
            self.windows_trend.append({"window_id": snapshot.window_id, "start": snapshot.start, "end": snapshot.end, "level": None,
                                       "availability": snapshot.availability.value, "degraded": True, "reason": analysis.reason})
        for det in self._window_detectors:
            try:
                for ev in det.on_window(analysis):
                    self._emit(ev)
            except Exception as exc:  # noqa: BLE001
                self._detector_error(det, exc)

    def _finish_detectors(self) -> list:
        out = []
        t = self.engine.stream_time or 0.0
        for det in self.detectors:
            try:
                out += det.finish(t)
            except Exception as exc:  # noqa: BLE001
                self._detector_error(det, exc)
        return out

    # -- incidents + evidence -----------------------------------------------------

    def _emit(self, ev) -> None:
        try:
            incident, created, changes = self.s.incidents.upsert(self.session_info, ev)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"incident persistence error: {type(exc).__name__}: {exc}")
            log.exception("incident upsert failed")
            return
        if created:
            self.events_emitted += 1
            if self.officer is not None:
                try:
                    self.officer.on_event(ev, incident, self._thumb())
                except Exception as exc:  # noqa: BLE001 - the officer view must never break incident handling
                    self.errors.append(f"officer alert error: {type(exc).__name__}: {exc}")
            self._capture_evidence(incident, ev, initial=True)
        elif "ended" in changes:
            self._capture_evidence(incident, ev, initial=False)

    def _banner(self, offset_s: float) -> str:
        utc = add_seconds(self.session_info.recording_start_utc, offset_s)
        tag = {"processing_clock": " (processing clock)", "live_capture": "", "declared_recording_start": ""}[self.session_info.time_basis]
        return (f"{self.camera['camera_id']} | video t={offset_s:.1f}s | {to_local_label(utc, self.session_info.utc_offset_minutes)}{tag}"
                + (" | SIMULATED" if self.simulated else ""))

    def _render_at(self, jpeg: bytes, t: float, highlight: set):
        img = decode_jpeg(jpeg)
        boxes = []
        for tid, cls, bbox in self.history.boxes_at(t):
            boxes.append((tid, f"{tid.rsplit('-t', 1)[-1]} {cls}" if tid in highlight else "", bbox))
        return annotate_frame(img, boxes, highlight, self._banner(t))

    def _capture_evidence(self, incident: dict, ev, initial: bool) -> None:
        iid = incident["incident_id"]
        highlight = set(ev.track_ids)
        try:
            frames = ev.key_frames if initial else [(ev.end_offset_s, "episode ended", ev.track_ids[0] if ev.track_ids else None)]
            thumb_done = False
            for offset, label, tid in frames:
                hit = self.ring.nearest(offset)
                if hit is None:
                    continue
                t, idx, jpeg = hit
                meta = {"label": label, "requested_offset_s": round(offset, 3), "actual_offset_s": round(t, 3),
                        "gap_s": round(abs(t - offset), 3), "banner": self._banner(t)}
                s_at = self.history.sample_at(tid, t) if tid else None
                if s_at is not None:
                    meta["track_bbox"] = [round(v, 1) for v in s_at.bbox]
                    meta["vehicle_class"] = s_at.vehicle_class
                self.s.evidence.add_bytes(iid, "frame", jpeg, ".jpg", "image/jpeg", offset_s=t, frame_index=idx, track_id=tid,
                                          meta=dict(meta, note="unannotated frame as captured"))
                ann = self._render_at(jpeg, t, highlight)
                ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    self.s.evidence.add_bytes(iid, "frame_annotated", buf.tobytes(), ".jpg", "image/jpeg", offset_s=t, frame_index=idx,
                                              track_id=tid, meta=dict(meta, note="boxes and banner drawn for review"))
                    if not thumb_done and initial:
                        h, w = ann.shape[:2]
                        small = cv2.resize(ann, (320, max(1, int(320 * h / w))))
                        ok2, b2 = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok2:
                            self.s.evidence.add_bytes(iid, "thumbnail", b2.tobytes(), ".jpg", "image/jpeg", offset_s=t, frame_index=idx,
                                                      track_id=tid, meta={"label": label})
                            thumb_done = True
            if initial and ev.clip_window is not None:
                t0, t1 = ev.clip_window
                items = self.ring.between(t0, t1)[:200]
                if len(items) >= 2:
                    clip_frames = [self._render_at(j, t, highlight) for (t, _i, j) in items]
                    self.s.evidence.add_clip(iid, clip_frames, fps=self._ring_fps, offset_s=items[0][0],
                                             meta={"start_offset_s": items[0][0], "end_offset_s": items[-1][0],
                                                   "requested_window_s": [round(t0, 2), round(t1, 2)],
                                                   "annotated": True, "source_session": self.session_id})
            assets = self.s.evidence.list(iid)
            level = self.s.incidents.set_evidence_completeness(iid, assets, bool(ev.track_ids))
            if initial:
                n = sum(1 for a in assets if a["kind"] == "frame")
                self.s.incidents.add_timeline(iid, "evidence", f"{n} key frame(s) and {sum(1 for a in assets if a['kind'] == 'clip')} clip captured "
                                              f"(completeness: {level})", "system")
        except Exception as exc:  # noqa: BLE001 - evidence failure must be visible, never silent
            self.evidence_failures += 1
            msg = f"evidence capture failed: {type(exc).__name__}: {exc}"
            self.errors.append(msg)
            log.exception(msg)
            self.s.incidents.add_timeline(iid, "evidence_error", msg, "system")

    # -- observability ------------------------------------------------------------

    def runtime(self) -> dict:
        with self._state_lock:
            state = self._state
        eng, src = self.engine, self.source
        health = src.health() if src is not None else None
        if state in ("connecting", "live") and eng is not None:
            if health is not None and health.state == "reconnecting":
                state = "reconnecting"
            elif src is not None and src.paused:
                state = "paused"
            elif eng.health == StreamHealth.LIVE and self._frames_in:
                state = "live"
            elif eng.health in (StreamHealth.STALLED, StreamHealth.DOWN):
                state = "stalled"
            elif self._frames_in == 0:
                state = "connecting"
        elif state == "paused":
            pass
        now = time.time()
        rec = eng.recorder.summary_ms() if eng is not None else {}
        counts: dict = {}
        for o in self._latest_obs:
            counts[o.vehicle_class] = counts.get(o.vehicle_class, 0) + 1
        unique = self._final_unique if self._final_unique is not None else self._unique_counts()
        return {
            "session_id": self.session_id, "camera_id": self.camera["camera_id"], "state": state, "error": self._error,
            "source_type": self.camera["source_type"], "data_class": self.camera["data_class"], "simulated": self.simulated,
            "tracker": self.tracker_label, "fps_nominal": self.fps,
            "frames_processed": eng.frames_processed if eng else 0, "frames_dropped": eng.frames_dropped if eng else 0,
            "windows_emitted": eng.windows_emitted if eng else 0,
            "stream_time_s": self._latest_t, "last_frame_age_s": (now - self.last_frame_wall) if self.last_frame_wall else None,
            "wall_elapsed_s": now - self.started_wall,
            "processing_fps": (eng.frames_processed / max(now - self.started_wall, 1e-6)) if eng else 0.0,
            "engine_health": eng.health.value if eng else "no_signal",
            "source_health": dataclasses.asdict(health) if health is not None else None,
            "queues": {"frames": self.pipeline.frame_queue.stats(), "windows": self.pipeline.window_queue.stats()} if self.pipeline else None,
            "latency": rec,
            "concurrent_vehicles": len(self._latest_obs), "concurrent_by_class": counts, "unique_tracks_by_class": unique,
            "unique_tracks_total": sum(unique.values()), "events_emitted": self.events_emitted,
            "evidence_failures": self.evidence_failures,
            "ring": {"frames": len(self.ring) if self.ring else 0, "bytes": self.ring.size_bytes() if self.ring else 0},
            "congestion_trend": list(self.windows_trend)[-30:],
            "supports_pause": bool(src.supports_pause) if src else False,
            "signal_state": self.signal.at(self._latest_t) if self.signal else None,
            "signal_mode": self.signal.mode if self.signal else None,
            "detectors": [d.name for d in self.detectors],
            "errors": list(self.errors)[-5:],
            "recording_start_utc": self.session_info.recording_start_utc if self.session_info else None,
            "time_basis": self.session_info.time_basis if self.session_info else None,
            "duration_s": self._duration_s,
            "progress": (min(1.0, self._latest_t / self._duration_s) if self._duration_s else None),
            "geometry": cams.geometry_label(self.profile),
            "speed_zone": bool(self.config.speed_zone),
            "live_tracks": sorted(self._live_tracks, key=lambda r: -r["age_s"])[:30],
        }

    @staticmethod
    def _source_duration(cam: dict, cfg):
        """Length of a recorded file in seconds (capped by max_stream_s); None for live sources."""
        if cam["source_type"] != "file":
            return None
        try:
            c = cv2.VideoCapture(str(cam["source_uri"]))
            n, f = c.get(cv2.CAP_PROP_FRAME_COUNT), c.get(cv2.CAP_PROP_FPS)
            c.release()
            full = n / f if n and f else None
        except Exception:  # noqa: BLE001
            full = None
        if full is None:
            return cfg.max_stream_s
        return min(full, cfg.max_stream_s) if cfg.max_stream_s else full

    def _final_stats(self, pipeline_stats: dict) -> dict:
        rt = self.runtime()
        keep = ("frames_processed", "frames_dropped", "windows_emitted", "processing_fps", "wall_elapsed_s", "events_emitted",
                "unique_tracks_total", "unique_tracks_by_class", "evidence_failures", "errors", "tracker", "latency")
        out = {k: rt[k] for k in keep if k in rt}
        out["engine"] = pipeline_stats.get("engine")
        out["queues"] = {"frames": pipeline_stats.get("frame_queue"), "windows": pipeline_stats.get("window_queue")}
        return out


class SessionManager:
    def __init__(self, services, tracker_factory=None, source_factory=None) -> None:
        self.s = services
        self._tracker_factory = tracker_factory or default_tracker_factory(services.config.model_path)
        self._source_factory = source_factory
        self._active: dict = {}
        self._lock = threading.Lock()
        self.mark_interrupted()

    def mark_interrupted(self) -> None:
        """Sessions left 'running' by a previous process died with it; say so instead of leaving them live."""
        with self.s.db.write() as conn:
            conn.execute("UPDATE processing_sessions SET state = 'interrupted', ended_at = ?, error = 'server restarted while the session was running' "
                         "WHERE state IN ('starting','running','connecting','live','paused')", (now_iso(),))

    def start(self, camera_id: str, user: str, role: str = None) -> dict:
        cam = self.s.cameras.get_raw(camera_id)
        if cam is None:
            raise LookupError(f"no such camera {camera_id!r}")
        if not cam["enabled"]:
            raise ValueError("camera is disabled")
        # Re-check the stored source: a camera row written before a source rule existed must not be started (or replayed).
        self.s.cameras.validate_registered_source(cam["source_type"], cam["source_uri"], cam["nominal_fps"], cam["data_class"])
        with self._lock:
            if any(p.camera["camera_id"] == camera_id and p.runtime()["state"] not in TERMINAL_STATES for p in self._active.values()):
                raise cams.ConfigError("camera already has an active session")
            self._evict_terminal()
            session_id = "SES-" + uuid.uuid4().hex[:10].upper()
            snapshot = {k: cam[k] for k in ("camera_id", "name", "source_type", "nominal_fps", "location_label", "data_class")}
            snapshot["source"] = cams.redact_uri(cam["source_uri"])
            snapshot["config"] = cams.loads(cam["config_json"], {})
            with self.s.db.write() as conn:
                conn.execute(
                    "INSERT INTO processing_sessions(session_id, camera_id, run_id, state, data_class, started_at, time_basis, recording_start_utc, "
                    "source_snapshot_json, requested_by) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (session_id, camera_id, session_id.replace("SES-", "").lower(), "starting", cam["data_class"], now_iso(),
                     "processing_clock", now_iso(), cams.dumps(snapshot), user))
            sess = ProcessingSession(self.s, cam, session_id, user, self._tracker_factory, self._source_factory)
            self._active[session_id] = sess
        self.s.audit.append(user, "session.start", "session", session_id,
                            {"camera_id": camera_id, "source_type": cam["source_type"], "data_class": cam["data_class"]}, actor_role=role)
        sess.start()
        return {"session_id": session_id, "camera_id": camera_id, "state": "starting"}

    MAX_TERMINAL_KEPT = 30

    def _evict_terminal(self) -> None:
        """Keep at most MAX_TERMINAL_KEPT finished (already released) sessions in memory; the DB row remains authoritative."""
        done = [sid for sid, p in self._active.items() if p.runtime()["state"] in TERMINAL_STATES]
        for sid in done[:max(0, len(done) - self.MAX_TERMINAL_KEPT)]:
            self._active.pop(sid, None)

    def get(self, session_id: str):
        return self._active.get(session_id)

    def stop(self, session_id: str, user: str, role: str = None) -> None:
        sess = self._active.get(session_id)
        if sess is None:
            raise LookupError(f"session {session_id} is not active")
        sess.stop()
        self.s.audit.append(user, "session.stop", "session", session_id, {}, actor_role=role)

    def pause(self, session_id: str, user: str, role: str = None) -> None:
        sess = self._active.get(session_id)
        if sess is None:
            raise LookupError(f"session {session_id} is not active")
        sess.pause()
        sess._set_state("paused")
        self.s.audit.append(user, "session.pause", "session", session_id, {}, actor_role=role)

    def resume(self, session_id: str, user: str, role: str = None) -> None:
        sess = self._active.get(session_id)
        if sess is None:
            raise LookupError(f"session {session_id} is not active")
        sess.resume()
        sess._set_state("live")
        self.s.audit.append(user, "session.resume", "session", session_id, {}, actor_role=role)

    def runtime(self, session_id: str):
        sess = self._active.get(session_id)
        return sess.runtime() if sess else None

    def active_runtimes(self) -> list:
        return [p.runtime() for p in list(self._active.values()) if p.runtime()["state"] not in TERMINAL_STATES]

    def shutdown(self) -> None:
        for sess in list(self._active.values()):
            sess.stop(join_timeout=10.0)
