"""Camera / source configuration and the source-adapter registry.

A camera row is (identity, source, location, data class, detection configuration). The
detection configuration (`CameraConfig`) is where the operator supplies the things ANVESH
cannot invent: direction zones, lane polygons with allowed classes, a stop line and signal
feed, prohibited turns, a speed limit and (optionally) a calibration. Each is validated
here so a malformed config is a 422 at the API, not a crash in a worker thread later.

SOURCE ADAPTERS. `file`, `webcam`, `rtsp`, `rtmp` and `http_stream` all end up as an
`anvesh.perception.ingestion.FrameSource`:
  - `file`       -> VideoFileFrameSource (paceable; pause/resume capable; finite)
  - `webcam`     -> RtspFrameSource with a DirectShow capture (reconnect, health, epochs)
  - `rtsp`/`rtmp`/`http_stream` -> RtspFrameSource over OpenCV/FFmpeg (the same reconnect,
    backoff, gap and epoch machinery; HLS/HTTP/RTMP are whatever the bundled FFmpeg demuxes)
  - `synthetic`  -> a deterministic generated clip; ALWAYS data class `simulated`
LIVE VALIDATION STATUS: only `file` has been exercised on real media here. The live adapters
are unit-tested against injected fake captures (reconnect/backoff/failure paths) and the RTSP
adapter additionally against a local fake stream; no real camera or RTSP/RTMP server has
been used. Registering a camera does not open it -- opening happens on the session thread,
never on an HTTP request thread (FFmpeg's open timeout can block for ~30 s).
"""

from __future__ import annotations

import dataclasses
import math
import re
import uuid
from pathlib import Path

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.timeutil import now_iso, parse_iso

SOURCE_TYPES = ("file", "webcam", "rtsp", "rtmp", "http_stream", "synthetic")
LIVE_TYPES = ("webcam", "rtsp", "rtmp", "http_stream")
DATA_CLASSES = ("operational", "demo", "simulated")
# Only these extensions may be registered as a recorded-file source (and therefore replayed through /sessions/<id>/video).
VIDEO_EXT = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,62}$")


class ConfigError(ValueError):
    pass


# -- detection configuration -------------------------------------------------


def _poly(value, name: str, min_points: int = 3) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) < min_points:
        raise ConfigError(f"{name}: need at least {min_points} [x, y] points")
    pts = []
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in p):
            raise ConfigError(f"{name}: every point must be [x, y] numbers, got {p!r}")
        pts.append((float(p[0]), float(p[1])))
    return tuple(pts)


def _vec(value, name: str) -> tuple:
    p = _poly([value], name, 1)[0]
    if p == (0.0, 0.0):
        raise ConfigError(f"{name}: direction vector must be non-zero")
    return p


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    direction_zones: tuple = ()  # dicts: zone_id, polygon, expected (dx,dy), min_displacement_px, min_duration_s
    lanes: tuple = ()  # dicts: lane_id, vertices, allowed_classes (tuple|None)
    signal: dict = None  # stop_line [[x,y],[x,y]], approach (dx,dy), mode, schedule
    turn_rules: tuple = ()  # dicts: rule_id, entry, exit, label
    speed_limit_kmh: float = None
    calibration: dict = None  # reference_points, validation (kept verbatim for the API)
    thresholds: dict = dataclasses.field(default_factory=dict)
    recording_start_utc: str = None
    realtime_factor: float = None  # file sources: None = as fast as possible
    stop_criterion: str = "step_speed"  # "displacement" is OPT-IN: its false-positive rate is not yet evaluated
    max_stream_s: float = None  # recorded sources: process only the first N seconds (long recordings)
    speed_zone: tuple = ()  # image polygon where the calibration is trusted; speed is neither shown nor judged outside it
    tracker: dict = dataclasses.field(default_factory=dict)  # optional {"conf": detector score floor, "config": repo-relative ByteTrack yaml}
    raw: dict = dataclasses.field(default_factory=dict, compare=False)

    def threshold(self, key: str, default):
        return self.thresholds.get(key, default)


_THRESHOLD_TYPES = {
    "window_s": float, "dwell_s": float, "long_stop_s": float, "max_capacity_vehicles": int,
    "collision_min_contact_s": float, "collision_min_iou": float, "min_confidence": float,
    "stop_radius_px": float,
    "queue_peer_stops": int,
    "congestion_close_grace_s": float,
    "jam_min_vehicles": int,
    "plate_reader_off": int,
    "plate_min_vehicle_px": int,
    "plates_only": int,
}


def parse_camera_config(raw: dict) -> CameraConfig:
    """Validate a camera's detection configuration. Any structural problem is a ConfigError (a 422 at the API), never a
    TypeError/AttributeError escaping from deep inside the parser."""
    try:
        return _parse_camera_config(raw)
    except ConfigError:
        raise
    except (TypeError, AttributeError, KeyError, ValueError, OverflowError) as exc:
        raise ConfigError(f"malformed config ({type(exc).__name__}: {exc})") from None


def _items(raw: dict, key: str) -> list:
    """`raw[key]` as a list of objects (or empty)."""
    v = raw.get(key)
    if v is None or v == []:
        return []
    if not isinstance(v, list) or not all(isinstance(i, dict) for i in v):
        raise ConfigError(f"{key} must be a list of objects")
    return v


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    return float(value)


def _parse_camera_config(raw: dict) -> CameraConfig:
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ConfigError("config must be an object")
    known = {"direction_zones", "lanes", "signal", "turn_rules", "speed_limit_kmh", "calibration", "thresholds",
             "recording_start_utc", "realtime_factor", "stop_criterion", "max_stream_s", "speed_zone", "tracker"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")

    zones = []
    for i, z in enumerate(_items(raw, "direction_zones")):
        zid = str(z.get("zone_id") or f"zone{i + 1}")
        zones.append({
            "zone_id": zid, "polygon": _poly(z.get("polygon"), f"direction_zones[{zid}].polygon"),
            "expected": _vec(z.get("expected"), f"direction_zones[{zid}].expected"),
            "min_displacement_px": _finite(z.get("min_displacement_px", 60.0), f"direction_zones[{zid}].min_displacement_px"),
            "min_duration_s": _finite(z.get("min_duration_s", 1.5), f"direction_zones[{zid}].min_duration_s"),
        })

    lanes = []
    for i, l in enumerate(_items(raw, "lanes")):
        lid = str(l.get("lane_id") or f"L{i + 1}")
        allowed = l.get("allowed_classes")
        if allowed is not None and (not isinstance(allowed, list) or not all(isinstance(a, str) for a in allowed)):
            raise ConfigError(f"lanes[{lid}].allowed_classes must be a list of class names")
        lanes.append({"lane_id": lid, "vertices": _poly(l.get("vertices"), f"lanes[{lid}].vertices"),
                      "allowed_classes": tuple(allowed) if allowed is not None else None})
    if len({l["lane_id"] for l in lanes}) != len(lanes):
        raise ConfigError("duplicate lane_id")

    signal = None
    if raw.get("signal"):
        s = raw["signal"]
        if not isinstance(s, dict):
            raise ConfigError("signal must be an object")
        mode = s.get("mode", "external")
        if mode not in ("external", "schedule", "manual"):
            raise ConfigError("signal.mode must be external, schedule or manual")
        signal = {"stop_line": _poly(s.get("stop_line"), "signal.stop_line", 2), "mode": mode,
                  "approach": _vec(s.get("approach", [0, 1]), "signal.approach")}
        if len(signal["stop_line"]) != 2:
            raise ConfigError("signal.stop_line must be exactly two points")
        if mode == "schedule":
            sch = s.get("schedule") or {}
            if not isinstance(sch, dict):
                raise ConfigError("signal.schedule must be an object")
            try:
                signal["schedule"] = {k: float(sch[k]) for k in ("red_s", "green_s", "amber_s")}
                signal["schedule"]["offset_s"] = float(sch.get("offset_s", 0.0))
            except (KeyError, TypeError, ValueError):
                raise ConfigError("signal.schedule needs red_s, green_s, amber_s") from None
            if min(signal["schedule"][k] for k in ("red_s", "green_s", "amber_s")) <= 0:
                raise ConfigError("signal.schedule durations must be > 0")

    rules = []
    for i, r in enumerate(_items(raw, "turn_rules")):
        rid = str(r.get("rule_id") or f"turn{i + 1}")
        rules.append({"rule_id": rid, "entry": _poly(r.get("entry"), f"turn_rules[{rid}].entry"),
                      "exit": _poly(r.get("exit"), f"turn_rules[{rid}].exit"), "label": str(r.get("label") or rid)})

    limit = raw.get("speed_limit_kmh")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 < limit < 400):
        raise ConfigError("speed_limit_kmh must be a number in (0, 400)")

    thresholds = {}
    raw_thresholds = raw.get("thresholds")
    if raw_thresholds is not None and not isinstance(raw_thresholds, dict):
        raise ConfigError("thresholds must be an object")
    for k, v in (raw_thresholds or {}).items():
        if k not in _THRESHOLD_TYPES:
            raise ConfigError(f"unknown threshold {k!r}; known: {sorted(_THRESHOLD_TYPES)}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ConfigError(f"threshold {k} must be a positive number")
        thresholds[k] = _THRESHOLD_TYPES[k](v)

    calibration = raw.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise ConfigError("calibration must be an object")
        build_calibration_profile("probe", "probe", calibration)  # validate eagerly

    start = raw.get("recording_start_utc")
    if start is not None:
        try:
            parse_iso(start)
        except ValueError as exc:
            raise ConfigError(f"recording_start_utc: {exc}") from None

    rt = raw.get("realtime_factor")
    if rt is not None and (isinstance(rt, bool) or not isinstance(rt, (int, float)) or not math.isfinite(rt) or rt <= 0):
        raise ConfigError("realtime_factor must be a number > 0 or null")

    max_s = raw.get("max_stream_s")
    if max_s is not None and (isinstance(max_s, bool) or not isinstance(max_s, (int, float)) or not math.isfinite(max_s) or max_s <= 0):
        raise ConfigError("max_stream_s must be a number > 0 or null")
    speed_zone = _poly(raw["speed_zone"], "speed_zone") if raw.get("speed_zone") else ()
    tracker = _parse_tracker(raw.get("tracker"))
    criterion = raw.get("stop_criterion", "step_speed")
    if criterion not in ("step_speed", "displacement"):
        raise ConfigError("stop_criterion must be 'step_speed' (default) or 'displacement' (opt-in)")

    return CameraConfig(
        direction_zones=tuple(zones), lanes=tuple(lanes), signal=signal, turn_rules=tuple(rules),
        speed_limit_kmh=float(limit) if limit is not None else None, calibration=calibration, thresholds=thresholds,
        recording_start_utc=start, realtime_factor=float(rt) if rt is not None else None, stop_criterion=criterion,
        max_stream_s=float(max_s) if max_s is not None else None, speed_zone=speed_zone, tracker=tracker, raw=dict(raw),
    )


def _parse_tracker(raw) -> dict:
    """`tracker`: {"conf": 0 < c < 1, "config": "<repo-relative>.yaml"}. Both optional; absent = the library defaults (conf 0.25,
    Ultralytics bytetrack.yaml). `conf` is the score floor of the detections handed to the tracker: ByteTrack's second association
    stage needs the LOW-score boxes (track_low_thresh, 0.1 by default), which a 0.25 floor removes before the tracker sees them."""
    if raw is None:
        return {}
    if not isinstance(raw, dict) or set(raw) - {"conf", "config"}:
        raise ConfigError("tracker must be an object with only 'conf' and/or 'config'")
    out = {}
    if "conf" in raw:
        c = raw["conf"]
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0.01 <= c < 1:
            raise ConfigError("tracker.conf must be a number in [0.01, 1)")
        out["conf"] = float(c)
    if "config" in raw:
        cfg = raw["config"]
        if not isinstance(cfg, str) or not cfg.endswith((".yaml", ".yml")) or cfg.startswith(("/", "\\")) or ":" in cfg or ".." in cfg.replace("\\", "/").split("/"):
            raise ConfigError("tracker.config must be a repo-relative .yaml path without '..'")
        out["config"] = cfg
    return out


def build_calibration_profile(profile_id: str, camera_id: str, calibration: dict):
    """A CalibrationProfile from a camera's `calibration` block. Fails closed: with no
    `validation` block the profile is `WORLD_UNVALIDATED` and yields no metric speed."""
    from anvesh.world.calibration import CalibrationPoint, MetricValidationRecord, fit_planar_homography

    try:
        pts = [CalibrationPoint(image_xy=tuple(p["image_xy"]), world_xy=tuple(p["world_xy"]))
               for p in calibration.get("reference_points", [])]
        validation = None
        if calibration.get("validation"):
            validation = MetricValidationRecord(**calibration["validation"])
        return fit_planar_homography(profile_id, camera_id, pts, validation=validation)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ConfigError(f"calibration: {exc}") from None


def geometry_label(profile) -> dict:
    """How the dashboard must describe a camera's geometry. Synthetic and real validation are
    kept visibly apart; an unvalidated fit is never called calibrated metric."""
    if profile is None:
        return {"validity": "image", "label": "Image space (pixels) - no calibration", "metric_speed": False, "synthetic": False}
    validity = profile.geometry_validity.value
    if profile.is_synthetic_validated:
        return {"validity": validity, "label": "SYNTHETIC calibration (test fixture) - not a real camera calibration",
                "metric_speed": True, "synthetic": True}
    if profile.is_real_camera_validated:
        return {"validity": validity, "label": "Metric - validated by survey hold-out", "metric_speed": True, "synthetic": False}
    if profile.is_marking_standard_validated:
        v = profile.validation
        return {"validity": validity, "approximate": True, "synthetic": False, "metric_speed": True,
                "label": f"APPROXIMATE metric - scaled from a published lane-marking module, hold-out RMSE {v.holdout_rmse_m:.2f} m "
                         f"({v.n_holdout_points} points); site NOT surveyed"}
    if validity == "world_validated":
        return {"validity": validity, "label": "Metric - validated (evidence kind: " + profile.validation.kind.value + ")",
                "metric_speed": True, "synthetic": False}
    return {"validity": validity, "label": "Homography fitted but UNVALIDATED - pixel quantities only, no km/h",
            "metric_speed": False, "synthetic": False}


# -- source adapters --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SourceAdapter:
    source_type: str
    live: bool
    supports_pause: bool
    description: str
    uri_example: str


ADAPTERS = {
    "file": SourceAdapter("file", False, True, "Recorded video file (uploaded or on disk); finite, pausable, replayable", "D:/videos/clip.mp4"),
    "webcam": SourceAdapter("webcam", True, False, "Local camera by device index (DirectShow on Windows)", "0"),
    "rtsp": SourceAdapter("rtsp", True, False, "RTSP stream over OpenCV/FFmpeg with reconnect/backoff", "rtsp://user:pass@host:554/stream"),
    "rtmp": SourceAdapter("rtmp", True, False, "RTMP stream via the bundled FFmpeg", "rtmp://host/app/stream"),
    "http_stream": SourceAdapter("http_stream", True, False, "HTTP/HTTPS MJPEG or HLS (.m3u8) via FFmpeg", "https://host/stream.m3u8"),
    "synthetic": SourceAdapter("synthetic", False, True, "Generated clip for workflow tests; always data class 'simulated'", "synthetic:stopped_car"),
}


def validate_source(source_type: str, uri: str, nominal_fps, data_class: str, allowed_roots: tuple = (), denied_roots: tuple = ()) -> None:
    if not isinstance(source_type, str) or source_type not in ADAPTERS:
        raise ConfigError(f"source_type must be one of {SOURCE_TYPES}")
    if not isinstance(data_class, str) or data_class not in DATA_CLASSES:
        raise ConfigError(f"data_class must be one of {DATA_CLASSES}")
    if uri is not None and not isinstance(uri, str):
        raise ConfigError("source_uri must be a string")
    uri = (uri or "").strip()
    if not uri:
        raise ConfigError("source_uri must be non-empty")
    if len(uri) > 2048:
        raise ConfigError("source_uri is too long")
    if source_type == "synthetic" and data_class != "simulated":
        raise ConfigError("synthetic sources must have data_class 'simulated'")
    if source_type == "file":
        check_file_source(uri, allowed_roots, denied_roots)
    elif source_type == "webcam":
        if not uri.isdigit():
            raise ConfigError("webcam source_uri must be a device index such as 0")
    elif source_type == "rtsp":
        if not uri.lower().startswith(("rtsp://", "rtsps://")):
            raise ConfigError("rtsp source_uri must start with rtsp:// or rtsps://")
    elif source_type == "rtmp":
        if not uri.lower().startswith(("rtmp://", "rtmps://")):
            raise ConfigError("rtmp source_uri must start with rtmp:// or rtmps://")
    elif source_type == "http_stream":
        if not uri.lower().startswith(("http://", "https://")):
            raise ConfigError("http_stream source_uri must start with http:// or https://")
    if source_type in LIVE_TYPES:
        # RtspFrameSource.fps() would otherwise probe the stream at start, and a camera that is down
        # at startup would raise instead of entering the reconnect loop.
        if (nominal_fps is None or isinstance(nominal_fps, bool) or not isinstance(nominal_fps, (int, float))
                or not math.isfinite(nominal_fps) or not 1.0 <= float(nominal_fps) <= 240.0):
            raise ConfigError("live sources require nominal_fps within [1, 240] so a camera that is down at start can still reconnect")


def check_file_source(uri: str, allowed_roots: tuple = (), denied_roots: tuple = ()) -> None:
    """A recorded-file source must be a video file inside an allowed media root, and must not be inside a protected tree
    (the application's own data directory: database, credentials, evidence, exports) unless it lies under a root that is
    itself explicitly allowed inside it (the uploads directory). The extension rule matters because the replay endpoint
    streams the registered path back to the browser: without it a camera could be registered on the database file or on
    the credentials file and downloaded by any user who may view evidence."""
    p = Path(uri)
    covering = []
    if allowed_roots:
        covering = [Path(r) for r in allowed_roots if _inside(p, Path(r))]
        if not covering:  # outside every allowed root is the more fundamental refusal, so it is reported first
            raise ConfigError("file source must be under an allowed media root (uploads, the repository, or ANVESH_MEDIA_ROOTS)")
    if p.suffix.lower() not in VIDEO_EXT:
        raise ConfigError(f"file source must be a video file ({', '.join(sorted(VIDEO_EXT))})")
    if allowed_roots:
        for denied in denied_roots:
            d = Path(denied)
            if _inside(p, d) and not any(_inside(r, d) and r.resolve() != d.resolve() for r in covering):
                raise ConfigError("file source is inside the application's protected data directory (only its uploads directory may be used)")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def redact_uri(uri: str) -> str:
    """Never return stream credentials to the dashboard or write them to logs."""
    return re.sub(r"(://)([^/@:]+)(:[^/@]*)?@", r"\1***:***@", uri)


def build_frame_source(source_type: str, uri: str, nominal_fps, config: CameraConfig, capture_factory=None):
    """Construct the FrameSource for a camera. Does NOT open it."""
    from anvesh.perception.ingestion import VideoFileFrameSource
    from anvesh.streaming.rtsp import RtspFrameSource

    if source_type == "file":
        return VideoFileFrameSource(uri, realtime_factor=config.realtime_factor)
    if source_type == "synthetic":
        from anvesh.app.synthetic import SyntheticFrameSource
        return SyntheticFrameSource(uri.split(":", 1)[-1] or "stopped_car")
    if source_type == "webcam":
        import cv2
        factory = capture_factory or (lambda u: cv2.VideoCapture(int(u), cv2.CAP_DSHOW))
        return RtspFrameSource(uri, nominal_fps=nominal_fps, capture_factory=factory)
    return RtspFrameSource(uri, nominal_fps=nominal_fps, capture_factory=capture_factory)


# -- service ------------------------------------------------------------------


def _clean_name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("name must be a non-empty string")
    if len(value.strip()) > 200:
        raise ConfigError("name is too long (max 200 characters)")
    return value.strip()


def _check_label(value) -> None:
    if value is not None and (not isinstance(value, str) or len(value) > 500):
        raise ConfigError("location_label must be a string of at most 500 characters")


def _check_geo(body: dict) -> None:
    for key, lo, hi in (("latitude", -90, 90), ("longitude", -180, 180)):
        v = body.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi):
            raise ConfigError(f"{key} must be within [{lo}, {hi}]")


def _check_offset(offset) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or not -720 <= offset <= 840:
        raise ConfigError("utc_offset_minutes must be an integer within [-720, 840]")


def _check_fps(fps) -> None:
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 240.0):
        raise ConfigError("nominal_fps must be a number in (0, 240]")


class CameraService:
    def __init__(self, db: AppDatabase, audit: AuditLog, media_roots: tuple = (), protected_roots: tuple = ()) -> None:
        self._db, self._audit, self._media_roots, self._protected_roots = db, audit, tuple(media_roots), tuple(protected_roots)

    def validate_registered_source(self, source_type, uri, nominal_fps, data_class) -> None:
        """Re-run the source rules against a stored camera (used when a session starts, so rows written before a rule existed are caught)."""
        validate_source(source_type, uri, nominal_fps, data_class, self._media_roots, self._protected_roots)

    def _present(self, row: dict) -> dict:
        row = dict(row)
        row["config"] = loads(row.pop("config_json"), {})
        row["source_uri_display"] = redact_uri(row["source_uri"])
        row.pop("source_uri", None)
        row["adapter"] = dataclasses.asdict(ADAPTERS[row["source_type"]])
        row["enabled"] = bool(row["enabled"])
        return row

    def get_raw(self, camera_id: str):
        return self._db.one("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))

    def get(self, camera_id: str):
        row = self.get_raw(camera_id)
        return self._present(row) if row else None

    def list(self) -> list:
        return [self._present(r) for r in self._db.query("SELECT * FROM cameras ORDER BY name")]

    @staticmethod
    def _refuse_demo_operational(actor_is_demo: bool, data_class) -> None:
        if actor_is_demo and data_class == "operational":
            raise PermissionError("demo accounts cannot register, change or reclassify operational cameras")

    def create(self, actor: str, body: dict, actor_role: str = None, actor_is_demo: bool = False) -> dict:
        if not isinstance(body, dict):
            raise ConfigError("camera definition must be an object")
        self._refuse_demo_operational(actor_is_demo, body.get("data_class"))
        raw_id = body.get("camera_id")
        if raw_id is not None and not isinstance(raw_id, str):
            raise ConfigError("camera_id must be a string")
        camera_id = (raw_id or "").strip() or "cam-" + uuid.uuid4().hex[:8]
        if not _ID_RE.match(camera_id):
            raise ConfigError("camera_id must be 2-63 chars of letters, digits, '.', '_' or '-'")
        name = _clean_name(body.get("name"))
        source_type = body.get("source_type")
        uri = body.get("source_uri")
        uri = uri.strip() if isinstance(uri, str) else uri
        data_class = body.get("data_class", "demo")
        fps = body.get("nominal_fps")
        _check_fps(fps)
        validate_source(source_type, uri, fps, data_class, self._media_roots, self._protected_roots)
        config = body.get("config")
        parse_camera_config(config)
        _check_label(body.get("location_label"))
        _check_geo(body)
        offset = body.get("utc_offset_minutes", 330)
        _check_offset(offset)
        now = now_iso()
        with self._db.write() as conn:
            if conn.execute("SELECT 1 FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone():
                raise ConfigError(f"camera {camera_id!r} already exists")
            conn.execute(
                "INSERT INTO cameras(camera_id, name, source_type, source_uri, nominal_fps, location_label, latitude, longitude, "
                "utc_offset_minutes, data_class, enabled, config_json, created_at, updated_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?,?,?)",
                (camera_id, name, source_type, uri, float(fps) if fps is not None else None, body.get("location_label"),
                 body.get("latitude"), body.get("longitude"), offset, data_class, dumps(config or {}), now, now, actor),
            )
        self._audit.append(actor, "camera.create", "camera", camera_id,
                           {"source_type": source_type, "data_class": data_class, "source": redact_uri(uri)}, actor_role=actor_role)
        return self.get(camera_id)

    def update(self, actor: str, camera_id: str, body: dict, actor_role: str = None, actor_is_demo: bool = False) -> dict:
        row = self.get_raw(camera_id)
        if row is None:
            raise LookupError(f"no such camera {camera_id!r}")
        if not isinstance(body, dict):
            raise ConfigError("camera update must be an object")
        # neither an operational camera may be touched, nor any camera turned operational, by a demo account
        self._refuse_demo_operational(actor_is_demo, row["data_class"])
        self._refuse_demo_operational(actor_is_demo, body.get("data_class"))
        sets, params, changed = [], [], []
        if "name" in body:
            sets.append("name = ?")
            params.append(_clean_name(body["name"]))
            changed.append("name")
        if "location_label" in body:
            _check_label(body["location_label"])
            sets.append("location_label = ?")
            params.append(body["location_label"])
            changed.append("location_label")
        if "latitude" in body or "longitude" in body:
            _check_geo(body)
            for key in ("latitude", "longitude"):
                if key in body:
                    sets.append(f"{key} = ?")
                    params.append(body[key])
                    changed.append(key)
        if "utc_offset_minutes" in body:
            _check_offset(body["utc_offset_minutes"])
            sets.append("utc_offset_minutes = ?")
            params.append(body["utc_offset_minutes"])
            changed.append("utc_offset_minutes")
        if "nominal_fps" in body:
            _check_fps(body["nominal_fps"])
            sets.append("nominal_fps = ?")
            params.append(float(body["nominal_fps"]) if body["nominal_fps"] is not None else None)
            changed.append("nominal_fps")
        if "enabled" in body:
            if not isinstance(body["enabled"], bool):
                raise ConfigError("enabled must be true or false")
            sets.append("enabled = ?")
            params.append(1 if body["enabled"] else 0)
            changed.append("enabled")
        if "config" in body:
            parse_camera_config(body["config"])
            sets.append("config_json = ?")
            params.append(dumps(body["config"] or {}))
            changed.append("config")
        if any(k in body for k in ("source_uri", "source_type", "data_class", "nominal_fps")):
            validate_source(body.get("source_type", row["source_type"]), body.get("source_uri", row["source_uri"]),
                            body.get("nominal_fps", row["nominal_fps"]), body.get("data_class", row["data_class"]),
                            self._media_roots, self._protected_roots)
            for key in ("source_type", "source_uri", "data_class"):
                if key in body:
                    value = body[key].strip() if key == "source_uri" else body[key]
                    sets.append(f"{key} = ?")
                    params.append(value)
                    changed.append(key if key != "source_uri" else "source_uri(redacted)")
        if not sets:
            return self.get(camera_id)
        sets.append("updated_at = ?")
        params.append(now_iso())
        with self._db.write() as conn:
            conn.execute(f"UPDATE cameras SET {', '.join(sets)} WHERE camera_id = ?", params + [camera_id])
        self._audit.append(actor, "camera.update", "camera", camera_id, {"changed": changed}, actor_role=actor_role)
        return self.get(camera_id)
