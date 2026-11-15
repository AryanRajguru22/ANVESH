"""Source adapters and session resilience: reconnect, outage, stall, retry exhaustion, bounded queues, pause/resume, and a REAL
HTTP MJPEG stream decoded by OpenCV/FFmpeg. None of this is a real RTSP/RTMP camera: those paths run against injected fake
captures (and the HTTP stream is a local test server). That limitation is deliberate and documented."""

import threading
import time

import cv2
import numpy as np
import pytest

from anvesh.app import cameras as cams
from anvesh.app.runner import SessionManager
from anvesh.perception.ingestion import VideoFileFrameSource
from anvesh.streaming.rtsp import RtspFrameSource


class NullTracker:
    lost_buffer_frames = 30

    def __init__(self, delay=0.0):
        self.delay = delay

    def reset(self):
        pass

    def update(self, image, frame_index, timestamp):
        if self.delay:
            time.sleep(self.delay)
        return [], []


class FakeCapture:
    """cv2.VideoCapture stand-in: yields `frames` frames, then raises/fails per `end`."""

    def __init__(self, frames, end="fail", opened=True, gap_at=None, gap_s=0.0):
        self._n, self._end, self._opened, self._i = frames, end, opened, 0
        self._gap_at, self._gap_s = gap_at, gap_s

    def isOpened(self):
        return self._opened

    def get(self, prop):
        return 10.0

    def release(self):
        pass

    def read(self):
        if self._gap_at is not None and self._i == self._gap_at:
            time.sleep(self._gap_s)
        if self._i >= self._n:
            if self._end == "fail":
                return False, None
            time.sleep(60)
        self._i += 1
        time.sleep(0.02)
        return True, np.zeros((72, 128, 3), np.uint8)


def make_mgr(services, factory=None, tracker=None):
    return SessionManager(services, tracker_factory=lambda c, cfg: tracker or NullTracker(), source_factory=factory)


def add_cam(services, cid, stype="rtsp", uri="rtsp://127.0.0.1:1/x", fps=10, cls="demo", config=None):
    services.cameras.create("admin", {"camera_id": cid, "name": cid, "source_type": stype, "source_uri": uri, "nominal_fps": fps, "data_class": cls,
                                      "config": config or {}})


def wait_for(pred, timeout=20.0, step=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return None


def test_camera_down_at_start_enters_reconnect_loop_then_goes_live_and_stops_cleanly(services):
    attempts = {"n": 0}

    def capture_factory(url):
        attempts["n"] += 1
        return FakeCapture(0, opened=False) if attempts["n"] <= 2 else FakeCapture(10 ** 6, end="block")

    def factory(cam, cfg):
        return RtspFrameSource(cam["source_uri"], nominal_fps=10, capture_factory=capture_factory, sleep=lambda s: time.sleep(0.05), backoff_s=(0.05,))

    add_cam(services, "rtsp-down")
    mgr = make_mgr(services, factory)
    sid = mgr.start("rtsp-down", "admin")["session_id"]
    seen = set()
    assert wait_for(lambda: (seen.add(mgr.runtime(sid)["state"]) or mgr.runtime(sid)["state"] == "live"), 20), seen
    assert wait_for(lambda: mgr.runtime(sid)["frames_processed"] > 0, 5)
    rt = mgr.runtime(sid)
    assert rt["source_health"]["reconnects"] == 0 and attempts["n"] == 3
    assert {"connecting", "reconnecting"} & seen  # it was visibly not-live before the third attempt succeeded
    mgr.stop(sid, "admin")
    mgr.get(sid)._thread.join(15)
    assert mgr.runtime(sid)["state"] == "stopped"
    row = services.db.one("SELECT state, ended_at FROM processing_sessions WHERE session_id = ?", (sid,))
    assert row["state"] == "stopped" and row["ended_at"]


def test_mid_stream_outage_reconnects_with_new_identity_session_and_counts_loss(services):
    calls = {"n": 0}

    def capture_factory(url):
        calls["n"] += 1
        return FakeCapture(25, end="fail") if calls["n"] == 1 else FakeCapture(10 ** 6, end="block")

    def factory(cam, cfg):
        return RtspFrameSource(cam["source_uri"], nominal_fps=10, capture_factory=capture_factory, sleep=lambda s: time.sleep(0.4), backoff_s=(0.4,))

    add_cam(services, "rtsp-flaky")
    mgr = make_mgr(services, factory)
    sid = mgr.start("rtsp-flaky", "admin")["session_id"]
    runner = mgr.get(sid)
    assert wait_for(lambda: runner.engine is not None and runner.engine.stats()["sessions_started"] == 2 and runner.engine.frames_processed > 30, 20)
    st = runner.engine.stats()
    assert st["sessions_started"] == 2  # a reconnect is a NEW identity session: raw tracker ids never carry over
    assert runner.source.health().reconnects == 1
    assert runner.engine.frames_dropped >= 1  # the outage is accounted for as lost frames, not hidden
    mgr.stop(sid, "admin")
    runner._thread.join(15)


def test_stalled_source_is_reported_stalled_then_recovers(services):
    def capture_factory(url):
        return FakeCapture(10 ** 6, end="block", gap_at=15, gap_s=3.5)

    def factory(cam, cfg):
        return RtspFrameSource(cam["source_uri"], nominal_fps=10, capture_factory=capture_factory)

    add_cam(services, "rtsp-stall")
    mgr = make_mgr(services, factory)
    sid = mgr.start("rtsp-stall", "admin")["session_id"]
    assert wait_for(lambda: mgr.runtime(sid)["state"] == "live", 15)
    assert wait_for(lambda: mgr.runtime(sid)["state"] == "stalled", 15), "a silent camera must be reported stalled, not live"
    assert wait_for(lambda: mgr.runtime(sid)["state"] == "live", 15)  # frames resumed
    mgr.stop(sid, "admin")
    mgr.get(sid)._thread.join(15)


def test_retry_exhaustion_fails_one_camera_without_affecting_another(services):
    def dead_factory(cam, cfg):
        return RtspFrameSource(cam["source_uri"], nominal_fps=10, capture_factory=lambda u: FakeCapture(0, opened=False), sleep=lambda s: None,
                               backoff_s=(0.01,), max_reconnects=3)

    add_cam(services, "rtsp-dead")
    services.cameras.create("admin", {"camera_id": "sim-ok", "name": "ok", "source_type": "synthetic", "source_uri": "synthetic:queue",
                                      "data_class": "simulated"})

    def factory(cam, cfg):
        return dead_factory(cam, cfg) if cam["camera_id"] == "rtsp-dead" else cams.build_frame_source(cam["source_type"], cam["source_uri"], cam["nominal_fps"], cfg)

    mgr = make_mgr(services, factory)
    dead = mgr.start("rtsp-dead", "admin")["session_id"]
    ok = mgr.start("sim-ok", "admin")["session_id"]
    mgr.get(dead)._thread.join(20)
    mgr.get(ok)._thread.join(60)
    d = mgr.runtime(dead)
    assert d["state"] == "failed" and "open failed" in (d["error"] or "") or "connect" in (d["error"] or "")
    assert mgr.runtime(ok)["state"] == "completed"  # the healthy camera was unaffected
    assert services.db.one("SELECT state FROM processing_sessions WHERE session_id = ?", (dead,))["state"] == "failed"


def test_live_queue_is_bounded_and_drops_are_counted_when_processing_is_slower_than_the_source(services):
    class FastCapture(FakeCapture):
        def read(self):
            if self._i >= self._n:
                time.sleep(60)
            self._i += 1
            time.sleep(0.002)
            return True, np.zeros((72, 128, 3), np.uint8)

    def factory(cam, cfg):
        return RtspFrameSource(cam["source_uri"], nominal_fps=100, capture_factory=lambda u: FastCapture(400))

    add_cam(services, "rtsp-fast", fps=100)
    mgr = make_mgr(services, factory, tracker=NullTracker(delay=0.05))
    sid = mgr.start("rtsp-fast", "admin")["session_id"]
    runner = mgr.get(sid)
    assert wait_for(lambda: runner.pipeline is not None and runner.pipeline.reader_frames >= 390, 20)
    q = runner.pipeline.frame_queue.stats()
    assert q["policy"] == "drop_oldest" and q["high_water"] <= q["capacity"] == 8
    assert q["dropped"] > 100  # most frames could not be processed in time -- and that is counted, not hidden
    assert runner.engine.frames_dropped > 0
    mgr.stop(sid, "admin")
    runner._thread.join(15)


def _video(path, n=60, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 72))
    for i in range(n):
        w.write(np.full((72, 128, 3), i, np.uint8))
    w.release()


def test_pause_and_resume_recorded_source_and_live_sources_refuse_pause(services, tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid, n=80, fps=10)
    services.db.execute("SELECT 1")
    services.media_roots  # noqa: B018
    from anvesh.app.cameras import CameraService
    services.cameras = CameraService(services.db, services.audit, media_roots=(tmp_path,))
    add_cam(services, "file-pause", stype="file", uri=str(vid), fps=None, config={"realtime_factor": 1.0})
    mgr = make_mgr(services)
    sid = mgr.start("file-pause", "admin")["session_id"]
    runner = mgr.get(sid)
    assert wait_for(lambda: runner.engine is not None and runner.engine.frames_processed >= 10, 15)
    mgr.pause(sid, "admin")
    time.sleep(0.6)
    frozen = runner.engine.frames_processed
    time.sleep(1.2)
    assert mgr.runtime(sid)["state"] == "paused" and runner.engine.frames_processed <= frozen + 2  # nothing advanced while paused
    mgr.resume(sid, "admin")
    assert wait_for(lambda: runner.engine.frames_processed > frozen + 5, 10)
    runner._thread.join(30)
    assert mgr.runtime(sid)["state"] == "completed" and runner.engine.frames_processed == 80
    # live adapters do not support pause
    assert cams.ADAPTERS["rtsp"].supports_pause is False and cams.ADAPTERS["file"].supports_pause is True


def test_max_stream_s_limits_recorded_processing(services, tmp_path):
    vid = tmp_path / "v.mp4"
    _video(vid, n=100, fps=10)
    from anvesh.app.cameras import CameraService
    services.cameras = CameraService(services.db, services.audit, media_roots=(tmp_path,))
    add_cam(services, "file-cap", stype="file", uri=str(vid), fps=None, config={"max_stream_s": 3.0})
    mgr = make_mgr(services)
    sid = mgr.start("file-cap", "admin")["session_id"]
    mgr.get(sid)._thread.join(30)
    rt = mgr.runtime(sid)
    assert rt["state"] == "completed" and 28 <= rt["frames_processed"] <= 32


def test_http_mjpeg_stream_through_the_http_stream_adapter_with_real_ffmpeg_decode(services):
    """A real network stream decoded by OpenCV/FFmpeg: a local MJPEG-over-HTTP server (NOT an RTSP/RTMP camera)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    stop = threading.Event()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            i = 0
            try:
                while not stop.is_set():
                    ok, buf = cv2.imencode(".jpg", np.full((72, 128, 3), (i * 5) % 255, np.uint8))
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(buf) + buf.tobytes() + b"\r\n")
                    i += 1
                    time.sleep(0.1)
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        add_cam(services, "http-cam", stype="http_stream", uri=f"http://127.0.0.1:{srv.server_address[1]}/stream", fps=10)
        mgr = make_mgr(services)
        sid = mgr.start("http-cam", "admin")["session_id"]
        assert wait_for(lambda: mgr.runtime(sid)["state"] == "live" and mgr.runtime(sid)["frames_processed"] >= 8, 30), mgr.runtime(sid)
        rt = mgr.runtime(sid)
        assert rt["source_type"] == "http_stream" and rt["source_health"]["state"] == "connected"
        mgr.stop(sid, "admin")
        mgr.get(sid)._thread.join(15)
    finally:
        stop.set()
        srv.shutdown()
        srv.server_close()


def test_source_validation_rejects_bad_uris_and_hides_credentials():
    with pytest.raises(cams.ConfigError):
        cams.validate_source("webcam", "front", None, "demo")
    with pytest.raises(cams.ConfigError):
        cams.validate_source("rtmp", "rtsp://x", 25, "demo")
    with pytest.raises(cams.ConfigError):
        cams.validate_source("rtsp", "rtsp://h/s", 500, "demo")
    cams.validate_source("webcam", "0", 30, "demo")
    assert cams.redact_uri("rtsp://admin:hunter2@10.1.1.1/s") == "rtsp://***:***@10.1.1.1/s"
    assert cams.redact_uri("rtsp://10.1.1.1/s") == "rtsp://10.1.1.1/s"
