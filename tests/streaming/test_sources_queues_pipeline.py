import threading
import time

import numpy as np
import pytest

from anvesh.perception.ingestion import VideoFileFrameSource
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.latency import LatencyRecorder
from anvesh.streaming.pipeline import StreamingPipeline
from anvesh.streaming.queues import BoundedQueue, DropPolicy, QueueClosed
from anvesh.streaming.rtsp import RtspFrameSource
from anvesh.streaming.windows import WindowSpec
from tests.perception.conftest import _write_tiny_video
from tests.streaming.support import FakeClock, ListFrameSource, ScriptedTracker, make_frames

FPS = 10.0


# ---------------------------------------------------------------------------
# BoundedQueue
# ---------------------------------------------------------------------------


def test_queue_block_policy_is_lossless_and_bounded():
    q = BoundedQueue(2, DropPolicy.BLOCK)
    assert q.put(1) and q.put(2)
    assert q.put(3, timeout=0.05) is False  # full: producer waits, nothing dropped
    assert q.dropped == 0 and len(q) == 2
    assert q.get() == 1
    assert q.put(3)
    assert [q.get(), q.get()] == [2, 3]


def test_queue_drop_oldest_counts_drops_and_never_grows():
    q = BoundedQueue(3, DropPolicy.DROP_OLDEST)
    for i in range(10):
        q.put(i)
    assert len(q) == 3 and q.dropped == 7 and q.high_water == 3
    assert [q.get(), q.get(), q.get()] == [7, 8, 9]  # freshest survive


def test_queue_drop_newest_refuses_and_counts():
    q = BoundedQueue(2, DropPolicy.DROP_NEWEST)
    assert q.put(1) and q.put(2) and q.put(3) is False
    assert q.dropped == 1 and [q.get(), q.get()] == [1, 2]


def test_queue_close_drains_then_raises_and_unblocks_producers():
    q = BoundedQueue(1, DropPolicy.BLOCK)
    q.put("x")
    raised = []
    t = threading.Thread(target=lambda: _try(q.put, "y", raised))
    t.start()
    time.sleep(0.05)
    q.close()
    t.join(1.0)
    assert raised == ["closed"]
    assert q.get() == "x"
    with pytest.raises(QueueClosed):
        q.get()


def _try(fn, arg, sink):
    try:
        fn(arg)
    except QueueClosed:
        sink.append("closed")


def test_queue_rejects_bad_capacity():
    with pytest.raises(ValueError):
        BoundedQueue(0)


# ---------------------------------------------------------------------------
# LatencyRecorder
# ---------------------------------------------------------------------------


def test_latency_recorder_percentiles_and_bounded_reservoir():
    rec = LatencyRecorder(reservoir=100, seed=1)
    for i in range(1, 101):
        rec.record("s", i / 1000.0)
    s = rec.summary_ms()["s"]
    assert s["count"] == 100 and s["percentiles_exact"]
    assert s["p50_ms"] == pytest.approx(50.5, abs=0.6)
    assert s["p95_ms"] == pytest.approx(95.05, abs=0.6)
    assert s["max_ms"] == pytest.approx(100.0)
    for i in range(100000):  # far past capacity: memory stays bounded
        rec.record("s", 0.01)
    assert len(rec._stages["s"].reservoir) == 100
    assert rec.summary_ms()["s"]["count"] == 100100


# ---------------------------------------------------------------------------
# VideoFileFrameSource: timing, EOF, pacing, health
# ---------------------------------------------------------------------------


def test_file_source_timestamps_eof_and_health(tmp_path):
    path = tmp_path / "t.mp4"
    _write_tiny_video(path, num_frames=7, fps=10.0)
    src = VideoFileFrameSource(path)
    assert src.health().state == "idle"
    frames = list(src)
    assert [f.index for f in frames] == list(range(7))
    assert [f.timestamp for f in frames] == pytest.approx([i / 10.0 for i in range(7)])
    assert all(f.decode_s is not None and f.capture_wall_s is not None and f.epoch == 0 for f in frames)
    h = src.health()
    assert h.state == "ended" and h.frames_read == 7  # clean EOF


def test_file_source_pacing_waits_until_the_frame_is_due(tmp_path):
    path = tmp_path / "t.mp4"
    _write_tiny_video(path, num_frames=5, fps=10.0)
    clock = FakeClock()
    src = VideoFileFrameSource(path, realtime_factor=1.0, clock=clock, sleep=clock.sleep)
    frames = list(src)
    assert len(frames) == 5
    assert clock.t == pytest.approx(0.4)  # last frame due at 4/10 s of replay
    assert [f.timestamp for f in frames] == pytest.approx([i / 10.0 for i in range(5)])  # pacing never alters timestamps
    fast = FakeClock()
    list(VideoFileFrameSource(path, realtime_factor=4.0, clock=fast, sleep=fast.sleep))
    assert fast.t == pytest.approx(0.1)
    unpaced = FakeClock()
    list(VideoFileFrameSource(path, clock=unpaced, sleep=unpaced.sleep))
    assert unpaced.sleeps == []


def test_file_source_missing_file_and_bad_factor(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileFrameSource(tmp_path / "nope.mp4")
    path = tmp_path / "t.mp4"
    _write_tiny_video(path)
    with pytest.raises(ValueError):
        VideoFileFrameSource(path, realtime_factor=0)


# ---------------------------------------------------------------------------
# RtspFrameSource (against fake captures -- NOT a real camera)
# ---------------------------------------------------------------------------


class FakeCapture:
    """Yields `n_ok` frames (advancing the shared clock by `dt` per read),
    then fails reads forever (a dropped connection)."""

    def __init__(self, clock, n_ok, dt=0.1, opens=True):
        self._clock, self._n, self._dt, self._opens = clock, n_ok, dt, opens
        self.released = False

    def isOpened(self):
        return self._opens

    def get(self, prop):
        return 10.0

    def read(self):
        if self._n <= 0:
            return False, None
        self._n -= 1
        self._clock.t += self._dt
        return True, np.zeros((12, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def _rtsp(clock, captures, **kw):
    it = iter(captures)
    created = []

    def factory(url):
        c = next(it)
        created.append(c)
        return c

    src = RtspFrameSource("rtsp://x/stream", nominal_fps=10.0, capture_factory=factory, clock=clock, sleep=clock.sleep, **kw)
    return src, created


def test_rtsp_reconnects_with_backoff_and_bumps_epoch():
    clock = FakeClock()
    caps = [FakeCapture(clock, 3), FakeCapture(clock, 0, opens=False), FakeCapture(clock, 3)]
    src, created = _rtsp(clock, caps, backoff_s=(1.0, 2.0), max_frames=6)
    frames = list(src)
    assert len(frames) == 6
    assert [f.epoch for f in frames] == [0, 0, 0, 1, 1, 1]
    assert [f.index for f in frames] == [0, 1, 2, 3, 4, 5]  # indices continue across reconnects
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts) and ts[3] - ts[2] > 2.0  # stream clock keeps running through the outage
    assert frames[3].gap_before > 10  # the outage is accounted as missing frames
    h = src.health()
    assert h.reconnects == 1 and h.epoch == 1 and h.source_gap_frames == frames[3].gap_before
    assert all(c.released for c in created)  # no leaked captures
    assert clock.sleeps[0] == 1.0  # first backoff after the failed read


def test_rtsp_gives_up_after_max_reconnects():
    clock = FakeClock()
    caps = [FakeCapture(clock, 0, opens=False) for _ in range(5)]
    src, _ = _rtsp(clock, caps, backoff_s=(0.5,), max_reconnects=3)
    assert list(src) == []
    assert src.health().state == "failed"
    assert "open failed" in src.health().last_error


def test_rtsp_jitter_and_gap_accounting():
    clock = FakeClock()

    class Jittery(FakeCapture):
        def read(self):
            self._dt = 0.1 if self._n % 4 else 0.45  # periodic stall
            return super().read()

    src, _ = _rtsp(clock, [Jittery(clock, 12)], max_frames=12)
    list(src)
    h = src.health()
    assert h.interval_jitter_s > 0.05
    assert h.source_gap_frames >= 2
    assert 1.0 < h.measured_fps < 20.0


def test_rtsp_stop_ends_cleanly():
    clock = FakeClock()
    src, created = _rtsp(clock, [FakeCapture(clock, 1000)])
    out = []
    for f in src:
        out.append(f)
        if len(out) == 5:
            src.close()
    assert len(out) == 5 and src.health().state == "stopped" and created[0].released


def test_rtsp_validates_arguments():
    with pytest.raises(ValueError):
        RtspFrameSource("")
    with pytest.raises(ValueError):
        RtspFrameSource("rtsp://x", nominal_fps=0.2)


# ---------------------------------------------------------------------------
# Threaded pipeline: parity with synchronous replay, drops, errors
# ---------------------------------------------------------------------------


def _churn(i):
    out = [("p", (30.0, 30.0))]
    for j in range(max(0, i // 20 - 1), i // 20 + 1):
        if 0 <= i - j * 20 < 45:
            out.append((f"v{j}", (60.0 + (i - j * 20) * 2, 40.0 + 5 * (j % 5))))
    return out


def _signature(snaps):
    return [
        (s.window_id, s.start, s.end, s.frames_observed, s.dropped_frames,
         tuple((t.track_id, t.first_seen, t.last_seen, t.speed_estimate.value, t.occlusion_state.value, len(t.position_history))
               for t in s.tracks))
        for s in snaps
    ]


def _sync_run(frames, windows):
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_churn), FPS, windows=windows, run_id="parity")
    snaps = []
    for f in frames:
        snaps.extend(eng.process_frame(f))
    snaps.extend(eng.finish())
    return snaps


def test_threaded_lossless_pipeline_is_bit_identical_to_synchronous_replay():
    frames = make_frames(600, FPS)
    expected = _signature(_sync_run(frames, WindowSpec(10.0)))

    got = []
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_churn), FPS, windows=WindowSpec(10.0), run_id="parity")
    pipe = StreamingPipeline(ListFrameSource(frames), eng, on_window=got.append, frame_queue_size=4,
                             frame_policy=DropPolicy.BLOCK)
    stats = pipe.run()

    assert _signature(got) == expected  # exact equality when nothing is dropped
    assert stats["frame_queue"]["dropped"] == 0 and stats["engine"]["frames_dropped"] == 0
    assert stats["engine"]["frames_processed"] == 600
    assert stats["frame_queue"]["high_water"] <= 4  # bounded


def test_live_drop_oldest_pipeline_accounts_for_every_dropped_frame():
    frames = make_frames(300, FPS)

    class SlowTracker(ScriptedTracker):
        def update(self, image, frame_index, timestamp):
            time.sleep(0.002)
            return super().update(image, frame_index, timestamp)

    got = []
    eng = StreamingPerceptionEngine("cam", SlowTracker(_churn), FPS, windows=WindowSpec(10.0))
    pipe = StreamingPipeline(ListFrameSource(frames), eng, on_window=got.append, frame_queue_size=2,
                             frame_policy=DropPolicy.DROP_OLDEST)
    stats = pipe.run()

    q = stats["frame_queue"]
    # every frame is either processed or counted as dropped -- none vanish
    assert stats["engine"]["frames_processed"] + q["dropped"] == 300
    assert stats["engine"]["frames_dropped"] <= q["dropped"]  # (a dropped tail frame has no later frame to expose the gap)
    assert q["high_water"] <= 2


def test_pipeline_surfaces_worker_errors_in_the_callers_thread():
    class Boom(ScriptedTracker):
        def update(self, image, frame_index, timestamp):
            if frame_index == 5:
                raise RuntimeError("detector exploded")
            return super().update(image, frame_index, timestamp)

    eng = StreamingPerceptionEngine("cam", Boom(_churn), FPS, windows=WindowSpec(10.0))
    pipe = StreamingPipeline(ListFrameSource(make_frames(50, FPS)), eng)
    with pytest.raises(RuntimeError, match="detector exploded"):
        pipe.run()


def test_pipeline_records_stage_latencies():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_churn), FPS, windows=WindowSpec(10.0))
    StreamingPipeline(ListFrameSource([_stamped(f) for f in make_frames(250, FPS)]), eng).run()
    s = eng.recorder.summary_ms()
    for stage in ("detect_track", "store_update", "windowing", "frame_processing", "frame_end_to_end", "decode"):
        assert stage in s and s[stage]["count"] > 0 and s[stage]["p95_ms"] >= s[stage]["p50_ms"] >= 0.0


def _stamped(frame):
    frame.capture_wall_s = time.perf_counter()
    frame.decode_s = 0.001
    return frame


# ---------------------------------------------------------------------------
# Batch (run_m1_pipeline) vs streaming engine: same identities/trajectories
# ---------------------------------------------------------------------------


def test_run_m1_pipeline_batch_replay_matches_the_streaming_engine(tmp_path):
    from anvesh.perception.pipeline import run_m1_pipeline

    path = tmp_path / "t.mp4"
    _write_tiny_video(path, num_frames=120, fps=10.0)
    batch = run_m1_pipeline(path, "cam", ScriptedTracker(_churn), "scripted", "0", run_id="parity")

    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_churn), FPS, archive=True, run_id="parity")
    for f in make_frames(120, FPS):
        eng.process_frame(f)
    eng.finish()
    streamed = eng.store.manager.all_tracks()

    key = lambda s: (s.track_id, s.raw_tracker_id, s.first_seen, s.last_seen, tuple(s.trajectory))  # noqa: E731
    assert [key(s) for s in batch.track_states] == [key(s) for s in streamed]
    assert batch.frame_count == 120 and len(batch.camera_observations) == 120
