"""Synthetic-stream reliability tests for the streaming core.

LABEL: every source here is a test double (an in-memory list, a generator, or an injected fake `cv2.VideoCapture`). These tests
establish SOFTWARE behaviour - incremental processing, bounded queues, drop accounting, isolation between sources, clean
shutdown, corrupt-frame handling. They are NOT webcam / RTSP / RTMP acceptance and say nothing about detector accuracy or
real-camera compatibility. No assertion depends on wall-clock speed, so they are not timing-flaky; throughput/latency numbers
come from `scripts/stream_bench.py`, which reports a spread over repeated runs.
"""

import threading
import time

import numpy as np
import pytest

from anvesh.perception.ingestion import VideoFileFrameSource
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.pipeline import StreamingPipeline
from anvesh.streaming.queues import DropPolicy
from anvesh.streaming.rtsp import RtspFrameSource
from anvesh.streaming.windows import WindowSpec

from .support import FakeClock, ListFrameSource, ScriptedTracker, make_frames

FPS = 10.0


def _churn(i):
    out = [("p", (30.0, 30.0))]
    for j in range(max(0, i // 20 - 1), i // 20 + 1):
        if 0 <= i - j * 20 < 45:
            out.append((f"v{j}", (60.0 + (i - j * 20) * 2, 40.0 + 5 * (j % 5))))
    return out


def _engine(camera="cam", tracker=None, run_id="r"):
    return StreamingPerceptionEngine(camera, tracker or ScriptedTracker(_churn), FPS, windows=WindowSpec(10.0), run_id=run_id)


def _signature(snaps):
    return [(s.window_id, s.start, s.end, s.frames_observed, s.dropped_frames,
             tuple((t.track_id, t.first_seen, t.last_seen) for t in s.tracks)) for s in snaps]


# -- incremental processing ----------------------------------------------------------------------------------------


def test_windows_are_delivered_before_the_source_ends_so_nothing_waits_for_the_whole_video():
    """The source refuses to yield frames beyond the first window's worth until a window result has been delivered. If the
    pipeline needed the whole video (or even the whole next window) before producing output this would deadlock and time out."""
    first_window_delivered = threading.Event()
    pulled_after_first_window = []
    frames = make_frames(400, FPS)

    def gated():
        for f in frames:
            if f.index == 250:  # well past the first 10 s window (frame 100) plus the engine's lateness allowance
                assert first_window_delivered.wait(timeout=10.0), "no window was delivered while frames were still being produced"
            if first_window_delivered.is_set():
                pulled_after_first_window.append(f.index)
            yield f

    class GatedSource(ListFrameSource):
        def __iter__(self):
            return gated()

    got = []

    def on_window(snap):
        got.append(snap)
        first_window_delivered.set()

    pipe = StreamingPipeline(GatedSource(frames), _engine(), on_window=on_window, frame_queue_size=4)
    stats = pipe.run()
    assert got and stats["engine"]["frames_processed"] == 400
    assert pulled_after_first_window, "the first window was only delivered after the source was exhausted"


def test_memory_of_the_pipeline_is_bounded_by_queue_capacity_not_by_stream_length():
    frames = make_frames(2000, FPS)
    pipe = StreamingPipeline(ListFrameSource(frames), _engine(), on_window=lambda s: None, frame_queue_size=3, window_queue_size=2)
    stats = pipe.run()
    assert stats["frame_queue"]["high_water"] <= 3 and stats["window_queue"]["high_water"] <= 2
    assert stats["engine"]["frames_processed"] == 2000


# -- slow consumers, drops and conservation ------------------------------------------------------------------------------


def test_stalled_window_consumer_never_grows_the_window_queue_and_every_window_is_accounted_for():
    delivered = []
    release = threading.Event()
    threading.Timer(0.4, release.set).start()  # the consumer is stuck for 0.4 s while the worker keeps producing windows

    def stalled(snap):
        release.wait(timeout=10)
        delivered.append(snap)

    eng = _engine()
    pipe = StreamingPipeline(ListFrameSource(make_frames(3000, FPS)), eng, on_window=stalled, frame_queue_size=3000, window_queue_size=2,
                             frame_policy=DropPolicy.DROP_OLDEST)
    stats = pipe.run()
    wq = stats["window_queue"]
    assert wq["dropped"] > 0, "the stalled consumer should have forced window drops (otherwise this test exercises nothing)"
    assert wq["high_water"] <= 2
    # conservation: windows the engine emitted = delivered + counted-dropped (+ nothing lost silently)
    assert stats["engine"]["windows_emitted"] == len(delivered) + wq["dropped"]
    fq = stats["frame_queue"]
    assert stats["engine"]["frames_processed"] + fq["dropped"] == 3000


def test_dropped_frames_are_conserved_with_a_slow_detector_for_several_queue_sizes():
    class Slow(ScriptedTracker):
        def update(self, image, frame_index, timestamp):
            time.sleep(0.001)
            return super().update(image, frame_index, timestamp)

    for qsize in (1, 2, 8):
        pipe = StreamingPipeline(ListFrameSource(make_frames(400, FPS)), _engine(tracker=Slow(_churn)), on_window=lambda s: None,
                                 frame_queue_size=qsize, frame_policy=DropPolicy.DROP_OLDEST)
        stats = pipe.run()
        assert stats["engine"]["frames_processed"] + stats["frame_queue"]["dropped"] == 400, qsize
        assert stats["frame_queue"]["high_water"] <= qsize, qsize


# -- per-source isolation ------------------------------------------------------------------------------------------------


def test_concurrent_sources_are_isolated_and_a_failing_one_does_not_disturb_the_others():
    """Eight pipelines run at once. Seven are lossless and must reproduce their own single-source result EXACTLY (no cross-talk
    of tracks, windows or counters); one is made to crash mid-stream and must fail alone."""
    n_sources, n_frames = 8, 400
    expected = {}
    for k in range(n_sources):
        snaps = []
        eng = _engine(camera=f"cam{k}", run_id=f"run{k}")
        for f in make_frames(n_frames, FPS):
            snaps.extend(eng.process_frame(f))
        snaps.extend(eng.finish())
        expected[k] = _signature(snaps)

    class Boom(ScriptedTracker):
        def update(self, image, frame_index, timestamp):
            if frame_index == 150:
                raise RuntimeError("detector exploded on this source only")
            return super().update(image, frame_index, timestamp)

    results, errors = {}, {}

    def run(k):
        got = []
        tracker = Boom(_churn) if k == 3 else ScriptedTracker(_churn)
        pipe = StreamingPipeline(ListFrameSource(make_frames(n_frames, FPS)), _engine(camera=f"cam{k}", tracker=tracker, run_id=f"run{k}"),
                                 on_window=got.append, frame_queue_size=4)
        try:
            pipe.run()
            results[k] = _signature(got)
        except Exception as exc:  # noqa: BLE001
            errors[k] = exc

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n_sources)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert set(errors) == {3} and "exploded" in str(errors[3])
    for k in range(n_sources):
        if k != 3:
            assert results[k] == expected[k], f"source {k} was affected by another source"


# -- shutdown ------------------------------------------------------------------------------------------------------------


def test_stop_ends_an_endless_source_promptly_and_leaves_no_pipeline_threads_behind():
    def endless():
        i = 0
        while True:
            yield make_frames(1, FPS, start_index=i)[0]
            i += 1
            time.sleep(0.0005)

    class Endless(ListFrameSource):
        def __iter__(self):
            return endless()

    pipe = StreamingPipeline(Endless([]), _engine(), on_window=lambda s: None, frame_queue_size=4)
    out = {}
    t = threading.Thread(target=lambda: out.update(stats=pipe.run()), name="runner")
    t.start()
    time.sleep(0.4)
    t0 = time.time()
    pipe.stop()
    t.join(timeout=10)
    assert not t.is_alive(), "run() did not return after stop()"
    assert time.time() - t0 < 5.0
    assert out["stats"]["engine"]["frames_processed"] > 0
    assert [th.name for th in threading.enumerate() if th.name.startswith("anvesh-")] == []


# -- corrupt frames and truncated files -------------------------------------------------------------------------------------


class _Cap:
    """Fake cv2.VideoCapture: `script` is a list of ('ok'|'none'|'empty'|'fail') and then the capture dies."""

    def __init__(self, clock, script, dt=0.1):
        self._clock, self._script, self._dt = clock, list(script), dt
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 10.0

    def read(self):
        if not self._script:
            return False, None
        kind = self._script.pop(0)
        self._clock.t += self._dt
        if kind == "ok":
            return True, np.zeros((12, 16, 3), np.uint8)
        if kind == "none":
            return True, None
        if kind == "empty":
            return True, np.zeros((0, 0, 3), np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_a_corrupt_frame_from_a_live_source_is_skipped_and_counted_not_fatal():
    clock = FakeClock()
    cap = _Cap(clock, ["ok", "ok", "none", "ok", "empty", "ok"])
    src = RtspFrameSource("rtsp://x/s", nominal_fps=10.0, capture_factory=lambda u: cap, clock=clock, sleep=clock.sleep, max_frames=4)
    frames = list(src)
    assert [f.index for f in frames] == [0, 1, 2, 3]  # corrupt reads never become frames
    assert all(f.image is not None and f.image.size > 0 for f in frames)
    h = src.health()
    assert h.corrupt_frames == 2 and h.epoch == 0 and h.reconnects == 0  # no reconnect for an isolated bad frame
    assert frames[2].gap_before >= 1 and frames[3].gap_before >= 1  # the missing time is accounted as lost frames


def test_a_run_of_corrupt_frames_is_treated_as_a_failed_connection_and_reconnects():
    clock = FakeClock()
    caps = iter([_Cap(clock, ["ok"] + ["none"] * 60), _Cap(clock, ["ok", "ok", "ok"])])
    src = RtspFrameSource("rtsp://x/s", nominal_fps=10.0, capture_factory=lambda u: next(caps), clock=clock, sleep=clock.sleep,
                          backoff_s=(0.5,), max_frames=4)
    frames = list(src)
    assert [f.epoch for f in frames] == [0, 1, 1, 1]
    assert src.health().reconnects == 1


def test_the_pipeline_survives_corrupt_frames_end_to_end():
    clock = FakeClock()
    cap = _Cap(clock, (["ok"] * 30 + ["none"]) * 6)
    src = RtspFrameSource("rtsp://x/s", nominal_fps=10.0, capture_factory=lambda u: cap, clock=clock, sleep=clock.sleep, max_frames=150)
    eng = _engine()
    stats = StreamingPipeline(src, eng, on_window=lambda s: None, frame_queue_size=8).run()
    assert stats["engine"]["frames_processed"] == 150 and src.health().corrupt_frames == 4  # 4 bad packets were read before frame 150


def _write_video(path, n, fps=10.0):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i % 255, np.uint8))
    w.release()


def test_a_file_that_ends_short_of_the_container_frame_count_is_flagged_deterministically(tmp_path, monkeypatch):
    from anvesh.perception import ingestion

    class ShortCapture:
        def __init__(self, path):
            self._n = 40

        def isOpened(self):
            return True

        def get(self, prop):
            return 10.0 if prop == ingestion.cv2.CAP_PROP_FPS else 100.0  # fps 10, the container claims 100 frames

        def read(self):
            if self._n <= 0:
                return False, None
            self._n -= 1
            return True, np.zeros((12, 16, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(ingestion.cv2, "VideoCapture", ShortCapture)
    path = tmp_path / "short.mp4"
    path.write_bytes(b"x")
    src = VideoFileFrameSource(path)
    assert len(list(src)) == 40
    h = src.health()
    assert h.state == "ended" and "premature end" in h.last_error and "40 of 100" in h.last_error


def test_a_complete_recording_ends_cleanly_without_a_warning(tmp_path):
    path = tmp_path / "ok.mp4"
    _write_video(path, 60)
    src = VideoFileFrameSource(path)
    assert len(list(src)) == 60
    h = src.health()
    assert h.state == "ended" and h.last_error == ""


def test_a_recording_cut_before_its_index_fails_loudly_instead_of_yielding_nothing(tmp_path):
    """mp4 written by OpenCV keeps its index (moov) at the end; cutting the tail makes the file unopenable. That must be an
    error at open (the session fails with a message), never a silent zero-frame 'completed' run."""
    full = tmp_path / "full.mp4"
    _write_video(full, 120)
    data = full.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: int(len(data) * 0.55)])
    with pytest.raises(RuntimeError, match="Failed to open video file"):
        list(VideoFileFrameSource(cut))
