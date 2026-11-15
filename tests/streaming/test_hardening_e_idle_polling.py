"""Hardening E: idle polling is automatic only for sources that expose stream_time(); explicit polling
on an unsupported source raises instead of silently doing nothing; the live frame policy is unchanged."""

import inspect
import threading
import time

import pytest

from anvesh.perception.ingestion import SourceHealth
from anvesh.streaming.engine import StreamHealth, StreamingPerceptionEngine
from anvesh.streaming.pipeline import IDLE_POLL_AUTO, MAX_AUTO_IDLE_POLL_S, StreamingPipeline
from anvesh.streaming.queues import DropPolicy
from anvesh.streaming.windows import WindowSpec
from tests.streaming.support import ListFrameSource, ScriptedTracker, make_frames

FPS = 10.0


def _engine(windows=None, fps=FPS):
    return StreamingPerceptionEngine("camE", ScriptedTracker(lambda i: []), fps=fps,
                                     windows=windows or WindowSpec(window_s=10.0))


class _ClockedSource:
    """n frames, then a silent camera (no more frames, no end of stream); exposes a controllable stream clock."""

    def __init__(self, n=50, with_clock=True):
        self._frames = make_frames(n, FPS)
        self.now = (n - 1) / FPS
        self._closed = threading.Event()
        if with_clock:
            self.stream_time = lambda: self.now

    def fps(self):
        return FPS

    def __iter__(self):
        yield from self._frames
        self._closed.wait(timeout=15.0)

    def health(self):
        return SourceHealth("connected", 0, len(self._frames), 0, 0, FPS, 0.0)

    def close(self):
        self._closed.set()


class _NoClockSource(_ClockedSource):
    def __init__(self, n=50):
        super().__init__(n, with_clock=False)


# ---------------------------------------------------------------- resolution of the setting ----------


def test_default_is_auto_and_the_file_source_gets_no_polling():
    assert inspect.signature(StreamingPipeline.__init__).parameters["idle_poll_s"].default == IDLE_POLL_AUTO
    pipe = StreamingPipeline(ListFrameSource(make_frames(10, FPS)), _engine())
    assert pipe.idle_poll_s is None and pipe.stats()["idle_poll_s"] is None


def test_default_enables_polling_for_a_source_with_a_stream_clock():
    pipe = StreamingPipeline(_ClockedSource(), _engine())
    assert pipe.idle_poll_s == pytest.approx(MAX_AUTO_IDLE_POLL_S)  # default silence threshold 1.0 s -> min(0.5, 0.5)


def test_the_auto_period_tracks_the_engines_silence_threshold():
    eng = _engine(WindowSpec(window_s=10.0, max_gap_s=0.2))  # threshold = max(0.2, 2.5 / 10 fps) = 0.25 s
    assert eng.silence_threshold_s == pytest.approx(0.25)
    assert StreamingPipeline(_ClockedSource(), eng).idle_poll_s == pytest.approx(0.125)
    slow = _engine(fps=0.5)  # threshold = max(1.0, 2.5 * 2 s) = 5 s -> capped at the 0.5 s maximum
    assert StreamingPipeline(_ClockedSource(), slow).idle_poll_s == pytest.approx(MAX_AUTO_IDLE_POLL_S)


def test_explicit_none_opts_out_even_when_the_source_has_a_clock():
    assert StreamingPipeline(_ClockedSource(), _engine(), idle_poll_s=None).idle_poll_s is None


def test_an_explicit_positive_poll_is_used_as_given_when_supported():
    assert StreamingPipeline(_ClockedSource(), _engine(), idle_poll_s=0.01).idle_poll_s == 0.01
    assert StreamingPipeline(_ClockedSource(), _engine(), idle_poll_s=2).idle_poll_s == 2


@pytest.mark.parametrize("value", [0.05, 1, 10.0])
def test_explicit_polling_on_a_source_without_stream_time_raises(value):
    for src in (_NoClockSource(), ListFrameSource(make_frames(5, FPS))):
        with pytest.raises(ValueError, match="stream_time"):
            StreamingPipeline(src, _engine(), idle_poll_s=value)


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, True, False, "fast", "AUTO", float("nan")])
def test_invalid_poll_values_are_rejected(bad):
    with pytest.raises(ValueError):
        StreamingPipeline(_ClockedSource(), _engine(), idle_poll_s=bad)


def test_the_live_frame_policy_default_is_unchanged():
    assert inspect.signature(StreamingPipeline.__init__).parameters["frame_policy"].default == DropPolicy.BLOCK
    pipe = StreamingPipeline(_ClockedSource(), _engine())  # a clocked (live-style) source still defaults to BLOCK
    assert pipe.frame_queue.stats()["policy"] == DropPolicy.BLOCK.value


# ---------------------------------------------------------------- behaviour (threads, bounded waits) ----


def _run_pipeline(src, eng, **kw):
    got = []
    pipe = StreamingPipeline(src, eng, on_window=got.append, frame_queue_size=64, **kw)
    t = threading.Thread(target=pipe.run, daemon=True)
    t.start()
    return pipe, t, got


def _wait(predicate, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


def test_by_default_a_dead_camera_is_noticed_without_any_opt_in():
    src, eng = _ClockedSource(), _engine()
    pipe, t, got = _run_pipeline(src, eng)  # NO idle_poll_s argument
    try:
        assert _wait(lambda: eng.frames_processed == 50, 5.0)
        src.now = 30.0  # the camera has been dark for ~25 s of stream time
        assert _wait(lambda: len(got) >= 3 and eng.health == StreamHealth.DOWN, 5.0), (len(got), eng.health)
        assert any(s.unavailable for s in got)
    finally:
        pipe.stop()
        t.join(timeout=5.0)
    assert not t.is_alive()


def test_explicit_opt_out_keeps_the_old_blind_behaviour():
    src, eng = _ClockedSource(), _engine()
    pipe, t, got = _run_pipeline(src, eng, idle_poll_s=None)
    try:
        assert _wait(lambda: eng.frames_processed == 50, 5.0)
        src.now = 30.0
        time.sleep(1.2)  # long enough for the default 0.5 s poll to have fired twice
        assert got == [] and eng.health == StreamHealth.LIVE  # nothing noticed: this is exactly what opting out means
    finally:
        pipe.stop()
        t.join(timeout=5.0)


def test_a_recorded_file_still_runs_to_completion_with_no_polling():
    eng = _engine()
    got = []
    StreamingPipeline(ListFrameSource(make_frames(250, FPS)), eng, on_window=got.append).run()
    assert len(got) == 3 and not any(s.unavailable for s in got)
