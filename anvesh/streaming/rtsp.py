"""RTSP / live-style frame source (M-A streaming core).

STATUS (be precise about what is validated): this is IMPLEMENTED and
UNIT-TESTED against injected fake captures (reconnect, backoff, epoch bump,
gap/jitter accounting, clean stop). It has NOT been run against a real RTSP
camera or server in this repository -- no RTSP server/camera was available.
Treat live behaviour (codec quirks, FFmpeg timeouts, real network loss) as
unvalidated.

Design for live sources:
  - STREAM CLOCK. A live source has no trustworthy file PTS, so a frame's
    `timestamp` is the monotonic capture clock (`time.perf_counter`) relative to the first frame,
    continuous across reconnects (so windows keep advancing and never go
    backwards). `Frame.capture_wall_s` is the same clock, absolute.
  - RECONNECT. A failed open or a failed/raised read releases the capture and
    reconnects with exponential backoff (`backoff_s`, last value repeated).
    Every successful (re)connection after the first bumps `Frame.epoch`; the
    consumer must treat a new epoch as a new tracker/identity session
    (raw tracker IDs of the old connection mean nothing in the new one).
    After `max_reconnects` consecutive failures the source ends in state
    "failed" rather than spinning forever.
  - JITTER / GAPS. Inter-frame intervals are tracked (EMA fps, stddev of the
    last 64 intervals). An interval above `gap_factor` x the nominal interval
    is reported as `Frame.gap_before` (estimated skipped frames) and summed
    in `health().source_gap_frames` -- "frames the source lost", distinct from
    frames the pipeline's bounded queue drops later.
  - HEALTH. `health()` is safe to call from another thread.
"""

from __future__ import annotations

import collections
import statistics
import time
from typing import Iterator

import cv2

from anvesh.perception.ingestion import SourceHealth
from anvesh.perception.types import Frame


def _plausible_fps(fps) -> bool:
    return bool(fps) and 1.0 <= float(fps) <= 240.0


class RtspFrameSource:
    def __init__(
        self,
        url: str,
        nominal_fps: float = None,
        capture_factory=None,
        clock=time.perf_counter,
        sleep=time.sleep,
        backoff_s: tuple = (0.5, 1.0, 2.0, 4.0, 8.0),
        max_reconnects: int = None,
        max_frames: int = None,
        gap_factor: float = 1.5,
    ) -> None:
        if not url:
            raise ValueError("url must be non-empty")
        if nominal_fps is not None and not _plausible_fps(nominal_fps):
            raise ValueError(f"nominal_fps must be within [1, 240], got {nominal_fps}")
        if not backoff_s:
            raise ValueError("backoff_s must be non-empty")
        self._url = url
        self._nominal_fps = float(nominal_fps) if nominal_fps else None
        self._factory = capture_factory or (lambda u: cv2.VideoCapture(u, cv2.CAP_FFMPEG))
        self._clock = clock
        self._sleep = sleep
        self._backoff = tuple(backoff_s)
        self._max_reconnects = max_reconnects
        self._max_frames = max_frames
        self._gap_factor = gap_factor
        self._stop = False
        self._state = "idle"
        self._epoch = -1
        self._frames_read = 0
        self._reconnects = 0
        self._gap_frames = 0
        self._last_error = ""
        self._intervals: collections.deque = collections.deque(maxlen=64)
        self._ema_fps = 0.0
        self._probed_fps = None
        self._t0 = None  # stream-clock origin (first frame ever delivered)

    # -- FrameSource -------------------------------------------------------

    def fps(self) -> float:
        if self._nominal_fps:
            return self._nominal_fps
        if self._probed_fps:
            return self._probed_fps
        capture = self._factory(self._url)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open stream to probe fps: {self._url}")
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        if not _plausible_fps(fps):
            raise RuntimeError(f"stream reports an implausible fps ({fps}); pass nominal_fps explicitly")
        self._probed_fps = float(fps)
        return self._probed_fps

    def stream_time(self):
        """The current STREAM time (seconds since the first frame, same clock as
        `Frame.timestamp`), or None before the first frame. Lets the pipeline keep
        windows closing while the camera is silent."""
        t0 = self._t0
        return None if t0 is None else self._clock() - t0

    def health(self) -> SourceHealth:
        intervals = list(self._intervals)
        jitter = statistics.pstdev(intervals) if len(intervals) > 1 else 0.0
        return SourceHealth(
            state=self._state, epoch=max(self._epoch, 0), frames_read=self._frames_read,
            reconnects=self._reconnects, source_gap_frames=self._gap_frames,
            measured_fps=self._ema_fps, interval_jitter_s=jitter, last_error=self._last_error,
        )

    def close(self) -> None:
        self._stop = True

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[Frame]:
        nominal = self.fps()
        nominal_dt = 1.0 / nominal
        t0 = None  # stream clock origin = first frame ever delivered
        index = 0
        last_wall = None
        failures = 0
        self._stop = False

        while not self._stop:
            capture = None
            try:
                capture = self._factory(self._url)
                if not capture.isOpened():
                    raise RuntimeError("open failed")
            except Exception as exc:  # noqa: BLE001 - any capture error is a connection failure
                self._last_error = f"connect: {exc}"
                if capture is not None:
                    capture.release()
                if not self._after_failure(failures):
                    return
                failures += 1
                continue

            self._epoch += 1
            if self._epoch > 0:
                self._reconnects += 1
            self._state = "connected"
            reconnect_downtime = None
            if self._epoch > 0 and last_wall is not None:
                reconnect_downtime = self._clock() - last_wall

            try:
                while not self._stop:
                    t_read = self._clock()
                    ok, image = capture.read()
                    now = self._clock()
                    if not ok:
                        raise RuntimeError("read failed")
                    failures = 0
                    if t0 is None:
                        t0 = now
                        self._t0 = now
                    gap = 0
                    if last_wall is not None:
                        dt = now - last_wall
                        if reconnect_downtime is not None:
                            gap = max(0, int(round(reconnect_downtime / nominal_dt)) - 1)
                            reconnect_downtime = None
                        else:
                            self._intervals.append(dt)
                            if dt > 0:
                                inst = 1.0 / dt
                                self._ema_fps = inst if self._ema_fps == 0.0 else 0.9 * self._ema_fps + 0.1 * inst
                            if dt > self._gap_factor * nominal_dt:
                                gap = max(0, int(round(dt / nominal_dt)) - 1)
                    self._gap_frames += gap
                    self._frames_read += 1
                    last_wall = now
                    yield Frame(
                        index=index, timestamp=now - t0, image=image,
                        capture_wall_s=now, decode_s=now - t_read, epoch=self._epoch, gap_before=gap,
                    )
                    index += 1
                    if self._max_frames is not None and index >= self._max_frames:
                        self._state = "ended"
                        return
            except GeneratorExit:
                raise
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"read: {exc}"
            finally:
                if capture is not None:
                    capture.release()

            if self._stop:
                break
            if not self._after_failure(failures):
                return
            failures += 1

        self._state = "stopped"

    def _after_failure(self, failures: int) -> bool:
        """Back off before reconnecting; False once the retry budget is spent."""
        if self._max_reconnects is not None and failures >= self._max_reconnects:
            self._state = "failed"
            return False
        self._state = "reconnecting"
        self._sleep(self._backoff[min(failures, len(self._backoff) - 1)])
        return True
