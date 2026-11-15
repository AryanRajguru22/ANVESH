"""Video ingestion (blueprint Part 3, module 1): read timestamped frames from
a camera source -- a recorded file OR a live stream -- through ONE
`FrameSource` interface (M-A: "replay = live").

Both kinds of source feed the same streaming pipeline
(`anvesh.streaming`); there is no batch-only and live-only perception path.
This module holds the interface, the recorded-file source and the shared
health record. The RTSP/live-style source lives in `anvesh.streaming.rtsp`.

Uses OpenCV's VideoCapture with an explicit open-check -- a video that
fails to open raises immediately instead of silently yielding zero frames
(blueprint Part 3 module 1 note, fixing V0_AUDIT SS13's silent-failure bug).

TIMING (see `anvesh.perception.types.Frame`): a file's frame timestamps are
`frame_index / fps` (t=0 at the first frame). That is deterministic, which
batch/stream parity relies on, but it also assumes a perfectly regular clock:
CityFlow's README warns that some frames are skipped or stuck in transmission,
so for such sources `index / fps` is only approximately the true capture
time. A file source can be PACED (`realtime_factor=1.0` emits frame `i` no
earlier than `i / fps` after the source started -- file-as-live-stream; `2.0`
= twice real time; `None` = as fast as the consumer pulls, "accelerated").
Pacing changes only WHEN a frame is emitted, never its `timestamp`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2

from anvesh.perception.types import Frame


@dataclass(frozen=True)
class SourceHealth:
    """Point-in-time health of a frame source (any kind).

    `state`: "idle" (not started) | "connected" | "reconnecting" | "ended"
    (the source stopped delivering frames; for a recorded file `last_error` is set when fewer frames were decoded than the
    container reports, i.e. the file may be truncated) | "stopped" (closed by the owner) | "failed".
    `frames_read`: frames successfully decoded and yielded.
    `source_gap_frames`: frames the source itself knows/estimates it skipped
    (inter-frame time far above nominal), NOT queue drops downstream.
    `measured_fps`/`interval_jitter_s`: EMA of the source's own delivery rate
    and the standard deviation of recent inter-frame intervals -- not to be
    confused with the processing FPS of the pipeline consuming it.
    """

    state: str
    epoch: int
    frames_read: int
    reconnects: int
    source_gap_frames: int
    measured_fps: float
    interval_jitter_s: float
    last_error: str = ""
    corrupt_frames: int = 0


class FrameSource(Protocol):
    """Abstraction over "a sequence of timestamped frames" for one camera."""

    def fps(self) -> float: ...

    def __iter__(self) -> Iterator[Frame]: ...

    def health(self) -> SourceHealth: ...

    def close(self) -> None: ...


class VideoFileFrameSource:
    """Reads timestamped frames from a recorded video file via OpenCV.

    `realtime_factor`: None = unpaced; otherwise the replay speed relative to
    the file's own clock (1.0 = real time). `clock`/`sleep` are injectable so
    pacing is testable without waiting.
    """

    def __init__(
        self,
        video_path: str | Path,
        realtime_factor: float = None,
        clock=time.perf_counter,
        sleep=time.sleep,
    ) -> None:
        self._video_path = Path(video_path)
        if not self._video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self._video_path}")
        if realtime_factor is not None and realtime_factor <= 0:
            raise ValueError(f"realtime_factor must be > 0 or None, got {realtime_factor}")
        self._realtime_factor = realtime_factor
        self._clock = clock
        self._sleep = sleep
        self._state = "idle"
        self._frames_read = 0
        self._last_error = ""
        self._stop = False

    @property
    def video_path(self) -> Path:
        return self._video_path

    def fps(self) -> float:
        capture = cv2.VideoCapture(str(self._video_path))
        try:
            self._require_opened(capture)
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        return self._require_valid_fps(fps)

    def health(self) -> SourceHealth:
        return SourceHealth(
            state=self._state, epoch=0, frames_read=self._frames_read, reconnects=0,
            source_gap_frames=0, measured_fps=0.0, interval_jitter_s=0.0, last_error=self._last_error,
        )

    def close(self) -> None:
        self._stop = True

    def _require_opened(self, capture: cv2.VideoCapture) -> None:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self._video_path}")

    def _require_valid_fps(self, fps: float) -> float:
        if not fps or fps <= 0:
            raise RuntimeError(f"Video reports an invalid FPS ({fps}): {self._video_path}")
        return float(fps)

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self._video_path))
        self._require_opened(capture)
        fps = self._require_valid_fps(capture.get(cv2.CAP_PROP_FPS))
        self._state = "connected"
        self._stop = False
        self._last_error = ""
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start = self._clock()
        try:
            index = 0
            while not self._stop:
                t_read = self._clock()
                ok, image = capture.read()
                decode_s = self._clock() - t_read
                if not ok:
                    # OpenCV reports a failed read exactly like a normal end of file, so a truncated or corrupt tail looks
                    # like a complete recording. The container's own frame count (exact for most mp4/avi, an estimate for some
                    # containers) lets us flag a clearly short read without failing the session.
                    self._state = "ended"
                    if expected > 0 and self._frames_read < expected - max(2, expected // 100):
                        self._last_error = (f"premature end of stream: decoded {self._frames_read} of {expected} frames reported by the "
                                            "container (the file may be truncated or corrupt; the count can be an estimate for some containers)")
                    break
                timestamp = index / fps
                if self._realtime_factor is not None:
                    due = start + timestamp / self._realtime_factor
                    wait = due - self._clock()
                    if wait > 0:
                        self._sleep(wait)
                self._frames_read += 1
                yield Frame(
                    index=index, timestamp=timestamp, image=image,
                    capture_wall_s=self._clock(), decode_s=decode_s,
                )
                index += 1
            if self._stop:
                self._state = "stopped"
        finally:
            capture.release()
