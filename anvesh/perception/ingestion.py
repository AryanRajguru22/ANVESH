"""Video ingestion (blueprint Part 3, module 1): read timestamped frames
from a recorded video file per camera.

Uses OpenCV's VideoCapture with an explicit open-check -- a video that
fails to open raises immediately instead of silently yielding zero frames
(blueprint Part 3 module 1 note, fixing V0_AUDIT §13's silent-failure bug).

Frame timestamps are relative to the start of the recording
(`frame_index / fps`, t=0 at the first frame) -- this is recorded/replayed
video, not a live feed, so there is no wall-clock/UTC capture time to
recover from the file itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

import cv2

from anvesh.perception.types import Frame


class FrameSource(Protocol):
    """Abstraction over "a sequence of timestamped frames" for one camera."""

    def fps(self) -> float: ...

    def __iter__(self) -> Iterator[Frame]: ...


class VideoFileFrameSource:
    """Reads timestamped frames from a recorded video file via OpenCV."""

    def __init__(self, video_path: str | Path) -> None:
        self._video_path = Path(video_path)
        if not self._video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self._video_path}")

    def fps(self) -> float:
        capture = cv2.VideoCapture(str(self._video_path))
        try:
            self._require_opened(capture)
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        return self._require_valid_fps(fps)

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
        try:
            index = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                yield Frame(index=index, timestamp=index / fps, image=image)
                index += 1
        finally:
            capture.release()
