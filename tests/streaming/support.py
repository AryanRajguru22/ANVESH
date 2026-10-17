"""Shared test doubles for streaming tests: an in-memory frame source, a
scripted tracker, and a bounded-latency fake clock. No YOLO, no GPU, no files."""

from __future__ import annotations

import numpy as np

from anvesh.perception.ingestion import SourceHealth
from anvesh.perception.types import Frame, RawDetection, TrackedDetection
from anvesh.storage.schemas import VehicleClass

TINY = np.zeros((12, 16, 3), dtype=np.uint8)


def make_frames(n: int, fps: float = 10.0, epoch: int = 0, start_index: int = 0, **kw) -> list:
    return [
        Frame(index=start_index + i, timestamp=(start_index + i) / fps, image=TINY, epoch=epoch, **kw)
        for i in range(n)
    ]


class ListFrameSource:
    """A FrameSource over a prebuilt list of Frames (any fps, any gaps)."""

    def __init__(self, frames: list, fps: float = 10.0) -> None:
        self._frames = frames
        self._fps = fps
        self.closed = False

    def fps(self) -> float:
        return self._fps

    def __iter__(self):
        for f in self._frames:
            if self.closed:
                return
            yield f

    def health(self) -> SourceHealth:
        return SourceHealth("connected", 0, len(self._frames), 0, 0, self._fps, 0.0)

    def close(self) -> None:
        self.closed = True


def detection(frame_index: int, timestamp: float, x: float, y: float, w: float = 10.0, h: float = 10.0):
    return RawDetection(
        frame_index=frame_index, timestamp=timestamp, class_name="car", vehicle_class=VehicleClass.CAR,
        confidence=0.9, bbox_xyxy=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
    )


class ScriptedTracker:
    """Deterministic tracker double. `script(frame_index) -> [(raw_id, (x, y))]`.
    Records `reset()` calls and declares ByteTrack-like `lost_buffer_frames`."""

    def __init__(self, script, lost_buffer_frames: int = 30) -> None:
        self._script = script
        self.lost_buffer_frames = lost_buffer_frames
        self.reset_calls = 0
        self.updates = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def update(self, image, frame_index: int, timestamp: float):
        self.updates += 1
        raw, tracked = [], []
        for raw_id, (x, y) in self._script(frame_index):
            d = detection(frame_index, timestamp, x, y)
            raw.append(d)
            if raw_id is not None:
                tracked.append(TrackedDetection(detection=d, track_id=str(raw_id)))
        return raw, tracked


class FakeClock:
    """Manually advanced clock for pacing/latency tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t
        self.sleeps: list = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
