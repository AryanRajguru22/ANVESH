"""Debug visualization (blueprint Part 3, module 8): draw bounding boxes,
vehicle class, persistent track ID, frame/time, and trajectory trails onto
frames. This is a plain OpenCV drawing utility, not a dashboard/UI --
`scripts/run_pipeline.py` uses it to produce an annotated demo video.
"""

from __future__ import annotations

from pathlib import Path

import cv2

_BOX_COLOR = (60, 200, 60)
_TEXT_COLOR = (255, 255, 255)
_TRAIL_COLOR = (60, 160, 255)


def draw_frame(
    image,
    frame_index: int,
    timestamp: float,
    tracked_detections: list,
    active_tracks: dict,
    trail_length: int = 30,
):
    annotated = image.copy()

    for td in tracked_detections:
        x1, y1, x2, y2 = (int(v) for v in td.detection.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"id={td.track_id} {td.detection.vehicle_class.value} {td.detection.confidence:.2f}"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA
        )

        state = active_tracks.get(td.track_id)
        if state is not None and len(state.trajectory) > 1:
            points = state.trajectory[-trail_length:]
            for (x0, y0, _), (x1p, y1p, _) in zip(points, points[1:]):
                cv2.line(annotated, (int(x0), int(y0)), (int(x1p), int(y1p)), _TRAIL_COLOR, 2)

    header = f"frame={frame_index} t={timestamp:.2f}s"
    cv2.putText(annotated, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 2, cv2.LINE_AA)
    return annotated


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter with an explicit open-check."""

    def __init__(self, output_path, fps: float, frame_size: tuple) -> None:
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._output_path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {self._output_path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
