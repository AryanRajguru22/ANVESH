from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_tiny_video(path: Path, num_frames: int = 5, fps: float = 10.0, size: tuple = (16, 12)) -> None:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "test setup: failed to open synthetic video writer"
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=i * 10 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def tiny_video_path(tmp_path) -> Path:
    """A small, synthetic (non-external) mp4 fixture for ingestion tests."""
    path = tmp_path / "tiny.mp4"
    _write_tiny_video(path, num_frames=5, fps=10.0, size=(16, 12))
    return path
