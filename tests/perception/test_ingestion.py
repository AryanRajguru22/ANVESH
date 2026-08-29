from pathlib import Path

import pytest

from anvesh.perception.ingestion import VideoFileFrameSource


def test_missing_file_raises_file_not_found(tmp_path: Path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        VideoFileFrameSource(missing)


def test_invalid_video_content_raises_runtime_error(tmp_path: Path):
    bad_file = tmp_path / "not_a_video.mp4"
    bad_file.write_text("this is not video data")
    source = VideoFileFrameSource(bad_file)
    with pytest.raises(RuntimeError):
        source.fps()
    with pytest.raises(RuntimeError):
        list(source)


def test_reads_expected_frame_count_and_timestamps(tiny_video_path: Path):
    source = VideoFileFrameSource(tiny_video_path)
    fps = source.fps()
    assert fps == pytest.approx(10.0, rel=0.2)

    frames = list(source)
    assert len(frames) == 5

    for expected_index, frame in enumerate(frames):
        assert frame.index == expected_index
        assert frame.timestamp == pytest.approx(expected_index / fps)
        assert frame.image is not None
        assert frame.image.shape[2] == 3


def test_frame_source_is_reiterable(tiny_video_path: Path):
    source = VideoFileFrameSource(tiny_video_path)
    first_pass = [f.index for f in source]
    second_pass = [f.index for f in source]
    assert first_pass == second_pass == [0, 1, 2, 3, 4]
