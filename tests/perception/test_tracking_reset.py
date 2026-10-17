"""Regression for `ByteTrackYoloTracker.reset()` (ANVESH_PRODUCT_GAP_AUDIT.md
finding 10): the old implementation set `predictor.trackers = None`, which made
the NEXT `update()` raise `TypeError: 'NoneType' object is not subscriptable` on
the installed Ultralytics. The suite's tracker fakes can never reproduce that,
so this test runs the REAL model/tracker on a synthetic image (skipped only if
the weights are absent -- they are git-ignored)."""

from pathlib import Path

import numpy as np
import pytest

from anvesh.perception.tracking import ByteTrackYoloTracker

WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "yolov8n.pt"
pytestmark = pytest.mark.skipif(not WEIGHTS.is_file(), reason="models/yolov8n.pt not present")


def _image():
    rng = np.random.default_rng(0)
    return (rng.random((360, 640, 3)) * 255).astype(np.uint8)


def test_reset_then_update_works_on_the_real_tracker():
    tracker = ByteTrackYoloTracker(model_name=str(WEIGHTS), confidence_threshold=0.05)
    img = _image()
    for i in range(3):
        tracker.update(img, i, i * 0.1)
    tracker.reset()
    raw, tracked = tracker.update(img, 3, 0.3)  # used to raise TypeError
    assert isinstance(raw, list) and isinstance(tracked, list)


def test_the_legacy_reset_really_did_crash_this_ultralytics():
    """Pins that the environment reproduces the original defect, so the test
    above is capable of failing."""
    tracker = ByteTrackYoloTracker(model_name=str(WEIGHTS), confidence_threshold=0.05)
    img = _image()
    tracker.update(img, 0, 0.0)
    tracker._model.predictor.trackers = None  # the old reset() body
    with pytest.raises(TypeError):
        tracker.update(img, 1, 0.1)


def test_tracker_declares_its_lost_buffer_for_the_occlusion_horizon():
    tracker = ByteTrackYoloTracker(model_name=str(WEIGHTS))
    assert tracker.lost_buffer_frames == 30  # bytetrack.yaml track_buffer


def test_identities_stay_unique_across_a_tracker_reset_through_the_engine():
    """After reset ByteTrack restarts raw ids at 1; the engine must fence the
    new run into a new session so identity keys never collide."""
    from anvesh.perception.types import Frame
    from anvesh.streaming.engine import StreamingPerceptionEngine

    class Wrapped:
        """Real tracker; frames of epoch 1 trigger the engine's reset()."""

        def __init__(self):
            self.inner = ByteTrackYoloTracker(model_name=str(WEIGHTS), confidence_threshold=0.05)
            self.lost_buffer_frames = self.inner.lost_buffer_frames
            self.resets = 0

        def update(self, image, i, t):
            return self.inner.update(image, i, t)

        def reset(self):
            self.resets += 1
            self.inner.reset()

    tracker = Wrapped()
    eng = StreamingPerceptionEngine("cam", tracker, 10.0, archive=True)
    img = _image()
    for i in range(3):
        eng.process_frame(Frame(i, i / 10.0, img, epoch=0))
    for i in range(3, 6):
        eng.process_frame(Frame(i, i / 10.0, img, epoch=1))
    eng.finish()
    assert tracker.resets == 1
    keys = [s.identity_key for s in eng.store.manager.all_tracks()]
    assert len(keys) == len(set(keys))
