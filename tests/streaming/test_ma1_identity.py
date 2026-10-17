"""M-A.1 Fix 4: session and run identity. An identity is a TRACKLET key
`(camera, session, serial)`; it must be unique within an engine's whole lifetime
(including repeated/non-monotonic epochs) and, by default, across runs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anvesh.perception.pipeline import run_m1_pipeline
from anvesh.perception.track_manager import TrackManager, derive_run_id
from anvesh.perception.types import TrackedDetection
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.windows import WindowSpec
from tests.streaming.support import ScriptedTracker, detection, make_frames

FPS = 10.0


def _one_car(i):
    return [("1", (50.0, 50.0))]  # the SAME raw tracker id in every frame/session


def _run(eng, frames):
    snaps = []
    for f in frames:
        snaps.extend(eng.process_frame(f))
    snaps.extend(eng.finish())
    return snaps


def _identity_keys(eng):
    return [s.identity_key for s in eng.store.manager.all_tracks()]


def test_manager_never_reuses_a_session_id():
    m = TrackManager(max_occlusion_s=1.0, camera_id="cam", session_id="s000")
    m.update(0.0, [TrackedDetection(detection(0, 0.0, 10, 10), "1")])
    with pytest.raises(ValueError, match="already used"):
        m.begin_session("s000")  # the current one
    m.begin_session("s001")
    with pytest.raises(ValueError, match="already used"):
        m.begin_session("s000")  # an EARLIER one (previously accepted: duplicate key s000/t00000)
    assert m.used_sessions == {"s000", "s001"}


def test_engine_epoch_sequence_0_1_0_never_collides():
    tracker = ScriptedTracker(_one_car)
    eng = StreamingPerceptionEngine("cam", tracker, FPS, windows=WindowSpec(10.0), run_id=None)
    frames = (make_frames(20, FPS, epoch=0) + make_frames(20, FPS, epoch=1, start_index=20)
              + make_frames(20, FPS, epoch=0, start_index=40))
    _run(eng, frames)
    keys = _identity_keys(eng)
    assert len(keys) == len(set(keys)) == 3  # same raw id "1" every time, three distinct tracklets
    assert [k[1] for k in keys] == ["s000", "s001", "s000e1"]
    assert tracker.reset_calls == 2 and eng.stats()["sessions_started"] == 3


def test_flapping_epochs_get_distinct_suffixes():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0), run_id=None)
    frames, idx = [], 0
    for epoch in (0, 1, 0, 1, 0):
        frames += make_frames(10, FPS, epoch=epoch, start_index=idx)
        idx += 10
    _run(eng, frames)
    sessions = [k[1] for k in _identity_keys(eng)]
    assert sessions == ["s000", "s001", "s000e1", "s001e1", "s000e2"]
    assert len(set(_identity_keys(eng))) == 5


def test_monotonic_reconnects_keep_the_plain_session_names():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0), run_id=None)
    frames, idx = [], 0
    for epoch in (0, 1, 2, 3):
        frames += make_frames(10, FPS, epoch=epoch, start_index=idx)
        idx += 10
    _run(eng, frames)
    assert [k[1] for k in _identity_keys(eng)] == ["s000", "s001", "s002", "s003"]


def test_unchanged_epoch_is_not_a_new_session():
    tracker = ScriptedTracker(_one_car)
    eng = StreamingPerceptionEngine("cam", tracker, FPS, windows=WindowSpec(10.0), run_id=None)
    _run(eng, make_frames(30, FPS, epoch=7))
    assert tracker.reset_calls == 0 and eng.stats()["sessions_started"] == 1 and len(_identity_keys(eng)) == 1


def test_repeated_raw_tracker_ids_across_a_reset_open_new_identities():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0), run_id="r1")
    snaps = _run(eng, make_frames(20, FPS, epoch=0) + make_frames(20, FPS, epoch=1, start_index=20))
    ids = {t.track_id for s in snaps for t in s.tracks}
    assert ids == {"cam:r1-s000-t00000", "cam:r1-s001-t00000"}
    assert {s.raw_tracker_id for s in eng.store.manager.all_tracks()} == {"1"}  # raw id kept for audit only


def test_two_default_engines_never_share_identities():
    def ids(eng):
        return {t.track_id for s in _run(eng, make_frames(30, FPS)) for t in s.tracks}

    a = ids(StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0)))
    b = ids(StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0)))
    assert a and b and not (a & b)  # default is run-scoped


def test_explicit_session_only_scope_is_an_opt_out_and_collides_across_runs():
    def ids(eng):
        return {t.track_id for s in _run(eng, make_frames(30, FPS)) for t in s.tracks}

    a = ids(StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0), run_id=None))
    b = ids(StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0), run_id=None))
    assert a == b  # why it is not the default: two runs mint identical ids


def test_an_explicit_run_id_is_reproducible():
    def ids(rid):
        eng = StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, windows=WindowSpec(10.0), run_id=rid)
        return {t.track_id for s in _run(eng, make_frames(30, FPS)) for t in s.tracks}

    assert ids("same") == ids("same") and not (ids("same") & ids("other"))


def test_identity_scope_is_reported():
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS)
    assert eng.stats()["identity_scope"] == "run" and len(eng.stats()["run_id"]) == 8
    eng = StreamingPerceptionEngine("cam", ScriptedTracker(_one_car), FPS, run_id=None)
    assert eng.stats()["identity_scope"] == "session" and eng.stats()["run_id"] is None


def _write_video(path: Path, frames: int, w: int = 32) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, 24))
    for i in range(frames):
        writer.write(np.full((24, w, 3), i % 255, dtype=np.uint8))
    writer.release()


def test_batch_pipeline_default_ids_are_deterministic_per_recording_and_distinct_across_recordings(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write_video(a, 20)
    _write_video(b, 40)

    def run(path):
        tracker = ScriptedTracker(_one_car)
        tracker.lost_buffer_frames = 0
        res = run_m1_pipeline(path, "cam", tracker, "scripted", "0")
        return {t.track_id for t in res.vehicle_tracks}

    first, again, other = run(a), run(a), run(b)
    assert first == again  # same recording -> same identities (reproducible)
    assert first and other and not (first & other)  # different recording -> no collision
    assert all(i.startswith("cam:" + derive_run_id("cam", "a.mp4", a.stat().st_size)) for i in first)
