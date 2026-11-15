"""End-to-end through the REAL streaming engine/pipeline with a scripted SIMULATED source:
camera -> session -> detectors -> incident -> evidence."""
import time

import numpy as np

import pytest

from anvesh.app.runner import SessionManager
from anvesh.app.synthetic import ScriptedTracker


def _tracker_factory(camera, config):
    return ScriptedTracker(camera["source_uri"].split(":", 1)[-1])


def run_scenario(services, scenario, config=None, wait_s=60.0):
    services.cameras.create("admin", {"camera_id": f"sim-{scenario}", "name": scenario, "source_type": "synthetic",
                                      "source_uri": f"synthetic:{scenario}", "data_class": "simulated", "config": config or {}})
    mgr = SessionManager(services, tracker_factory=_tracker_factory)
    sess = mgr.start(f"sim-{scenario}", "admin")
    runner = mgr.get(sess["session_id"])
    runner._thread.join(wait_s)
    assert not runner._thread.is_alive(), "session did not finish"
    return mgr, runner


def test_stopped_car_produces_incident_with_evidence(services):
    mgr, runner = run_scenario(services, "stopped_car", {"stop_criterion": "displacement"})
    rt = runner.runtime()
    assert rt["state"] == "completed", rt
    assert rt["frames_processed"] == 300 and rt["frames_dropped"] == 0
    incidents, total = services.incidents.list({"session_id": runner.session_id})
    stops = [i for i in incidents if i["event_type"] in ("stopped_vehicle", "long_duration_stop")]
    assert stops, [i["event_type"] for i in incidents]
    inc = stops[0]
    assert inc["data_class"] == "simulated"
    assert inc["motion_space"] == "image" and inc["speed_basis"] == "image_px_displacement_heuristic"
    assert any("IMAGE space" in u for u in inc["uncertainty"])
    assert inc["established_cause"] is None
    assert all(v["role"] == "observed" for v in inc["involved"])
    assets = services.evidence.list(inc["incident_id"])
    kinds = {a["kind"] for a in assets}
    assert {"frame", "frame_annotated", "clip", "thumbnail"} <= kinds
    assert inc["evidence_completeness"] == "full"
    for a in assets:
        assert services.evidence.verify(a["asset_id"])["ok"]
    # the timeline starts with the detection
    full = services.incidents.get(inc["incident_id"], with_timeline=True)
    assert full["timeline"][0]["kind"] == "detected"
    assert services.audit.verify()["ok"]


def test_session_state_and_completion_recorded_in_db(services):
    mgr, runner = run_scenario(services, "queue")
    row = services.db.one("SELECT * FROM processing_sessions WHERE session_id = ?", (runner.session_id,))
    assert row["state"] == "completed" and row["ended_at"] and row["time_basis"] == "processing_clock"
    assert row["data_class"] == "simulated"


def test_reprocessing_same_source_creates_new_incidents_not_collisions(services):
    _, r1 = run_scenario(services, "stopped_car", {"stop_criterion": "displacement"})
    services.cameras.update("admin", "sim-stopped_car", {"enabled": True})
    mgr = SessionManager(services, tracker_factory=_tracker_factory)
    s2 = mgr.start("sim-stopped_car", "admin")
    r2 = mgr.get(s2["session_id"])
    r2._thread.join(60)
    a, _ = services.incidents.list({"session_id": r1.session_id})
    b, _ = services.incidents.list({"session_id": r2.session_id})
    assert a and b
    assert {i["incident_id"] for i in a}.isdisjoint({i["incident_id"] for i in b})
    assert {i["natural_key"] for i in a}.isdisjoint({i["natural_key"] for i in b})


def test_one_active_session_per_camera(services):
    from anvesh.app.cameras import ConfigError
    services.cameras.create("admin", {"camera_id": "sim-q", "name": "q", "source_type": "synthetic", "source_uri": "synthetic:queue",
                                      "data_class": "simulated", "config": {"realtime_factor": 1.0}})
    mgr = SessionManager(services, tracker_factory=_tracker_factory)
    first = mgr.start("sim-q", "admin")
    with pytest.raises(ConfigError):
        mgr.start("sim-q", "admin")
    mgr.stop(first["session_id"], "admin")
    mgr.get(first["session_id"])._thread.join(20)
    assert mgr.runtime(first["session_id"])["state"] == "stopped"


def test_finished_session_releases_ring_history_preview_and_model_but_keeps_counts(services):
    mgr, runner = run_scenario(services, "stopped_car", {"stop_criterion": "displacement"})
    assert runner.released and len(runner.ring) == 0 and runner.history.tracks_count() == 0 and runner.last_jpeg is None
    assert runner.engine._tracker is None and runner.tracker._inner is None  # the detector (and its GPU memory) is dropped
    rt = runner.runtime()
    assert rt["state"] == "completed" and rt["frames_processed"] == 300 and rt["unique_tracks_total"] >= 3 and rt["ring"]["bytes"] == 0
    # evidence written before release is untouched
    inc = services.incidents.list({"session_id": runner.session_id})[0][0]
    assert all(services.evidence.verify(a["asset_id"])["ok"] for a in services.evidence.list(inc["incident_id"]))


def test_terminal_sessions_are_evicted_from_memory_beyond_the_cap(services):
    mgr, runner = run_scenario(services, "queue")
    mgr.MAX_TERMINAL_KEPT = 0
    mgr._evict_terminal()
    assert mgr.get(runner.session_id) is None  # DB row remains the record
    assert services.db.one("SELECT state FROM processing_sessions WHERE session_id = ?", (runner.session_id,))["state"] == "completed"


def test_missing_detector_weights_fail_closed_without_any_download(services, tmp_path):
    import cv2
    from anvesh.app.cameras import CameraService
    from anvesh.app.runner import default_tracker_factory
    vid = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
    for _ in range(20):
        w.write(np.zeros((36, 64, 3), np.uint8))
    w.release()
    services.cameras = CameraService(services.db, services.audit, media_roots=(tmp_path,))
    services.cameras.create("admin", {"camera_id": "no-weights", "name": "n", "source_type": "file", "source_uri": str(vid), "data_class": "demo"})
    missing = tmp_path / "does_not_exist" / "yolov8n.pt"  # a name Ultralytics would try to download
    mgr = SessionManager(services, tracker_factory=default_tracker_factory(str(missing)))
    sid = mgr.start("no-weights", "admin")["session_id"]
    mgr.get(sid)._thread.join(30)
    rt = mgr.runtime(sid)
    assert rt["state"] == "failed" and "detector weights not found" in rt["error"] and "never downloads" in rt["error"]
    assert not missing.exists() and not list(tmp_path.rglob("*.pt"))  # nothing was fetched
    row = services.db.one("SELECT state, error FROM processing_sessions WHERE session_id = ?", (sid,))
    assert row["state"] == "failed" and "detector weights not found" in row["error"]
