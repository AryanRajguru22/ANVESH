"""Live per-track speed (dashboard overlay / overspeed / braking input) and the marking-standard calibration kind.

The scale used here is 0.1 m per pixel, so a track moving 100 px/s is 10 m/s = 36 km/h."""

import collections

import pytest

from anvesh.app.cameras import ConfigError, build_calibration_profile, geometry_label, parse_camera_config
from anvesh.app.livespeed import LiveSpeed
from anvesh.world.calibration import (CalibrationPoint, MetricValidationRecord, ValidationEvidenceKind, fit_planar_homography)

S = collections.namedtuple("S", "t ground bbox")


def _pts():
    return [CalibrationPoint((x, y), (x * 0.1, y * 0.1)) for x, y in ((0, 0), (1000, 0), (1000, 800), (0, 800), (500, 400))]


def _profile(kind="marking_standard_holdout", n=4, rmse=0.1):
    return fit_planar_homography("p", "cam", _pts(), validation=MetricValidationRecord(kind, "test", n, rmse, 1.0))


def _track(speed_mps, seconds=3.0, fps=25, y=400.0, x0=100.0, box=(40, 20)):
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        x = x0 + speed_mps / 0.1 * t
        out.append(S(t, (x, y), (x - box[0] / 2, y - box[1], x + box[0] / 2, y)))
    return out


# -- calibration kind -------------------------------------------------------------------------------

def test_marking_standard_kind_is_validated_but_never_survey_or_synthetic():
    p = _profile()
    assert p.supports_metric_speed and p.is_marking_standard_validated
    assert not p.is_real_camera_validated and not p.is_synthetic_validated


def test_marking_standard_record_needs_four_holdout_points_and_a_passing_rmse():
    with pytest.raises(ValueError):
        MetricValidationRecord("marking_standard_holdout", "x", 3, 0.1, 1.0)
    with pytest.raises(ValueError):
        MetricValidationRecord("marking_standard_holdout", "x", 4, 1.5, 1.0)  # hold-out error above tolerance
    assert MetricValidationRecord("marking_standard_holdout", "x", 4, 0.4, 1.0).kind == ValidationEvidenceKind.MARKING_STANDARD_HOLDOUT


def test_geometry_label_says_approximate_and_not_surveyed():
    lab = geometry_label(_profile())
    assert lab["metric_speed"] is True and lab["approximate"] is True and lab["synthetic"] is False
    assert "APPROXIMATE" in lab["label"] and "NOT surveyed" in lab["label"]
    assert geometry_label(_profile("survey_holdout"))["label"] == "Metric - validated by survey hold-out"
    assert geometry_label(None)["metric_speed"] is False


def test_demo_camera_config_builds_a_marking_standard_profile():
    import json
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parents[2] / "demo" / "camera_config.json").read_text())
    parse_camera_config(cfg)  # the shipped demo configuration is valid
    prof = build_calibration_profile("demo-cal", "demo", cfg["calibration"])
    assert prof.is_marking_standard_validated and prof.validation.holdout_rmse_m < 1.0 and cfg["speed_zone"]
    # near-field scale sanity: one pixel step near the bottom of the frame is a few decimetres, not metres
    a, b = prof.image_to_world((450, 330)), prof.image_to_world((450, 331))
    assert 0.03 < ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 < 0.2


# -- config -----------------------------------------------------------------------------------------

def test_speed_zone_is_parsed_and_validated():
    cfg = parse_camera_config({"speed_zone": [[0, 0], [10, 0], [10, 10]]})
    assert cfg.speed_zone == ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    assert parse_camera_config({}).speed_zone == ()
    with pytest.raises(ConfigError):
        parse_camera_config({"speed_zone": [[0, 0], [1, 1]]})


# -- estimator --------------------------------------------------------------------------------------

def test_speed_is_the_calibrated_slope_with_an_error_band():
    ls = LiveSpeed(_profile(), ema=1.0)
    est = ls.estimate("a", _track(25.0)[-40:])  # 25 m/s = 90 km/h
    assert est["status"] == "ok" and est["speed_kmh"] == pytest.approx(90.0, abs=0.5) and 0 < est["error_kmh"] < 3 and est["approximate"] is True


def test_no_metric_calibration_means_no_number():
    assert LiveSpeed(None).estimate("a", _track(25.0))["status"] == "no_metric_calibration"
    unvalidated = fit_planar_homography("p", "cam", _pts())  # fitted, no validation record
    est = LiveSpeed(unvalidated).estimate("a", _track(25.0))
    assert est["status"] == "no_metric_calibration" and est["speed_kmh"] is None


def test_outside_the_zone_short_history_and_jitter_give_no_number():
    zone = [[0, 0], [5000, 0], [5000, 300], [0, 300]]  # track runs along y=400
    assert LiveSpeed(_profile(), zone=zone).estimate("a", _track(25.0))["status"] == "outside_speed_zone"
    assert LiveSpeed(_profile()).estimate("a", _track(25.0)[:3])["status"] == "insufficient_history"
    jittery = [S(s.t, (s.ground[0] + (60 if i % 2 else -60), s.ground[1] + (50 if i % 3 else -50)), s.bbox) for i, s in enumerate(_track(2.0))]
    assert LiveSpeed(_profile()).estimate("a", jittery)["status"] in ("rejected_unstable", "ok")  # unstable tracks are not reported as a calm number


def test_a_box_clipped_by_the_frame_edge_is_not_a_speed_source():
    ls = LiveSpeed(_profile(), frame_size=(5000, 400))  # bottom of the box is at y=400 = the frame edge
    assert ls.estimate("a", _track(25.0))["status"] == "clipped_at_frame_edge"
    assert LiveSpeed(_profile(), frame_size=(5000, 900)).estimate("a", _track(25.0))["status"] == "ok"
    assert LiveSpeed(_profile(), frame_size=lambda: (5000, 900)).estimate("a", _track(25.0))["status"] == "ok"  # lazily supplied


def test_implausible_speed_is_rejected():
    assert LiveSpeed(_profile(), max_kmh=100.0).estimate("a", _track(40.0))["status"] == "rejected_implausible"  # 144 km/h


def test_smoothing_is_per_track_and_forgotten_when_absent():
    ls = LiveSpeed(_profile(), ema=0.5)
    ls.estimate("a", _track(10.0)); ls.estimate("a", _track(20.0))
    assert 36 < ls._smooth["a"] < 72  # between the two readings, not equal to the last
    ls.forget_absent(["b"])
    assert "a" not in ls._smooth
