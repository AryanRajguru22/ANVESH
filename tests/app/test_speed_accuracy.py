"""Regression protection for the speed estimator (`LiveSpeed`): known distance/time, unit conversion, irregular and dropped frames, box jitter,
track discontinuities, a REAL perspective transform, invalid calibration, outliers, smoothing, unreliable states and - the defect found by the
speed audit - samples OUTSIDE the calibrated zone must not influence the speed.

Geometry of the affine test profile: 0.1 m per pixel (so 100 px/s = 10 m/s = 36 km/h). Perspective profile: a pinhole-style projective map."""
import collections
import random

import pytest

from anvesh.app.livespeed import LiveSpeed
from anvesh.world.calibration import CalibrationPoint, MetricValidationRecord, fit_planar_homography

S = collections.namedtuple("S", "t ground bbox")


def _affine_profile():
    pts = [CalibrationPoint((x, y), (x * 0.1, y * 0.1)) for x, y in ((0, 0), (1000, 0), (1000, 800), (0, 800), (500, 400))]
    return fit_planar_homography("p", "cam", pts, validation=MetricValidationRecord("marking_standard_holdout", "t", 4, 0.1, 1.0))


def _s(t, x, y, w=40, h=20):
    return S(t, (x, y), (x - w / 2, y - h, x + w / 2, y))


def _line(speed_mps, times, x0=100.0, y=400.0):
    return [_s(t, x0 + speed_mps / 0.1 * t, y) for t in times]


def _times(seconds=1.6, fps=25):
    return [i / fps for i in range(int(seconds * fps) + 1)]


# ---------------------------------------------------------------- distance / time / units
def test_known_distance_and_time_give_the_known_speed_and_the_km_h_conversion():
    est = LiveSpeed(_affine_profile(), ema=1.0).estimate("a", _line(20.0, _times()))  # 20 m/s
    assert est["status"] == "ok" and est["speed_kmh"] == pytest.approx(72.0, abs=0.2)  # 20 m/s x 3.6
    est = LiveSpeed(_affine_profile(), ema=1.0).estimate("a", _line(27.78, _times()))
    assert est["speed_kmh"] == pytest.approx(100.0, abs=0.3)


def test_irregular_frame_intervals_use_the_real_timestamps_not_the_frame_count():
    rnd = random.Random(1)
    t, times = 0.0, []
    while t < 1.6:
        times.append(t)
        t += rnd.choice((0.02, 0.04, 0.06, 0.08))  # irregular but all below the gap limit
    est = LiveSpeed(_affine_profile(), ema=1.0).estimate("a", _line(20.0, times))
    assert est["status"] == "ok" and est["speed_kmh"] == pytest.approx(72.0, abs=0.3)
    # the same positions with a wrong, uniform clock would give a different number: the code must not assume 1/fps
    uniform = [i * 0.04 for i in range(len(times))]
    wrong = LiveSpeed(_affine_profile(), ema=1.0).estimate("a", [_s(u, s.ground[0], s.ground[1]) for u, s in zip(uniform, _line(20.0, times))])
    assert abs(wrong["speed_kmh"] - 72.0) > 3.0


def test_dropped_frames_do_not_change_the_speed_and_a_long_gap_ends_the_fitted_run():
    rnd = random.Random(2)
    kept = [t for t in _times(2.0) if rnd.random() > 0.3]  # ~30 % of the frames missing, timestamps kept
    kept = [t for i, t in enumerate(kept) if i == 0 or t - kept[i - 1] <= 0.12]
    est = LiveSpeed(_affine_profile(), ema=1.0).estimate("a", _line(20.0, kept))
    assert est["status"] == "ok" and est["speed_kmh"] == pytest.approx(72.0, abs=0.3)
    gap = [t for t in _times(2.0) if not 0.6 < t < 1.3]  # a 0.7 s hole: only the samples after it may be used
    ls = LiveSpeed(_affine_profile(), ema=1.0, min_samples=6, min_span_s=0.3)
    est = ls.estimate("a", _line(20.0, gap))
    assert est["status"] == "ok" and est["span_s"] <= 0.75  # the run restarts after the gap
    assert LiveSpeed(_affine_profile(), ema=1.0, min_samples=30).estimate("a", _line(20.0, gap))["status"] == "insufficient_history"


# ---------------------------------------------------------------- noise, outliers, discontinuities
def test_box_jitter_gives_a_speed_inside_its_own_error_band():
    rnd = random.Random(3)
    track = [_s(t, 100 + 200.0 * t + rnd.gauss(0, 10.0), 400 + rnd.gauss(0, 10.0)) for t in _times(2.0)]  # 1 m rms jitter, 20 m/s
    est = LiveSpeed(_affine_profile(), ema=1.0, window_s=2.0).estimate("a", track)
    assert est["status"] == "ok" and abs(est["speed_kmh"] - 72.0) <= est["error_kmh"]


def test_an_isolated_outlier_is_removed_and_many_outliers_make_the_track_unstable():
    track = _line(20.0, _times())
    bad = list(track)
    bad[20] = _s(bad[20].t, bad[20].ground[0] + 40.0, bad[20].ground[1] + 20.0)  # one box jumps ~4.5 m
    est = LiveSpeed(_affine_profile(), ema=1.0).estimate("a", bad)
    assert est["status"] == "ok" and est["outliers_removed"] == 1 and est["speed_kmh"] == pytest.approx(72.0, abs=1.0)
    many = [s if i % 3 else _s(s.t, s.ground[0] + 30.0, s.ground[1] + 25.0) for i, s in enumerate(track)]
    assert LiveSpeed(_affine_profile(), ema=1.0).estimate("a", many)["status"] in ("rejected_unstable", "ok")
    heavy = [_s(s.t, s.ground[0] + (30.0 if i % 2 else 0.0), s.ground[1] + (25.0 if i % 2 else 0.0)) for i, s in enumerate(track)]
    est = LiveSpeed(_affine_profile(), ema=1.0, max_error_kmh=10.0).estimate("a", heavy)  # the app's precision gate
    assert est["status"] != "ok" or est["error_kmh"] > 4.0  # either withheld, or shown with an honestly wide error band


def test_a_track_id_swap_jump_is_not_averaged_into_the_speed():
    first = _line(20.0, _times(1.0))                                   # one vehicle ...
    second = [_s(s.t + 1.04, s.ground[0] + 900.0, s.ground[1] + 0.0) for s in _line(20.0, _times(0.8))]  # ... then another 90 m ahead under the same id
    est = LiveSpeed(_affine_profile(), ema=1.0, min_samples=8, min_span_s=0.4).estimate("a", first + second)
    assert est["status"] == "ok" and est["speed_kmh"] == pytest.approx(72.0, abs=1.0) and est["span_s"] <= 0.81  # only the newer vehicle's samples


# ---------------------------------------------------------------- perspective, zone, invalid calibration
def _persp():
    """Pinhole-style projective map: v = 120 + 1500/(Y+4), u = 320 + 600 X/(Y+4); metres X across, Y along the road."""
    def to_img(X, Y):
        return 320 + 600 * X / (Y + 4), 120 + 1500 / (Y + 4)
    pts = [CalibrationPoint(to_img(X, Y), (X, Y)) for X in (-3.0, 0.0, 3.0) for Y in (2.0, 6.0, 12.0, 20.0, 30.0, 45.0)]
    return fit_planar_homography("p", "cam", pts, validation=MetricValidationRecord("marking_standard_holdout", "t", 4, 0.1, 1.0)), to_img


def test_perspective_transform_gives_the_same_speed_at_every_distance_from_the_camera():
    prof, to_img = _persp()
    speed = 30.0  # m/s = 108 km/h
    track = []
    for i in range(0, 90):
        t = i / 25
        u, v = to_img(1.5, 45.0 - speed * t * 0.5 - 0.0)  # approaching, slower than real to stay inside the calibrated range
        track.append(_s(t, u, v, w=60, h=40))
    # constant world speed = 15 m/s here; check several windows at very different image rows (scale changes ~10x between them)
    out = []
    for end in (30, 50, 70, 89):
        est = LiveSpeed(prof, ema=1.0, window_s=0.8, min_span_s=0.4, min_samples=8).estimate("a", track[: end + 1])
        assert est["status"] == "ok"
        out.append(est["speed_kmh"])
    assert max(out) - min(out) < 0.02 * 54.0 and out[0] == pytest.approx(54.0, rel=0.01)  # 15 m/s = 54 km/h, flat across the picture


def test_samples_outside_the_calibrated_zone_do_not_influence_the_speed():
    # a vehicle at 40 m/s outside the zone (x < 600 px), then 20 m/s inside it (x >= 600): the zone speed is what must be reported
    pts, x, t = [], 100.0, 0.0
    while x < 1000.0:
        pts.append(_s(t, x, 400.0))
        x += (400.0 if x < 600.0 else 200.0) * 0.04  # px per frame at 40 m/s then 20 m/s (0.1 m/px)
        t += 0.04
    zone = [[600, 0], [1000, 0], [1000, 800], [600, 800]]
    t_in = next(s.t for s in pts if s.ground[0] >= 600)
    pts = [s for s in pts if s.t <= t_in + 0.5]  # evaluate half a second after entering the zone, while the window still reaches outside it
    inside = LiveSpeed(_affine_profile(), zone=zone, ema=1.0, min_samples=8, min_span_s=0.4).estimate("a", pts)
    assert inside["status"] == "ok" and inside["speed_kmh"] == pytest.approx(72.0, abs=1.0)
    blended = LiveSpeed(_affine_profile(), zone=zone, ema=1.0, min_samples=8, min_span_s=0.4, in_zone_only=False, max_gap_s=1.0, max_jump_m=50.0).estimate("a", pts)
    assert blended["speed_kmh"] > 80.0  # the old behaviour: far-field samples inflate the number


def test_invalid_calibration_is_refused_and_unvalidated_geometry_gives_no_number():
    collinear = [CalibrationPoint((i * 10, i * 10), (i, i)) for i in range(6)]
    with pytest.raises(ValueError):
        fit_planar_homography("p", "cam", collinear)
    unvalidated = fit_planar_homography("p", "cam", [CalibrationPoint((x, y), (x * 0.1, y * 0.1)) for x, y in ((0, 0), (1000, 0), (1000, 800), (0, 800))])
    est = LiveSpeed(unvalidated).estimate("a", _line(20.0, _times()))
    assert est["status"] == "no_metric_calibration" and est["speed_kmh"] is None


# ---------------------------------------------------------------- smoothing and unreliable states
def test_no_smoothing_by_default_in_the_app_and_a_failed_reading_is_not_blended_into_the_next_one():
    ls = LiveSpeed(_affine_profile(), ema=1.0)
    assert ls.estimate("a", _line(10.0, _times()))["speed_kmh"] == pytest.approx(36.0, abs=0.3)
    assert ls.estimate("a", _line(20.0, _times()))["speed_kmh"] == pytest.approx(72.0, abs=0.3)  # genuine change is not hidden
    ls2 = LiveSpeed(_affine_profile(), ema=0.5)
    ls2.estimate("a", _line(10.0, _times()))
    ls2.estimate("a", _line(20.0, _times()[:3]))  # too short -> failure -> state forgotten
    assert "a" not in ls2._smooth


def test_an_imprecise_estimate_is_withheld_instead_of_shown():
    rnd = random.Random(4)
    track = [_s(t, 100 + 200.0 * t + rnd.gauss(0, 4.0), 400 + rnd.gauss(0, 4.0)) for t in _times(0.6)]
    est = LiveSpeed(_affine_profile(), ema=1.0, min_samples=8, min_span_s=0.4, max_error_kmh=3.0).estimate("a", track)
    assert est["status"] == "insufficient_precision" and est["speed_kmh"] is None


def test_a_clipped_newest_box_shows_no_number_even_if_older_samples_were_fine():
    track = _line(20.0, _times())
    track[-1] = S(track[-1].t, track[-1].ground, (0, 0, 10, 400))  # newest box touches the frame bottom (400 = frame height)
    est = LiveSpeed(_affine_profile(), ema=1.0, frame_size=(5000, 400)).estimate("a", track)
    assert est["status"] == "clipped_at_frame_edge" and est["speed_kmh"] is None


def test_the_officer_speed_tile_shows_the_median_and_the_range_not_the_fastest(services):
    from .test_officer_layer import _lanes, _run
    o = _run(services, "show_accident", {"lanes": _lanes()}).officer
    o._speed_now = [80.0, 100.0, 120.0, 140.0]
    tile = {t["key"]: t for t in o.tiles(0.0, {"active": False}, 112.7, True, True, [])}["speed"]
    assert tile["value"] == "110 km/h" and "median of 4" in tile["detail"] and "80-140" in tile["detail"] and "scale not surveyed" in tile["detail"]
