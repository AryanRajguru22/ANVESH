"""Hardening D: frame completeness gates hysteresis and scales reliability (neutral at 1.0)."""

import math

import pytest

from anvesh.evidence.reliability import compute_reliability
from anvesh.perception.traffic_state import CongestionStateTracker
from anvesh.pipeline import CameraWindowInput, run_corridor_window
from anvesh.storage.schemas import (
    CameraRole,
    CongestionLevel,
    GeometryValidity,
    Measurement,
    MotionSpace,
    OcclusionState,
    VehicleClass,
    VehicleTrack,
)
from anvesh.streaming.engine import StreamingPerceptionEngine
from anvesh.streaming.reasoning import AnalysisStatus, WindowReasoner
from anvesh.streaming.windows import (
    FRAME_ALIGNMENT_TOLERANCE_FRAMES,
    WindowAvailability,
    WindowSnapshot,
    WindowSpec,
    frame_completeness,
)
from tests.streaming.support import ScriptedTracker, make_frames
from tests.test_ma1_missing_speed_and_geometry import CAM, WINDOW, _window_inputs, _world_track

FPS = 10.0


# ------------------------------------------------------------------ the pure function / tolerance ----


def test_the_alignment_tolerance_is_exactly_one_frame():
    assert FRAME_ALIGNMENT_TOLERANCE_FRAMES == 1


@pytest.mark.parametrize("observed, expected, want", [
    (100, 100, 1.0),
    (99, 100, 1.0),            # boundary: ONE frame short is alignment, not loss
    (98, 100, 0.99),           # boundary: two short is real (and proportionate, not a cliff)
    (21, 100, 0.22),
    (0, 100, 0.01),
    (150, 100, 1.0),           # more than expected never exceeds 1
    (0, 1, 1.0),               # a one-frame window cannot be called incomplete by alignment
    (0, 2, 0.5),
    (5, None, 1.0), (5, 0, 1.0), (5, -3, 1.0),  # unknown expectation: nothing to compare against
])
def test_frame_completeness_boundaries(observed, expected, want):
    assert frame_completeness(observed, expected) == pytest.approx(want)


# ------------------------------------------------------------------ engine integration ----------------


def _snapshots(frames, fps=FPS, window_s=10.0):
    eng = StreamingPerceptionEngine("camC", ScriptedTracker(lambda i: []), fps=fps, windows=WindowSpec(window_s=window_s))
    out = []
    for f in frames:
        out += eng.process_frame(f)
    return out + eng.finish()


def test_a_complete_stream_is_fully_complete_at_integer_and_fractional_fps():
    for fps in (10.0, 25.0, 29.97):
        snaps = _snapshots(make_frames(int(fps * 60), fps=fps), fps=fps)
        full = [s for s in snaps if not s.final_partial]
        assert len(full) >= 5
        assert all(s.frame_completeness == 1.0 for s in full), (fps, [s.frame_completeness for s in full])
        assert all(s.observation_completeness == 1.0 and s.availability == WindowAvailability.OBSERVED for s in full)


def test_uniform_frame_loss_is_invisible_to_coverage_but_not_to_completeness():
    kept = [f for f in make_frames(300, fps=FPS) if f.index % 5 == 0]  # 80% lost, never a hole > 0.5 s
    snaps = [s for s in _snapshots(kept) if not s.final_partial]
    assert snaps
    for s in snaps:
        assert s.coverage == 1.0  # the old signal: blind to this loss
        assert s.frame_completeness == pytest.approx((20 + FRAME_ALIGNMENT_TOLERANCE_FRAMES) / 100)
        assert s.observation_completeness == s.frame_completeness
        assert s.availability == WindowAvailability.DEGRADED


def test_windows_cut_post_hoc_default_to_complete():
    s = WindowSnapshot(window_index=0, window_id="c:w0", camera_id="c", start=0.0, end=10.0, tracks=(),
                       frames_observed=0, dropped_frames=0, sessions=(), closed_at_stream_time=10.0)
    assert s.frame_completeness == 1.0 and s.observation_completeness == 1.0


# ------------------------------------------------------------------ reasoner: hysteresis gate ---------


def _heavy_image_tracks(n=20):
    return tuple(
        VehicleTrack(track_id=f"{CAM}:{i}", camera_id=CAM, first_seen=0.0, last_seen=10.0, vehicle_class=VehicleClass.CAR,
                     occlusion_state=OcclusionState.VISIBLE, position_history=[], speed_estimate=Measurement(1.0, 0.1),
                     motion_space=MotionSpace.IMAGE, geometry_validity=GeometryValidity.IMAGE)
        for i in range(n)
    )


def _snap(completeness, coverage=1.0, dropped=0, k=0):
    return WindowSnapshot(window_index=k, window_id=f"{CAM}:w{k:06d}", camera_id=CAM, start=10.0 * k, end=10.0 * k + 10.0,
                          tracks=_heavy_image_tracks(), frames_observed=100, dropped_frames=dropped, sessions=(),
                          closed_at_stream_time=10.0 * k + 10.0, coverage=coverage, frame_completeness=completeness)


def test_hysteresis_advances_on_a_complete_window():
    r = WindowReasoner(CAM)
    a = r.process(_snap(1.0))
    assert a.status == AnalysisStatus.OBSERVED and a.congestion_advanced and r.congestion_state != CongestionLevel.FREE_FLOW


def test_uniform_frame_loss_no_longer_advances_hysteresis():
    r = WindowReasoner(CAM)
    before = r.congestion_state
    a = r.process(_snap(0.21, coverage=1.0, dropped=80))  # exactly the repro: coverage 1.0, 80% lost
    assert a.status == AnalysisStatus.DEGRADED and not a.congestion_advanced
    assert r.congestion_state == before
    assert any("NOT advanced" in n for n in a.result.limitations)


@pytest.mark.parametrize("completeness, advances", [
    (0.90, True),    # boundary: exactly the existing min_update_coverage still advances
    (0.8999, False),  # boundary: just under does not
    (0.95, True),
    (0.5, False),
])
def test_the_advance_threshold_is_the_existing_min_update_coverage(completeness, advances):
    r = WindowReasoner(CAM)  # default min_update_coverage = 0.9
    before = r.congestion_state
    a = r.process(_snap(completeness, dropped=5))
    assert a.congestion_advanced is advances
    assert (r.congestion_state != before) is advances


def test_the_worse_of_coverage_and_frame_completeness_decides():
    assert _snap(0.95, coverage=0.7).observation_completeness == pytest.approx(0.7)
    r = WindowReasoner(CAM)
    assert not r.process(_snap(0.95, coverage=0.7, dropped=3)).congestion_advanced


def test_a_materially_incomplete_window_is_flagged_even_if_no_loss_was_reported():
    r = WindowReasoner(CAM)
    a = r.process(_snap(0.6, coverage=1.0, dropped=0))  # e.g. a declared fps the source does not deliver
    assert a.degraded and a.status == AnalysisStatus.DEGRADED and not a.congestion_advanced


def test_a_slightly_incomplete_but_unreported_window_is_not_flagged():
    r = WindowReasoner(CAM)
    a = r.process(_snap(0.95, coverage=1.0, dropped=0))  # small fps drift: absorbed by the 0.9 threshold
    assert a.status == AnalysisStatus.OBSERVED and a.congestion_advanced


# ------------------------------------------------------------------ reliability multiplier -----------


def _agg(tracks=None):
    tracks = tracks if tracks is not None else [_world_track(i) for i in range(4)]
    return tracks, _window_inputs(tracks)


def test_reliability_is_exactly_neutral_at_completeness_one():
    tracks, agg = _agg()
    base = compute_reliability(CAM, *WINDOW, tracks, agg)
    same = compute_reliability(CAM, *WINDOW, tracks, agg, observation_completeness=1.0)
    assert same == base and "observation_completeness" not in same.factors
    assert set(base.factors) == {"observation_count", "tracking_continuity", "calibration", "temporal_coverage"}


@pytest.mark.parametrize("c", [0.0, 0.2, 0.5, 0.999])
def test_reliability_scales_linearly_below_one_and_records_the_factor(c):
    tracks, agg = _agg()
    base = compute_reliability(CAM, *WINDOW, tracks, agg)
    r = compute_reliability(CAM, *WINDOW, tracks, agg, observation_completeness=c)
    assert r.score == pytest.approx(base.score * c)
    assert r.factors["observation_completeness"] == c
    assert {k: v for k, v in r.factors.items() if k != "observation_completeness"} == base.factors


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_reliability_rejects_an_out_of_range_completeness(bad):
    tracks, agg = _agg()
    with pytest.raises(ValueError, match="observation_completeness"):
        compute_reliability(CAM, *WINDOW, tracks, agg, observation_completeness=bad)


def test_camera_window_input_validates_and_defaults_to_one():
    tracks, agg = _agg()
    assert CameraWindowInput(camera_id=CAM, role=CameraRole.UPSTREAM, aggregation=agg, tracks=tracks).observation_completeness == 1.0
    with pytest.raises(ValueError):
        CameraWindowInput(camera_id=CAM, role=CameraRole.UPSTREAM, aggregation=agg, tracks=tracks, observation_completeness=1.5)


def test_loss_discounts_the_ranked_belief_through_reliability():
    tracks = [_world_track(i, speed=6.0) for i in range(18)]
    agg = _window_inputs(tracks)

    def run(c):
        ci = CameraWindowInput(camera_id=CAM, role=CameraRole.UPSTREAM, aggregation=agg, tracks=tracks, observation_completeness=c)
        return run_corridor_window(WINDOW, (ci,))

    full, lossy = run(1.0), run(0.5)
    assert lossy.reliability_by_camera[CAM].score == pytest.approx(full.reliability_by_camera[CAM].score * 0.5)
    b_full = full.candidate_ranking.ranked_list[0].belief
    b_lossy = lossy.candidate_ranking.ranked_list[0].belief
    assert b_lossy == pytest.approx(b_full * 0.5) and b_lossy < b_full


def test_the_reasoner_passes_completeness_into_reliability():
    full = WindowReasoner(CAM).process(_snap(1.0))
    lossy = WindowReasoner(CAM).process(_snap(0.5, dropped=50))
    assert lossy.result.reliability_by_camera[CAM].score == pytest.approx(full.result.reliability_by_camera[CAM].score * 0.5)
    assert "observation_completeness" not in full.result.reliability_by_camera[CAM].factors
