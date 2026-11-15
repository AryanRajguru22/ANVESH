"""Per-track live speed for the dashboard overlay, derived from the track's image ground points through the camera's
VALIDATED calibration. It never returns a number without one:

  - no calibration, or one that does not support metric speed  -> status `no_metric_calibration`
  - ground point outside the configured `speed_zone`             -> `outside_speed_zone`
  - too few / too short samples                                  -> `insufficient_history`
  - jittery or physically implausible track                      -> `rejected_unstable` / `rejected_implausible`

The value is the least-squares slope of world position over the last `window_s` seconds (not a two-point difference),
with a ~95 % band derived from the fit residual. It is an estimate with a stated RANDOM error, not a speed-gun reading; with a
marking-standard calibration it is APPROXIMATE (see `geometry_label`) and the scale error is NOT in the band.

SAMPLES USED (speed audit, docs/ANVESH_SPEED_AUDIT_FINDINGS.md). Only samples that are (a) inside the `zone` (the region the
calibration actually covers), (b) not clipped by the frame edge, (c) unbroken in time (no gap longer than `max_gap_s`) and (d) not
across an impossible jump (a track-id swap) are fitted. Before this only the NEWEST sample was zone-tested: at motorway speed the
1.6 s window reached ~48 m back into the extrapolated far field and speeds were inflated by ~13 % (median) and more for the fastest
vehicles. Isolated outlier samples are removed (median-absolute-deviation rule) and the fit repeated. A result whose error band is
wider than `max_error_kmh` is not shown (`insufficient_precision`).
"""

from __future__ import annotations

import math

MPS_TO_KMH = 3.6


def _in_polygon(pt, poly) -> bool:
    x, y = pt
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


class LiveSpeed:
    EDGE_PX = 3  # a box touching the frame edge is clipped: its bottom-centre is NOT the ground contact point

    def __init__(self, profile, zone=None, window_s: float = 1.6, min_span_s: float = 0.8, min_samples: int = 6,
                 max_kmh: float = 250.0, max_residual_m: float = 2.5, ema: float = 0.5, frame_size=None,
                 in_zone_only: bool = True, max_gap_s: float = 0.3, max_jump_m: float = 2.0, max_error_kmh: float = None,
                 outlier_floor_m: float = 0.25, outlier_k: float = 3.0, max_outlier_fraction: float = 0.25) -> None:
        """`frame_size`: None, (w, h), or a zero-argument callable returning either (read lazily: the first frame sets it).
        `in_zone_only`: with a zone, fit only samples inside it (the calibrated region). `max_gap_s`: a longer time gap between
        consecutive samples ends the fitted run (dropped frames / lost track). `max_jump_m`: extra distance allowed between two
        consecutive samples beyond `max_kmh * dt` (larger = a different vehicle took over the id). `max_error_kmh`: None = no gate."""
        self.frame_size = frame_size
        self.profile, self.zone = profile, (tuple(zone) if zone else None)
        self.window_s, self.min_span_s, self.min_samples = window_s, min_span_s, min_samples
        self.max_kmh, self.max_residual_m, self.alpha = max_kmh, max_residual_m, ema
        self.in_zone_only, self.max_gap_s, self.max_jump_m, self.max_error_kmh = in_zone_only, max_gap_s, max_jump_m, max_error_kmh
        self.outlier_floor_m, self.outlier_k, self.max_outlier_fraction = outlier_floor_m, outlier_k, max_outlier_fraction
        self._smooth: dict = {}

    @property
    def metric(self) -> bool:
        return bool(self.profile is not None and getattr(self.profile, "supports_metric_speed", False))

    def _fail(self, track_id: str, status: str) -> dict:
        self._smooth.pop(track_id, None)  # an unreliable reading must not be blended into a later one
        return {"status": status, "speed_kmh": None}

    @staticmethod
    def _slope(pts: list):
        n = len(pts)
        tbar = sum(p[0] for p in pts) / n
        stt = sum((p[0] - tbar) ** 2 for p in pts)
        if stt <= 1e-9:
            return None
        vx = sum((p[0] - tbar) * p[1] for p in pts) / stt
        vy = sum((p[0] - tbar) * p[2] for p in pts) / stt
        xbar, ybar = sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n
        res = [math.hypot(p[1] - (xbar + vx * (p[0] - tbar)), p[2] - (ybar + vy * (p[0] - tbar))) for p in pts]
        return vx, vy, stt, res

    def estimate(self, track_id: str, samples: list) -> dict:
        """`samples`: the track's history (objects with .t, .ground and optionally .bbox), oldest first."""
        if not self.metric:
            return {"status": "no_metric_calibration", "speed_kmh": None}
        if not samples:
            return {"status": "insufficient_history", "speed_kmh": None}
        newest = samples[-1]
        if self._is_clipped(newest):  # a clipped newest box has no valid ground point: show nothing rather than a stale number
            return self._fail(track_id, "clipped_at_frame_edge")
        if self.zone is not None and not _in_polygon(newest.ground, self.zone):
            return self._fail(track_id, "outside_speed_zone")
        zone_only = self.in_zone_only and self.zone is not None
        run = [newest]
        for s in reversed(samples[:-1]):
            if newest.t - s.t > self.window_s or run[-1].t - s.t > self.max_gap_s:
                break
            if self._is_clipped(s) or (zone_only and not _in_polygon(s.ground, self.zone)):
                break
            run.append(s)
        run.reverse()
        try:
            pts = [(s.t, *self.profile.image_to_world(s.ground)) for s in run]
        except (ValueError, RuntimeError):
            return self._fail(track_id, "no_metric_calibration")
        # cut the run at the newest PERSISTENT impossible jump (a different vehicle took over the id). A single sample that jumps away and
        # comes straight back is a box glitch, not a swap: it is left for the outlier step below.
        def _far(i, j):
            dt = abs(pts[j][0] - pts[i][0])
            return math.hypot(pts[j][1] - pts[i][1], pts[j][2] - pts[i][2]) > self.max_kmh / MPS_TO_KMH * dt + self.max_jump_m

        jumped = False
        for k in range(len(pts) - 1, 0, -1):
            if _far(k - 1, k) and not (k + 1 < len(pts) and not _far(k - 1, k + 1)):
                pts, jumped = pts[k:], True
                break
        if len(pts) < self.min_samples:
            return self._fail(track_id, "rejected_unstable" if jumped else "insufficient_history")
        fit = self._slope(pts)
        if fit is None:
            return self._fail(track_id, "insufficient_history")
        res = fit[3]
        # robust step: remove isolated outliers (> max(k * scaled MAD, floor)) and refit; too many outliers = an unstable track
        s_mad = 1.4826 * sorted(res)[len(res) // 2]
        cut = max(self.outlier_k * s_mad, self.outlier_floor_m)
        keep = [p for p, r in zip(pts, res) if r <= cut]
        dropped = len(pts) - len(keep)
        if dropped:
            if dropped > self.max_outlier_fraction * len(pts) or len(keep) < self.min_samples:
                return self._fail(track_id, "rejected_unstable")
            pts = keep
            fit = self._slope(pts)
            if fit is None:
                return self._fail(track_id, "insufficient_history")
        if pts[-1][0] - pts[0][0] < self.min_span_s:
            return self._fail(track_id, "insufficient_history")
        vx, vy, stt, res = fit
        n = len(pts)
        rms = math.sqrt(sum(r * r for r in res) / n)
        if rms > self.max_residual_m:
            return self._fail(track_id, "rejected_unstable")
        v = math.hypot(vx, vy) * MPS_TO_KMH
        if v > self.max_kmh:
            return self._fail(track_id, "rejected_implausible")
        err = 2.0 * (rms / math.sqrt(stt)) * MPS_TO_KMH + 1.0  # ~95 % band of the slope + 1 km/h floor (RANDOM error only)
        if self.max_error_kmh is not None and err > self.max_error_kmh:
            return self._fail(track_id, "insufficient_precision")
        prev = self._smooth.get(track_id)
        v_s = v if prev is None else self.alpha * v + (1 - self.alpha) * prev
        self._smooth[track_id] = v_s
        return {"status": "ok", "speed_kmh": round(v_s, 1), "error_kmh": round(err, 1), "n": n, "span_s": round(pts[-1][0] - pts[0][0], 2),
                "outliers_removed": dropped, "approximate": bool(getattr(self.profile, "is_marking_standard_validated", False))}

    def _is_clipped(self, s) -> bool:
        """A box touching the frame edge is clipped (a lorry leaving the frame looks like it is braking if its clipped bottom edge
        is read as the ground point)."""
        fs = self.frame_size() if callable(self.frame_size) else self.frame_size
        b = getattr(s, "bbox", None)
        if not fs or b is None:
            return False
        w, h = fs
        return b[0] <= self.EDGE_PX or b[1] <= 0 or b[2] >= w - self.EDGE_PX or b[3] >= h - self.EDGE_PX

    def forget_absent(self, live_ids) -> None:
        live = set(live_ids)
        for k in [k for k in self._smooth if k not in live]:
            self._smooth.pop(k, None)
