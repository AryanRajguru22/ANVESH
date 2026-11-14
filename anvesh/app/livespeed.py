"""Per-track live speed for the dashboard overlay, derived from the track's image ground points through the camera's
VALIDATED calibration. It never returns a number without one:

  - no calibration, or one that does not support metric speed  -> status `no_metric_calibration`
  - ground point outside the configured `speed_zone`             -> `outside_speed_zone`
  - too few / too short samples                                  -> `insufficient_history`
  - jittery or physically implausible track                      -> `rejected_unstable` / `rejected_implausible`

The value is the least-squares slope of world position over the last `window_s` seconds (not a two-point difference),
with a 1-sigma standard error derived from the fit residual, then smoothed per track (EMA). It is an estimate with a
stated error, not a speed-gun reading; with a marking-standard calibration it is APPROXIMATE (see `geometry_label`).
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
                 max_kmh: float = 250.0, max_residual_m: float = 2.5, ema: float = 0.5, frame_size=None) -> None:
        """`frame_size`: None, (w, h), or a zero-argument callable returning either (read lazily: the first frame sets it)."""
        self.frame_size = frame_size
        self.profile, self.zone = profile, (tuple(zone) if zone else None)
        self.window_s, self.min_span_s, self.min_samples = window_s, min_span_s, min_samples
        self.max_kmh, self.max_residual_m, self.alpha = max_kmh, max_residual_m, ema
        self._smooth: dict = {}

    @property
    def metric(self) -> bool:
        return bool(self.profile is not None and getattr(self.profile, "supports_metric_speed", False))

    def estimate(self, track_id: str, samples: list) -> dict:
        """`samples`: the track's history (objects with .t and .ground), oldest first."""
        if not self.metric:
            return {"status": "no_metric_calibration", "speed_kmh": None}
        if not samples:
            return {"status": "insufficient_history", "speed_kmh": None}
        samples = self._unclipped(samples)
        if not samples:
            return {"status": "clipped_at_frame_edge", "speed_kmh": None}
        last = samples[-1]
        if self.zone is not None and not _in_polygon(last.ground, self.zone):
            self._smooth.pop(track_id, None)
            return {"status": "outside_speed_zone", "speed_kmh": None}
        win = [s for s in samples if last.t - s.t <= self.window_s]
        if len(win) < self.min_samples or win[-1].t - win[0].t < self.min_span_s:
            return {"status": "insufficient_history", "speed_kmh": None}
        try:
            pts = [(s.t, *self.profile.image_to_world(s.ground)) for s in win]
        except (ValueError, RuntimeError):
            return {"status": "no_metric_calibration", "speed_kmh": None}
        n = len(pts)
        tbar = sum(p[0] for p in pts) / n
        stt = sum((p[0] - tbar) ** 2 for p in pts)
        if stt <= 1e-9:
            return {"status": "insufficient_history", "speed_kmh": None}
        vx = sum((p[0] - tbar) * p[1] for p in pts) / stt
        vy = sum((p[0] - tbar) * p[2] for p in pts) / stt
        xbar, ybar = sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n
        res = [math.hypot(p[1] - (xbar + vx * (p[0] - tbar)), p[2] - (ybar + vy * (p[0] - tbar))) for p in pts]
        rms = math.sqrt(sum(r * r for r in res) / n)
        if rms > self.max_residual_m:
            return {"status": "rejected_unstable", "speed_kmh": None}
        v = math.hypot(vx, vy) * MPS_TO_KMH
        if v > self.max_kmh:
            return {"status": "rejected_implausible", "speed_kmh": None}
        err = 2.0 * (rms / math.sqrt(stt)) * MPS_TO_KMH + 1.0  # ~95 % band of the slope + 1 km/h floor
        prev = self._smooth.get(track_id)
        v_s = v if prev is None else self.alpha * v + (1 - self.alpha) * prev
        self._smooth[track_id] = v_s
        return {"status": "ok", "speed_kmh": round(v_s, 1), "error_kmh": round(err, 1), "n": n,
                "approximate": bool(getattr(self.profile, "is_marking_standard_validated", False))}

    def _unclipped(self, samples: list) -> list:
        """Drop samples whose box touches the frame edge (a lorry leaving the frame looks like it is braking if its clipped
        bottom edge is read as the ground point)."""
        fs = self.frame_size() if callable(self.frame_size) else self.frame_size
        if not fs:
            return samples
        w, h = fs
        out = []
        for s in samples:
            b = getattr(s, "bbox", None)
            if b is not None and (b[0] <= self.EDGE_PX or b[1] <= 0 or b[2] >= w - self.EDGE_PX or b[3] >= h - self.EDGE_PX):
                continue
            out.append(s)
        return out

    def forget_absent(self, live_ids) -> None:
        live = set(live_ids)
        for k in [k for k in self._smooth if k not in live]:
            self._smooth.pop(k, None)
