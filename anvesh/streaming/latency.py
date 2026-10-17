"""Bounded-memory latency recorder: exact count/mean/max plus p50/p95/p99 from
a fixed-size deterministic reservoir, so a 34-minute (51k-frame) soak records
latency without growing."""

from __future__ import annotations

import random
import threading


class _Stage:
    __slots__ = ("count", "total", "maximum", "reservoir", "_rng", "_cap")

    def __init__(self, capacity: int, seed: int) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.reservoir: list = []
        self._rng = random.Random(seed)
        self._cap = capacity

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value > self.maximum:
            self.maximum = value
        if len(self.reservoir) < self._cap:
            self.reservoir.append(value)
        else:  # Algorithm R: uniform sample of everything seen so far
            j = self._rng.randrange(self.count)
            if j < self._cap:
                self.reservoir[j] = value


def _percentile(sorted_values: list, q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


class LatencyRecorder:
    """Thread-safe per-stage latency recorder (values in SECONDS)."""

    def __init__(self, reservoir: int = 20000, seed: int = 0) -> None:
        self._capacity = reservoir
        self._seed = seed
        self._stages: dict = {}
        self._lock = threading.Lock()

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            st = self._stages.get(stage)
            if st is None:
                st = self._stages[stage] = _Stage(self._capacity, self._seed)
            st.add(seconds)

    def summary_ms(self) -> dict:
        """{stage: {count, mean_ms, p50_ms, p95_ms, p99_ms, max_ms}}.
        Percentiles are exact until `reservoir` samples, then sampled."""
        with self._lock:
            out = {}
            for name, st in self._stages.items():
                vals = sorted(st.reservoir)
                out[name] = {
                    "count": st.count,
                    "mean_ms": 1000.0 * st.total / st.count if st.count else 0.0,
                    "p50_ms": 1000.0 * _percentile(vals, 0.50),
                    "p95_ms": 1000.0 * _percentile(vals, 0.95),
                    "p99_ms": 1000.0 * _percentile(vals, 0.99),
                    "max_ms": 1000.0 * st.maximum,
                    "percentiles_exact": st.count <= st._cap,
                }
            return out
