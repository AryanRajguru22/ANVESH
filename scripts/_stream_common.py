"""Shared helpers for the real-data streaming scripts (parity, latency
benchmark, memory soak). Not a public API."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.perception.detection import COCO_CLASS_NAMES, COCO_TO_VEHICLE_CLASS  # noqa: E402
from anvesh.perception.types import RawDetection, TrackedDetection  # noqa: E402

OUT = REPO_ROOT / "outputs" / "m_a"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = str(REPO_ROOT / "models" / "yolov8n.pt")


class CachedTracker:
    """Feeds the REAL tracker output cached by `cityflow_cache_perception.py`
    as if it were live, so decode -> identity -> store -> windows can be
    compared across execution modes without detector nondeterminism."""

    lost_buffer_frames = 30

    def __init__(self, cache_path: Path) -> None:
        self._frames = {}
        with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if "meta" not in obj:
                    self._frames[obj["i"]] = obj
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def update(self, image, frame_index: int, timestamp: float):
        fr = self._frames.get(frame_index)
        raw, tracked = [], []
        if fr is None:
            return raw, tracked
        for b in fr["trk"]:
            cls = int(b[5])
            vc = COCO_TO_VEHICLE_CLASS.get(cls)
            if vc is None:
                continue
            d = RawDetection(
                frame_index=frame_index, timestamp=timestamp, class_name=COCO_CLASS_NAMES.get(cls, str(cls)),
                vehicle_class=vc, confidence=min(1.0, max(0.0, float(b[4]))),
                bbox_xyxy=tuple(float(v) for v in b[:4]),
            )
            raw.append(d)
            if b[6] is not None:
                tracked.append(TrackedDetection(detection=d, track_id=str(int(b[6]))))
        return raw, tracked


def observability_signature(snap) -> tuple:
    """M-A.1 window observability + stop-episode lifecycle (element 7 of the window signature):
    how much of the window was observed, and which stop episodes touched it in which phase."""
    return (
        snap.availability.value, round(snap.coverage, 9), round(snap.silent_s, 9), snap.expected_frames,
        tuple(
            (v.phase.value, v.episode.episode_id, v.episode.start, v.episode.confirmed_at)
            for v in (snap.stop_episodes or ())
        ),
    )


def window_signature(snap) -> tuple:
    """World-model content of one window, timing excluded."""
    return (
        snap.window_id, snap.start, snap.end, snap.frames_observed, snap.dropped_frames, snap.final_partial,
        tuple(
            (t.track_id, t.first_seen, t.last_seen, t.vehicle_class.value, t.occlusion_state.value,
             t.motion_space.value, t.geometry_validity.value, t.speed_available,
             round(t.speed_estimate.value, 9), len(t.position_history))
            for t in snap.tracks
        ),
        observability_signature(snap),
    )


def opened_episodes(sigs: list) -> list:
    """(episode_id, start, confirmed_at) of every episode OPENED in the given window signatures."""
    return [(e[1], e[2], e[3]) for s in sigs if len(s) > 7 for e in s[7][4] if e[0] == "opened"]


def diff_windows(a: list, b: list) -> dict:
    """Quantify differences between two window lists (signatures)."""
    out = {"windows_a": len(a), "windows_b": len(b), "identical": a == b, "differing_windows": 0,
           "tracks_only_in_a": 0, "tracks_only_in_b": 0, "tracks_with_field_diffs": 0,
           "max_abs_speed_diff": 0.0, "occlusion_state_only_diffs": 0}
    for wa, wb in zip(a, b):
        if wa == wb:
            continue
        out["differing_windows"] += 1
        ta = {t[0]: t for t in wa[6]}
        tb = {t[0]: t for t in wb[6]}
        out["tracks_only_in_a"] += len(ta.keys() - tb.keys())
        out["tracks_only_in_b"] += len(tb.keys() - ta.keys())
        for k in ta.keys() & tb.keys():
            if ta[k] != tb[k]:
                out["tracks_with_field_diffs"] += 1
                out["max_abs_speed_diff"] = max(out["max_abs_speed_diff"], abs(ta[k][8] - tb[k][8]))
                strip = lambda t: t[:4] + t[5:]  # noqa: E731  (drop occlusion_state)
                if strip(ta[k]) == strip(tb[k]):
                    out["occlusion_state_only_diffs"] += 1
    return out


class GpuPoller(threading.Thread):
    """Polls `nvidia-smi` for GPU utilisation / memory while a run is in
    progress. Utilisation is nvidia-smi's own "% of time a kernel was running
    over the last sample period", not SM occupancy."""

    def __init__(self, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: list = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                util, mem = (float(x) for x in out.split(","))
                self.samples.append((time.time(), util, mem))
            except Exception:  # noqa: BLE001 - telemetry must never break a run
                pass
            self._halt.wait(self.interval_s)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        if not self.samples:
            return {"available": False}
        utils = sorted(s[1] for s in self.samples)
        return {
            "available": True, "samples": len(utils),
            "util_mean_pct": sum(utils) / len(utils), "util_p50_pct": utils[len(utils) // 2],
            "util_max_pct": utils[-1], "mem_max_mib": max(s[2] for s in self.samples),
        }
