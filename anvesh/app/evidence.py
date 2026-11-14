"""Evidence capture and storage.

FRAME RING. Stop episodes (and every other window-level event) surface only when a window
CLOSES, i.e. up to a window length after the event began. To save frames from the event
itself the session keeps a bounded ring of recent frames, JPEG-encoded (a raw 1080p frame
is 6 MB; a JPEG is ~100 KB) and throttled to `max_fps`. The ring is sized by SECONDS, so
memory is bounded by construction, and it must cover window + dwell + margin (the session
computes that).

EVIDENCE ASSETS are immutable: written once under `<evidence_dir>/<incident_id>/`, hashed
(SHA-256), recorded in `evidence_assets` (which rejects UPDATE/DELETE at the SQL level), and
re-verifiable. Raw key frames (kind `frame`) are the evidentiary originals; annotated frames
and the review clip are review AIDS rendered from them. Nothing is ever overwritten.
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import hashlib
import os
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.timeutil import now_iso

# Preference order. H.264 and VP9 play natively in Chrome/Edge/Firefox; MPEG-4 part 2 ("mp4v") generally
# does not, so it is a last resort and is flagged in the asset metadata.
_CODECS = (("avc1", ".mp4", "video/mp4", True), ("VP80", ".webm", "video/webm", True),
           ("VP90", ".webm", "video/webm", True), ("mp4v", ".mp4", "video/mp4", False))
# Measured on this build (150 frames, 640x360): avc1 ~1 s, VP80 ~9 s, VP90 ~77 s, mp4v 0.5 s. VP9 is so slow it stalled
# the window consumer, hence H.264 first; the next codecs are only reached where H.264 encoding is unavailable.


class FrameRing:
    def __init__(self, max_seconds: float = 30.0, max_fps: float = 10.0, jpeg_quality: int = 85) -> None:
        if max_seconds <= 0 or max_fps <= 0:
            raise ValueError("max_seconds and max_fps must be > 0")
        self._max_seconds, self._min_dt, self._q = max_seconds, 1.0 / max_fps, int(jpeg_quality)
        self._items: collections.deque = collections.deque()  # (timestamp, frame_index, jpeg_bytes)
        self._lock = threading.Lock()
        self._last_ts = None
        self.dropped_by_throttle = 0

    def add(self, frame_index: int, timestamp: float, image) -> bool:
        if self._last_ts is not None and timestamp - self._last_ts < self._min_dt - 1e-9:
            self.dropped_by_throttle += 1
            return False
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._q])
        if not ok:
            return False
        with self._lock:
            self._items.append((timestamp, frame_index, buf.tobytes()))
            self._last_ts = timestamp
            while self._items and timestamp - self._items[0][0] > self._max_seconds:
                self._items.popleft()
        return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def size_bytes(self) -> int:
        with self._lock:
            return sum(len(i[2]) for i in self._items)

    def span(self):
        with self._lock:
            return (self._items[0][0], self._items[-1][0]) if self._items else None

    def between(self, t0: float, t1: float) -> list:
        with self._lock:
            return [i for i in self._items if t0 - 1e-9 <= i[0] <= t1 + 1e-9]

    def nearest(self, t: float):
        with self._lock:
            if not self._items:
                return None
            times = [i[0] for i in self._items]
            k = bisect.bisect_left(times, t)
            cands = [self._items[j] for j in (k - 1, k) if 0 <= j < len(self._items)]
            return min(cands, key=lambda i: abs(i[0] - t))


def decode_jpeg(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def annotate_frame(image, boxes: list, highlight: set = (), banner: str = None):
    """Draw boxes on a COPY. `boxes` = [(track_id, label, (x1,y1,x2,y2))]; ids in `highlight` get a
    thick red box, the rest a thin neutral one."""
    out = image.copy()
    for track_id, label, (x1, y1, x2, y2) in boxes:
        hot = track_id in highlight
        colour = (40, 40, 255) if hot else (200, 180, 60)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3 if hot else 1)
        if hot or label:
            text = label or track_id
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(int(y1) - 4, th + 2)
            cv2.rectangle(out, (int(x1), ty - th - 3), (int(x1) + tw + 4, ty + 3), colour, -1)
            cv2.putText(out, text, (int(x1) + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(out, banner, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_clip(path_stem: Path, frames: list, fps: float) -> tuple:
    """Encode BGR `frames` to a browser-playable file next to `path_stem` (suffix chosen by codec).
    Returns (path, mime, codec, browser_playable). Reads the file back to confirm it is decodable."""
    if not frames:
        raise ValueError("no frames to encode")
    h, w = frames[0].shape[:2]
    last_error = None
    for fourcc, ext, mime, playable in _CODECS:
        path = path_stem.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        if not writer.isOpened():
            writer.release()
            continue
        for f in frames:
            writer.write(f)
        writer.release()
        cap = cv2.VideoCapture(str(path))
        readable = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= max(1, len(frames) - 1)
        cap.release()
        if readable:
            return path, mime, fourcc, playable
        last_error = f"{fourcc}: written file not decodable"
        with contextlib.suppress(OSError):
            path.unlink()
    raise RuntimeError(f"no video codec could write a readable clip ({last_error or 'no writer opened'})")


class EvidenceStore:
    def __init__(self, db: AppDatabase, audit: AuditLog, evidence_dir) -> None:
        self._db, self._audit = db, audit
        self.root = Path(evidence_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _new_path(self, incident_id: str, asset_id: str, ext: str) -> Path:
        d = self.root / incident_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{asset_id}{ext}"

    def _register(self, incident_id, kind, path: Path, mime, offset_s, frame_index, track_id, meta, actor) -> dict:
        data_hash = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                data_hash.update(chunk)
        asset_id = path.stem
        rel = path.relative_to(self.root).as_posix()
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO evidence_assets(asset_id, incident_id, kind, relpath, sha256, size_bytes, mime, offset_s, frame_index, "
                "track_id, meta_json, created_at, created_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (asset_id, incident_id, kind, rel, data_hash.hexdigest(), path.stat().st_size, mime, offset_s, frame_index,
                 track_id, dumps(meta or {}), now_iso(), actor),
            )
        self._audit.append(actor, "evidence.add", "incident", incident_id,
                           {"asset_id": asset_id, "kind": kind, "sha256": data_hash.hexdigest()})
        return self.get(asset_id)

    def add_bytes(self, incident_id: str, kind: str, data: bytes, ext: str, mime: str, offset_s=None, frame_index=None,
                  track_id=None, meta=None, actor: str = "system") -> dict:
        asset_id = "AST-" + uuid.uuid4().hex[:12]
        path = self._new_path(incident_id, asset_id, ext)
        with path.open("xb") as fh:  # exclusive create: never overwrites
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        return self._register(incident_id, kind, path, mime, offset_s, frame_index, track_id, meta, actor)

    def add_clip(self, incident_id: str, frames: list, fps: float, offset_s=None, meta=None, actor: str = "system") -> dict:
        asset_id = "AST-" + uuid.uuid4().hex[:12]
        stem = self._new_path(incident_id, asset_id, "")
        path, mime, codec, playable = write_clip(stem, frames, fps)
        meta = dict(meta or {}, codec=codec, browser_playable=playable, frames=len(frames), fps=fps)
        return self._register(incident_id, "clip", path, mime, offset_s, None, None, meta, actor)

    def get(self, asset_id: str):
        row = self._db.one("SELECT * FROM evidence_assets WHERE asset_id = ?", (asset_id,))
        if row:
            row["meta"] = loads(row.pop("meta_json"), {})
        return row

    def list(self, incident_id: str) -> list:
        rows = self._db.query("SELECT * FROM evidence_assets WHERE incident_id = ? ORDER BY created_at, asset_id", (incident_id,))
        purged = {p["asset_id"]: p["purged_at"] for p in
                  self._db.query("SELECT asset_id, purged_at FROM evidence_purges WHERE incident_id = ?", (incident_id,))}
        for r in rows:
            r["meta"] = loads(r.pop("meta_json"), {})
            r["purged_at"] = purged.get(r["asset_id"])
        return rows

    def purge_record(self, asset_id: str):
        """The retention tombstone for an asset whose file was removed under the retention policy, or None."""
        return self._db.one("SELECT * FROM evidence_purges WHERE asset_id = ?", (asset_id,))

    def path_of(self, asset: dict) -> Path:
        """Absolute path of an asset, refusing anything that resolves outside the evidence root."""
        p = (self.root / asset["relpath"]).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            raise PermissionError("asset path escapes the evidence root") from None
        return p

    def verify(self, asset_id: str) -> dict:
        asset = self.get(asset_id)
        if asset is None:
            raise LookupError(asset_id)
        purge = self.purge_record(asset_id)
        if purge is not None:
            return {"asset_id": asset_id, "ok": False, "purged": True, "reason": f"purged under the retention policy at {purge['purged_at']}",
                    "registered_sha256": asset["sha256"]}
        path = self.path_of(asset)
        if not path.is_file():
            return {"asset_id": asset_id, "ok": False, "reason": "file missing"}
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ok = h.hexdigest() == asset["sha256"]
        return {"asset_id": asset_id, "ok": ok, "reason": None if ok else "sha256 mismatch"}

    def read_verified(self, asset: dict) -> tuple:
        """(bytes or None, state): state is 'ok', 'purged', 'missing' or 'mismatch'. Bytes are returned only when they match the
        registered SHA-256, so an export can never carry a file that differs from what was recorded."""
        if self.purge_record(asset["asset_id"]) is not None:
            return None, "purged"
        path = self.path_of(asset)
        if not path.is_file():
            return None, "missing"
        data = path.read_bytes()
        return (data, "ok") if hashlib.sha256(data).hexdigest() == asset["sha256"] else (None, "mismatch")
