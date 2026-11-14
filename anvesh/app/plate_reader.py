"""Live number-plate reading for the officer screen: plate DETECTOR (YOLOv8n fine-tuned on the plates dataset) + OCR (RapidOCR).

HONEST BY CONSTRUCTION
  * A plate is only attempted on a vehicle whose box is wide enough for a plate to hold legible characters
    (`min_vehicle_px`, default 110 px). On far-field footage (a 640x360 motorway camera, vehicles ~30 px wide, plates ~8 px) the reader
    never fires and the screen says "vehicles too small to read a plate" - it does not guess.
  * A result is accepted only when the detector is confident, the OCR confidence is high and the text looks like a plate
    (4-10 letters/digits). Otherwise the vehicle stays "not readable". The best read per vehicle is kept; a second matching
    read raises it to `confirmed`.
  * It runs on its own thread, one crop at a time, throttled per vehicle, so it can never stall the video.
  * Plate text is a CANDIDATE read for a human to check - the existing manual plate verification workflow is unchanged.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "models" / "plates" / "plate_v1" / "weights" / "best.pt"
_PLATE_RE = re.compile(r"^[A-Z0-9]{4,10}$")


_SHARED: dict = {}  # model path -> (detector, ocr): loaded once per process (warm-up) and shared by every session
_SHARED_LOCK = threading.Lock()  # guards loading and inference: the models are not used from two threads at once


def normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


class PlateReader:
    def __init__(self, model_path=None, min_vehicle_px: int = 110, min_det_conf: float = 0.35, min_ocr_conf: float = 0.65,
                 retry_s: float = 1.5, max_tries: int = 8, device=None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL)
        self.min_vehicle_px = int(min_vehicle_px)
        self.min_det_conf = float(min_det_conf)
        self.min_ocr_conf = float(min_ocr_conf)
        self.retry_s = float(retry_s)
        self.max_tries = int(max_tries)
        self.device = device
        self.reason = None  # why the reader is unavailable, if it is
        self.results: dict = {}  # track_id -> {"text","conf","reads","confirmed","t"}
        self._tries: dict = {}  # track_id -> (count, last_wall)
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._det = None
        self._ocr = None
        self.attempts = 0
        self.skipped_small = 0
        self.max_vehicle_px_seen = 0
        self._check()

    # -- availability --------------------------------------------------------------------------------
    def _check(self) -> None:
        if not self.model_path.is_file():
            self.reason = f"plate detector weights not found ({self.model_path.name})"
            return
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except Exception:  # noqa: BLE001
            self.reason = "OCR engine (rapidocr-onnxruntime) is not installed"

    @property
    def available(self) -> bool:
        return self.reason is None

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="anvesh-plates", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- producer (frame thread; never blocks) -------------------------------------------------------
    def offer(self, track_id: str, image, bbox, vehicle_class: str) -> None:
        w = bbox[2] - bbox[0]
        self.max_vehicle_px_seen = max(self.max_vehicle_px_seen, int(w))
        if not self.available:
            return
        if w < self.min_vehicle_px:
            self.skipped_small += 1
            return
        if vehicle_class == "bicycle":
            return
        now = time.time()
        with self._lock:
            done = self.results.get(track_id)
            if done and done.get("confirmed"):
                return
            n, last = self._tries.get(track_id, (0, 0.0))
            if n >= self.max_tries or now - last < self.retry_s:
                return
            self._tries[track_id] = (n + 1, now)
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, wi = image.shape[:2]
        pad = int(0.06 * (x2 - x1))
        crop = image[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(wi, x2 + pad)]
        if crop.size == 0:
            return
        try:
            self._q.put_nowait((track_id, crop.copy(), (max(0, x1 - pad), max(0, y1 - pad))))
        except queue.Full:
            with self._lock:  # busy: do not spend a try
                self._tries[track_id] = (n, last)

    # -- worker ------------------------------------------------------------------------------------
    def _load(self) -> bool:
        try:
            with _SHARED_LOCK:
                key = str(self.model_path)
                if key not in _SHARED:
                    from rapidocr_onnxruntime import RapidOCR
                    from ultralytics import YOLO

                    _SHARED[key] = (YOLO(key), RapidOCR())
                self._det, self._ocr = _SHARED[key]
            return True
        except Exception as exc:  # noqa: BLE001
            self.reason = f"plate reader failed to load: {type(exc).__name__}: {exc}"
            log.exception("plate reader load failed")
            return False

    def warm_up(self) -> None:
        """Load both models and run them once on a blank crop, so the first real plate is not slow."""
        import numpy as np

        if self.available and self._load():
            self.read_crop(np.full((240, 320, 3), 90, np.uint8))

    def _loop(self) -> None:
        if not self._load():
            return
        while not self._stop.is_set():
            try:
                tid, crop, origin = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.attempts += 1
                self._read(tid, crop, origin)
            except Exception:  # noqa: BLE001 - a bad crop must not kill the reader
                log.exception("plate read failed")

    def read_crop(self, crop):
        """Detect + OCR on one vehicle crop (also used by the tests / the benchmark). Returns (text, conf, det_conf, plate_box) or None."""
        import cv2

        if self._det is None and not self._load():
            return None
        h, w = crop.shape[:2]
        scale = max(1.0, 480.0 / max(h, w))
        big = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
        with _SHARED_LOCK:
            res = self._det.predict(big, imgsz=640, conf=self.min_det_conf, verbose=False, device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        det_conf = float(res.boxes.conf[i])
        x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i].tolist())
        px, py = int(0.08 * (x2 - x1)) + 2, int(0.15 * (y2 - y1)) + 2
        plate = big[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
        if plate.size == 0 or plate.shape[1] < 24:
            return None
        ph = plate.shape[0]
        if ph < 64:
            f = 64.0 / ph
            plate = cv2.resize(plate, (int(plate.shape[1] * f), 64), interpolation=cv2.INTER_CUBIC)
        with _SHARED_LOCK:
            out, _ = self._ocr(plate)
        if not out:
            return None
        out = sorted(out, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))  # top-to-bottom, left-to-right
        text = normalise("".join(r[1] for r in out))
        conf = sum(float(r[2]) for r in out) / len(out)
        pbox = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)
        return text, conf, det_conf, pbox

    def _read(self, tid: str, crop, origin=(0, 0)) -> None:
        got = self.read_crop(crop)
        if got is None:
            return
        text, conf, det_conf, pbox = got
        box = [round(pbox[0] + origin[0]), round(pbox[1] + origin[1]), round(pbox[2] + origin[0]), round(pbox[3] + origin[1])]
        if conf < self.min_ocr_conf or not _PLATE_RE.match(text):
            return
        with self._lock:
            prev = self.results.get(tid)
            if prev is None:
                self.results[tid] = {"text": text, "conf": round(conf, 2), "det_conf": round(det_conf, 2), "reads": 1, "confirmed": False,
                                     "t": time.time(), "box": box}
            else:
                if text == prev["text"]:
                    prev["reads"] += 1
                    prev["confirmed"] = prev["reads"] >= 2
                    prev["conf"] = max(prev["conf"], round(conf, 2))
                    prev["box"] = box
                elif conf > prev["conf"]:
                    self.results[tid] = {"text": text, "conf": round(conf, 2), "det_conf": round(det_conf, 2), "reads": 1, "confirmed": False,
                                         "t": time.time(), "box": box}

    # -- consumer ----------------------------------------------------------------------------------
    def get(self, track_id: str):
        with self._lock:
            r = self.results.get(track_id)
            return dict(r) if r else None

    def summary(self) -> dict:
        with self._lock:
            reads = [dict(v, track_id=k) for k, v in self.results.items()]
        reads.sort(key=lambda r: -r["t"])
        return {"available": self.available, "reason": self.reason, "plates_read": len(reads), "recent": reads[:6], "attempts": self.attempts,
                "min_vehicle_px": self.min_vehicle_px, "largest_vehicle_px_seen": self.max_vehicle_px_seen}
