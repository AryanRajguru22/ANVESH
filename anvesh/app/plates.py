"""Plate readings: OCR integration boundary, normalisation, and the officer verification trail.

WHAT RUNS HERE, HONESTLY.
  - Format validation/normalisation of Indian registration marks: REAL, tested.
  - Plate LOCALISATION: needs a trained plate detector. None ships with ANVESH. `YoloPlateDetector`
    loads one if you place weights at `models/plate_detector.pt` (or set ANVESH_PLATE_MODEL); until
    then the detector reports `unavailable` and nothing is localised.
  - OCR: `EasyOcrEngine` runs if `easyocr` is installed AND its recognition weights are already on
    disk (it is created with downloads disabled so it can never fetch a model silently). Otherwise the
    engine reports `unavailable` with setup instructions. Nothing is faked.
  - Officer-entered plates (`manual_entry`) are always available and always carry that provenance.
The available footage (wide-angle, 640x360 / 1280x720 traffic scenes) does not resolve legible plates,
so the OCR path is not demonstrated on it; the dashboard says so.

INVARIANTS (tested; `plate_readings` rows are immutable at the SQL level):
  - the original reading is never overwritten: a verification or correction is a NEW row that
    `supersedes` the previous one, with its author and note;
  - an OCR reading is never a verified identity. Only `officer_verified` / `officer_corrected` count as
    verified, and OCR confidence is never converted into verification;
  - low-confidence or ill-formed readings are `needs_review`; an empty read is `unreadable`;
  - a verifier cannot be the same user who entered/read the plate when four-eyes is on.
"""

from __future__ import annotations

import dataclasses
import math
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from anvesh.app.audit import AuditLog
from anvesh.app.db import AppDatabase, dumps, loads
from anvesh.app.timeutil import now_iso

STATE_CODES = frozenset("AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UK UP WB".split())
_STANDARD = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$")
_BH = re.compile(r"^(\d{2})BH(\d{4})([A-Z]{1,2})$")
VERIFIED_STATUSES = ("officer_verified", "officer_corrected")
OCR_CANDIDATE_MIN_CONF = 0.85

_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def normalize(text: str) -> str:
    """Upper-case, alphanumerics only. The caller keeps the original string separately."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def validate_format(norm: str) -> dict:
    """{valid, kind, state_code, reason}. Structural validity only: a well-formed mark may not exist."""
    m = _STANDARD.match(norm)
    if m:
        if m.group(1) not in STATE_CODES:
            return {"valid": False, "kind": "standard", "state_code": m.group(1), "reason": f"unknown state/UT code {m.group(1)}"}
        return {"valid": True, "kind": "standard", "state_code": m.group(1), "reason": None}
    if _BH.match(norm):
        return {"valid": True, "kind": "bharat_series", "state_code": "BH", "reason": None}
    return {"valid": False, "kind": None, "state_code": None, "reason": "does not match a standard Indian registration format"}


def positional_alternatives(norm: str, limit: int = 4) -> list:
    """Format-repairing alternatives from common OCR confusions (0/O, 1/I, 5/S, 8/B ...), derived
    positionally from the standard layout. They are SUGGESTIONS flagged `derived`, never the reading."""
    if validate_format(norm)["valid"] or not 8 <= len(norm) <= 11:
        return []
    out = []
    for n_mid in (1, 2):
        series = len(norm) - 2 - n_mid - 4
        if not 0 <= series <= 3:
            continue
        layout = "L" * 2 + "D" * n_mid + "L" * series + "D" * 4
        chars = []
        for ch, kind in zip(norm, layout):
            if kind == "D" and not ch.isdigit():
                chars.append(_TO_DIGIT.get(ch, ch))
            elif kind == "L" and ch.isdigit():
                chars.append(_TO_LETTER.get(ch, ch))
            else:
                chars.append(ch)
        cand = "".join(chars)
        if validate_format(cand)["valid"] and cand != norm and cand not in out:
            out.append(cand)
    return out[:limit]


@dataclasses.dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float
    alternatives: tuple = ()
    engine: str = ""


class UnavailableOcr:
    available = False

    def __init__(self, reason: str, setup: str) -> None:
        self.reason, self.setup, self.name = reason, setup, "unavailable"

    def read(self, image):
        raise RuntimeError(f"OCR unavailable: {self.reason}")

    def describe(self) -> dict:
        return {"available": False, "engine": None, "reason": self.reason, "setup": self.setup}


class EasyOcrEngine:
    """EasyOCR with downloads DISABLED: it runs only on weights already present locally."""

    available = True
    name = "easyocr"

    def __init__(self, model_dir: str = None) -> None:
        import easyocr  # raises ImportError if not installed

        self._reader = easyocr.Reader(["en"], gpu=False, model_storage_directory=model_dir, download_enabled=False, verbose=False)

    def read(self, image) -> OcrResult:
        results = self._reader.readtext(image, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1, paragraph=False)
        if not results:
            return OcrResult("", 0.0, (), self.name)
        results = sorted(results, key=lambda r: r[0][0][0])
        text = "".join(r[1] for r in results)
        conf = float(min(r[2] for r in results))
        return OcrResult(text, conf, (), self.name)

    def describe(self) -> dict:
        return {"available": True, "engine": "easyocr (en, CPU, downloads disabled)", "reason": None, "setup": ""}


SETUP_OCR = ("pip install easyocr, then place its 'craft_mlt_25k.pth' and 'english_g2.pth' in ~/.EasyOCR/model (or set "
             "ANVESH_OCR_MODEL_DIR). ANVESH never downloads OCR weights on its own; this needs your approval.")


def load_ocr_engine(model_dir: str = None):
    try:
        return EasyOcrEngine(model_dir)
    except ImportError:
        return UnavailableOcr("the 'easyocr' package is not installed", SETUP_OCR)
    except Exception as exc:  # noqa: BLE001 - missing weights etc.
        return UnavailableOcr(f"EasyOCR could not start ({type(exc).__name__}: {exc})", SETUP_OCR)


class UnavailablePlateDetector:
    available = False

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def detect(self, image):
        raise RuntimeError(f"plate detector unavailable: {self.reason}")

    def describe(self) -> dict:
        return {"available": False, "model": None, "reason": self.reason,
                "setup": "Place a YOLO licence-plate detector at models/plate_detector.pt (or set ANVESH_PLATE_MODEL). None is shipped; "
                         "choosing and licensing one needs your decision."}


class YoloPlateDetector:
    available = True

    def __init__(self, weights: str) -> None:
        from ultralytics import YOLO

        self._model, self._weights = YOLO(weights), weights

    def detect(self, image) -> list:
        res = self._model.predict(image, verbose=False)[0]
        return [(tuple(float(v) for v in b), float(c)) for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]

    def describe(self) -> dict:
        return {"available": True, "model": Path(self._weights).name, "reason": None, "setup": ""}


def load_plate_detector(weights: str = None):
    path = Path(weights) if weights else None
    if path is None or not path.is_file():
        return UnavailablePlateDetector(f"no plate detector weights at {path or 'models/plate_detector.pt'}")
    try:
        return YoloPlateDetector(str(path))
    except Exception as exc:  # noqa: BLE001
        return UnavailablePlateDetector(f"weights could not be loaded ({type(exc).__name__}: {exc})")


def preprocess_crop(crop):
    """Grayscale, contrast-normalise and upscale a plate crop to a legible height before OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h = gray.shape[0]
    if h < 64:
        scale = 64.0 / max(h, 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)


def sanitize_ocr_result(res) -> tuple:
    """(OcrResult, [problems]). An engine's output is untrusted: text must be a string, confidence a finite number in [0, 1]
    (otherwise it is dropped to None, which can only ever yield `needs_review`), alternatives a tuple of strings."""
    problems = []
    text = res.text if isinstance(res.text, str) else str(res.text or "")
    conf = res.confidence
    if conf is None:
        pass  # "no confidence" is allowed; it can only ever yield needs_review
    elif isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
        problems.append(f"the OCR engine returned an invalid confidence ({conf!r}); it was discarded")
        conf = None
    alts = tuple(a for a in (res.alternatives or ()) if isinstance(a, str))
    return OcrResult(text, conf, alts, res.engine if isinstance(res.engine, str) else ""), problems


def reading_status(raw_text: str, confidence: float, fmt: dict) -> str:
    """OCR-derived reading -> status. NEVER a verified status."""
    if not normalize(raw_text):
        return "unreadable"
    if fmt["valid"] and confidence is not None and confidence >= OCR_CANDIDATE_MIN_CONF:
        return "ocr_candidate"
    return "needs_review"


class PlateService:
    def __init__(self, db: AppDatabase, audit: AuditLog, evidence, ocr=None, detector=None, require_distinct: bool = True) -> None:
        self._db, self._audit, self._evidence = db, audit, evidence
        self.ocr = ocr or UnavailableOcr("not configured", SETUP_OCR)
        self.detector = detector or UnavailablePlateDetector("not configured")
        self._distinct = require_distinct

    def status(self) -> dict:
        return {"ocr": self.ocr.describe() if hasattr(self.ocr, "describe") else {"available": bool(self.ocr.available)},
                "detector": self.detector.describe() if hasattr(self.detector, "describe") else {"available": bool(self.detector.available)},
                "manual_entry": {"available": True}}

    # -- reading ---------------------------------------------------------------

    def _insert(self, **c) -> dict:
        reading_id = "PLT-" + uuid.uuid4().hex[:10].upper()
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO plate_readings(reading_id, incident_id, track_id, source, raw_text, normalized_text, format_valid, confidence, "
                "alternatives_json, status, engine, crop_asset_id, frame_index, offset_s, supersedes, note, created_at, created_by) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (reading_id, c["incident_id"], c.get("track_id"), c["source"], c.get("raw_text"), c.get("normalized_text"),
                 1 if c.get("format_valid") else 0, c.get("confidence"), dumps(c.get("alternatives", [])), c["status"], c.get("engine"),
                 c.get("crop_asset_id"), c.get("frame_index"), c.get("offset_s"), c.get("supersedes"), c.get("note", ""), now_iso(), c["created_by"]))
        self._audit.append(c["created_by"], f"plate.{c['source']}", "incident", c["incident_id"],
                           {"reading_id": reading_id, "status": c["status"], "supersedes": c.get("supersedes"),
                            "normalized": c.get("normalized_text")}, actor_role=c.get("role"))
        return self.get(reading_id)

    def get(self, reading_id: str):
        row = self._db.one("SELECT * FROM plate_readings WHERE reading_id = ?", (reading_id,))
        if row:
            row["alternatives"] = loads(row.pop("alternatives_json"), [])
            row["format_valid"] = bool(row["format_valid"])
            row["verified"] = row["status"] in VERIFIED_STATUSES
        return row

    def history(self, incident_id: str) -> list:
        return [self.get(r["reading_id"]) for r in
                self._db.query("SELECT reading_id FROM plate_readings WHERE incident_id = ? ORDER BY created_at, rowid", (incident_id,))]

    def current(self, incident_id: str, track_id: str = None) -> list:
        """Readings not superseded by a later one (the head of each chain)."""
        rows = self.history(incident_id)
        superseded = {r["supersedes"] for r in rows if r["supersedes"]}
        heads = [r for r in rows if r["reading_id"] not in superseded]
        return [r for r in heads if track_id is None or r["track_id"] == track_id]

    def read_from_evidence(self, incident_id: str, track_id: str, user: str, role: str = None) -> dict:
        """Run localisation + OCR on the incident's saved frames for `track_id`. Returns what happened,
        including `unavailable` when the models are not installed. Creates readings only from real OCR output."""
        if not self.detector.available or not self.ocr.available:
            why = []
            if not self.detector.available:
                why.append("plate detector: " + self.detector.reason)
            if not self.ocr.available:
                why.append("OCR: " + self.ocr.reason)
            self._audit.append(user, "plate.ocr_unavailable", "incident", incident_id, {"reasons": why}, actor_role=role)
            return {"status": "unavailable", "reasons": why, "readings": [], "next_step": "enter the plate manually from the evidence frame"}
        readings, errors = [], []
        for asset in [a for a in self._evidence.list(incident_id) if a["kind"] == "frame" and a["track_id"] == track_id]:
            bbox = asset["meta"].get("track_bbox")
            if not bbox or asset.get("purged_at"):
                continue
            data, state = self._evidence.read_verified(asset)
            if data is None:  # purged, missing or failing its integrity check: never read plates from a file that differs from the record
                errors.append({"asset_id": asset["asset_id"], "stage": "frame", "error": f"evidence frame not usable ({state})"})
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                errors.append({"asset_id": asset["asset_id"], "stage": "frame", "error": "evidence frame could not be decoded"})
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            vehicle = frame[max(0, y1):y2, max(0, x1):x2]
            if vehicle.size == 0:
                continue
            try:
                found = self.detector.detect(vehicle)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not abort the others
                errors.append({"asset_id": asset["asset_id"], "stage": "detector", "error": f"{type(exc).__name__}: {exc}"})
                continue
            for (px1, py1, px2, py2), conf in found:
                crop = vehicle[max(0, int(py1)):int(py2), max(0, int(px1)):int(px2)]
                if crop.size == 0:
                    continue
                ok, buf = cv2.imencode(".png", crop)
                crop_asset = self._evidence.add_bytes(incident_id, "plate_crop", buf.tobytes(), ".png", "image/png", offset_s=asset["offset_s"],
                                                      frame_index=asset["frame_index"], track_id=track_id,
                                                      meta={"source_frame_asset": asset["asset_id"], "detector_confidence": conf}, actor=user)
                try:
                    res, problems = sanitize_ocr_result(self.ocr.read(preprocess_crop(crop)))
                except Exception as exc:  # noqa: BLE001 - an engine failure on one crop is recorded, not fatal
                    err = f"{type(exc).__name__}: {exc}"
                    errors.append({"asset_id": asset["asset_id"], "stage": "ocr", "error": err})
                    self._audit.append(user, "plate.ocr_error", "incident", incident_id, {"asset_id": asset["asset_id"], "error": err[:300]}, actor_role=role)
                    readings.append(self._insert(
                        incident_id=incident_id, track_id=track_id, source="ocr", raw_text="", normalized_text="", format_valid=False,
                        confidence=None, alternatives=[], status="unreadable", engine=getattr(self.ocr, "name", None),
                        crop_asset_id=crop_asset["asset_id"], frame_index=asset["frame_index"], offset_s=asset["offset_s"],
                        note=f"OCR engine error on this crop: {err[:200]}", created_by=user, role=role))
                    continue
                norm = normalize(res.text)
                fmt = validate_format(norm)
                alts = [{"text": a, "kind": "derived_positional"} for a in positional_alternatives(norm)] + \
                       [{"text": normalize(a), "kind": "ocr_alternative"} for a in res.alternatives]
                notes = ([] if fmt["valid"] else [f"format: {fmt['reason']}"]) + problems
                readings.append(self._insert(
                    incident_id=incident_id, track_id=track_id, source="ocr", raw_text=res.text, normalized_text=norm,
                    format_valid=fmt["valid"], confidence=res.confidence, alternatives=alts, status=reading_status(res.text, res.confidence, fmt),
                    engine=res.engine, crop_asset_id=crop_asset["asset_id"], frame_index=asset["frame_index"], offset_s=asset["offset_s"],
                    note="; ".join(notes), created_by=user, role=role))
        return {"status": "ok" if readings else ("error" if errors else "no_plate_found"), "readings": readings, "errors": errors}

    # -- officer actions ------------------------------------------------------------

    def manual_entry(self, incident_id: str, text: str, user: str, role: str, track_id: str = None, note: str = "",
                     crop_asset_id: str = None, supersedes: str = None) -> dict:
        raw = (text or "").strip()
        norm = normalize(raw)
        if not norm:
            raise ValueError("plate text must not be empty")
        fmt = validate_format(norm)
        return self._insert(incident_id=incident_id, track_id=track_id, source="manual_entry", raw_text=raw, normalized_text=norm,
                            format_valid=fmt["valid"], confidence=None, alternatives=[], status="officer_entered", engine=None,
                            crop_asset_id=crop_asset_id, supersedes=supersedes,
                            note=note or ("" if fmt["valid"] else f"format: {fmt['reason']}"), created_by=user, role=role)

    def verify(self, reading_id: str, user: str, role: str, note: str = "", allow_nonstandard: bool = False) -> dict:
        prev = self.get(reading_id)
        if prev is None:
            raise LookupError(reading_id)
        if prev["verified"]:
            raise ValueError("this reading is already verified")
        if prev["status"] in ("unreadable",) or not prev["normalized_text"]:
            raise ValueError("an unreadable reading cannot be verified; enter or correct the plate instead")
        if not prev["format_valid"] and not allow_nonstandard:
            raise ValueError("the plate does not match a standard Indian format; correct it, or confirm explicitly that it is a non-standard mark")
        if self._distinct and prev["created_by"] == user:
            raise ValueError("four-eyes: the user who entered/read the plate cannot also verify it")
        if self._superseded(prev):
            raise ValueError("this reading has been superseded; verify the current one")
        return self._insert(incident_id=prev["incident_id"], track_id=prev["track_id"], source="officer_verification",
                            raw_text=prev["raw_text"], normalized_text=prev["normalized_text"], format_valid=prev["format_valid"],
                            confidence=prev["confidence"], alternatives=prev["alternatives"], status="officer_verified", engine=prev["engine"],
                            crop_asset_id=prev["crop_asset_id"], frame_index=prev["frame_index"], offset_s=prev["offset_s"],
                            supersedes=reading_id, note=note, created_by=user, role=role)

    def correct(self, reading_id: str, new_text: str, user: str, role: str, note: str = "", can_verify: bool = False) -> dict:
        prev = self.get(reading_id)
        if prev is None:
            raise LookupError(reading_id)
        if self._superseded(prev):
            raise ValueError("this reading has been superseded; correct the current one")
        norm = normalize(new_text)
        if not norm:
            raise ValueError("corrected plate text must not be empty")
        if not (note or "").strip():
            raise ValueError("a note explaining the correction is required")
        fmt = validate_format(norm)
        if self._distinct and can_verify and prev["created_by"] == user and prev["source"] != "ocr":
            can_verify = False  # correcting your own entry does not make it independently verified
        return self._insert(incident_id=prev["incident_id"], track_id=prev["track_id"], source="officer_correction", raw_text=new_text.strip(),
                            normalized_text=norm, format_valid=fmt["valid"], confidence=None, alternatives=[],
                            status="officer_corrected" if can_verify else "officer_entered", engine=None, crop_asset_id=prev["crop_asset_id"],
                            frame_index=prev["frame_index"], offset_s=prev["offset_s"], supersedes=reading_id,
                            note=f"corrected from '{prev['normalized_text']}': {note}", created_by=user, role=role)

    def _superseded(self, reading: dict) -> bool:
        return self._db.one("SELECT 1 AS x FROM plate_readings WHERE supersedes = ?", (reading["reading_id"],)) is not None
