import sqlite3

import pytest

from anvesh.app.plates import (OCR_CANDIDATE_MIN_CONF, PlateService, UnavailableOcr, normalize, positional_alternatives, reading_status,
                               validate_format)

from .conftest import seed_incident


@pytest.mark.parametrize("text,valid", [
    ("MH12AB1234", True), ("mh 12 ab 1234", True), ("DL8CAF5030", True), ("KA01MX9999", True), ("22BH1234AA", True),
    ("XX12AB1234", False), ("MH12AB123", False), ("1234", False), ("", False),
])
def test_format_validation(text, valid):
    assert validate_format(normalize(text))["valid"] is valid


def test_unknown_state_code_is_explained():
    r = validate_format("ZZ12AB1234")
    assert not r["valid"] and "unknown state" in r["reason"]


def test_positional_alternatives_repair_confusions_but_are_only_suggestions():
    alts = positional_alternatives(normalize("MHI2A81234"))
    assert "MH12AB1234" in alts
    assert positional_alternatives("MH12AB1234") == []  # already valid: nothing to suggest


def test_ocr_reading_status_is_never_verified():
    ok = validate_format("MH12AB1234")
    assert reading_status("MH12AB1234", 0.99, ok) == "ocr_candidate"
    assert reading_status("MH12AB1234", OCR_CANDIDATE_MIN_CONF - 0.01, ok) == "needs_review"
    assert reading_status("MH12A", 0.99, validate_format("MH12A")) == "needs_review"
    assert reading_status("", 0.9, ok) == "unreadable"
    assert reading_status(" -- ", 0.9, ok) == "unreadable"


def test_ocr_unavailable_is_reported_and_creates_no_reading(services, users):
    inc = seed_incident(services)
    out = services.plates.read_from_evidence(inc["incident_id"], "run-s000-t00001", "inv1")
    assert out["status"] == "unavailable" and out["readings"] == []
    assert any("OCR" in r or "detector" in r for r in out["reasons"])
    assert services.plates.history(inc["incident_id"]) == []
    st = services.plates.status()
    assert st["ocr"]["available"] is False and st["manual_entry"]["available"] is True and st["ocr"]["setup"]


def test_real_loaders_report_unavailable_with_setup_when_not_installed(tmp_path):
    from anvesh.app.plates import load_ocr_engine, load_plate_detector
    engine = load_ocr_engine()
    if not engine.available:  # easyocr is not part of the pinned environment
        d = engine.describe()
        assert d["available"] is False and "easyocr" in d["setup"] and d["reason"]
    det = load_plate_detector(str(tmp_path / "missing.pt"))
    assert det.available is False and "models/plate_detector.pt" in det.describe()["setup"]


class _FakeDetector:
    available = True

    def detect(self, image):
        h, w = image.shape[:2]
        return [((0, 0, w, h), 0.9)]

    def describe(self):
        return {"available": True}


class _FakeOcr:
    available = True

    def __init__(self, text, conf):
        self._r = (text, conf)

    def read(self, image):
        from anvesh.app.plates import OcrResult
        return OcrResult(self._r[0], self._r[1], (), "fake-ocr")

    def describe(self):
        return {"available": True}


def test_ocr_path_with_injected_engine_keeps_original_crop_and_needs_review_when_ill_formed(services, users):
    inc = seed_incident(services)
    svc = PlateService(services.db, services.audit, services.evidence, ocr=_FakeOcr("MHI2A81234", 0.91), detector=_FakeDetector())
    out = svc.read_from_evidence(inc["incident_id"], "run-s000-t00001", "inv1")
    assert out["status"] == "ok" and len(out["readings"]) == 2  # two raw key frames for the track
    r = out["readings"][0]
    assert r["raw_text"] == "MHI2A81234" and r["status"] == "needs_review" and not r["verified"]
    assert r["engine"] == "fake-ocr" and r["confidence"] == 0.91 and r["crop_asset_id"]
    assert any(a["text"] == "MH12AB1234" and a["kind"] == "derived_positional" for a in r["alternatives"])
    crop = services.evidence.get(r["crop_asset_id"])
    assert crop["kind"] == "plate_crop" and services.evidence.verify(crop["asset_id"])["ok"]


def test_high_confidence_well_formed_ocr_is_candidate_not_verified(services, users):
    inc = seed_incident(services)
    svc = PlateService(services.db, services.audit, services.evidence, ocr=_FakeOcr("MH12AB1234", 0.97), detector=_FakeDetector())
    r = svc.read_from_evidence(inc["incident_id"], "run-s000-t00001", "inv1")["readings"][0]
    assert r["status"] == "ocr_candidate" and r["verified"] is False
    # an unreviewed OCR candidate cannot be sent to a registry
    assert "verified" not in r["status"]


def test_manual_entry_has_distinct_provenance_and_original_is_preserved_on_correction(services, users):
    inc = seed_incident(services)
    p = services.plates
    e = p.manual_entry(inc["incident_id"], "mh12 ab 1243", "inv1", "investigator", track_id="run-s000-t00001")
    assert e["source"] == "manual_entry" and e["status"] == "officer_entered" and e["confidence"] is None and e["normalized_text"] == "MH12AB1243"
    c = p.correct(e["reading_id"], "MH12AB1234", "rev1", "reviewer", note="digits transposed", can_verify=True)
    assert c["supersedes"] == e["reading_id"] and c["status"] == "officer_corrected" and c["verified"]
    orig = p.get(e["reading_id"])
    assert orig["raw_text"] == "mh12 ab 1243" and orig["normalized_text"] == "MH12AB1243"  # untouched
    assert [r["reading_id"] for r in p.current(inc["incident_id"])] == [c["reading_id"]]
    assert len(p.history(inc["incident_id"])) == 2
    with pytest.raises(ValueError):
        p.correct(e["reading_id"], "MH12AB9999", "rev1", "reviewer", note="x")  # superseded
    with pytest.raises(ValueError):
        p.correct(c["reading_id"], "MH12AB9999", "rev1", "reviewer", note=" ")  # note required


def test_investigator_correction_is_not_verification(services, users):
    inc = seed_incident(services)
    e = services.plates.manual_entry(inc["incident_id"], "MH12AB1243", "inv1", "investigator")
    c = services.plates.correct(e["reading_id"], "MH12AB1234", "inv1", "investigator", note="typo", can_verify=False)
    assert c["status"] == "officer_entered" and not c["verified"]


def test_verification_requires_a_different_user_and_valid_format(services, users):
    inc = seed_incident(services)
    p = services.plates
    e = p.manual_entry(inc["incident_id"], "MH12AB1234", "inv1", "investigator")
    with pytest.raises(ValueError, match="four-eyes"):
        p.verify(e["reading_id"], "inv1", "investigator")
    v = p.verify(e["reading_id"], "rev1", "reviewer", note="matches frame")
    assert v["verified"] and v["supersedes"] == e["reading_id"] and v["source"] == "officer_verification"
    with pytest.raises(ValueError, match="already verified"):
        p.verify(v["reading_id"], "rev2", "reviewer")
    bad = p.manual_entry(inc["incident_id"], "ABC123", "inv1", "investigator", track_id="other")
    with pytest.raises(ValueError, match="standard Indian format"):
        p.verify(bad["reading_id"], "rev1", "reviewer")
    assert p.verify(bad["reading_id"], "rev1", "reviewer", allow_nonstandard=True)["verified"]


def test_plate_readings_are_immutable_at_sql_level(services, users):
    inc = seed_incident(services)
    e = services.plates.manual_entry(inc["incident_id"], "MH12AB1234", "inv1", "investigator")
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute("UPDATE plate_readings SET normalized_text = 'XX00XX0000' WHERE reading_id = ?", (e["reading_id"],))
    with pytest.raises(sqlite3.DatabaseError):
        services.db.execute("DELETE FROM plate_readings WHERE reading_id = ?", (e["reading_id"],))


def test_empty_manual_entry_rejected(services, users):
    inc = seed_incident(services)
    with pytest.raises(ValueError):
        services.plates.manual_entry(inc["incident_id"], "  - ", "inv1", "investigator")
