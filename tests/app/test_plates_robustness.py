"""Plate reading robustness: confidence thresholds, several candidate plates, unreadable crops, engine failures, bad engine output,
manual correction and uncertainty reporting.

The OCR engine and plate detector here are TEST DOUBLES injected into PlateService. Nothing in this file shows that a real OCR engine
is installed, connected or accurate - none is (see docs/ANVESH_EXECUTION_REPORT.md, section 9.E).
"""

import math

import pytest

from anvesh.app.plates import OCR_CANDIDATE_MIN_CONF, OcrResult, PlateService, reading_status, sanitize_ocr_result, validate_format

from .conftest import seed_incident

TRACK = "run-s000-t00001"


class Detector:
    available = True

    def __init__(self, boxes=((0, 0, 1.0, 1.0),), raises=None):
        self._boxes, self._raises = boxes, raises

    def detect(self, image):
        if self._raises:
            raise self._raises
        h, w = image.shape[:2]
        return [((x1 * w, y1 * h, x2 * w, y2 * h), 0.9) for x1, y1, x2, y2 in self._boxes]

    def describe(self):
        return {"available": True}


class Ocr:
    """Returns the scripted results in order (the last one repeats); an Exception instance in the script is raised."""

    available = True
    name = "scripted-ocr"

    def __init__(self, *script):
        self._script, self.calls = list(script), 0

    def read(self, image):
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def describe(self):
        return {"available": True}


def svc_with(services, ocr, detector=None):
    return PlateService(services.db, services.audit, services.evidence, ocr=ocr, detector=detector or Detector())


def readings(out):
    return out["readings"]


# -- confidence thresholds ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text,conf,status", [
    ("MH12AB1234", OCR_CANDIDATE_MIN_CONF, "ocr_candidate"),                      # exactly on the threshold, well formed
    ("MH12AB1234", OCR_CANDIDATE_MIN_CONF - 1e-9, "needs_review"),               # just below
    ("MH12AB1234", 1.0, "ocr_candidate"),
    ("MH12AB1234", 0.0, "needs_review"),
    ("MH12AB123", 0.99, "needs_review"),                                          # high confidence cannot rescue a malformed mark
    ("ZZ12AB1234", 0.99, "needs_review"),                                         # unknown state code
    ("", 0.99, "unreadable"),
    ("  - . ", 0.99, "unreadable"),                                                # nothing alphanumeric survives normalisation
])
def test_status_from_confidence_and_format_is_never_a_verified_status(text, conf, status):
    from anvesh.app.plates import normalize

    got = reading_status(text, conf, validate_format(normalize(text)))
    assert got == status and got not in ("officer_verified", "officer_corrected", "officer_entered")


def test_a_missing_confidence_can_never_make_a_candidate():
    assert reading_status("MH12AB1234", None, validate_format("MH12AB1234")) == "needs_review"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5, True, "0.99", None])
def test_invalid_engine_confidence_is_discarded_not_trusted(bad):
    res, problems = sanitize_ocr_result(OcrResult("MH12AB1234", bad, (), "x"))
    assert res.confidence is None
    assert (not problems) if bad is None else problems  # None is "no confidence", not a defect
    assert reading_status(res.text, res.confidence, validate_format("MH12AB1234")) == "needs_review"


def test_engine_output_with_wrong_types_is_coerced_safely():
    res, _ = sanitize_ocr_result(OcrResult(12345, 0.9, ("MH12AB1234", 7, None), None))
    assert res.text == "12345" and res.alternatives == ("MH12AB1234",) and res.engine == ""


def test_invalid_confidence_through_the_service_yields_needs_review_with_a_note(services, users):
    inc = seed_incident(services)
    out = svc_with(services, Ocr(OcrResult("MH12AB1234", math.nan, (), "x"))).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    r = readings(out)[0]
    assert r["status"] == "needs_review" and r["confidence"] is None and "invalid confidence" in r["note"]


# -- several candidate plates, unreadable crops, engine failures -----------------------------------------------------------


def test_several_plates_in_one_vehicle_crop_give_independent_readings_each_with_its_own_crop(services, users):
    inc = seed_incident(services)
    ocr = Ocr(OcrResult("MH12AB1234", 0.95, (), "x"), OcrResult("KA01XY9999", 0.60, (), "x"))
    out = svc_with(services, ocr, Detector(boxes=((0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)))).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    rs = readings(out)
    assert len(rs) == 4  # two key frames x two candidate plates
    assert len({r["crop_asset_id"] for r in rs}) == 4
    assert {r["status"] for r in rs} == {"ocr_candidate", "needs_review"}
    assert not any(r["verified"] for r in rs)  # several candidates never become a verified one by themselves


def test_an_unreadable_crop_is_kept_as_unreadable_cannot_be_verified_and_can_be_entered_manually(services, users):
    inc = seed_incident(services)
    out = svc_with(services, Ocr(OcrResult("", 0.0, (), "x"))).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    r = readings(out)[0]
    assert r["status"] == "unreadable" and r["normalized_text"] == "" and r["crop_asset_id"]
    with pytest.raises(ValueError):
        services.plates.verify(r["reading_id"], "rev1", "reviewer")  # an unreadable reading cannot be verified
    # the officer enters the plate; the machine record is untouched and linked
    manual = services.plates.correct(r["reading_id"], "MH12AB1234", "rev1", "reviewer", note="read from the crop by eye", can_verify=False)
    assert manual["status"] == "officer_entered" and manual["supersedes"] == r["reading_id"] and manual["crop_asset_id"] == r["crop_asset_id"]
    assert services.plates.get(r["reading_id"])["status"] == "unreadable"


def test_an_ocr_engine_exception_on_one_crop_is_recorded_and_the_other_crops_are_still_read(services, users):
    inc = seed_incident(services)
    ocr = Ocr(RuntimeError("CUDA out of memory"), OcrResult("MH12AB1234", 0.97, (), "x"))
    out = svc_with(services, ocr).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    rs = readings(out)
    assert out["status"] == "ok" and [r["status"] for r in rs] == ["unreadable", "ocr_candidate"]
    assert "CUDA out of memory" in rs[0]["note"] and out["errors"] and out["errors"][0]["stage"] == "ocr"
    assert any(a["action"] == "plate.ocr_error" for a in services.audit.query(entity_type="incident", entity_id=inc["incident_id"])[0])
    assert services.audit.verify()["ok"]


def test_every_crop_failing_reports_errors_not_a_crash(services, users):
    inc = seed_incident(services)
    out = svc_with(services, Ocr(ValueError("bad tensor"))).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    assert out["status"] == "ok" and all(r["status"] == "unreadable" for r in readings(out))  # recorded as unreadable readings
    assert len(out["errors"]) == len(readings(out))


def test_a_plate_detector_exception_is_reported_per_frame_and_creates_no_reading(services, users):
    inc = seed_incident(services)
    out = svc_with(services, Ocr(OcrResult("MH12AB1234", 0.99, (), "x")), Detector(raises=RuntimeError("weights corrupt"))).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    assert out["status"] == "error" and out["readings"] == [] and {e["stage"] for e in out["errors"]} == {"detector"}
    assert services.plates.history(inc["incident_id"]) == []


def test_a_tampered_evidence_frame_is_not_read(services, users):
    inc = seed_incident(services)
    frame = next(a for a in services.evidence.list(inc["incident_id"]) if a["kind"] == "frame")
    services.evidence.path_of(frame).write_bytes(b"tampered")
    ocr = Ocr(OcrResult("MH12AB1234", 0.99, (), "x"))
    out = svc_with(services, ocr).read_from_evidence(inc["incident_id"], TRACK, "inv1")
    assert any("mismatch" in e["error"] for e in out["errors"])  # the altered frame is refused
    assert len(readings(out)) == 1  # the other, intact key frame was still read


# -- uncertainty reporting -------------------------------------------------------------------------------------------------


def test_format_repair_suggestions_are_labelled_derived_and_never_replace_the_reading(services, users):
    inc = seed_incident(services)
    ocr = Ocr(OcrResult("MH12AB1Z34", 0.91, ("MH12A81234",), "x"))
    r = readings(svc_with(services, ocr).read_from_evidence(inc["incident_id"], TRACK, "inv1"))[0]
    assert r["normalized_text"] == "MH12AB1Z34"  # exactly what the engine read
    kinds = {a["kind"] for a in r["alternatives"]}
    assert "derived_positional" in kinds and "ocr_alternative" in kinds
    derived = [a["text"] for a in r["alternatives"] if a["kind"] == "derived_positional"]
    assert "MH12AB1234" in derived and "MH12AB1Z34" not in derived
    assert r["status"] == "needs_review" and "format:" in r["note"]


def test_manual_correction_chain_keeps_every_step_and_only_a_second_person_verifies(services, users):
    inc = seed_incident(services)
    r = readings(svc_with(services, Ocr(OcrResult("MH12AB1Z34", 0.9, (), "x"))).read_from_evidence(inc["incident_id"], TRACK, "inv1"))[0]
    fixed = services.plates.correct(r["reading_id"], "MH12AB1234", "inv1", "investigator", note="Z is a 2", can_verify=False)
    assert fixed["status"] == "officer_entered"  # the person who corrected it cannot be the verifier
    with pytest.raises(ValueError):
        services.plates.verify(fixed["reading_id"], "inv1", "investigator")
    verified = services.plates.verify(fixed["reading_id"], "rev1", "reviewer", note="matches the crop")
    assert verified["verified"] and verified["supersedes"] == fixed["reading_id"]
    history = services.plates.history(inc["incident_id"])
    chain = [h["status"] for h in history if h["reading_id"] in {r["reading_id"], fixed["reading_id"], verified["reading_id"]}]
    assert chain == ["needs_review", "officer_entered", "officer_verified"]
    assert [h["reading_id"] for h in services.plates.current(inc["incident_id"], TRACK) if h["supersedes"] == fixed["reading_id"]] == [verified["reading_id"]]
