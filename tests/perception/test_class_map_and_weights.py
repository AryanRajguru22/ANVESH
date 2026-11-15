"""Checkpoint-agnostic class mapping, fail-closed weights resolution, and tracker/detector
integration with a model whose class indices differ from COCO (the ANVESH fine-tuned layout)."""

import sys
import types
from pathlib import Path

import pytest

from anvesh.perception.detection import COCO_TO_VEHICLE_CLASS, ClassMap, parse_detection_results
from anvesh.perception.tracking import parse_tracked_results
from anvesh.perception.weights import (
    WeightsNotFoundError,
    checkpoint_sha256,
    checkpoint_version_label,
    require_weights_file,
    resolve_detector_weights,
)
from anvesh.storage.schemas import VehicleClass
from tests.perception.fakes import FakeBoxes, FakeResult

ANVESH_NAMES = {0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}


# --- class mapping ---------------------------------------------------------------------------

def test_coco_names_resolve_to_the_legacy_coco_mapping():
    coco_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light"}
    cm = ClassMap.from_model_names(coco_names)
    assert cm.id_to_vehicle_class == COCO_TO_VEHICLE_CLASS
    assert cm.class_ids == [1, 2, 3, 5, 7]


def test_anvesh_five_class_layout_resolves_by_name_not_index():
    cm = ClassMap.from_model_names(ANVESH_NAMES)
    assert cm.id_to_vehicle_class == {
        0: VehicleClass.BICYCLE, 1: VehicleClass.CAR, 2: VehicleClass.MOTORCYCLE,
        3: VehicleClass.BUS, 4: VehicleClass.TRUCK}
    # The legacy COCO filter [1,2,3,5,7] would silently drop class 0 and reach past the model's 5 classes.
    assert cm.class_ids == [0, 1, 2, 3, 4]
    assert cm.class_ids != sorted(COCO_TO_VEHICLE_CLASS)


def test_names_given_as_a_list_and_with_mixed_case_and_spaces():
    cm = ClassMap.from_model_names(["Car", "Auto Rickshaw", "person"])
    assert cm.id_to_vehicle_class == {0: VehicleClass.CAR, 1: VehicleClass.AUTO_RICKSHAW}


def test_model_without_any_vehicle_class_is_rejected():
    with pytest.raises(ValueError, match="no recognised ANVESH vehicle classes"):
        ClassMap.from_model_names({0: "person", 1: "dog"})


def test_parse_detection_results_uses_the_given_class_map():
    cm = ClassMap.from_model_names(ANVESH_NAMES)
    boxes = FakeBoxes(xyxy=[[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 9.0, 9.0]], cls=[0.0, 4.0], conf=[0.8, 0.7])
    dets = parse_detection_results(FakeResult(boxes), 1, 0.1, cm)
    assert [(d.vehicle_class, d.class_name) for d in dets] == [
        (VehicleClass.BICYCLE, "bicycle"), (VehicleClass.TRUCK, "truck")]


def test_default_class_map_is_still_coco_so_existing_callers_are_unchanged():
    boxes = FakeBoxes(xyxy=[[0.0, 0.0, 10.0, 10.0]], cls=[2.0], conf=[0.9])
    assert parse_detection_results(FakeResult(boxes), 0, 0.0)[0].vehicle_class == VehicleClass.CAR


def test_tracked_results_use_the_given_class_map_and_keep_unconfirmed_ids_out_of_tracks():
    cm = ClassMap.from_model_names(ANVESH_NAMES)

    boxes = FakeBoxes(xyxy=[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 5.0, 5.0]], cls=[3.0, 1.0], conf=[0.9, 0.6],
                      ids=[7.0, None])
    raw, tracked = parse_tracked_results(FakeResult(boxes), 4, 0.4, cm)
    assert [d.vehicle_class for d in raw] == [VehicleClass.BUS, VehicleClass.CAR]
    assert [(t.track_id, t.detection.vehicle_class) for t in tracked] == [("7", VehicleClass.BUS)]


# --- weights resolution ------------------------------------------------------------------------

def test_missing_weights_fail_clearly(tmp_path):
    with pytest.raises(WeightsNotFoundError, match="never downloads"):
        require_weights_file(tmp_path / "nope.pt")


def test_explicit_finetuned_path_that_is_missing_does_not_fall_back_to_pretrained(tmp_path):
    pretrained = tmp_path / "yolov8n.pt"
    pretrained.write_bytes(b"pre")
    with pytest.raises(WeightsNotFoundError):
        resolve_detector_weights(str(tmp_path / "finetuned.pt"), str(pretrained))


def test_unset_finetuned_path_falls_back_to_pretrained(tmp_path):
    pretrained = tmp_path / "yolov8n.pt"
    pretrained.write_bytes(b"pre")
    for unset in (None, ""):
        path, source = resolve_detector_weights(unset, str(pretrained))
        assert (path, source) == (pretrained, "pretrained")


def test_existing_finetuned_path_is_used(tmp_path):
    pretrained, ft = tmp_path / "yolov8n.pt", tmp_path / "ft.pt"
    pretrained.write_bytes(b"pre")
    ft.write_bytes(b"ft")
    assert resolve_detector_weights(str(ft), str(pretrained)) == (ft, "finetuned")


def test_pretrained_fallback_missing_is_also_an_error(tmp_path):
    with pytest.raises(WeightsNotFoundError):
        resolve_detector_weights(None, str(tmp_path / "gone.pt"))


def test_version_label_hashes_the_actual_file(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    la, lb = checkpoint_version_label(a, "finetuned"), checkpoint_version_label(b, "finetuned")
    assert la != lb and la.startswith("finetuned:sha256-")
    assert la.endswith(checkpoint_sha256(a)[:12])
    assert checkpoint_version_label(tmp_path / "missing.pt").endswith("unreadable")


# --- config ------------------------------------------------------------------------------------

def test_app_config_prefers_finetuned_env_over_model_env_over_default(tmp_path):
    from anvesh.app.config import load_config
    base = load_config(env={}, data_dir=tmp_path)
    assert base.model_source == "pretrained" and base.model_path.endswith("yolov8n.pt")
    over = load_config(env={"ANVESH_MODEL": "x/pre.pt"}, data_dir=tmp_path)
    assert (over.model_path, over.model_source) == ("x/pre.pt", "pretrained")
    ft = load_config(env={"ANVESH_MODEL": "x/pre.pt", "ANVESH_FINETUNED_MODEL": "x/ft.pt"}, data_dir=tmp_path)
    assert (ft.model_path, ft.model_source) == ("x/ft.pt", "finetuned")


def test_default_tracker_factory_refuses_missing_weights_instead_of_downloading(tmp_path):
    from anvesh.app.runner import default_tracker_factory
    factory = default_tracker_factory(str(tmp_path / "yolov8n.pt"))
    with pytest.raises(WeightsNotFoundError):
        factory({"source_type": "file", "source_uri": "x.mp4"}, None)


# --- tracker / detector wiring with a stand-in model whose layout is NOT COCO ----------------------

class _FakeYolo:
    """Stands in for `ultralytics.YOLO`: a 5-class model that records the `classes=` filter it is given."""
    last = None

    def __init__(self, path):
        self.path = path
        self.names = ANVESH_NAMES
        self.calls = []
        _FakeYolo.last = self

    def _result(self):
        return FakeResult(FakeBoxes(xyxy=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]], cls=[0.0, 3.0],
                                    conf=[0.9, 0.8], ids=[1.0, 2.0]))

    def predict(self, image, **kw):
        self.calls.append(kw)
        return [self._result()]

    track = predict


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYolo
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return mod


def test_detector_and_tracker_pass_a_model_derived_filter_and_map_classes(tmp_path, fake_ultralytics):
    from anvesh.perception.detection import YoloDetector
    from anvesh.perception.tracking import ByteTrackYoloTracker
    w = tmp_path / "ft.pt"
    w.write_bytes(b"x")

    det = YoloDetector(str(w))
    out = det.detect(None, 0, 0.0)
    assert [d.vehicle_class for d in out] == [VehicleClass.BICYCLE, VehicleClass.BUS]
    assert _FakeYolo.last.calls[-1]["classes"] == [0, 1, 2, 3, 4]

    trk = ByteTrackYoloTracker(str(w))
    raw, tracked = trk.update(None, 0, 0.0)
    assert [d.vehicle_class for d in raw] == [VehicleClass.BICYCLE, VehicleClass.BUS]
    assert len(tracked) == 2
    assert _FakeYolo.last.calls[-1]["classes"] == [0, 1, 2, 3, 4]
    trk.reset()  # rebuilds the model; the class map must survive
    assert trk.update(None, 1, 0.1)[0][0].vehicle_class == VehicleClass.BICYCLE


def test_detector_and_tracker_refuse_missing_weights(tmp_path, fake_ultralytics):
    from anvesh.perception.detection import YoloDetector
    from anvesh.perception.tracking import ByteTrackYoloTracker
    with pytest.raises(WeightsNotFoundError):
        YoloDetector(str(tmp_path / "yolov8n.pt"))
    with pytest.raises(WeightsNotFoundError):
        ByteTrackYoloTracker(str(tmp_path / "yolov8n.pt"))


# --- real checkpoints (skipped when absent) -------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
PRETRAINED = REPO / "models" / "yolov8n.pt"
FINETUNED = REPO / "models" / "finetuned" / "anvesh_mio5_yolov8n_best.pt"


@pytest.mark.skipif(not PRETRAINED.is_file(), reason="pretrained weights not present")
def test_real_pretrained_checkpoint_resolves_to_the_coco_mapping():
    from ultralytics import YOLO
    assert ClassMap.from_model_names(YOLO(str(PRETRAINED)).names).id_to_vehicle_class == COCO_TO_VEHICLE_CLASS


@pytest.mark.skipif(not FINETUNED.is_file(), reason="fine-tuned checkpoint not present")
def test_real_finetuned_checkpoint_class_layout_and_inference_through_the_tracker():
    import numpy as np
    from ultralytics import YOLO
    from anvesh.perception.tracking import ByteTrackYoloTracker
    names = YOLO(str(FINETUNED)).names
    cm = ClassMap.from_model_names(names)
    assert {v.value for v in cm.id_to_vehicle_class.values()} == {"bicycle", "car", "motorcycle", "bus", "truck"}
    # The 6-class model also has `other_vehicle` (MIO-TCD motorized_vehicle): trained but not an ANVESH class, so ignored.
    assert sorted(names.values()) == sorted(["bicycle", "car", "motorcycle", "bus", "truck", "other_vehicle"])
    assert all(names[i] != "other_vehicle" for i in cm.class_ids) and len(cm.class_ids) == 5
    tracker = ByteTrackYoloTracker(str(FINETUNED))
    raw, tracked = tracker.update(np.zeros((480, 640, 3), dtype=np.uint8), 0, 0.0)  # blank frame: runs end to end
    assert raw == [] and tracked == []
