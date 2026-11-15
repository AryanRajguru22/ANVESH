"""MIO-TCD localisation -> YOLO label conversion for ANVESH's taxonomy.

Source: Luo et al., "MIO-TCD: A new benchmark dataset for vehicle classification and localization", IEEE TIP 2018,
https://tcd.miovision.com/ . Licence (README.txt in the archive and the dataset page): CC BY-NC-SA 4.0.
`gt_train.csv` rows: `image_id,class,x1,y1,x2,y2` (pixel corners). The 27,743 "test" images ship WITHOUT labels
(they were the challenge's hidden set), so every split below is carved out of the 110,000 labelled train images.

Class mapping (a judgement call, documented in the report):

  MIO-TCD class          -> training class    why
  car                    -> car
  work_van               -> car               small commercial van; COCO annotators label vans "car"
  bus                    -> bus
  articulated_truck      -> truck
  single_unit_truck      -> truck
  pickup_truck           -> truck             COCO's "truck" includes pickups, so the COCO baseline is scored on
                                              the same convention; NOTE the "truck" class is mostly pickups
  motorcycle             -> motorcycle
  bicycle                -> bicycle
  motorized_vehicle      -> other_vehicle     a vehicle too occluded/oblique to classify. Kept as its own training
                                              class so it is NOT taught as background; ANVESH ignores it at run time
                                              (its name is not an ANVESH vehicle class) and the evaluator treats it
                                              as a don't-care region.
  pedestrian             -> (not trained)     not an ANVESH class; evaluator don't-care region
  non-motorized_vehicle  -> (not trained)     trailers / carts; evaluator don't-care region

`auto_rickshaw` (an ANVESH taxonomy class) does not exist in this dataset and is NOT produced by the fine-tuned model.
"""

from __future__ import annotations

TRAIN_CLASSES = ("bicycle", "car", "motorcycle", "bus", "truck", "other_vehicle")  # id = index; 0..4 = shared taxonomy

CLASS_MAP = {
    "bicycle": "bicycle", "car": "car", "work_van": "car", "motorcycle": "motorcycle", "bus": "bus",
    "articulated_truck": "truck", "single_unit_truck": "truck", "pickup_truck": "truck",
    "motorized_vehicle": "other_vehicle",
}
IGNORE_ONLY = ("pedestrian", "non-motorized_vehicle")  # evaluator don't-care regions, never trained
NATIVE_CLASSES = tuple(sorted(set(CLASS_MAP) | set(IGNORE_ONLY)))


def _yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    return f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def convert_rows(rows, w: int, h: int, min_side_px: float = 1.0):
    """rows: [(native_class, x1, y1, x2, y2)] for ONE image -> (label_lines, ignore_lines, dropped).
    Boxes are clipped to the image; boxes that are degenerate after clipping are dropped and counted.
    An unknown native class raises (a silently skipped class would become unlabeled background)."""
    labels, ignore, dropped = [], [], 0
    for native, x1, y1, x2, y2 in rows:
        if native not in CLASS_MAP and native not in IGNORE_ONLY:
            raise ValueError(f"unknown MIO-TCD class {native!r}")
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < min_side_px or y2 - y1 < min_side_px:
            dropped += 1
            continue
        if native in CLASS_MAP:
            labels.append(_yolo_line(TRAIN_CLASSES.index(CLASS_MAP[native]), x1, y1, x2, y2, w, h))
        else:
            ignore.append(_yolo_line(0, x1, y1, x2, y2, w, h))
    return labels, ignore, dropped
