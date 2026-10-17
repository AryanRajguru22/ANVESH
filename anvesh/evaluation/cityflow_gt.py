"""CityFlowV2 (AIC22) ground-truth / baseline-result loaders.

IMPORTANT GROUND-TRUTH CAVEAT (from the dataset's own ReadMe.txt, item 6):
`gt/gt.txt` lists "Only vehicles that pass through at least 2 cameras". A
vehicle that is visible in only ONE camera is NOT annotated. Consequences for
single-camera scoring, enforced in how `scripts/eval_cityflow.py` labels its
output:
  - RECALL, IDR, ID switches and fragmentations measure behaviour on annotated
    vehicles and are NOT inflated/deflated by the missing annotations.
  - PRECISION, IDP and IDF1 count every correct-but-unannotated prediction as a
    false positive, so they are LOWER BOUNDS on the true values. True precision
    cannot be computed from this GT and is reported as unavailable.

Frame numbering: MOTChallenge files are 1-based (`frame=1` is the first video
frame). `FRAME_OFFSET` maps a 0-based video frame index to the GT frame
number; `scripts/eval_cityflow.py` verifies the alignment empirically (mean
IoU at offsets -1/0/+1) instead of trusting this constant.
"""

from __future__ import annotations

from pathlib import Path

DATASET_ROOT = Path(r"E:\traffic dataset")
FRAME_OFFSET = 1  # 0-based video frame index -> 1-based MOT frame number


def camera_dir(scenario: str = "S02", camera: str = "c006", split: str = "validation", root=DATASET_ROOT) -> Path:
    return Path(root) / split / scenario / camera


def load_mot_file(path) -> dict:
    """MOTChallenge text -> {frame: [(obj_id, x, y, w, h)]}. `obj_id` is the
    int ID column (-1 for detection files). Confidence (column 7) is returned
    separately by `load_mot_file_with_conf`."""
    table: dict = {}
    for row in load_mot_file_with_conf(path).items():
        frame, rows = row
        table[frame] = [(oid, x, y, w, h) for (oid, x, y, w, h, _c) in rows]
    return table


def load_mot_file_with_conf(path) -> dict:
    """-> {frame: [(obj_id, x, y, w, h, conf)]}."""
    table: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            oid = int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            table.setdefault(frame, []).append((oid, x, y, w, h, conf))
    return table


def load_roi_mask(path):
    """roi.jpg -> boolean numpy array (True = inside the region where
    annotated vehicles' whole bodies lie)."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read ROI image: {path}")
    return image > 127


def inside_roi(mask, x: float, y: float, w: float, h: float) -> bool:
    """True iff the box's bottom-centre ground point lies inside the ROI
    (the AIC convention is about where vehicles are annotatable; the ground
    point is what the rest of ANVESH treats as a vehicle's position)."""
    height, width = mask.shape
    gx = int(min(max(x + w / 2.0, 0), width - 1))
    gy = int(min(max(y + h, 0), height - 1))
    return bool(mask[gy, gx])


def filter_table_by_roi(table: dict, mask) -> dict:
    out: dict = {}
    for frame, rows in table.items():
        kept = [r for r in rows if inside_roi(mask, r[1], r[2], r[3], r[4])]
        if kept:
            out[frame] = kept
    return out
