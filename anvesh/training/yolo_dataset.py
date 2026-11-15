"""Validation of a YOLO-format detection dataset before it is trained on or reported on.

    errors   -> the dataset must not be used until fixed (the CLI and tests treat any error as failure)
    warnings -> real but non-fatal, reported so they cannot be missed

What is checked:
  * data.yaml parses, `names` is a contiguous 0..N-1 map, every split directory exists;
  * every image is readable and has a plausible size; every label file belongs to an image;
  * label rows: 5 numeric fields, a class id inside `names`, normalised box inside [0, 1] with positive width/height,
    no exact duplicate rows;
  * MISSING vs EMPTY label file: a missing file is "never labelled" (reported separately, and an error under `strict`);
    an empty file is "a person verified there is nothing here" (a negative image). The two are never merged;
  * LEAKAGE: byte-identical images that appear in more than one split (SHA-256), and identical base ids across splits
    (Roboflow-style `<name>_jpg.rf.<hash>` suffixes are stripped first). This finds EXACT duplicates only; near-duplicate
    frames of the same scene (adjacent video frames) need the thumbnail audit in `anvesh.training.splitting` and, for
    anything cut from video, a split by SOURCE VIDEO - see docs/ANVESH_ACCIDENT_RESEARCH_PIPELINE.md.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ROBOFLOW_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)


def _read_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def base_id(stem: str) -> str:
    return _ROBOFLOW_SUFFIX.sub("", stem)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"image path has no 'images' directory: {image}")
    return Path(*parts).with_suffix(".txt")


def _image_size(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        return im.size


def validate_yolo_dataset(data_yaml, strict: bool = False, check_images: bool = True, max_examples: int = 5) -> dict:
    data_yaml = Path(data_yaml)
    errors, warnings = [], []

    def err(msg):
        errors.append(msg)

    def warn(msg):
        warnings.append(msg)

    report = {"data_yaml": str(data_yaml), "splits": {}, "errors": errors, "warnings": warnings}
    if not data_yaml.is_file():
        err(f"data.yaml not found: {data_yaml}")
        report["ok"] = False
        return report
    cfg = _read_yaml(data_yaml)
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names = cfg.get("names")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    if not isinstance(names, dict) or not names:
        err("data.yaml has no `names` mapping")
        names = {}
    if sorted(int(k) for k in names) != list(range(len(names))):
        err(f"class ids must be contiguous 0..{len(names) - 1}, got {sorted(names)}")
    report["classes"] = {int(k): v for k, v in names.items()}
    n_classes = len(names)

    hashes: dict = {}  # sha256 -> [(split, image)]
    bases: dict = defaultdict(set)  # base id -> {split}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel is None:
            if split == "train":
                err("data.yaml has no `train` entry")
            continue
        img_dir = (root / rel).resolve()
        if not img_dir.is_dir():
            err(f"split `{split}`: image directory not found: {img_dir}")
            continue
        images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        stats = Counter()
        class_boxes = Counter()
        examples = defaultdict(list)

        def note(kind, what):
            stats[kind] += 1
            if len(examples[kind]) < max_examples:
                examples[kind].append(what)

        for img in images:
            bases[base_id(img.stem)].add(split)
            if check_images:
                try:
                    w, h = _image_size(img)
                    if w < 8 or h < 8:
                        note("tiny_images", img.name)
                except Exception as exc:  # noqa: BLE001
                    note("unreadable_images", f"{img.name}: {type(exc).__name__}")
                    continue
            hashes.setdefault(_sha256(img), []).append((split, img.name))
            lab = _label_path(img)
            if not lab.is_file():
                note("missing_label_files", img.name)
                continue
            text = lab.read_text(encoding="utf-8").strip()
            if not text:
                stats["empty_label_files"] += 1  # a verified negative image
                continue
            seen_rows = set()
            n_rows = 0
            for ln, row in enumerate(text.splitlines(), 1):
                parts = row.split()
                try:
                    if len(parts) != 5:
                        raise ValueError(f"{len(parts)} fields (need 5)")
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    if not 0 <= cls < n_classes:
                        raise ValueError(f"class id {cls} outside 0..{n_classes - 1}")
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        raise ValueError("box outside the unit square or non-positive size")
                    if cx - bw / 2 < -1e-6 or cx + bw / 2 > 1 + 1e-6 or cy - bh / 2 < -1e-6 or cy + bh / 2 > 1 + 1e-6:
                        raise ValueError("box extends beyond the image")
                except ValueError as exc:
                    note("bad_label_rows", f"{lab.name}:{ln}: {exc}")
                    continue
                if row.strip() in seen_rows:
                    note("duplicate_label_rows", f"{lab.name}:{ln}")
                seen_rows.add(row.strip())
                n_rows += 1
                class_boxes[cls] += 1
                if bw * bh < 1e-5:
                    note("tiny_boxes", f"{lab.name}:{ln}")
            stats["labelled_images"] += 1
            stats["boxes"] += n_rows
        try:
            lab_root = _label_path(img_dir / "x.jpg").parent
        except ValueError:
            lab_root = None
            err(f"split `{split}`: image directory {img_dir} is not under an `images` directory, so label files cannot be located")
        if lab_root is not None and lab_root.is_dir():
            expected = {_label_path(i) for i in images}
            for lab in lab_root.rglob("*.txt"):
                if lab not in expected:
                    note("orphan_label_files", lab.name)
        report["splits"][split] = {"images": len(images), **dict(stats), "class_boxes": {int(k): v for k, v in sorted(class_boxes.items())},
                                   "examples": dict(examples)}
        for kind in ("unreadable_images", "bad_label_rows", "orphan_label_files"):
            if stats[kind]:
                err(f"split `{split}`: {stats[kind]} {kind.replace('_', ' ')} (e.g. {examples[kind][:3]})")
        for kind in ("duplicate_label_rows", "tiny_images", "tiny_boxes"):
            if stats[kind]:
                warn(f"split `{split}`: {stats[kind]} {kind.replace('_', ' ')} (e.g. {examples[kind][:3]})")
        if stats["missing_label_files"]:
            msg = (f"split `{split}`: {stats['missing_label_files']} images have NO label file (never labelled - NOT the same as an empty "
                   f"label file, which means verified-empty); e.g. {examples['missing_label_files'][:3]}")
            (err if strict else warn)(msg)
        if images and not stats["boxes"]:
            warn(f"split `{split}`: no boxes at all")

    cross_hash = [(h, locs) for h, locs in hashes.items() if len({s for s, _ in locs}) > 1]
    cross_base = {b: sorted(s) for b, s in bases.items() if len(s) > 1}
    report["leakage"] = {"identical_images_across_splits": len(cross_hash), "base_ids_across_splits": len(cross_base),
                         "examples": [locs[:3] for _, locs in cross_hash[:max_examples]],
                         "scope": "exact (SHA-256) and filename-base-id duplicates only; near-duplicate frames of the same scene are NOT detected here"}
    if cross_hash:
        err(f"{len(cross_hash)} byte-identical images appear in more than one split (train/val/test leakage)")
    if cross_base:
        err(f"{len(cross_base)} image base ids appear in more than one split")
    report["ok"] = not errors
    return report
