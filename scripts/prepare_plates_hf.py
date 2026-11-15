"""Convert the extracted Roboflow/HF `keremberke/license-plate-object-detection` COCO export to a YOLO dataset.

    .venv\\Scripts\\python.exe -I scripts\\prepare_plates_hf.py

Reads  datasets/plates_hf/extracted/{train,valid,test}/_annotations.coco.json (never modified)
Writes datasets/plates_yolo/{images,labels}/{train,val,test}, data.yaml and dataset_manifest.json.

Checks (the script fails rather than guessing): exactly one category; every annotation maps to a known image;
no image base-id appears in two splits (filenames carry a `_jpg.rf.<hash>` suffix that is stripped first).
Licence: the export README says CC BY 4.0 ("Provided by undefined"); the attribution chain and the plate region are
NOT verified here (the sampled images are Vietnamese plates from a parking-barrier camera). See the manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "datasets" / "plates_hf"
DST = REPO_ROOT / "datasets" / "plates_yolo"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DST.exists():
        raise SystemExit(f"{DST} already exists; refusing to overwrite (remove or rename it deliberately)")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(SRC), "splits": {},
                "licence_note": "CC BY 4.0 per export README ('Provided by undefined'); attribution chain and plate region unverified",
                "source_zips": {z.name: sha256(z) for z in sorted((SRC / "download").glob("*.zip"))}}
    seen: dict = {}
    for src_split, dst_split in SPLITS.items():
        coco = json.loads((SRC / "extracted" / src_split / "_annotations.coco.json").read_text(encoding="utf-8"))
        if [c["name"] for c in coco["categories"]] != ["license_plate"]:
            raise SystemExit(f"unexpected categories in {src_split}: {coco['categories']}")
        by_img: dict = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a["bbox"])
        (DST / "images" / dst_split).mkdir(parents=True)
        (DST / "labels" / dst_split).mkdir(parents=True)
        n_box = 0
        for im in coco["images"]:
            base = re.sub(r"_jpg\.rf\..*$", "", im["file_name"])
            if base in seen:
                raise SystemExit(f"image id {base} appears in {seen[base]} and {dst_split}")
            seen[base] = dst_split
            w, h = im["width"], im["height"]
            lines = []
            for x, y, bw, bh in by_img.get(im["id"], []):
                x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + bw, w), min(y + bh, h)
                if x1 - x0 <= 1 or y1 - y0 <= 1:
                    continue
                lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            stem = Path(im["file_name"]).stem
            shutil.copy2(SRC / "extracted" / src_split / im["file_name"], DST / "images" / dst_split / im["file_name"])
            (DST / "labels" / dst_split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_box += len(lines)
        manifest["splits"][dst_split] = {"images": len(coco["images"]), "boxes": n_box}
    (DST / "data.yaml").write_text(
        f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n",
        encoding="utf-8")
    (DST / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
