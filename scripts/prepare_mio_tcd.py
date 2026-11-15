"""Build ANVESH's leakage-aware YOLO dataset from the extracted MIO-TCD localisation archive.

Run in order (each stage is resumable and deterministic):

    .venv\\Scripts\\python.exe -I scripts\\prepare_mio_tcd.py features
    .venv\\Scripts\\python.exe -I scripts\\prepare_mio_tcd.py probe
    .venv\\Scripts\\python.exe -I scripts\\prepare_mio_tcd.py build --threshold 8 --seed 0
    .venv\\Scripts\\python.exe -I scripts\\prepare_mio_tcd.py subset --size 30000   # optional smaller train list

`features`  thumbnail + size of every labelled image (cached in datasets/mio_tcd/work/features.npz)
`probe`     how large the near-duplicate groups get at several distance thresholds (to choose --threshold)
`build`     group -> split whole groups 80/10/10 -> write images (hard links), labels, don't-care files, data.yaml and
            dataset_manifest.json (class counts per split, leakage audit, source hashes)
`subset`    optional smaller train list + data_subset_<N>.yaml (val/test untouched)

Inputs are untrusted downloaded data: they are read from datasets/mio_tcd/extracted and never executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anvesh.training.mio_tcd import CLASS_MAP, IGNORE_ONLY, NATIVE_CLASSES, TRAIN_CLASSES, convert_rows  # noqa: E402
from anvesh.training.splitting import (  # noqa: E402
    assign_groups_to_splits, group_size_summary, near_duplicate_groups, nearest_distance, thumbnail_feature)

ROOT = REPO_ROOT / "datasets" / "mio_tcd"
SRC = ROOT / "extracted" / "MIO-TCD-Localization"
WORK = ROOT / "work"
OUT = ROOT / "yolo5"
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def read_gt() -> dict:
    per_image = defaultdict(list)
    with (SRC / "gt_train.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            per_image[r[0]].append((r[1], float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    return per_image


def _feat(path: str):
    import cv2
    from PIL import Image

    with Image.open(path) as im:  # header only: true size without a full decode
        w, h = im.size
    small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    if small is None:
        return None
    return w, h, thumbnail_feature(small)


def stage_features(args) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / "features.npz"
    if f.is_file():
        print(f"{f} exists; delete it to recompute")
        return
    ids = sorted(read_gt())
    paths = [str(SRC / "train" / f"{i}.jpg") for i in ids]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} labelled images missing from extraction, e.g. {missing[:2]}")
    with Pool(args.workers) as pool:
        out = pool.map(_feat, paths, chunksize=256)
    bad = [i for i, o in zip(ids, out) if o is None]
    if bad:
        raise SystemExit(f"{len(bad)} unreadable images, e.g. {bad[:3]}")
    np.savez_compressed(f, ids=np.array(ids), wh=np.array([[o[0], o[1]] for o in out]),
                        feats=np.stack([o[2] for o in out]))
    print(f"wrote {f}: {len(ids)} images")


def load_features():
    z = np.load(WORK / "features.npz")
    return z["ids"], z["wh"], z["feats"]


def stage_probe(args) -> None:
    ids, _, feats = load_features()
    for thr in args.thresholds:
        s = group_size_summary(near_duplicate_groups(feats, thr))
        print(f"threshold {thr:>5}: groups={s['n_groups']:>6} largest={s['largest_group']:>6} "
              f"({100 * s['largest_group_fraction']:.2f}% of images) singletons={s['singleton_groups']}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def write_data_yaml(path: Path, train_entry: str) -> None:
    lines = [f"path: {OUT.as_posix()}", f"train: {train_entry}", "val: images/val", "test: images/test",
             f"nc: {len(TRAIN_CLASSES)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(TRAIN_CLASSES)]
    path.write_text("\n".join(lines) + "\n")


def stage_subset(args) -> None:
    """Pick a smaller TRAIN subset for a modest compute budget: ALL train images containing a rare class
    (motorcycle/bicycle) plus a seeded random sample of the rest. Val and test are never subsampled."""
    members = [l for l in (OUT / "train_full.txt").read_text().splitlines() if l]
    rng = np.random.default_rng(args.seed)
    rare = {"motorcycle", "bicycle"}
    rare_imgs, rest = [], []
    for p in members:
        classes = {TRAIN_CLASSES[int(line.split()[0])] for line in (OUT / "labels" / "train" / f"{Path(p).stem}.txt").read_text().splitlines()}
        (rare_imgs if classes & rare else rest).append(p)
    take = max(0, args.size - len(rare_imgs))
    picked = sorted(rare_imgs + [rest[i] for i in rng.choice(len(rest), min(take, len(rest)), replace=False)])
    name = f"train_subset_{args.size}.txt"
    (OUT / name).write_text("\n".join(picked) + "\n")
    write_data_yaml(OUT / f"data_subset_{args.size}.yaml", name)
    counts = Counter()
    for p in picked:
        for line in (OUT / "labels" / "train" / f"{Path(p).stem}.txt").read_text().splitlines():
            counts[TRAIN_CLASSES[int(line.split()[0])]] += 1
    info = {"size": len(picked), "rare_class_images_included": len(rare_imgs), "random_fill": len(picked) - len(rare_imgs),
            "seed": args.seed, "rule": "all train images containing motorcycle or bicycle + seeded random sample of the rest",
            "class_instances": {c: counts[c] for c in TRAIN_CLASSES}}
    mf = OUT / "dataset_manifest.json"
    m = json.loads(mf.read_text())
    m.setdefault("train_subsets", {})[str(args.size)] = info
    mf.write_text(json.dumps(m, indent=2))
    print(json.dumps(info, indent=2))


def stage_build(args) -> None:
    ids, wh, feats = load_features()
    gt = read_gt()
    labels = near_duplicate_groups(feats, args.threshold)
    group_summary = group_size_summary(labels)
    print("groups:", group_summary)
    if group_summary["largest_group_fraction"] > args.max_group_fraction:
        raise SystemExit(f"largest group is {100 * group_summary['largest_group_fraction']:.1f}% of images (> "
                         f"{100 * args.max_group_fraction:.0f}%): lower --threshold, single-linkage chaining is merging scenes")
    split_of = assign_groups_to_splits(labels, FRACTIONS, args.seed)

    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; refusing to overwrite a dataset that may already have been trained on")
    for kind in ("images", "labels", "ignore"):
        for sp in FRACTIONS:
            (OUT / kind / sp).mkdir(parents=True)

    native = {sp: Counter() for sp in FRACTIONS}
    mapped = {sp: Counter() for sp in FRACTIONS}
    ignored = {sp: Counter() for sp in FRACTIONS}
    imgs_with = {sp: Counter() for sp in FRACTIONS}
    dropped_total, link_modes, members = 0, Counter(), {sp: [] for sp in FRACTIONS}
    for k, (iid, (w, h)) in enumerate(zip(ids, wh)):
        sp = str(split_of[k])
        lab, ign, dropped = convert_rows(gt[str(iid)], int(w), int(h))
        dropped_total += dropped
        link_modes[_link_or_copy(SRC / "train" / f"{iid}.jpg", OUT / "images" / sp / f"{iid}.jpg")] += 1
        (OUT / "labels" / sp / f"{iid}.txt").write_text("\n".join(lab) + ("\n" if lab else ""))
        (OUT / "ignore" / sp / f"{iid}.txt").write_text("\n".join(ign) + ("\n" if ign else ""))
        for row in gt[str(iid)]:
            native[sp][row[0]] += 1
        for line in lab:
            mapped[sp][TRAIN_CLASSES[int(line.split()[0])]] += 1
        ignored[sp]["ignore_regions_pedestrian_nonmotorized"] += len(ign)
        for c in {TRAIN_CLASSES[int(line.split()[0])] for line in lab}:
            imgs_with[sp][c] += 1
        members[sp].append(str(OUT / "images" / sp / f"{iid}.jpg"))

    train_list = OUT / "train_full.txt"
    train_list.write_text("\n".join(members["train"]) + "\n")
    write_data_yaml(OUT / "data.yaml", "train_full.txt")

    # leakage audit: nearest-thumbnail distance from held-out images to TRAIN images, versus a plain random split.
    sp_arr = np.array([str(s) for s in split_of])
    tr = feats[sp_arr == "train"]
    audit = {"metric": "RMS pixel distance between 16x16x3 thumbnails (0-255 scale); 'near-duplicate' = nearest train image closer than tau",
             "taus": list(args.audit_taus), "grouped_split": {}, "random_split_same_sizes": {}}
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(feats))
    n_val, n_test = (sp_arr == "val").sum(), (sp_arr == "test").sum()
    rand = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}
    for name, held, ref in (("grouped_split", {s: feats[sp_arr == s] for s in ("val", "test")}, tr),
                            ("random_split_same_sizes", {s: feats[rand[s]] for s in ("val", "test")}, feats[rand["train"]])):
        for s, q in held.items():
            d = nearest_distance(q, ref)
            audit[name][s] = {f"frac_with_train_neighbour_below_{t}": float((d < t).mean()) for t in args.audit_taus}
            audit[name][s]["median_nearest_distance"] = float(np.median(d))

    manifest = {
        "dataset": "MIO-TCD localisation (labelled train portion), re-split by ANVESH", "utc": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://tcd.miovision.com/challenge/dataset.html",
        "licence": "CC BY-NC-SA 4.0 (archive README.txt and dataset page)",
        "citation": "Luo et al., MIO-TCD: A new benchmark dataset for vehicle classification and localization, IEEE TIP 2018",
        "archive_sha256": _sha256(ROOT / "download" / "MIO-TCD-Localization.tar"),
        "gt_train_csv_sha256": _sha256(SRC / "gt_train.csv"),
        "note_original_test_images": "27,743 original 'test' images have no public labels and are NOT used",
        "n_labelled_images": int(len(ids)), "split_fractions_target": FRACTIONS, "seed": args.seed,
        "grouping": {"feature": "16x16x3 thumbnail", "rms_threshold": args.threshold, **group_summary},
        "split_images": {sp: int((sp_arr == sp).sum()) for sp in FRACTIONS},
        "native_class_instances": {sp: dict(sorted(native[sp].items())) for sp in FRACTIONS},
        "training_class_instances": {sp: {c: mapped[sp][c] for c in TRAIN_CLASSES} for sp in FRACTIONS},
        "images_containing_class": {sp: {c: imgs_with[sp][c] for c in TRAIN_CLASSES} for sp in FRACTIONS},
        "dont_care_boxes": {sp: dict(ignored[sp]) for sp in FRACTIONS},
        "class_map": CLASS_MAP, "ignore_only": list(IGNORE_ONLY), "native_classes": list(NATIVE_CLASSES),
        "degenerate_boxes_dropped": dropped_total, "image_storage": dict(link_modes),
        "leakage_audit": audit,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("split_images", "training_class_instances")}, indent=2))
    print("leakage audit:", json.dumps(audit, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("features")
    f.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("probe")
    p.add_argument("--thresholds", type=float, nargs="+", default=[3, 5, 8, 12, 16])
    b = sub.add_parser("build")
    b.add_argument("--threshold", type=float, required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max-group-fraction", type=float, default=0.05)
    b.add_argument("--audit-taus", type=float, nargs="+", default=[3, 6, 10])
    sb = sub.add_parser("subset")
    sb.add_argument("--size", type=int, required=True)
    sb.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"features": stage_features, "probe": stage_probe, "build": stage_build, "subset": stage_subset}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
