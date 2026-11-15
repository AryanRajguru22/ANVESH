"""Turn a finished training run + comparison JSONs into report tables and figures (read-only on the evidence).

    .venv\\Scripts\\python.exe scripts\\summarize_training.py --run models\\finetuned\\runs\\anvesh_mio5_v1 ^
        --compare-val outputs\\training\\compare_val.json --compare-test outputs\\training\\compare_test.json ^
        --cityflow outputs\\training\\compare_cityflow.json --figures docs\\training_figures

Prints Markdown to stdout; every number comes from the run's results.csv / manifests / comparison JSONs, none is typed in.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_results(run: Path) -> list:
    with (run / "results.csv").open(newline="", encoding="utf-8") as fh:
        return [{k.strip(): float(v) for k, v in row.items() if v.strip() != ""} for row in csv.DictReader(fh)]


def md_table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def plots(rows: list, fig_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    ep = [int(r["epoch"]) for r in rows]
    made = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, k in zip(axes, ("box_loss", "cls_loss", "dfl_loss")):
        ax.plot(ep, [r[f"train/{k}"] for r in rows], marker="o", label="train")
        ax.plot(ep, [r[f"val/{k}"] for r in rows], marker="s", label="validation")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png", dpi=130)
    plt.close(fig)
    made.append("loss_curves.png")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, lab in (("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall"),
                   ("metrics/mAP50(B)", "mAP@50"), ("metrics/mAP50-95(B)", "mAP@50-95")):
        ax.plot(ep, [r[k] for r in rows], marker="o", label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_title("validation metrics per epoch (6-class model, Ultralytics val)")
    fig.tight_layout()
    fig.savefig(fig_dir / "val_metrics.png", dpi=130)
    plt.close(fig)
    made.append("val_metrics.png")
    return made


def confusion_plot(cmj: dict, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(cmj["rows_pred_cols_true"], dtype=float)
    norm = cm / np.maximum(cm.sum(0, keepdims=True), 1)  # column-normalised: share of each TRUE class
    labels = cmj["labels"]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("true class")
    ax.set_ylabel("predicted class")
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if norm[i, j] > .5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare-val")
    ap.add_argument("--compare-test")
    ap.add_argument("--cityflow")
    ap.add_argument("--figures", required=True)
    a = ap.parse_args()
    run, figs = Path(a.run), Path(a.figures)
    man = json.loads((run / "training_manifest.json").read_text())
    rows = read_results(run)
    print(f"## Run `{run.name}`: status {man['status']}, epochs completed {man['epochs_completed']}/{man['epochs_requested']}, "
          f"{man['elapsed_seconds'] / 3600:.2f} h\n")
    print(md_table(["epoch", "train box", "train cls", "train dfl", "val box", "val cls", "val dfl", "P", "R", "mAP50", "mAP50-95"],
                   [[int(r["epoch"]), f3(r["train/box_loss"]), f3(r["train/cls_loss"]), f3(r["train/dfl_loss"]),
                     f3(r["val/box_loss"]), f3(r["val/cls_loss"]), f3(r["val/dfl_loss"]), f3(r["metrics/precision(B)"]),
                     f3(r["metrics/recall(B)"]), f3(r["metrics/mAP50(B)"]), f3(r["metrics/mAP50-95(B)"])] for r in rows]))
    print("\nfigures:", plots(rows, figs))
    for tag, path in (("val", a.compare_val), ("test", a.compare_test)):
        if not path:
            continue
        c = json.loads(Path(path).read_text())
        print(f"\n### Shared-taxonomy comparison on the {tag} split ({c['n_images']} images)\n")
        names = list(c["models"])
        print(md_table(["model", "P (max-F1)", "R (max-F1)", "mAP50", "mAP50-95", "P@0.25", "R@0.25", "F1@0.25"],
                       [[n, f3(c["models"][n]["metrics_conf0.001"]["precision"]), f3(c["models"][n]["metrics_conf0.001"]["recall"]),
                         f3(c["models"][n]["metrics_conf0.001"]["map50"]), f3(c["models"][n]["metrics_conf0.001"]["map50_95"]),
                         f3(c["models"][n]["operating_point"]["precision"]), f3(c["models"][n]["operating_point"]["recall"]),
                         f3(c["models"][n]["operating_point"]["f1"])] for n in names]))
        print("\nPer class (n_gt, P, R, AP50, AP50-95):\n")
        classes = c["shared_classes"]
        print(md_table(["class", "n_gt"] + [f"{n}: P / R / AP50 / AP50-95" for n in names],
                       [[cl, c["models"][names[0]]["metrics_conf0.001"]["per_class"][cl]["n_gt"]] +
                        [" / ".join(f3(c["models"][n]["metrics_conf0.001"]["per_class"][cl].get(k))
                                    for k in ("precision", "recall", "ap50", "ap50_95")) for n in names] for cl in classes]))
        for key in ("bootstrap_map50", "bootstrap_map50_95"):
            if key in c:
                b = c[key]
                print(f"\nBootstrap {b['metric']} (n={b['n_boot']}, image-level, 95% CI): " +
                      "; ".join(f"{n} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]" for n, v in b["models"].items()) +
                      "; paired diff " + "; ".join(f"{k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                                  for k, v in b["paired_difference"].items()))
        if "ultralytics_val_crosscheck" in c:
            print("\nCross-check vs Ultralytics val() per-class AP50:", json.dumps(c["ultralytics_val_crosscheck"]))
        figs.mkdir(parents=True, exist_ok=True)
        for n in names:
            confusion_plot(c["models"][n]["confusion_matrix_conf0.25_iou0.45"], f"{n} - {tag} split (conf>=0.25, IoU 0.45)",
                           figs / f"confusion_{tag}_{n}.png")
    if a.cityflow:
        c = json.loads(Path(a.cityflow).read_text())
        print("\n### CityFlow S02 (class-agnostic recall; precision not computable)\n")
        names = list(c["models"])
        print(md_table(["camera", "GT boxes"] + [f"{n}: recall / det per frame" for n in names],
                       [[cam, v["gt_boxes_in_sampled_frames"]] + [f"{f3(v['models'][n]['recall_class_agnostic'])} / "
                                                                   f"{v['models'][n]['detections_per_frame']:.1f}" for n in names]
                        for cam, v in c["cameras"].items()] +
                       [["pooled", ""] + [f3(c["pooled"][n]["recall_class_agnostic"]) for n in names]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
