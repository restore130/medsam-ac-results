"""Plots for the five-model pixel-uncertainty evaluation.

Reads the CSVs produced by uncertainty_metrics.py and renders:
  - SegECE / AUROC comparison bar charts (val + test)
  - Dice-vs-SegECE scatter
  - reliability curves (pooled confidence vs accuracy, with y=x)
  - case heatmaps (original image, GT, 5 predictions, 5 entropy maps)

Case-selection rule (documented, not cherry-picked by eye):
  high Dice  -> val sample with the highest mean Dice across the 5 models
  low Dice   -> val sample with the lowest mean Dice across the 5 models
  boundary   -> val sample with the highest fraction of error pixels within
                5 px of the GT boundary, computed on the original model
                (most errors -> most reliable boundary-concentration signal),
                requiring >= 1000 error pixels.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.load_result import load_result
from uncertainty_metrics import entropy_normalized, load_gt

MODELS = ["original", "linear_probe", "adapter", "lora_r8", "lora_r16"]
MODEL_LABELS = ["Original", "LinearProbe", "Adapter", "LoRA-r8", "LoRA-r16"]
MODEL_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
FIG = ROOT / "figures"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_index(path: Path = ROOT / "indices/all_samples.csv") -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# comparison + reliability (from CSVs)
# --------------------------------------------------------------------------- #
def plot_comparisons(summary_rows, out_dir: Path) -> None:
    s = {(r["model_setting"], r["split"]): r for r in summary_rows}

    def grouped(metric: str, fname: str, ylabel: str) -> None:
        x = np.arange(len(MODELS))
        width = 0.36
        val = [float(s[(m, "val")][metric]) for m in MODELS]
        test = [float(s[(m, "test")][metric]) for m in MODELS]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, val, width, label="val", color="#4c72b0")
        ax.bar(x + width / 2, test, width, label="test", color="#c44e52")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by model (lower is better)" if "ECE" in ylabel
                     else f"{ylabel} by model (higher is better)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    grouped("seg_ece_per_image_mean", "comparison_segece.png", "SegECE (per-image mean)")
    grouped("auroc_mean", "comparison_auroc.png", "Pixel error AUROC")

    # Dice vs SegECE scatter (val)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for m, c in zip(MODELS, MODEL_COLORS):
        ax.scatter(float(s[(m, "val")]["seg_ece_per_image_mean"]),
                   float(s[(m, "val")]["dice_mean"]),
                   s=120, color=c, label=dict(zip(MODELS, MODEL_LABELS))[m], zorder=3)
    ax.set_xlabel("SegECE (lower = better calibrated)")
    ax.set_ylabel("Dice (higher = better segmentation)")
    ax.set_title("Dice vs SegECE (val)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dice_segece_scatter.png", dpi=150)
    plt.close(fig)


def plot_reliability(rel_rows, split: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for m, c in zip(MODELS, MODEL_COLORS):
        rows = [r for r in rel_rows
                if r["model_setting"] == m and r["split"] == split and r["n_pixels"] != "0"]
        if not rows:
            continue
        conf = [float(r["mean_conf"]) for r in rows]
        acc = [float(r["mean_acc"]) for r in rows]
        ax.plot(conf, acc, "-o", color=c, markersize=4,
                label=dict(zip(MODELS, MODEL_LABELS))[m])
    lims = [0.5, 1.0]
    ax.plot(lims, lims, "k--", lw=1, label="ideal y=x")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Mean confidence (per bin)")
    ax.set_ylabel("Pixel accuracy (per bin)")
    ax.set_title(f"Reliability curve ({split}, pooled pixels, 10 bins)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability_{split}.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# case selection + heatmaps
# --------------------------------------------------------------------------- #
def select_high_low(per_sample: list[dict], split: str = "val"):
    by_sample = defaultdict(list)
    for r in per_sample:
        if r["split"] == split:
            by_sample[r["sample_id"]].append(float(r["dice"]))
    avg = {sid: float(np.mean(ds)) for sid, ds in by_sample.items()}
    high = max(avg, key=avg.get)
    low = min(avg, key=avg.get)
    return high, avg[high], low, avg[low]


def _boundary_error_frac(pred: np.ndarray, gt: np.ndarray, radius: int = 5):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    err = pred != gt
    n_err = int(err.sum())
    if n_err == 0:
        return 0.0, 0
    border = gt ^ binary_erosion(gt)
    if border.sum() == 0:
        return 0.0, n_err
    near = distance_transform_edt(~border) <= radius
    return float((err & near).sum()) / n_err, n_err


def select_boundary(index_rows, split: str = "val", min_errors: int = 1000, radius: int = 5):
    best = None
    for r in index_rows:
        if r["model_setting"] != "original" or r["split"] != split:
            continue
        res = load_result("original", split, r["sample_id"])
        pred = res["prediction_mask"].astype(bool)
        gt = load_gt(res["gt_path"])
        frac, n_err = _boundary_error_frac(pred, gt, radius)
        if n_err < min_errors:
            continue
        if best is None or frac > best[1]:
            best = (r["sample_id"], frac, n_err)
    return best


def _dice_lookup(per_sample: list[dict], split: str, sid: str) -> dict[str, float]:
    return {r["model_setting"]: float(r["dice"])
            for r in per_sample if r["split"] == split and r["sample_id"] == sid}


def plot_case(sid: str, split: str, dice_map: dict[str, float], out_path: Path) -> None:
    image_path = ROOT / "methods/C/data/raw/Kvasir-SEG/images" / f"{sid}.jpg"
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))
    gt = load_gt(str(ROOT / "methods/C/data/raw/Kvasir-SEG/masks" / f"{sid}.jpg"))

    preds, ents = [], []
    for m in MODELS:
        res = load_result(m, split, sid)
        preds.append(res["prediction_mask"].astype(bool))
        ents.append(entropy_normalized(res["probability"].astype(np.float64)))

    fig, axes = plt.subplots(3, 5, figsize=(20, 11))

    # row 0: original image + GT
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image", fontsize=10)
    axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground truth", fontsize=10)
    for j in range(2, 5):
        axes[0, j].axis("off")

    # row 1: predictions with FP/FN overlay; row 2: entropy
    for j, m in enumerate(MODELS):
        axp = axes[1, j]
        axp.imshow(preds[j], cmap="gray", vmin=0, vmax=1)
        if gt.any():
            axp.contour(gt.astype(float), levels=[0.5], colors="lime", linewidths=1.0)
        fp = (preds[j] & ~gt).astype(float)
        fn = (gt & ~preds[j]).astype(float)
        if fp.any():
            axp.contourf(fp, levels=[0.5, 1.5], colors=["red"], alpha=0.45)
        if fn.any():
            axp.contourf(fn, levels=[0.5, 1.5], colors=["blue"], alpha=0.45)
        axp.set_title(f"{dict(zip(MODELS, MODEL_LABELS))[m]}\nDice={dice_map[m]:.3f}",
                      fontsize=10)
        axp.set_xticks([])
        axp.set_yticks([])

        axe = axes[2, j]
        im = axe.imshow(ents[j], cmap="viridis", vmin=0, vmax=1)
        axe.set_title(f"{dict(zip(MODELS, MODEL_LABELS))[m]}\nentropy", fontsize=10)
        axe.set_xticks([])
        axe.set_yticks([])

    for ax in axes[0, :2]:
        ax.set_xticks([])
        ax.set_yticks([])
    # shared entropy colorbar (0..1)
    fig.colorbar(im, ax=axes[2, :], fraction=0.02, pad=0.01, label="normalized entropy")
    fig.suptitle(f"{split} / {sid}   (green=GT boundary, red=FP, blue=FN)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "summary_metrics.csv"))
    ap.add_argument("--reliability", default=str(ROOT / "reliability_curves.csv"))
    ap.add_argument("--per-sample", default=str(ROOT / "per_sample_metrics.csv"))
    ap.add_argument("--out-dir", default=str(FIG))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(Path(args.summary))
    rel = read_csv(Path(args.reliability))
    per_sample = read_csv(Path(args.per_sample))

    plot_comparisons(summary, out_dir)
    for split in ("val", "test"):
        plot_reliability(rel, split, out_dir)

    index_rows = load_index()
    high_sid, high_d, low_sid, low_d = select_high_low(per_sample, "val")
    bnd = select_boundary(index_rows, "val")
    bnd_sid, bnd_frac = bnd[0], bnd[1]

    print(f"high-Dice case : {high_sid} (mean dice {high_d:.4f})")
    print(f"low-Dice case  : {low_sid} (mean dice {low_d:.4f})")
    print(f"boundary case  : {bnd_sid} (boundary err frac {bnd_frac:.3f}, err px {bnd[2]})")

    for tag, sid in [("high_dice", high_sid), ("low_dice", low_sid), ("boundary", bnd_sid)]:
        dice_map = _dice_lookup(per_sample, "val", sid)
        plot_case(sid, "val", dice_map, out_dir / f"case_{tag}_{sid}.png")
        print(f"  wrote case_{tag}_{sid}.png")

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
