"""Pixel-level uncertainty metrics for the five-model MedSAM evaluation.

Unified interface: every model exports an original-size foreground
`probability` (NOT logits) plus a binary mask.  GT masks are JPEG and are
thresholded at >127.  All five models are scored with the same functions.

Metrics implemented here (per do.txt):
  - binary entropy H(p) = -p log p - (1-p) log(1-p)
  - SegECE over fixed 10 confidence bins shared across models
  - per-image pixel AUROC: label = error pixel, score = entropy H(p)
  - Dice / IoU / HD95 (symmetric surface distance, 95th pct, max of both
    directions, original-image pixel units) recomputed with one code path
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.load_result import load_result

EPS = 1e-12
N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


# --------------------------------------------------------------------------- #
# core metric functions (pure numpy, no I/O)
# --------------------------------------------------------------------------- #
def binary_entropy(p, eps: float = EPS) -> np.ndarray:
    """Two-class prediction entropy in nats (clipped to avoid log(0))."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def entropy_normalized(p, eps: float = EPS) -> np.ndarray:
    """Entropy divided by log(2) -> [0, 1] for a shared 5-model colormap."""
    return binary_entropy(p, eps) / np.log(2.0)


def dice_coef(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    inter = int((pred & gt).sum())
    denom = int(pred.sum()) + int(gt.sum())
    return (2.0 * inter / denom) if denom > 0 else float("nan")


def iou_coef(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return (inter / union) if union > 0 else float("nan")


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray):
    """Distance of every boundary pixel of a to the nearest boundary pixel of b."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    sa = a ^ binary_erosion(a)
    sb = b ^ binary_erosion(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return None
    dt = distance_transform_edt(~sb)
    return dt[sa]


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance (original pixel units)."""
    d_pg = _surface_distances(pred, gt)
    d_gp = _surface_distances(gt, pred)
    if d_pg is None or d_gp is None:
        return float("nan")
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def seg_ece(p: np.ndarray, gt: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error over fixed confidence bins."""
    p = np.asarray(p, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=bool).ravel()
    conf = np.maximum(p, 1.0 - p)
    acc = (p > 0.5) == gt
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = conf.size
    for b in range(n_bins):
        m = bin_idx == b
        nb = int(m.sum())
        if nb > 0:
            ece += (nb / n) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(ece)


def ece_bin_accumulate(p: np.ndarray, gt: np.ndarray, n_bins: int = N_BINS):
    """Per-bin (count, sum_conf, sum_acc) for pooling across images."""
    p = np.asarray(p, dtype=np.float64).ravel()
    gt = np.asarray(gt, dtype=bool).ravel()
    conf = np.maximum(p, 1.0 - p)
    acc = ((p > 0.5) == gt).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    count = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    sum_conf = np.bincount(bin_idx, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(bin_idx, weights=acc, minlength=n_bins)
    return count, sum_conf, sum_acc


def error_auroc(p: np.ndarray, gt: np.ndarray) -> float:
    """AUROC of entropy H(p) for discriminating error pixels.

    Returns NaN when the image has a single error label (all correct or all
    wrong), matching the do.txt rule of not substituting 0 or 1.
    """
    p = np.asarray(p, dtype=np.float64)
    gt = np.asarray(gt, dtype=bool)
    err = ((p > 0.5) != gt).astype(np.int8).ravel()
    h = binary_entropy(p).ravel()
    if err.sum() == 0 or err.sum() == err.size:
        return float("nan")
    return float(roc_auc_score(err, h))


# --------------------------------------------------------------------------- #
# per-sample helpers
# --------------------------------------------------------------------------- #
def load_gt(gt_path: str) -> np.ndarray:
    with Image.open(gt_path) as im:
        return np.asarray(im.convert("L")) > 127


def compute_sample_metrics(p: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> dict:
    p = np.asarray(p, dtype=np.float64)
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    return {
        "dice": dice_coef(pred, gt),
        "iou": iou_coef(pred, gt),
        "hd95": hd95(pred, gt),
        "seg_ece": seg_ece(p, gt),
        "error_auroc": error_auroc(p, gt),
        "n_pixels": int(p.size),
        "n_errors": int((pred != gt).sum()),
        "entropy_mean": float(entropy_normalized(p).mean()),
    }


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def read_index(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _safe_bin_mean(sum_: np.ndarray, count: np.ndarray) -> np.ndarray:
    out = np.full(count.shape, np.nan, dtype=np.float64)
    m = count > 0
    out[m] = sum_[m] / count[m]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "indices/all_samples.csv"))
    ap.add_argument("--out-per-sample", default=str(ROOT / "per_sample_metrics.csv"))
    ap.add_argument("--out-summary", default=str(ROOT / "summary_metrics.csv"))
    ap.add_argument("--out-reliability", default=str(ROOT / "reliability_curves.csv"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()

    rows = read_index(Path(args.index))
    if args.models:
        rows = [r for r in rows if r["model_setting"] in args.models]
    if args.splits:
        rows = [r for r in rows if r["split"] in args.splits]

    fieldnames = [
        "model_setting", "split", "sample_id",
        "dice", "iou", "hd95", "seg_ece", "error_auroc",
        "n_pixels", "n_errors", "entropy_mean",
    ]

    # pool per (model, split) -> {count, sum_conf, sum_acc} arrays of length N_BINS
    pool: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    per_sample: list[dict] = []
    for i, row in enumerate(rows, 1):
        res = load_result(row["model_setting"], row["split"], row["sample_id"])
        p = res["probability"].astype(np.float64)
        pred = res["prediction_mask"].astype(bool)
        gt = load_gt(res["gt_path"])

        m = compute_sample_metrics(p, pred, gt)
        m["model_setting"] = row["model_setting"]
        m["split"] = row["split"]
        m["sample_id"] = row["sample_id"]
        per_sample.append(m)

        key = (row["model_setting"], row["split"])
        count, sum_conf, sum_acc = ece_bin_accumulate(p, gt)
        if key not in pool:
            pool[key] = {"count": count, "sum_conf": sum_conf, "sum_acc": sum_acc}
        else:
            pool[key]["count"] += count
            pool[key]["sum_conf"] += sum_conf
            pool[key]["sum_acc"] += sum_acc

        if i % 300 == 0:
            print(f"  processed {i}/{len(rows)}", flush=True)

    # ---- write per-sample CSV ----
    with open(args.out_per_sample, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_sample)

    # ---- aggregate summary (per model x split) ----
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in per_sample:
        groups[(m["model_setting"], m["split"])].append(m)

    summary_rows = []
    for key in sorted(groups):
        ms, split = key
        g = groups[key]
        n = len(g)
        dice = np.array([x["dice"] for x in g])
        iou = np.array([x["iou"] for x in g])
        hd = np.array([x["hd95"] for x in g])
        hd_valid = hd[~np.isnan(hd)]
        auroc = np.array([x["error_auroc"] for x in g])
        auroc_valid = auroc[~np.isnan(auroc)]

        pc = pool[key]
        count = pc["count"]
        total = int(count.sum())
        mean_conf = _safe_bin_mean(pc["sum_conf"], count)
        mean_acc = _safe_bin_mean(pc["sum_acc"], count)
        weights = count / total
        pooled_ece = float(np.nansum(weights * np.abs(mean_acc - mean_conf)))

        summary_rows.append({
            "model_setting": ms,
            "split": split,
            "n_samples": n,
            "dice_mean": float(np.mean(dice)),
            "dice_std": float(np.std(dice)),
            "iou_mean": float(np.mean(iou)),
            "hd95_mean": float(np.mean(hd_valid)) if hd_valid.size else float("nan"),
            "hd95_valid": int(hd_valid.size),
            "seg_ece_per_image_mean": float(np.mean([x["seg_ece"] for x in g])),
            "seg_ece_pooled": pooled_ece,
            "auroc_mean": float(np.mean(auroc_valid)) if auroc_valid.size else float("nan"),
            "auroc_valid": int(auroc_valid.size),
        })

    summary_fields = list(summary_rows[0].keys())
    with open(args.out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # ---- write reliability-curve bins (pooled) ----
    rel_rows = []
    for key in sorted(pool):
        ms, split = key
        pc = pool[key]
        count = pc["count"]
        mean_conf = _safe_bin_mean(pc["sum_conf"], count)
        mean_acc = _safe_bin_mean(pc["sum_acc"], count)
        for b in range(N_BINS):
            rel_rows.append({
                "model_setting": ms,
                "split": split,
                "bin": b,
                "edge_low": float(BIN_EDGES[b]),
                "edge_high": float(BIN_EDGES[b + 1]),
                "n_pixels": int(count[b]),
                "mean_conf": float(mean_conf[b]) if count[b] > 0 else "",
                "mean_acc": float(mean_acc[b]) if count[b] > 0 else "",
            })

    rel_fields = [
        "model_setting", "split", "bin", "edge_low", "edge_high",
        "n_pixels", "mean_conf", "mean_acc",
    ]
    with open(args.out_reliability, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rel_fields)
        w.writeheader()
        w.writerows(rel_rows)

    print(f"\nDone. {len(per_sample)} samples -> {args.out_per_sample}")
    print(f"summary -> {args.out_summary}")
    print(f"reliability -> {args.out_reliability}")


if __name__ == "__main__":
    main()
