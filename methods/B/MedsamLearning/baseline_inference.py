"""Run the official MedSAM ViT-B baseline on a fixed dataset split."""

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from medsam_dataset import create_dataloader
from medsam_model import file_md5, load_medsam, predict_from_boxes


def segmentation_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum(dtype=np.int64)
    pred_sum = prediction.sum(dtype=np.int64)
    target_sum = target.sum(dtype=np.int64)
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection) / max(pred_sum + target_sum, 1)
    iou = intersection / max(union, 1)
    return float(dice), float(iou)


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/baseline.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-features", action=argparse.BooleanOptionalAction)
    parser.add_argument("--verify-checksum", action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def choose(cli_value, config, key):
    return config[key] if cli_value is None else cli_value


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config = load_config(config_path)

    checkpoint = Path(choose(args.checkpoint, config, "checkpoint"))
    output_dir = Path(choose(args.output_dir, config, "output_dir"))
    if not checkpoint.is_absolute():
        checkpoint = project_dir / checkpoint
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    split = choose(args.split, config, "split")
    device_name = choose(args.device, config, "device")
    max_samples = choose(args.max_samples, config, "max_samples")
    num_workers = choose(args.num_workers, config, "num_workers")
    save_features = choose(args.save_features, config, "save_features")
    verify_checksum = choose(args.verify_checksum, config, "verify_checksum")
    use_amp = choose(args.amp, config, "amp")

    run_dir = output_dir / split
    masks_dir = run_dir / "masks"
    features_dir = run_dir / "features"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if save_features:
        features_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_medsam(checkpoint, device_name, verify_checksum)
    loader = create_dataloader(
        project_dir, split, batch_size=1, num_workers=num_workers
    )
    rows = []
    started = time.perf_counter()
    total = len(loader) if max_samples is None else min(len(loader), max_samples)

    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        sample_started = time.perf_counter()
        sample_id = batch["sample_id"][0]
        image = batch["image"].to(device, non_blocking=True)
        box = batch["bbox_1024"].to(device, non_blocking=True)
        height, width = (int(value) for value in batch["original_size"][0])
        masks, probabilities, embeddings = predict_from_boxes(
            model, image, box, (height, width), use_amp=use_amp
        )
        prediction = masks[0, 0].cpu().numpy().astype(np.uint8)
        target = batch["gt_mask"][0].numpy()
        dice, iou = segmentation_metrics(prediction, target)
        Image.fromarray(prediction * 255, mode="L").save(masks_dir / f"{sample_id}.png")
        if save_features:
            np.savez_compressed(
                features_dir / f"{sample_id}.npz",
                image_embedding=embeddings[0].cpu().numpy().astype(np.float16),
                bbox_1024=box[0].cpu().numpy().astype(np.float32),
                original_size=np.array([height, width], dtype=np.int64),
            )
        rows.append(
            {
                "sample_id": sample_id,
                "dice": dice,
                "iou": iou,
                "foreground_probability_mean": float(probabilities.mean().item()),
                "elapsed_seconds": time.perf_counter() - sample_started,
            }
        )
        print(
            f"[{index + 1}/{total}] {sample_id} "
            f"Dice={dice:.4f} IoU={iou:.4f}"
        )

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "sample_count": len(rows),
        "dice_mean": float(np.mean([row["dice"] for row in rows])),
        "dice_std": float(np.std([row["dice"] for row in rows])),
        "iou_mean": float(np.mean([row["iou"] for row in rows])),
        "iou_std": float(np.std([row["iou"] for row in rows])),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "checkpoint_md5": file_md5(checkpoint),
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "save_features": save_features,
        "amp": bool(use_amp and device.type == "cuda"),
        "bbox_rule": "tight ground-truth box, xyxy, no perturbation",
        "threshold": 0.5,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
