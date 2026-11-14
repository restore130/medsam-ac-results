"""Export original MedSAM predictions and mask decoder embeddings."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline_inference import segmentation_metrics
from medsam_dataset import create_dataloader
from medsam_model import file_md5, load_medsam, predict_from_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", default="weights/medsam_vit_b.pth")
    parser.add_argument("--output-dir", default="results/original")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")

    root = Path(__file__).resolve().parent
    checkpoint = root / args.checkpoint
    run_dir = root / args.output_dir / args.split
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {run_dir}")

    model, device = load_medsam(checkpoint, args.device, verify_checksum=True)
    model.requires_grad_(False)
    loader = create_dataloader(root, args.split, batch_size=1)
    run_dir.mkdir(parents=True, exist_ok=True)
    captured = []
    hook = model.mask_decoder.output_upscaling.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach())
    )
    rows = []
    start = time.perf_counter()
    try:
        for index, batch in enumerate(loader):
            if args.max_samples is not None and index >= args.max_samples:
                break
            sample_start = time.perf_counter()
            sample_id = batch["sample_id"][0]
            image = batch["image"].to(device)
            box = batch["bbox_1024"].to(device)
            height, width = (int(v) for v in batch["original_size"][0])
            captured.clear()
            masks, probabilities, _ = predict_from_boxes(
                model, image, box, (height, width), use_amp=args.amp
            )
            if len(captured) != 1 or captured[0].ndim != 4 or captured[0].shape[:2] != (1, 32):
                raise RuntimeError("Unexpected mask decoder embedding output")
            probability = probabilities[0, 0].float().cpu().numpy()
            prediction = masks[0, 0].cpu().numpy().astype(np.uint8)
            embedding = captured[0][0].float().cpu().numpy()
            dice, iou = segmentation_metrics(prediction, batch["gt_mask"][0].numpy())
            np.savez_compressed(
                run_dir / f"{sample_id}.npz",
                sample_id=np.array(sample_id), probability=probability,
                prediction_mask=prediction, mask_embedding=embedding,
                bbox_1024=box[0].cpu().numpy().astype(np.float32),
                bbox_original=batch["bbox_original"][0].numpy().astype(np.float32),
                original_size=np.array([height, width], dtype=np.int64),
            )
            rows.append({"sample_id": sample_id, "dice": dice, "iou": iou,
                         "embedding_l2_norm": float(np.linalg.norm(embedding)),
                         "elapsed_seconds": time.perf_counter() - sample_start})
            print(f"{index + 1}: {sample_id} Dice={dice:.4f} IoU={iou:.4f}", flush=True)
    finally:
        hook.remove()

    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"model": "original_medsam_vit_b", "split": args.split,
               "sample_count": len(rows), "checkpoint": str(checkpoint.resolve()),
               "checkpoint_md5": file_md5(checkpoint), "device": str(device),
               "amp": bool(args.amp and device.type == "cuda"),
               "threshold": 0.5, "bbox_rule": "tight ground-truth box, xyxy, no perturbation",
               "elapsed_seconds": time.perf_counter() - start,
               "dice_mean": float(np.mean([row["dice"] for row in rows])),
               "iou_mean": float(np.mean([row["iou"] for row in rows]))}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
