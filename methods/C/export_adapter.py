"""Run Adapter MedSAM inference and export probabilities and mask embeddings."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapter_model import build_adapter_model, load_adapter_checkpoint
from adapter_numerics import amp_dtype_from_config, require_finite
from baseline_inference import segmentation_metrics
from medsam_dataset import create_dataloader
from medsam_model import file_md5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/adapter.json")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", default="checkpoints/adapter/best.pt")
    parser.add_argument("--output-dir", default="results/adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_checkpoint = root / config["checkpoint"]
    adapter_checkpoint = Path(args.checkpoint)
    if not adapter_checkpoint.is_absolute():
        adapter_checkpoint = root / adapter_checkpoint
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    run_dir = output_root / args.split
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {run_dir}")

    model, device = build_adapter_model(
        base_checkpoint,
        device=args.device,
        bottleneck_dimension=config["bottleneck_dimension"],
        verify_checksum=True,
    )
    payload = load_adapter_checkpoint(model, adapter_checkpoint, map_location="cpu")
    model.eval()
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    amp_dtype = amp_dtype_from_config(config, device)
    loader = create_dataloader(root, args.split, batch_size=1, shuffle=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    captured = []
    hook = model.mask_decoder.output_upscaling.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach())
    )
    rows = []
    started = time.perf_counter()
    try:
        for index, batch in enumerate(loader):
            if args.max_samples is not None and index >= args.max_samples:
                break
            sample_started = time.perf_counter()
            sample_id = batch["sample_id"][0]
            image = batch["image"].to(device, non_blocking=True)
            box = batch["bbox_1024"].to(device, non_blocking=True)
            height, width = (int(v) for v in batch["original_size"][0])
            captured.clear()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                image_embedding = model.image_encoder(image)
                boxes = box.float()
                if boxes.ndim == 2:
                    boxes = boxes[:, None, :]
                sparse, dense = model.prompt_encoder(
                    points=None, boxes=boxes, masks=None
                )
                logits, _ = model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                probabilities = torch.nn.functional.interpolate(
                    torch.sigmoid(logits.float()),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            require_finite(logits, "export decoder logits", sample_id=sample_id)
            require_finite(probabilities, "export probabilities", sample_id=sample_id)
            if len(captured) != 1 or captured[0].shape[:2] != (1, 32):
                raise RuntimeError("Unexpected mask decoder embedding output")
            require_finite(captured[0], "export mask embedding", sample_id=sample_id)
            probability = probabilities[0, 0].cpu().numpy()
            prediction = (probability > 0.5).astype(np.uint8)
            embedding = captured[0][0].float().cpu().numpy()
            dice, iou = segmentation_metrics(
                prediction, batch["gt_mask"][0].numpy()
            )
            np.savez_compressed(
                run_dir / f"{sample_id}.npz",
                sample_id=np.array(sample_id),
                probability=probability.astype(np.float32, copy=False),
                prediction_mask=prediction,
                mask_embedding=embedding,
                bbox_1024=box[0].cpu().numpy().astype(np.float32),
                bbox_original=batch["bbox_original"][0].numpy().astype(np.float32),
                original_size=np.array([height, width], dtype=np.int64),
            )
            rows.append({
                "sample_id": sample_id,
                "dice": dice,
                "iou": iou,
                "embedding_l2_norm": float(np.linalg.norm(embedding)),
                "elapsed_seconds": time.perf_counter() - sample_started,
            })
            print(
                f"{index + 1}: {sample_id} Dice={dice:.4f} IoU={iou:.4f}",
                flush=True,
            )
    finally:
        hook.remove()

    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model_setting": "adapter",
        "split": args.split,
        "sample_count": len(rows),
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_md5": file_md5(base_checkpoint),
        "adapter_checkpoint": str(adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256(adapter_checkpoint),
        "trained_epoch": payload.get("epoch"),
        "best_val_dice": payload.get("best_val_dice"),
        "device": str(device),
        "amp": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "threshold": 0.5,
        "bbox_rule": "tight ground-truth box, xyxy, no perturbation",
        "elapsed_seconds": time.perf_counter() - started,
        "dice_mean": float(np.mean([row["dice"] for row in rows])),
        "dice_std": float(np.std([row["dice"] for row in rows])),
        "iou_mean": float(np.mean([row["iou"] for row in rows])),
        "iou_std": float(np.std([row["iou"] for row in rows])),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
