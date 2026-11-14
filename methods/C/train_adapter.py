"""Train bottleneck adapters on the fixed MedSAM dataset split."""

import argparse
import csv
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapter_model import (
    adapter_parameters,
    adapter_state_dict,
    build_adapter_model,
    set_gradient_checkpointing,
)
from adapter_numerics import (
    NumericalError,
    amp_dtype_from_config,
    nonfinite_tensor_names,
    optimizer_tensors,
    require_finite,
    require_finite_tensors,
    require_valid_scale,
    validate_checkpoint,
)
from baseline_inference import segmentation_metrics
from medsam_dataset import create_dataloader


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_bce_loss(logits, target, smooth):
    target = F.interpolate(
        target.float(), size=logits.shape[-2:], mode="nearest"
    )
    logits_float = logits.float()
    target_flat = target.flatten(start_dim=1)
    probabilities = torch.sigmoid(logits_float).flatten(start_dim=1)
    intersection = (probabilities * target_flat).sum(dim=1)
    denominator = probabilities.sum(dim=1) + target_flat.sum(dim=1)
    soft_dice_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
    dice_loss = soft_dice_loss.mean()
    bce_loss = F.binary_cross_entropy_with_logits(logits_float, target)
    return 0.5 * dice_loss + 0.5 * bce_loss, dice_loss.detach(), bce_loss.detach()


def forward_logits(model, images, boxes):
    embeddings = model.image_encoder(images)
    boxes = boxes.to(dtype=torch.float32, device=images.device)
    if boxes.ndim == 2:
        boxes = boxes[:, None, :]
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
        image_pe = model.prompt_encoder.get_dense_pe()
    logits, _ = model.mask_decoder(
        image_embeddings=embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return logits


def resize_probabilities(logits, size):
    return F.interpolate(
        torch.sigmoid(logits.float()),
        size=tuple(size),
        mode="bilinear",
        align_corners=False,
    )


def save_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log_numerical_event(path, event):
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def save_checkpoint(path, model, optimizer, scaler, epoch, best_dice, config):
    payload = {
        "format_version": 1,
        "model_setting": "adapter",
        "adapter_bottleneck_dimension": config["bottleneck_dimension"],
        "adapter_state": adapter_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val_dice": best_dice,
        "config": config,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    validate_checkpoint(payload, training=True)
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(path, model, optimizer, scaler, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_setting") != "adapter":
        raise ValueError(f"Not an Adapter training checkpoint: {path}")
    if payload["adapter_bottleneck_dimension"] != model.image_encoder.blocks[0].attention_adapter.down.out_features:
        raise ValueError("Resume checkpoint bottleneck dimension differs from config")
    validate_checkpoint(payload, training=True)
    from adapter_model import load_adapter_state_dict

    load_adapter_state_dict(model, payload["adapter_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload.get("scaler_state", {}))
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    # epoch stores the count of completed epochs; range() uses a zero-based index.
    return int(payload["epoch"]), float(payload["best_val_dice"])


def preserve_resume_best(resume_path, checkpoint_dir, completed_epochs, best_val_dice):
    source = Path(resume_path).parent / "best.pt"
    destination = Path(checkpoint_dir) / "best.pt"
    candidate = destination if destination.exists() else source
    if not candidate.exists():
        raise ValueError("Resume requires the corresponding best.pt alongside the checkpoint")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, training=True)
    if int(payload["epoch"]) > completed_epochs or float(payload["best_val_dice"]) != best_val_dice:
        raise ValueError("Existing best.pt does not match this resume checkpoint; use fresh run directories")
    if not destination.exists():
        shutil.copy2(candidate, destination)
    return payload


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled, smooth, max_samples=None,
                    epoch=None, event_path=None, amp_dtype=torch.float16):
    model.eval()  # frozen base stays in evaluation mode; adapters have no train/eval layers
    loss_values, dice_values, bce_values, metric_dice = [], [], [], []
    trainable = adapter_parameters(model)
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        context = {"phase": "train", "epoch": epoch, "batch": index + 1,
                   "sample_id": str(batch["sample_id"][0])}
        images = batch["image"].to(device, non_blocking=True)
        boxes = batch["bbox_1024"].to(device, non_blocking=True)
        targets = batch["gt_mask_1024"].to(device, non_blocking=True).unsqueeze(1).float()
        require_finite(images, "input images", **context)
        require_finite(boxes, "input boxes", **context)
        require_finite(targets, "target masks", **context)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = forward_logits(model, images, boxes)
        require_finite(logits, "decoder logits", **context)
        loss, dice_loss, bce_loss = dice_bce_loss(logits, targets, smooth)
        require_finite(loss, "training loss", **context)
        scale_before = scaler.get_scale()
        require_valid_scale(scale_before, **context)
        scaler.scale(loss).backward()
        if index == 0:
            if not any(p.grad is not None for p in trainable.values()):
                raise RuntimeError("No Adapter parameter received a gradient")
            if any(p.grad is not None for _, p in frozen):
                names = [name for name, p in frozen if p.grad is not None][:5]
                raise RuntimeError(f"Frozen parameters received gradients: {names}")
        scaler.unscale_(optimizer)
        gradients = {name: p.grad for name, p in trainable.items() if p.grad is not None}
        invalid_gradients = nonfinite_tensor_names(gradients)
        if invalid_gradients and not scaler.is_enabled():
            raise NumericalError("Nonfinite gradients without AMP; optimizer step cancelled",
                                 invalid_tensors=invalid_gradients[:12], **context)
        # GradScaler normally skips an overflowing step and reduces its scale.
        # Keep that behavior, but record it and never let scale underflow silently.
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        require_valid_scale(scale_after, **context)
        require_finite_tensors(trainable, "Adapter parameters after optimizer step", **context)
        require_finite_tensors(optimizer_tensors(optimizer.state_dict()),
                              "optimizer state after step", **context)
        event = {**context, "loss": float(loss.detach()), "scale_before": scale_before,
                 "scale_after": scale_after, "update_skipped": bool(invalid_gradients)}
        if invalid_gradients or index == 0 or (index + 1) % 25 == 0:
            log_numerical_event(event_path, event)
        if invalid_gradients:
            print(f"AMP skipped overflowing gradients: sample={context['sample_id']} "
                  f"scale={scale_before:g}->{scale_after:g}", flush=True)

        heights, widths = batch["original_size"][0]
        probabilities = resize_probabilities(logits.detach(), (int(heights), int(widths)))
        require_finite(probabilities, "training probabilities", **context)
        prediction = probabilities[0, 0].cpu().numpy() > 0.5
        target_original = batch["gt_mask"][0].numpy()
        sample_dice, _ = segmentation_metrics(prediction, target_original)
        loss_values.append(float(loss.detach().cpu()))
        dice_values.append(float(dice_loss.cpu()))
        bce_values.append(float(bce_loss.cpu()))
        metric_dice.append(sample_dice)
        if index == 0:
            print(
                "First backward check passed: "
                f"trainable_tensors={sum(p.grad is not None for p in trainable.values())}/"
                f"{len(trainable)} frozen_gradients=0",
                flush=True,
            )
        if (index + 1) % 25 == 0:
            print(f"train batches {index + 1}/{len(loader)} AMP_scale={scale_after:g}", flush=True)
    if not loss_values:
        raise ValueError("Training loader produced no samples")
    return {
        "train_loss": float(np.mean(loss_values)),
        "train_dice_loss": float(np.mean(dice_values)),
        "train_bce_loss": float(np.mean(bce_values)),
        "train_dice": float(np.mean(metric_dice)),
        "samples": len(loss_values),
    }


@torch.inference_mode()
def evaluate(model, loader, device, amp_enabled, max_samples=None, epoch=None,
             amp_dtype=torch.float16):
    model.eval()
    dice_values, iou_values = [], []
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        context = {"phase": "val", "epoch": epoch, "batch": index + 1,
                   "sample_id": str(batch["sample_id"][0])}
        images = batch["image"].to(device, non_blocking=True)
        boxes = batch["bbox_1024"].to(device, non_blocking=True)
        require_finite(images, "validation input images", **context)
        require_finite(boxes, "validation input boxes", **context)
        height, width = (int(v) for v in batch["original_size"][0])
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = forward_logits(model, images, boxes)
        require_finite(logits, "validation decoder logits", **context)
        probabilities = resize_probabilities(logits, (height, width))
        require_finite(probabilities, "validation probabilities", **context)
        probability = probabilities[0, 0].cpu().numpy()
        prediction = probability > 0.5
        target = batch["gt_mask"][0].numpy()
        dice, iou = segmentation_metrics(prediction, target)
        dice_values.append(dice)
        iou_values.append(iou)
        if (index + 1) % 25 == 0:
            print(f"val samples {index + 1}/{len(loader)}", flush=True)
    if not dice_values:
        raise ValueError("Validation loader produced no samples")
    return {
        "val_dice_mean": float(np.mean(dice_values)),
        "val_dice_std": float(np.std(dice_values)),
        "val_iou_mean": float(np.mean(iou_values)),
        "val_iou_std": float(np.std(iou_values)),
        "val_samples": len(dice_values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/adapter.json")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_json(config_path)
    device_name = args.device or "auto"
    epochs = args.epochs or config["epochs"]
    if epochs < 1:
        parser.error("--epochs must be positive")
    for limit in (args.max_train_samples, args.max_val_samples):
        if limit is not None and limit < 1:
            parser.error("sample limits must be positive")
    output_dir = Path(args.output_dir or "results/adapter/train")
    checkpoint_dir = Path(args.checkpoint_dir or "checkpoints/adapter")
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = root / checkpoint_dir
    if args.resume is None:
        for directory in (output_dir, checkpoint_dir):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(
                    f"Run directory is not empty: {directory}; choose a fresh directory or --resume"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["training_seed"])
    model, device = build_adapter_model(
        root / config["checkpoint"],
        device=device_name,
        bottleneck_dimension=config["bottleneck_dimension"],
        verify_checksum=True,
    )
    set_gradient_checkpointing(model, config.get("gradient_checkpointing", False))
    parameters = adapter_parameters(model)
    trainable_count = sum(p.numel() for p in parameters.values())
    total_count = sum(p.numel() for p in model.parameters())
    if not parameters or trainable_count <= 0:
        raise RuntimeError("No trainable Adapter parameters found")
    names_path = output_dir / "trainable_parameters.txt"
    names_path.write_text(
        "\n".join(f"{name}\t{parameter.numel()}" for name, parameter in parameters.items())
        + f"\n\ntrainable_parameters={trainable_count}\ntotal_parameters={total_count}\n",
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        list(parameters.values()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    amp_dtype = amp_dtype_from_config(config, device)
    # BF16 has FP32's exponent range and does not use FP16 gradient scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    train_loader = create_dataloader(
        root,
        "train",
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        root, "val", batch_size=1, shuffle=False, num_workers=0
    )

    start_epoch, best_val_dice = 0, -float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        start_epoch, best_val_dice = load_resume(
            resume_path, model, optimizer, scaler, device
        )
        if start_epoch >= epochs:
            raise ValueError("--epochs must exceed the completed epoch count in the resume checkpoint")
        best_payload = preserve_resume_best(resume_path, checkpoint_dir, start_epoch, best_val_dice)

    metrics_path = output_dir / "epochs.csv"
    fieldnames = [
        "epoch", "train_loss", "train_dice_loss", "train_bce_loss", "train_dice",
        "train_samples", "val_dice_mean", "val_dice_std", "val_iou_mean",
        "val_iou_std", "val_samples", "epoch_seconds", "peak_gpu_memory_mib",
    ]
    existing_rows = []
    if args.resume and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        if existing_rows and int(existing_rows[-1]["epoch"]) != start_epoch:
            raise ValueError("Existing log ends at a different epoch; use fresh output/checkpoint directories")
    save_json(output_dir / "config_used.json", config)
    save_json(output_dir / "run_arguments.json", {
        "config_path": str(config_path.resolve()), "epochs": epochs,
        "max_train_samples": args.max_train_samples, "max_val_samples": args.max_val_samples,
        "resume": str(resume_path.resolve()) if args.resume else None,
        "start_epoch": start_epoch + 1, "device": str(device),
        "amp": amp_enabled, "amp_dtype": str(amp_dtype),
    })
    mode = "a" if existing_rows else "w"
    run_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"Adapter training on {device}; trainable={trainable_count:,}/{total_count:,}; "
        f"epochs={start_epoch + 1}..{epochs}; AMP={amp_enabled}; dtype={amp_dtype}; "
        f"GradScaler={scaler.is_enabled()}",
        flush=True,
    )
    if args.resume:
        previous_precision = best_payload.get("selection_precision", {
            "amp": best_payload["config"].get("amp", False),
            "amp_dtype": best_payload["config"].get("amp_dtype", "float16"),
        })
        current_precision = {"amp": amp_enabled, "amp_dtype": config.get("amp_dtype", "float16")}
        if previous_precision != current_precision:
            # Compare future epochs with the existing best under the same precision.
            from adapter_model import load_adapter_state_dict

            resumed_state = adapter_state_dict(model)
            load_adapter_state_dict(model, best_payload["adapter_state"])
            print("Revalidating saved best checkpoint under the selected precision...", flush=True)
            try:
                with torch.random.fork_rng():
                    baseline = evaluate(model, val_loader, device, amp_enabled,
                                        args.max_val_samples, epoch=best_payload["epoch"],
                                        amp_dtype=amp_dtype)
            except NumericalError as error:
                save_json(output_dir / "numerical_failure.json", error.details)
                raise
            finally:
                load_adapter_state_dict(model, resumed_state)
            best_val_dice = baseline["val_dice_mean"]
            best_payload["best_val_dice"] = best_val_dice
            best_payload["selection_precision"] = current_precision
            best_payload["selection_validation"] = baseline
            temporary = checkpoint_dir / "best.pt.tmp"
            torch.save(best_payload, temporary)
            temporary.replace(checkpoint_dir / "best.pt")
            save_json(output_dir / "resume_baseline.json", {
                "checkpoint_epoch": best_payload["epoch"],
                "previous_precision": previous_precision, "current_precision": current_precision,
                **baseline,
            })
            print(f"Saved best baseline Dice={best_val_dice:.6f}", flush=True)
    with metrics_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            print(f"Starting epoch {epoch + 1}/{epochs}", flush=True)
            try:
                train_result = train_one_epoch(
                    model, train_loader, optimizer, scaler, device, amp_enabled,
                    config["loss"]["smooth"], args.max_train_samples,
                    epoch=epoch + 1, event_path=output_dir / "numerical_events.jsonl",
                    amp_dtype=amp_dtype,
                )
                val_result = evaluate(
                    model, val_loader, device, amp_enabled, args.max_val_samples, epoch=epoch + 1,
                    amp_dtype=amp_dtype,
                )
            except NumericalError as error:
                save_json(output_dir / "numerical_failure.json", error.details)
                print("Numerical failure: stopping before writing epoch metrics/checkpoints. "
                      f"See {output_dir / 'numerical_failure.json'}", flush=True)
                raise
            peak_memory = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            )
            train_samples = train_result.pop("samples")
            row = {
                "epoch": epoch + 1,
                **train_result,
                **val_result,
                "train_samples": train_samples,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "peak_gpu_memory_mib": peak_memory,
            }
            writer.writerow(row)
            handle.flush()
            score = row["val_dice_mean"]
            is_best = score > best_val_dice  # ties keep the earlier epoch
            if is_best:
                best_val_dice = score
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, scaler,
                    epoch + 1, best_val_dice, config,
                )
            save_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, scaler,
                epoch + 1, best_val_dice, config,
            )
            print(
                f"epoch {epoch + 1}/{epochs} train_loss={row['train_loss']:.5f} "
                f"train_Dice={row['train_dice']:.4f} val_Dice={score:.4f} "
                f"time={row['epoch_seconds']:.1f}s "
                f"peak_GPU={peak_memory if peak_memory is not None else 'n/a'} MiB"
                f"{' best' if is_best else ''}",
                flush=True,
            )

    total_training_seconds = time.perf_counter() - run_started
    summary = {
        "model_setting": "adapter",
        "completed_epochs": min(epochs, start_epoch + max(0, epochs - start_epoch)),
        "best_val_dice": best_val_dice,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_ratio": trainable_count / total_count,
        "training_seconds_this_invocation": total_training_seconds,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda" else None
        ),
        "device": str(device),
        "amp": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "checkpoint_dir": str(checkpoint_dir),
        "config": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "config_used.json", config)


if __name__ == "__main__":
    main()
