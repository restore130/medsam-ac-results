"""Train and export MedSAM LoRA using the first group's Kvasir-SEG project."""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class QueryValueLoRA(nn.Module):
    def __init__(self, qkv, rank, alpha, dropout):
        super().__init__()
        width = qkv.in_features
        if qkv.out_features != 3 * width:
            raise ValueError("Expected SAM's fused query/key/value projection")
        self.qkv = qkv
        self.down_q = nn.Linear(width, rank, bias=False)
        self.up_q = nn.Linear(rank, width, bias=False)
        self.down_v = nn.Linear(width, rank, bias=False)
        self.up_v = nn.Linear(rank, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.zeros_(self.up_q.weight)
        nn.init.zeros_(self.up_v.weight)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return torch.cat((q + self.up_q(self.down_q(self.dropout(x))) * self.scale,
                          k,
                          v + self.up_v(self.down_v(self.dropout(x))) * self.scale), dim=-1)


def forward(model, image, box, checkpoint_blocks):
    encoder = model.image_encoder
    if checkpoint_blocks:
        x = encoder.patch_embed(image)
        if encoder.pos_embed is not None:
            x = x + encoder.pos_embed
        for block in encoder.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        features = encoder.neck(x.permute(0, 3, 1, 2))
    else:
        features = encoder(image)
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=box[:, None, :], masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=features, image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return logits


def loss_fn(logits, target):
    target = F.interpolate(target[:, None].float(), logits.shape[-2:], mode="nearest")
    logits = logits.float()
    p = logits.sigmoid()
    intersection = (p * target).sum(dim=(1, 2, 3))
    volume = (p + target).sum(dim=(1, 2, 3))
    return F.binary_cross_entropy_with_logits(logits, target) + (
        1 - (2 * intersection + 1e-5) / (volume + 1e-5)
    ).mean()


def prediction(logits, batch):
    size = tuple(batch["original_size"][0].tolist())
    probs = F.interpolate(logits.float().sigmoid(), size=size, mode="bilinear",
                          align_corners=False)[0, 0].detach().cpu().numpy()
    return probs, probs > 0.5


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_weights(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def csv_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def hd95(pred, gt):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if not pred.any() or not gt.any():
        return None if pred.any() != gt.any() else 0.0
    surface_pred = pred ^ binary_erosion(pred)
    surface_gt = gt ^ binary_erosion(gt)
    # Symmetric surface HD95 in original-image pixels (4-connected contours).
    return float(max(np.percentile(distance_transform_edt(~surface_gt)[surface_pred], 95),
                     np.percentile(distance_transform_edt(~surface_pred)[surface_gt], 95)))


@torch.inference_mode()
def evaluate(model, loader, device, amp, metrics, destination=None, limit=None):
    model.eval()
    rows, captures = [], []
    handle = None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
        handle = model.mask_decoder.output_upscaling.register_forward_hook(
            lambda _module, _inputs, result: captures.append(result.detach())
        )
    try:
        for batch in loader:
            with torch.autocast(device.type, enabled=amp):
                logits = forward(model, batch["image"].to(device), batch["bbox_1024"].to(device), False)
            probs, mask = prediction(logits, batch)
            gt = batch["gt_mask"][0].numpy().astype(bool)
            sample_id = batch["sample_id"][0]
            dice, iou = metrics(mask, gt)
            row = dict(sample_id=sample_id, dice=dice, iou=iou)
            if destination:
                embedding = captures.pop().float()
                if embedding.ndim != 4 or embedding.shape[:2] != (1, 32):
                    raise RuntimeError(f"Unexpected mask embedding shape: {embedding.shape}")
                if not torch.isfinite(embedding).all() or not np.isfinite(probs).all():
                    raise ValueError(f"Non-finite output for {sample_id}")
                norms = torch.linalg.vector_norm(embedding[0], dim=0).cpu().numpy()
                row.update(hd95_px=hd95(mask, gt), embedding_norm_mean=float(norms.mean()),
                           embedding_norm_std=float(norms.std()),
                           **{f"embedding_norm_p{p}": float(np.percentile(norms, p))
                              for p in (5, 50, 95)})
                np.savez_compressed(
                    destination / f"{sample_id}.npz", probability=probs,
                    prediction=mask.astype(np.uint8), mask_embedding=embedding[0].cpu().numpy(),
                    bbox_1024=batch["bbox_1024"][0].numpy(),
                    original_size=batch["original_size"][0].numpy(),
                    sample_id=np.asarray(sample_id),
                )
                Image.fromarray(mask.astype(np.uint8) * 255).save(destination / f"{sample_id}.png")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    finally:
        if handle:
            handle.remove()
    return rows


def plot_curve(path, history):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    epochs = [r["epoch"] for r in history]
    plt.plot(epochs, [r["train_dice"] for r in history], label="Train Dice")
    plt.plot(epochs, [r["val_dice"] for r in history], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--export-only", type=Path, metavar="BEST_PT")
    parser.add_argument("--allow-formal-r16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    from baseline_inference import segmentation_metrics
    from medsam_dataset import create_dataloader
    from medsam_model import MEDSAM_VIT_B_MD5, load_medsam

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.export_only and (args.resume or args.smoke or args.stop_after_epoch):
        parser.error("--export-only requires a fresh output directory")
    checkpoint_blocks = cfg["checkpoint_blocks"] and not args.no_gradient_checkpointing
    if args.resume and args.smoke:
        parser.error("--resume and --smoke are mutually exclusive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    out = args.output
    if out.resolve() == (Path(__file__).resolve().parent / "runs" / "formal-r16"):
        if (out.parent / ".disable-auto-r16").is_file() and not args.allow_formal_r16:
            print("Automatic formal r16 training is disabled.", flush=True)
            return
    if out.exists() and not args.resume:
        raise FileExistsError(f"Choose a fresh output directory or use --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model, device = load_medsam(project / "weights" / "medsam_vit_b.pth",
                                args.device, verify_checksum=True)
    model.requires_grad_(False)
    for block in model.image_encoder.blocks:
        block.attn.qkv = QueryValueLoRA(block.attn.qkv, cfg["rank"], cfg["alpha"],
                                        cfg["dropout"]).to(device)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    parameter_count = sum(p.numel() for p in trainable.values())
    expected_count = 12 * 4 * 768 * cfg["rank"]
    if parameter_count != expected_count:
        raise RuntimeError(f"Unexpected number of LoRA parameters: {parameter_count}")
    optimizer = torch.optim.AdamW(trainable.values(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"], fused=False)
    cuda = device.type == "cuda"
    amp = cfg["amp"] and cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    environment = dict(config=cfg, torch=str(torch.__version__), device=str(device),
                       gpu=torch.cuda.get_device_name(device) if cuda else None,
                       pretrained_md5=MEDSAM_VIT_B_MD5, trainable_count=parameter_count,
                       cuda_version=torch.version.cuda,
                       hypothesis="r16 increases Dice and representation drift, and worsens SegECE relative to r8",
                       trainable_parameters={name: list(p.shape) for name, p in trainable.items()})
    save_json(out / "run_config.json", environment)

    def load_adapter(weights):
        if weights.keys() != trainable.keys():
            raise ValueError("Checkpoint adapter does not match configuration")
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(weights[name])

    def adapter():
        return {name: p.detach().cpu().clone() for name, p in trainable.items()}

    history, best = [], None
    if args.export_only:
        best = torch.load(args.export_only, map_location="cpu", weights_only=True)
        if best["config"] != cfg or best["pretrained_md5"] != MEDSAM_VIT_B_MD5:
            raise ValueError("Export weights do not match the configuration or base model")
    if args.resume:
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if state["config"] != cfg:
            raise ValueError("Resume requires the original configuration")
        load_adapter(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best = state["history"], state["best"]
    train = create_dataloader(project, "train", batch_size=cfg["batch"])
    val = create_dataloader(project, "val", batch_size=cfg["batch"])
    best_score = best["dice"] if best else -1.0
    print(f"rank={cfg['rank']} device={device} amp={amp} "
          f"gradient_checkpointing={checkpoint_blocks} trainable={parameter_count}", flush=True)

    last_epoch = 0 if args.export_only else 1 if args.smoke else cfg["epochs"]
    if args.stop_after_epoch is not None:
        last_epoch = min(last_epoch, args.stop_after_epoch)
    for epoch in range(len(history) + 1, last_epoch + 1):
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.train()
        rng = torch.Generator().manual_seed(cfg["seed"] + epoch)
        indices = torch.randperm(len(train.dataset), generator=rng).tolist()
        limit = 4 if args.smoke else len(indices)
        shuffled = torch.utils.data.DataLoader(train.dataset, batch_size=cfg["batch"],
                                               sampler=indices[:limit],
                                               pin_memory=cuda)
        losses, dices = [], []
        for step, batch in enumerate(shuffled, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = forward(model, batch["image"].to(device), batch["bbox_1024"].to(device),
                                 checkpoint_blocks)
            loss = loss_fn(logits, batch["gt_mask_1024"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
            scaler.scale(loss).backward()
            check_update = args.smoke or (epoch == 1 and step == 1)
            if check_update:
                if any(p.grad is None for p in trainable.values()):
                    raise RuntimeError("A LoRA branch did not receive gradients")
                first_up = next(p for n, p in trainable.items() if ".up_q." in n)
                before = first_up.detach().clone()
            scaler.step(optimizer)
            scaler.update()
            if check_update and torch.equal(first_up, before) and not amp:
                raise RuntimeError("The optimizer did not update LoRA")
            if args.smoke and not torch.equal(first_up, before):
                print(f"LoRA update verified step={step}", flush=True)
            losses.append(loss.item())
            _, mask = prediction(logits, batch)
            dices.append(segmentation_metrics(mask, batch["gt_mask"][0].numpy())[0])
            if step == 1 or step % 25 == 0:
                print(f"epoch={epoch} step={step}/{limit} loss={losses[-1]:.5f} "
                      f"elapsed_s={time.perf_counter()-start:.1f}", flush=True)
        if args.smoke and not any(torch.count_nonzero(p).item() for n, p in trainable.items()
                                  if ".up_q." in n or ".up_v." in n):
            raise RuntimeError("No LoRA up-projection was updated during smoke run")
        train_s = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if cuda else None
        scores = evaluate(model, val, device, amp, segmentation_metrics, limit=1 if args.smoke else None)
        val_dice = float(np.mean([row["dice"] for row in scores]))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                            train_dice=float(np.mean(dices)), val_dice=val_dice,
                            train_seconds=train_s, val_seconds=time.perf_counter()-start-train_s,
                            train_peak_cuda_bytes=peak))
        if val_dice > best_score:
            best_score = val_dice
            best = dict(epoch=epoch, dice=val_dice, adapter=adapter(), config=cfg,
                        pretrained_md5=MEDSAM_VIT_B_MD5)
        save_weights(out / "last.pt", dict(config=cfg, adapter=adapter(),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     history=history, best=best))
        save_weights(out / "best.pt", best)
        csv_rows(out / "history.csv", history)
        plot_curve(out / "dice.png", history)
        print(f"epoch={epoch} val_dice={val_dice:.6f} peak_bytes={peak}", flush=True)

    if args.stop_after_epoch is not None and last_epoch < cfg["epochs"]:
        return

    if args.smoke:
        batch = next(iter(val))
        image = batch["image"].to(device)
        box = batch["bbox_1024"].to(device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=amp):
            before_reload = forward(model, image, box, False).float().cpu()
        with torch.no_grad():
            for parameter in trainable.values():
                parameter.zero_()
    restored = best if args.export_only else torch.load(
        out / "best.pt", map_location="cpu", weights_only=True)
    load_adapter(restored["adapter"])
    if args.smoke:
        with torch.inference_mode(), torch.autocast(device.type, enabled=amp):
            after_reload = forward(model, image, box, False).float().cpu()
        torch.testing.assert_close(before_reload, after_reload, rtol=1e-3, atol=1e-3)
    for split in (["val"] if args.smoke else ["val", "test"]):
        loader = create_dataloader(project, split, batch_size=cfg["batch"])
        started = time.perf_counter()
        dest = out / split
        rows = evaluate(model, loader, device, amp, segmentation_metrics, destination=dest,
                        limit=1 if args.smoke else None)
        csv_rows(dest / "metrics.csv", rows)
        save_json(dest / "summary.json", dict(
            samples=len(rows), elapsed_seconds=time.perf_counter()-started,
            epoch=restored["epoch"], mean_dice=float(np.mean([r["dice"] for r in rows])),
            mean_iou=float(np.mean([r["iou"] for r in rows])),
            hd95_valid_samples=sum(r["hd95_px"] is not None for r in rows),
            mean_hd95_px=float(np.mean([r["hd95_px"] for r in rows if r["hd95_px"] is not None]))
            if any(r["hd95_px"] is not None for r in rows) else None,
        ))
    print("Smoke passed" if args.smoke else "Training and export complete", flush=True)


if __name__ == "__main__":
    main()
