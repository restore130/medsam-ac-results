"""Task A: frozen MedSAM with trainable output_hypernetworks_mlps only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from medsam_dataset import create_dataloader
from medsam_model import load_medsam, MEDSAM_VIT_B_MD5


def atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load(path):
    # Only load checkpoints produced by this project, not untrusted pickle files.
    return torch.load(path, map_location='cpu', weights_only=False)


def forward(model, batch, device, amp):
    cache_dir = Path(__file__).resolve().parent / 'image_feature_cache_v1'
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / (batch['sample_id'][0] + '.pt')
    boxes = batch['bbox_1024'].to(device)
    with torch.autocast('cuda', enabled=amp):
        with torch.no_grad():
            if cache_file.exists():
                features = torch.load(cache_file, map_location='cpu', weights_only=True).to(device)
            else:
                features = model.image_encoder(batch['image'].to(device))
                atomic_save(features.detach().cpu(), cache_file)
            sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=features, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False)
    return logits.float()


def probabilities(logits, batch):
    size = tuple(int(x) for x in batch['original_size'][0])
    return F.interpolate(logits.sigmoid(), size=size, mode='bilinear', align_corners=False)


def loss_fn(logits, target):
    target = F.interpolate(target[:, None].float(), logits.shape[-2:], mode='nearest')
    prob = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (prob * target).sum(dims) + 1e-5) / (prob.sum(dims) + target.sum(dims) + 1e-5)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()


def dice_score(prob, target):
    pred = prob > .5
    target = target.bool()
    return (2 * (pred & target).sum().double() / (pred.sum() + target.sum()).clamp_min(1)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--export', choices=['val', 'test'])
    p.add_argument('--smoke', action='store_true', help='Two samples per split; never a reportable experiment')
    a = p.parse_args()
    print('Starting Linear Probe; checking GPU and original weights...', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required by this Colab training entry. Enable GPU first.')
    device = torch.device('cuda'); amp = True
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    project = Path(a.project); out = Path(a.output)
    if a.smoke: out = out / 'smoke_only'
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_medsam(project / 'weights/medsam_vit_b.pth', 'cuda', True)
    print('Original weights loaded. Frozen image features will be cached locally.', flush=True)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('mask_decoder.output_hypernetworks_mlps.')
    named = {n: v for n, v in model.named_parameters() if v.requires_grad}
    assert named
    # eval mode does not disable gradients; frozen backbone stays deterministic.
    model.eval()
    (out / 'trainable_parameters.txt').write_text('\n'.join(f'{n}: {tuple(v.shape)}' for n,v in named.items()), encoding='utf-8')
    split_hashes = {s: hashlib.sha256((project / f'data/splits/{s}.csv').read_bytes()).hexdigest() for s in ['train','val','test']}
    config = dict(seed=a.seed, lr=a.lr, weight_decay=a.weight_decay, epochs=a.epochs,
                  batch_size=1, amp=amp, method='linear_probe', head='output_hypernetworks_mlps',
                  loss='BCEWithLogits + soft Dice, equal weights, smooth=1e-5',
                  checkpoint_md5=MEDSAM_VIT_B_MD5, split_sha256=split_hashes, smoke=a.smoke)
    if (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        if old != config: raise ValueError('Config differs from existing run. Use a new output directory.')
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    loader = lambda split, shuffle=False: create_dataloader(project, split, batch_size=1, shuffle=shuffle, num_workers=0)

    if a.export:
        ck = load(out / 'best.pth')
        model.mask_decoder.output_hypernetworks_mlps.load_state_dict(ck['head'])
        dest = out / a.export; dest.mkdir(exist_ok=True)
        captured = {}
        hook = model.mask_decoder.output_upscaling.register_forward_hook(lambda m,i,o: captured.update(embedding=o.detach()))
        rows=[]; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                for i,b in enumerate(loader(a.export)):
                    if a.smoke and i >= 2: break
                    logits=forward(model,b,device,amp); prob=probabilities(logits,b)[0,0].cpu().numpy()
                    emb=captured.pop('embedding')[0].float().cpu().numpy()
                    assert emb.ndim == 3 and emb.shape[0] == 32
                    sid=b['sample_id'][0]; pred=(prob>.5).astype(np.uint8)
                    np.savez_compressed(dest/f'{sid}.npz', sample_id=sid, method='linear_probe', split=a.export,
                        probability=prob.astype(np.float32), low_res_logits=logits[0,0].cpu().numpy(),
                        mask=pred, mask_embedding=emb.astype(np.float16),
                        original_size=b['original_size'][0].numpy(), bbox_1024=b['bbox_1024'][0].numpy())
                    Image.fromarray(pred*255).save(dest/f'{sid}.png')
                    rows.append(dict(sample_id=sid, dice=dice_score(torch.from_numpy(prob),b['gt_mask'][0]),
                        embedding_norm_mean=float(np.linalg.norm(emb,axis=0).mean())))
                    print(f'Export {a.export} {i+1}: {sid}', flush=True)
        finally: hook.remove()
        torch.cuda.synchronize()
        write_csv(dest/'metrics.csv', rows)
        summary=dict(sample_count=len(rows), dice_mean=float(np.mean([r['dice'] for r in rows])),
                     embedding_norm_mean=float(np.mean([r['embedding_norm_mean'] for r in rows])),
                     elapsed_seconds=time.perf_counter()-started, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                     gpu=torch.cuda.get_device_name(), embedding_storage='float16, raw, no normalization',
                     selected_epoch=ck['epoch'])
        (dest/'summary.json').write_text(json.dumps(summary,indent=2))
        return

    opt=torch.optim.AdamW(named.values(), lr=a.lr, weight_decay=a.weight_decay)
    scaler=torch.amp.GradScaler('cuda', enabled=amp)
    start=0; best=-1.; history=[]; total_seconds=0.; peak=0
    if a.resume:
        ck=load(out/'last.pth')
        model.mask_decoder.output_hypernetworks_mlps.load_state_dict(ck['head'])
        opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start=ck['epoch']; best=ck['best']; history=ck['history']; total_seconds=ck['total_seconds']; peak=ck['peak']
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif (out/'last.pth').exists():
        raise RuntimeError('Run already exists: use --resume or a new output directory.')
    torch.cuda.reset_peak_memory_stats()
    # First forward also verifies the exact required decoder feature shape.
    verified={}
    def check_embedding(m,i,o):
        assert o.ndim==4 and o.shape[1]==32, f'Unexpected embedding {o.shape}'
        verified['shape']=list(o.shape)
    hook=model.mask_decoder.output_upscaling.register_forward_hook(check_embedding)
    for epoch in range(start,a.epochs):
        tick=time.perf_counter(); losses=[]
        for i,b in enumerate(loader('train',True)):
            if a.smoke and i>=2: break
            opt.zero_grad(set_to_none=True)
            logits=forward(model,b,device,amp)
            loss=loss_fn(logits,b['gt_mask_1024'].to(device))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            if i==0:
                if not any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in named.values()):
                    raise RuntimeError('No finite nonzero head gradients')
                print('Verified output head gradients and embedding:',verified,flush=True)
            scaler.step(opt); scaler.update(); losses.append(loss.item())
            if (i+1)%50==0: print(f'Epoch {epoch+1}, batch {i+1}, loss={loss.item():.4f}',flush=True)
        scores=[]
        with torch.no_grad():
            for i,b in enumerate(loader('val')):
                if a.smoke and i>=2: break
                prob=probabilities(forward(model,b,device,amp),b)[0,0].cpu()
                scores.append(dice_score(prob,b['gt_mask'][0]))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-tick; total_seconds+=elapsed
        peak=max(peak,torch.cuda.max_memory_allocated())
        score=float(np.mean(scores)); improved=score>best
        best=max(best,score)
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_dice=score,lr=a.lr,elapsed_seconds=elapsed))
        ck=dict(epoch=epoch+1,head=model.mask_decoder.output_hypernetworks_mlps.state_dict(),
            optimizer=opt.state_dict(),scaler=scaler.state_dict(),best=best,history=history,
            total_seconds=total_seconds,peak=peak,config=config,python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if improved: atomic_save(ck,out/'best.pth')
        atomic_save(ck,out/'last.pth'); write_csv(out/'train_log.csv',history)
        print(f'Epoch {epoch+1}: val Dice={score:.5f}, best={best:.5f}, seconds={elapsed:.1f}',flush=True)
    hook.remove()
    best_ck=load(out/'best.pth')
    model.mask_decoder.output_hypernetworks_mlps.load_state_dict(best_ck['head'])
    resources=dict(trainable_parameters=sum(v.numel() for v in named.values()),
        total_parameters=sum(v.numel() for v in model.parameters()),training_seconds=total_seconds,
        peak_vram_bytes=peak,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
        best_epoch=best_ck['epoch'],best_val_dice=best)
    resources['trainable_fraction']=resources['trainable_parameters']/resources['total_parameters']
    (out/'resource_summary.json').write_text(json.dumps(resources,indent=2))
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]); axs[0].set_title('Training loss')
    axs[1].plot([r['epoch'] for r in history],[r['val_dice'] for r in history]); axs[1].set_title('Validation Dice')
    for ax in axs: ax.set_xlabel('Epoch'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=180); plt.close(fig)


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__': main()
