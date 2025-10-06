#!/usr/bin/env python3
import argparse, os, re, math
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet.registry import MODELS, DATASETS
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader

# -------------------- utils: data & model --------------------

def get_split_cfg(cfg, split):
    if split == 'train' and hasattr(cfg, 'train_dataloader'):
        return cfg.train_dataloader
    if split == 'val' and hasattr(cfg, 'val_dataloader'):
        return cfg.val_dataloader
    if split == 'test' and hasattr(cfg, 'test_dataloader'):
        return cfg.test_dataloader
    # Fallback: prefer val, then train
    return getattr(cfg, 'val_dataloader', getattr(cfg, 'train_dataloader', None))

def build_loader(dl_cfg, limit=None):
    ds = DATASETS.build(dl_cfg['dataset'])
    if limit:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=dl_cfg.get('batch_size', 2),
        num_workers=dl_cfg.get('num_workers', 4),
        collate_fn=pseudo_collate,      # returns list[dict]
        shuffle=False,
        pin_memory=True,
    )
    return loader, ds

def get_preproc_params(cfg):
    prep = cfg.model.get('data_preprocessor', {})
    mean = prep.get('mean', [123.675, 116.28, 103.53])
    std  = prep.get('std',  [58.395, 57.12, 57.375])
    bgr  = prep.get('bgr_to_rgb', True)
    return mean, std, bgr

def to_float_tensor(batch, device, mean, std, bgr_to_rgb):
    """Handle pseudo_collate (list[dict]) and dict forms."""
    if isinstance(batch, list):
        imgs = torch.stack([
            (s['inputs'][0] if isinstance(s['inputs'], (list, tuple)) else s['inputs'])
            for s in batch
        ], dim=0)
    else:
        x = batch['inputs']
        if isinstance(x, (list, tuple)):
            imgs = torch.stack(x, dim=0)
        else:
            imgs = x.unsqueeze(0)

    imgs = imgs.to(device, non_blocking=True).float()
    if bgr_to_rgb:
        imgs = imgs[:, [2,1,0], :, :]
    mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1,3,1,1)
    std  = torch.tensor(std,  device=device, dtype=torch.float32).view(1,3,1,1)
    imgs = (imgs - mean) / std
    return imgs

def get_backbone(model):
    if hasattr(model, 'backbone'):
        return model.backbone
    raise AttributeError('Model has no .backbone')

def get_patch_multiple(backbone: nn.Module):
    # Try common attributes
    if hasattr(backbone, 'patch_size'):
        p = backbone.patch_size
        return p if isinstance(p, int) else int(p[0])
    if hasattr(backbone, 'patch_embed'):
        pe = backbone.patch_embed
        if hasattr(pe, 'proj') and hasattr(pe.proj, 'kernel_size'):
            ks = pe.proj.kernel_size
            return ks[0] if isinstance(ks, (list, tuple)) else int(ks)
        if hasattr(pe, 'patch_size'):
            p = pe.patch_size
            return p if isinstance(p, int) else int(p[0])
    # Fallback (no constraint)
    return 1

def pad_to_multiple(imgs: torch.Tensor, multiple: int):
    if multiple <= 1:
        return imgs
    _, _, H, W = imgs.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return imgs
    return F.pad(imgs, (0, pad_w, 0, pad_h))

def flatten_feat(t: torch.Tensor):
    with torch.no_grad():
        if t.dim() == 4:     # BCHW -> (B,C)
            return t.mean(dim=(2,3))
        if t.dim() == 3:     # BND -> (B,D)
            return t.mean(dim=1)
        return t.view(t.size(0), -1)

# -------------------- hook discovery --------------------

def enumerate_hook_modules(backbone: nn.Module):
    """
    Prefer per-block outputs:
    - ViTs commonly expose backbone.blocks (ModuleList)
    - Some CNNs have 'stages' or 'layers'
    Fallback: hook the backbone itself (single layer).
    Returns list[(name, module)]
    """
    # ViT-style
    if hasattr(backbone, 'blocks') and isinstance(backbone.blocks, (nn.ModuleList, list)):
        return [ (f'block_{i}', m) for i, m in enumerate(backbone.blocks) ]
    # Swin/ResNet-ish
    for attr in ['stages', 'layers', 'res_layers']:
        if hasattr(backbone, attr) and isinstance(getattr(backbone, attr), (nn.ModuleList, list, tuple)):
            mods = []
            for i, stage in enumerate(getattr(backbone, attr)):
                name = f'{attr}_{i}'
                mods.append((name, stage))
            if mods:
                return mods
    # Fallback
    return [('backbone', backbone)]

class FeatCollector:
    def __init__(self):
        self.buffers = defaultdict(list)
        self.handles = []

    def hook_for(self, key):
        def fn(_m, _inp, out):
            try:
                feat = flatten_feat(out.detach())
                self.buffers[key].append(feat.cpu())
            except Exception:
                pass
        return fn

    def register(self, modules):
        for name, mod in modules:
            h = mod.register_forward_hook(self.hook_for(name))
            self.handles.append(h)

    def clear(self):
        self.buffers.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stacked(self):
        """Return dict: name -> (N, D)"""
        out = {}
        for k, vs in self.buffers.items():
            out[k] = torch.cat(vs, dim=0)
        return out

# -------------------- CKA (linear) --------------------

def center(X: torch.Tensor):
    # X: (N,D), center columns
    X = X - X.mean(dim=0, keepdim=True)
    return X

def gram_linear(X: torch.Tensor):
    # (N,D) -> (N,N)
    return X @ X.T

def cka_linear(X: torch.Tensor, Y: torch.Tensor):
    """
    Centered Kernel Alignment with linear kernel.
    X, Y: (N, D1), (N, D2)
    """
    Xc = center(X)
    Yc = center(Y)
    K = gram_linear(Xc)
    L = gram_linear(Yc)
    hsic = (K * L).sum()
    norm_x = (K * K).sum().sqrt()
    norm_y = (L * L).sum().sqrt()
    if norm_x.item() == 0.0 or norm_y.item() == 0.0:
        return torch.tensor(0.0)
    return hsic / (norm_x * norm_y)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description='Layer-wise CKA between two MMDet backbones')
    ap.add_argument('--configA', required=True)
    ap.add_argument('--ckptA',   required=True)
    ap.add_argument('--configB', required=True)
    ap.add_argument('--ckptB',   required=True)

    ap.add_argument('--split', choices=['train','val','test'], default='val')
    ap.add_argument('--limit', type=int, default=0, help='limit number of samples (0=all in split)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-batches', type=int, default=10, help='number of batches to use (0=all)')
    ap.add_argument('--out', default='', help='optional path prefix to save CKA matrix (.npy) and heatmap (.png)')
    args = ap.parse_args()

    # --- Configs & loaders ---
    cfgA = Config.fromfile(args.configA)
    cfgB = Config.fromfile(args.configB)

    dlA = get_split_cfg(cfgA, args.split)
    dlB = get_split_cfg(cfgB, args.split)
    if dlA is None or dlB is None:
        raise RuntimeError('Could not find dataloader config for the requested split.')

    loaderA, dsA = build_loader(dlA, limit=args.limit)
    loaderB, dsB = build_loader(dlB, limit=args.limit)
    if len(dsA) != len(dsB):
        print(f"[WARN] Dataset sizes differ ({len(dsA)} vs {len(dsB)}). Will iterate over the shorter.")

    # --- Build models ---
    modelA = MODELS.build(cfgA.model)
    modelB = MODELS.build(cfgB.model)
    _ = load_checkpoint(modelA, args.ckptA, map_location='cpu')
    _ = load_checkpoint(modelB, args.ckptB, map_location='cpu')
    device = torch.device(args.device)
    modelA.to(device).eval()
    modelB.to(device).eval()

    bbA = get_backbone(modelA)
    bbB = get_backbone(modelB)

    meanA, stdA, bgrA = get_preproc_params(cfgA)
    meanB, stdB, bgrB = get_preproc_params(cfgB)

    pA = get_patch_multiple(bbA)
    pB = get_patch_multiple(bbB)

    # --- Hooks ---
    modsA = enumerate_hook_modules(bbA)
    modsB = enumerate_hook_modules(bbB)
    colA = FeatCollector()
    colB = FeatCollector()
    colA.register(modsA)
    colB.register(modsB)

    # --- Iterate (paired) ---
    itA = iter(loaderA)
    itB = iter(loaderB)

    n_batches = 0
    with torch.no_grad():
        while True:
            if args.max_batches and n_batches >= args.max_batches:
                break
            try:
                batchA = next(itA)
                batchB = next(itB)
            except StopIteration:
                break

            imgsA = to_float_tensor(batchA, device, meanA, stdA, bgrA)
            imgsB = to_float_tensor(batchB, device, meanB, stdB, bgrB)

            imgsA = pad_to_multiple(imgsA, pA)
            imgsB = pad_to_multiple(imgsB, pB)

            # forward to trigger hooks
            _ = bbA(imgsA)
            _ = bbB(imgsB)

            n_batches += 1

    # --- Stack features ---
    featsA = colA.stacked()  # dict name -> (N, DA)
    featsB = colB.stacked()  # dict name -> (N, DB)

    colA.close()
    colB.close()

    if not featsA or not featsB:
        raise RuntimeError("No features captured. Check hook discovery in enumerate_hook_modules().")

    # Align sample counts across A/B (truncate to min N)
    Ns = []
    for k in featsA:
        Ns.append(featsA[k].shape[0])
    for k in featsB:
        Ns.append(featsB[k].shape[0])
    Nmin = min(Ns)

    for k in featsA:
        featsA[k] = featsA[k][:Nmin]
    for k in featsB:
        featsB[k] = featsB[k][:Nmin]

    # --- Build layer lists with stable ordering ---
    namesA = sorted(featsA.keys(), key=lambda s: (len(s), s))
    namesB = sorted(featsB.keys(), key=lambda s: (len(s), s))

    Xs = [featsA[n].to(device) for n in namesA]
    Ys = [featsB[n].to(device) for n in namesB]

    # --- Compute LA x LB CKA matrix ---
    LA, LB = len(Xs), len(Ys)
    C = torch.zeros((LA, LB), dtype=torch.float32)
    for i in range(LA):
        for j in range(LB):
            C[i, j] = cka_linear(Xs[i], Ys[j]).detach().cpu()

    C_np = C.numpy()

    # --- Save outputs ---
    if args.out:
        out_npy = args.out if args.out.endswith('.npy') else (args.out + '.npy')
        os.makedirs(os.path.dirname(out_npy) or '.', exist_ok=True)
        np.save(out_npy, {'cka': C_np, 'layersA': namesA, 'layersB': namesB})
        # optional quick heatmap (no seaborn dependency)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(max(6, LB*0.5), max(4, LA*0.5)))
            plt.imshow(C_np, vmin=0, vmax=1, aspect='auto')
            plt.colorbar(label='CKA (linear)')
            plt.yticks(range(LA), namesA, fontsize=16)
            plt.xticks(range(LB), namesB, fontsize=16, rotation=90)
            plt.tight_layout()
            png_path = out_npy[:-4] + '.png'
            plt.savefig(png_path, dpi=200)
            plt.close()
            print(f"Saved CKA matrix to {out_npy} and heatmap to {png_path}")
        except Exception as e:
            print(f"Saved CKA matrix to {out_npy}. (Heatmap skipped: {e})")
    else:
        # Print a small summary if no output path given
        print("CKA (linear) matrix shape:", C_np.shape)
        print("Top-left 3x3 preview:\n", np.round(C_np[:3,:3], 3))

if __name__ == '__main__':
    main()
