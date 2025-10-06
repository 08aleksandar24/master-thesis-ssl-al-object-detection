#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from torch.utils.data import DataLoader

from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet.registry import MODELS, DATASETS
from mmengine.runner.checkpoint import load_checkpoint


# ----------------------------- utilities -----------------------------

def pick_device(device_arg: str):
    """
    Respect CUDA_VISIBLE_DEVICES from the shell.
    - If user sets CUDA_VISIBLE_DEVICES=1 and passes --device cuda, that maps to cuda:0.
    - If they pass --device cpu, we use cpu.
    - If they pass --device cuda:0 explicitly, we'll trust it.
    """
    if device_arg.lower().startswith('cpu'):
        return torch.device('cpu')
    if device_arg.lower().startswith('cuda:'):
        return torch.device(device_arg)
    # default 'cuda' -> cuda:0 among visible devices
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def build_loader(cfg, split='val', batch_size=1):
    dl_cfg = cfg.val_dataloader if split == 'val' else cfg.train_dataloader
    ds = DATASETS.build(dl_cfg['dataset'])
    # For visualization we iterate one-by-one; robust collate.
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=dl_cfg.get('num_workers', 4),
        collate_fn=pseudo_collate,  # typically list[dict], but we'll handle dict too
        shuffle=False,
        pin_memory=True,
    )
    return loader


def build_model(cfg, ckpt_path, device):
    model = MODELS.build(cfg.model)
    _ = load_checkpoint(model, ckpt_path, map_location='cpu')
    model.to(device).eval()
    return model


def get_preproc_from_cfg(cfg, device):
    prep = cfg.model.get('data_preprocessor', {})
    mean = torch.tensor(prep.get('mean', [123.675, 116.28, 103.53]), dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(prep.get('std',  [58.395, 57.12, 57.375]),   dtype=torch.float32, device=device).view(1, 3, 1, 1)
    bgr_to_rgb = prep.get('bgr_to_rgb', True)
    return mean, std, bgr_to_rgb


def tensor_to_pil(img_tensor, mean, std, bgr_to_rgb=True):
    """
    img_tensor: (1,3,H,W) float32 normalized by (x-mean)/std.
    We invert normalization and convert to uint8 PIL.
    """
    x = img_tensor.detach().cpu() * std.cpu() + mean.cpu()
    x = x[0]  # (3,H,W)
    if bgr_to_rgb:
        x = x[[2,1,0], ...]
    x = torch.clamp(x, 0, 255).byte().permute(1, 2, 0).numpy()  # (H,W,3) uint8
    return Image.fromarray(x)


# ----------------------- attention capture hooks ----------------------

class AttnCollector:
    """
    Collects attention weight tensors (B, heads, N, N) from ViT-like modules.
    Will store them per forward pass; call clear() between images.
    """
    def __init__(self):
        self.stack = []
        self._hooks = []

    def _hook(self, module, inputs, output):
        # Try common names for attention probabilities inside modules
        # If not available, try to grab from module attributes set during forward.
        attn = None
        if hasattr(module, 'attn_probs'):   # some impls expose this
            attn = module.attn_probs
        elif hasattr(module, 'attn'):       # e.g., DINOv2 Attention stores .attn after softmax
            maybe = getattr(module, 'attn')
            if torch.is_tensor(maybe):
                attn = maybe
        elif isinstance(output, (tuple, list)):
            # sometimes the module returns (out, attn)
            for obj in output:
                if torch.is_tensor(obj) and obj.dim() == 4 and obj.shape[-1] == obj.shape[-2]:
                    attn = obj
                    break

        if attn is not None and torch.is_tensor(attn) and attn.dim() == 4:
            self.stack.append(attn.detach())

    def register(self, backbone: nn.Module):
        """
        Register hooks on known attention submodules inside a ViT backbone.
        We heuristically attach to modules named 'attn' inside blocks.
        """
        # DINOv2 / timm-style: backbone.blocks[i].attn
        for name, module in backbone.named_modules():
            # lightweight filter to avoid too many hooks
            # we want modules that are likely self-attn blocks
            if name.endswith('.attn') or name.endswith('attention') or name.endswith('self_attn'):
                h = module.register_forward_hook(self._hook)
                self._hooks.append(h)

    def clear(self):
        self.stack.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get(self):
        if len(self.stack) == 0:
            return None
        # stack per-layer attentions in forward order: list[(B, H, N, N)]
        # we’ll assume B=1 for rollout/visualization
        return self.stack


def attention_rollout(attn_stack, add_identity=True, head_fuse='mean'):
    """
    attn_stack: list of (1, heads, N, N) tensors from input->output order.
    Returns a single rollout vector over tokens: (N,) normalized to [0,1].
    """
    with torch.no_grad():
        # fuse heads
        fused = []
        for A in attn_stack:  # (1,H,N,N)
            A = A[0]          # (H,N,N)
            if head_fuse == 'mean':
                A = A.mean(dim=0)  # (N,N)
            elif head_fuse == 'max':
                A, _ = A.max(dim=0)
            else:
                raise ValueError(f'Unknown head_fuse: {head_fuse}')
            if add_identity:
                I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
                A = A + I
            # row-normalize
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            fused.append(A)
        # rollout: multiply from first to last
        R = fused[0]
        for A in fused[1:]:
            R = A @ R
        # take the attention from [CLS] token to all tokens, assume cls at index 0
        rollout_vec = R[0]  # (N,)
        # normalize to [0,1]
        rollout_vec = (rollout_vec - rollout_vec.min()) / (rollout_vec.max() - rollout_vec.min() + 1e-6)
        return rollout_vec


def tokens_to_map(rollout_vec, grid_hw):
    """
    Convert token rollout (N=1+HW) to an HxW map (skip CLS).
    grid_hw: (Hp, Wp) in tokens (patch grid).
    """
    Hp, Wp = grid_hw
    tok = rollout_vec[1:]  # drop CLS
    if tok.numel() != Hp * Wp:
        # try best-effort reshape (sometimes extra tokens exist)
        side = int(math.sqrt(tok.numel()))
        Hp = Wp = side
        tok = tok[:Hp*Wp]
    m = tok.view(Hp, Wp).detach().cpu().numpy()
    # normalize to [0,1]
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return m


def resize_map_to_img(m, img_hw):
    from PIL import Image
    H, W = img_hw
    m_img = (m * 255.0).astype(np.uint8)
    m_img = Image.fromarray(m_img)
    m_img = m_img.resize((W, H), resample=Image.BILINEAR)
    return np.array(m_img).astype(np.float32) / 255.0


def overlay_heatmap(pil_rgb, heat, alpha=0.5):
    """
    pil_rgb: PIL Image RGB (HxW)
    heat: HxW float in [0,1]
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    heat_color = (cmap(heat)[..., :3] * 255).astype(np.uint8)  # HxWx3
    base = np.array(pil_rgb).astype(np.uint8)
    over = (alpha * heat_color + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(over)


# ------------------------------ main logic ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--num', type=int, default=64, help='how many images to dump')
    ap.add_argument('--split', choices=['val','train'], default='val')
    ap.add_argument('--device', default='cuda', help="cpu | cuda | cuda:0 etc. With CUDA_VISIBLE_DEVICES, use 'cuda' to mean cuda:0 among visible.")
    ap.add_argument('--alpha', type=float, default=0.5, help='overlay strength')
    args = ap.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    mean, std, bgr_to_rgb = get_preproc_from_cfg(cfg, device)

    # Build dataset and loader
    loader = build_loader(cfg, split=args.split, batch_size=1)

    # Build model
    model = build_model(cfg, args.ckpt, device)
    backbone = getattr(model, 'backbone', model)

    # Prepare attention collector (hooks). We keep this flexible:
    # 1) If your backbone already sets `backbone.last_attn` each forward, we can use it directly.
    # 2) Else we register hooks on common attention modules.
    attn_collector = AttnCollector()
    attn_collector.register(backbone)

    saved = 0
    for idx, batch in enumerate(loader):
        if saved >= args.num:
            break

        # support list[dict] and dict batches
        if isinstance(batch, list):
            sample = batch[0]
        elif isinstance(batch, dict):
            sample = batch
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        x = sample['inputs']  # Tensor or list[Tensor]
        if isinstance(x, torch.Tensor):
            img = x.unsqueeze(0)  # (1,C,H,W)
        elif isinstance(x, (list, tuple)):
            img = x[0].unsqueeze(0) if isinstance(x[0], torch.Tensor) else torch.stack(x, dim=0)
        else:
            raise TypeError(f"Unexpected 'inputs' type: {type(x)}")

        # preprocess (match DetDataPreprocessor)
        img = img.to(device, non_blocking=True).float()
        if bgr_to_rgb:
            img = img[:, [2,1,0], ...]
        img = (img - mean) / std

        # clear previous attentions
        attn_collector.clear()
        if hasattr(backbone, 'last_attn'):
            setattr(backbone, 'last_attn', None)

        with torch.no_grad():
            _ = backbone(img)

        # Option A: prefer explicit backbone.last_attn if present
        attn_stack = []
        last_attn = getattr(backbone, 'last_attn', None)
        if last_attn is not None and torch.is_tensor(last_attn) and last_attn.dim() == 4:
            attn_stack = [last_attn.detach()]
        else:
            # Option B: use collected stack from hooks
            attn_stack = attn_collector.get()

        if not attn_stack:
            print(f"[WARN] No attention captured on sample {idx}. Check your backbone/hook wiring.")
            continue

        # rollout over layers
        rollout_vec = attention_rollout(attn_stack, add_identity=True, head_fuse='mean')

        # try to guess token grid from backbone if available
        # Many ViTs (patch_size P, image size HxW) -> token grid Hp=H/P, Wp=W/P
        H, W = img.shape[-2], img.shape[-1]
        # try common attributes
        patch = getattr(backbone, 'patch_size', None)
        if patch is None and hasattr(backbone, 'patch_embed') and hasattr(backbone.patch_embed, 'proj'):
            # conv proj stride is often the patch size
            st = backbone.patch_embed.proj.stride
            if isinstance(st, tuple):
                patch = st[0]
            else:
                patch = int(st)
        if patch is None:
            # fallback: assume square 14 or 16
            patch = 16

        Hp, Wp = H // patch, W // patch
        m = tokens_to_map(rollout_vec, (Hp, Wp))
        m = resize_map_to_img(m, (H, W))

        # reconstruct a nice RGB image for overlay
        pil_img = tensor_to_pil(img, mean, std, bgr_to_rgb=bgr_to_rgb)
        overlay = overlay_heatmap(pil_img, m, alpha=args.alpha)

        # save
        out_path = Path(args.outdir) / f"attn_{idx:05d}.png"
        overlay.save(out_path)
        saved += 1
        print(f"[{saved}/{args.num}] saved {out_path}")

    attn_collector.remove()
    print("Done.")


if __name__ == "__main__":
    main()
