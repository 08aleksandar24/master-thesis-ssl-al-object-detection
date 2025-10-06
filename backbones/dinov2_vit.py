from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from mmdet.registry import MODELS
from mmengine.model import BaseModule

# You still need to import DINOv2 blocks etc. from dinov2.layers
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

logger = logging.getLogger("dinov2")
class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

@MODELS.register_module()
class DinoVisionTransformer(BaseModule):
    """
    MMDetection-compatible DINOv2 ViT backbone.
    Args:
        out_indices (tuple): Indices of transformer blocks to extract features for FPN.
        init_cfg (dict, optional): Support MMDet-style pretrained loading.
        ... (all original ViT args supported)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1e-6,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        out_indices=(3, 5, 7, 11),
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        LOAD_FROM_HUB = False
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.out_indices = out_indices

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError
        if patch_size == 14 or patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))


        print("LOADING")
        # Load checkpoint if init_cfg is set (MMDet style)
        if LOAD_FROM_HUB:
            print("HUB")
            self.init_weights = self.init_weights_hub
        elif self.init_cfg is not None and self.init_cfg.get('type', None) == 'Pretrained':
            print("CHECKPOINT")
            self.ckpt_path = self.init_cfg['checkpoint']
            self.init_weights = self._load_pretrained
        else:
            print("NO CHECKPOINT")
            # Weight init
            self.init_weights = self.init_weights_scratch()
        self.init_weights()
        for n, p in self.named_parameters():
            #if not n.startswith("fpn"):
                p.requires_grad = False
        print("Frozen backbone")

    def init_weights_hub(self):
        import torch
        # Download model from torch.hub (this will download if missing)
        #hub_model = torch.hub.load('facebookresearch/dinov2', hub_name)
        import timm
        model = timm.create_model(
        model_name='vit_base_patch14_dinov2.lvd142m',
        #features_only=True,
        pretrained=True,
        in_chans=3,
        #out_indices=(-1,),
        checkpoint_path="",
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_size=16,
        #freeze=True,
        #dynamic_img_size=True,
        #dynamic_img_pad=True,

        )
        
        #model = model.eval()
        #for param in model.parameters():
            #param.requires_grad = False
        #for n, p in self.named_parameters():
         #   if not n.startswith("fpn"):
          #      p.requires_grad = False
        hub_state = model.state_dict()
        # Optionally remove/resize pos_embed here if you want!
        missing, unexpected = self.load_state_dict(hub_state, strict=False)
        print('[DINOv2-ViT] Loaded torch.hub weights, missing:', missing, 'unexpected:', unexpected)
        #for n, p in self.named_parameters():
            #if not n.startswith("fpn"):
                #p.requires_grad = False
    def init_weights_scratch(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _load_pretrained(self):
        # Robust loading for .pth and .ckpt
        state_dict = torch.load(self.ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = 1  # Usually cls token
            orig_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
            new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                print(f"Interpolating position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
                # Only interpolate the spatial tokens, not the cls token
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, embedding_size)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                state_dict['pos_embed'] = new_pos_embed
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print('[DINOv2-ViT] Loaded pretrained, missing:', missing, 'unexpected:', unexpected)
        #for n, p in self.named_parameters():
            #if not n.startswith("fpn"):
                #p.requires_grad = False

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward(self, x):
        """
        Standard MMDet-style forward: returns tuple of features from out_indices for FPN.
        """
        #print("[Backbone] Input:", x.shape, x.dtype, x.min().item(), x.max().item())

        B, C, H, W = x.shape
        x = self.prepare_tokens_with_masks(x, masks=None)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                #print(f"DEBUG: backbone feat {i}: shape {x.shape}, mean={x.mean().item()}, std={x.std().item()}")

                # Remove cls token, reshape patch tokens to [B, C, h, w]
                patch_tokens = x[:, 1:self.patch_embed.num_patches+1]
                # Shape: [B, N, C]
                n_patches = patch_tokens.shape[1]
                size = int(n_patches ** 0.5)
                patch_tokens = self.norm(patch_tokens)
                patch_tokens = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, size, size)
                #print(f"[Backbone] Output feat {i}: {patch_tokens.shape}, dtype={patch_tokens.dtype}, min={patch_tokens.min().item()}, max={patch_tokens.max().item()}")
                features.append(patch_tokens.contiguous())
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            #features.append(ops[i](xp))
            features[i] = ops[i](features[i])
        #print("DEBUGGG",[i.shape for i in features])
        return tuple(features)

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
