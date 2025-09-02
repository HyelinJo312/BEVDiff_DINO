# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from LayoutDiffusion
#   (https://github.com/ZGCTroy/LayoutDiffusion)
# Copyright (c) 2023 LayoutDiffusion authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from abc import abstractmethod
import os
import safetensors
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

class SiLU(nn.Module):  # export-friendly version of SiLU()
    @staticmethod
    def forward(x):
        return x * th.sigmoid(x)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond_kwargs=None):
        extra_output = None
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, (AttentionBlock, ObjectAwareCrossAttention)):
                x, extra_output = layer(x, cond_kwargs)
            else:
                x = layer(x)
        return x, extra_output


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, out_size=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.out_size = out_size
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.out_size is None:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(
                    x, (x.shape[2], self.out_size, self.out_size), mode="nearest"
                )
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=self.out_size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            out_size=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, out_size=out_size)
            self.x_upd = Upsample(channels, False, dims, out_size=out_size)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=False
    ):
        super().__init__()
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.randn(channels // self.num_heads, resolution ** 2) / channels ** 0.5)  # [C,L1]
        else:
            self.positional_embedding = None

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)

        self.qkv = conv_nd(1, channels, channels * 3, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.encoder_channels = encoder_channels
        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs=None):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        qkv = self.qkv(self.norm(x))  # N x 3C x L1, 其中L1=H*W
        if cond_kwargs is not None and self.encoder_channels is not None:
            kv_for_encoder_out = self.encoder_kv(cond_kwargs['xf_out'])  # xf_out: (N x encoder_channels x L2) -> (N x 2C x L2), 其中L2=max_obj_num
            h = self.attention(qkv, kv_for_encoder_out, positional_embedding=self.positional_embedding)
        else:
            h = self.attention(qkv, positional_embedding=self.positional_embedding)
        h = self.proj_out(h)
        output = (x + h).reshape(b, c, *spatial)

        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({
                'type': self.type,
                'ds': self.ds,
                'resolution': self.resolution,
                'num_heads': self.num_heads,
                'num_channels': self.channels,
                'image_query_embeddings': qkv[:, :self.channels, :].detach(),  # N x C x L1
            })
            if cond_kwargs is not None:
                extra_output.update({
                    'layout_key_embeddings': kv_for_encoder_out[:, : self.channels, :].detach()  # N x C x L2
                })

        return output, extra_output


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None, positional_embedding=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Q_T, K_T, and V_T.
        :param encoder_kv: an [N x (H * 2 * C) x S] tensor of K_E, and V_E.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if positional_embedding is not None:
            q = q + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]
            k = k + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]

        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)




class DINOContextAdapter(nn.Module):
    """
    Adapt DINOv2 global context (CLS tokens) to UNet time-embedding size.

    Input:
      - context: (B, V, C_in) or (B, C_in)
    Output:
      - context_proj: (B, C_emb)

    Args:
      c_in:   DINO hidden size (e.g., 768 for vitb14)
      c_emb:  UNet time-embedding size (e.g., 1024)
      pool:   'mean' | 'max' | 'attn'  (how to pool across views)
      dropout: dropout in the MLP head
      ln_first / ln_after: LayerNorm before/after projection
    """
    def __init__(self, c_in: int, c_emb: int,
                 pool: str = 'mean',
                 dropout: float = 0.0,
                 ln_first: bool = True,
                 ln_after: bool = True):
        super().__init__()
        assert pool in ['mean', 'max', 'attn']
        self.pool = pool

        self.ln_first = nn.LayerNorm(c_in) if ln_first else None

        # lightweight projection head to emb dim
        self.proj = nn.Sequential(
            nn.Linear(c_in, c_emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_emb, c_emb),
        )
        self.ln_after = nn.LayerNorm(c_emb) if ln_after else None

        if pool == 'attn':
            # content-based attention over views
            self.q = nn.Linear(c_in, c_in)
            self.k = nn.Linear(c_in, c_in)
            self.v = nn.Linear(c_in, c_in)

    def forward(self, context, mask=None):
        """
        context: (B, V, C_in) or (B, C_in)
        mask:    optional (B, V) boolean; False = ignore that view
        returns: (B, C_emb)
        """
        if context.ndim == 2:
            # (B, C_in) -> add V=1
            context = context.unsqueeze(1)  # (B, 1, C_in)
        elif context.ndim != 3:
            raise ValueError(f"context must be (B,C) or (B,V,C), got {context.shape}")

        B, V, C = context.shape

        x = context
        if self.ln_first is not None:
            x = self.ln_first(x)  # LN over C

        if self.pool == 'mean':
            if mask is not None:
                w = mask.float().clamp_min(1e-6)  # (B,V)
                x_sum = (x * w.unsqueeze(-1)).sum(dim=1)
                denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
                g = x_sum / denom                    # (B, C)
            else:
                g = x.mean(dim=1)                    # (B, C)

        elif self.pool == 'max':
            if mask is not None:
                # set masked views to very small value before max
                neg_inf = th.finfo(x.dtype).min
                x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
                g, _ = x_masked.max(dim=1)           # (B, C)
            else:
                g, _ = x.max(dim=1)                  # (B, C)

        else:  # 'attn'
            # single-query attention over views
            q = self.q(x.mean(dim=1, keepdim=True))   # (B,1,C)
            k = self.k(x)                              # (B,V,C)
            v = self.v(x)                              # (B,V,C)
            attn = (q @ k.transpose(1, 2)) / (C ** 0.5)  # (B,1,V)
            if mask is not None:
                attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
            attn = th.softmax(attn, dim=-1)        # (B,1,V)
            g = (attn @ v).squeeze(1)                 # (B, C)

        g = self.proj(g)                               # (B, C_emb)
        if self.ln_after is not None:
            g = self.ln_after(g)
        return g


class GroupReducer(nn.Module):
    """
    Channel reduction by learnable softmax-weighted mean inside small groups.
    Example: 768 -> 256 with group size g=3 (per output channel 3 weights).
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        assert c_in % c_out == 0, f"{c_in} not divisible by {c_out}"
        self.c_in = c_in
        self.c_out = c_out
        self.g = c_in // c_out                              # e.g., 3
        # logits for per-group weights (c_out, g), init ~ 0 -> softmax ~ uniform
        self.logits = nn.Parameter(th.zeros(c_out, self.g))

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: (B, Q, c_in) -> (B, Q, c_out)
        """
        B, Q, C = x.shape
        x = x.view(B, Q, self.c_out, self.g)               # (B,Q,c_out,g)
        w = th.softmax(self.logits, dim=-1)             # (c_out,g)
        y = (x * w.view(1,1,self.c_out,self.g)).sum(dim=-1)  # (B,Q,c_out)
        return y


class LinearReducer(nn.Module):
    """
    Standard learnable linear projection with group-mean initialization.
    """
    def __init__(self, c_in: int, c_out: int, init_group_mean: bool = True, use_weight_norm: bool = False):
        super().__init__()
        proj = nn.Linear(c_in, c_out, bias=True)
        if init_group_mean:
            with th.no_grad():
                assert c_in % c_out == 0, f"{c_in} not divisible by {c_out}"
                g = c_in // c_out                             # e.g., 3
                W = th.zeros(c_out, c_in)
                for i in range(c_out):
                    W[i, i*g:(i+1)*g] = 1.0 / g              # group-mean initialization
                proj.weight.copy_(W)
                proj.bias.zero_()
        self.proj = nn.utils.weight_norm(proj) if use_weight_norm else proj

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: (B, Q, c_in) -> (B, Q, c_out)
        """
        return self.proj(x)


class MLPReducer(nn.Module):
    """
    Small non-linear projection for extra capacity with few params.
    """
    def __init__(self, c_in: int, c_out: int, hidden: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, c_out),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: (B, Q, c_in) -> (B, Q, c_out)
        """
        return self.net(x)


class DINOBevAligner(nn.Module):
    """
    Self-contained BEV aligner for DINOv2 last_tokens using BEVFormer-style reference generation.

    Inputs:
      - last_tokens: (B, V, N, C_dino)
      - patch_hw:    (Hp, Wp) with Hp*Wp == N
      - img_metas:   list of dicts (len=B), each with:
          * 'lidar2img': (V, 4, 4)
          * 'img_shape' : ((H, W, 3),) or similar; we use [0][0]=H, [0][1]=W

    Returns:
      - bev_feat_ctx: (B, C_ctx, bev_h, bev_w)

    Learnable:
      - Per-view scalar weights (softplus -> positive)
      - Reducer parameters (depending on reducer mode)
      - LayerNorm γ/β if pre_ln_affine/post_ln_affine are True
    """
    def __init__(
        self,
        bev_h=50,
        bev_w=50,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        input_size=518,             # DINO square resize S
        c_ctx=256,                  # output channels
        reducer='group',            # 'group' | 'linear' | 'mlp'
        pre_norm=True,
        post_norm=True,
        pre_ln_affine=False,       # turn to True if you want learnable γ/β before sampling
        post_ln_affine=True,       # recommended True (stability + capacity)
        linear_use_weight_norm=False,
        mlp_hidden=128,
        mlp_dropout=0.1,
        eps=1e-6,
    ):
        super().__init__()
        assert reducer in ['group', 'linear', 'mlp']
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.S = input_size
        self.c_ctx = c_ctx
        self.reducer_kind = reducer
        self.eps = eps

        # Norms are created lazily with correct feature dim
        self.pre_norm_enabled  = pre_norm
        self.post_norm_enabled = post_norm
        self.pre_ln_affine  = pre_ln_affine
        self.post_ln_affine = post_ln_affine
        self.pre_ln  = None
        self.post_ln = None

        # Per-view weights (initialized lazily with V)
        self._w_view = None

        # Channel reducer (created lazily with C_dino)
        self.reducer = None
        self.linear_use_weight_norm = linear_use_weight_norm
        self.mlp_hidden = mlp_hidden
        self.mlp_dropout = mlp_dropout

    # ---------- BEVFormer-style reference generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=th.float32):
        if dim == '3d':
            zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = th.stack((xs, ys, zs), -1)                 # (D,H,W,3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (D, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)              # (bs, D, H*W, 3)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = th.meshgrid(
                th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = th.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)          # (bs, H*W, 1, 2)
            return ref_2d
        else:
            raise ValueError("dim must be '3d' or '2d'")

    @th.no_grad()
    def _point_sampling(self, reference_points_3d, img_metas):
        allow_tf32 = th.backends.cuda.matmul.allow_tf32
        th.backends.cuda.matmul.allow_tf32 = False
        th.backends.cudnn.allow_tf32 = False

        device = reference_points_3d.device
        dtype  = reference_points_3d.dtype
        B      = reference_points_3d.shape[0]
        D      = reference_points_3d.shape[1]
        Q      = reference_points_3d.shape[2]

        # lidar2img
        lidar2img = []
        for m in img_metas:
            lidar2img.append(m['lidar2img'])
        lidar2img = np.asarray(lidar2img)                              # (B,V,4,4)
        lidar2img = th.as_tensor(lidar2img, device=device, dtype=th.float32)
        V = lidar2img.size(1)

        # scale to pc_range
        pc = self.pc_range
        ref = reference_points_3d.clone()          # (B,D,Q,3) in [0,1]
        ref[..., 0:1] = ref[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc[5] - pc[2]) + pc[2]
        ref = th.cat((ref, th.ones_like(ref[..., :1])), dim=-1)  # (B,D,Q,4)

        # reshape and tile for cams
        ref = ref.permute(1,0,2,3).view(D, B, 1, Q, 4, 1)              # (D,B,1,Q,4,1)
        lidar2img = lidar2img.view(1, B, V, 1, 4, 4).repeat(D,1,1,Q,1,1)

        # projection
        pts = th.matmul(lidar2img, ref).squeeze(-1)                  # (D,B,V,Q,4)
        eps = 1e-5
        bev_mask = (pts[..., 2:3] > eps)                                 # (D,B,V,Q,1)

        uv = pts[..., 0:2] / th.maximum(pts[..., 2:3], th.ones_like(pts[..., 2:3]) * eps)  # (D,B,V,Q,2)
        H_img = img_metas[0]['img_shape'][0][0]
        W_img = img_metas[0]['img_shape'][0][1]
        uv[..., 0] = uv[..., 0] / W_img
        uv[..., 1] = uv[..., 1] / H_img

        bev_mask = (bev_mask &
                    (uv[..., 1:2] > 0.0) &
                    (uv[..., 1:2] < 1.0) &
                    (uv[..., 0:1] < 1.0) &
                    (uv[..., 0:1] > 0.0))

        uv = uv.permute(2,1,3,0,4).contiguous()                          # (V,B,Q,D,2)
        bev_mask = bev_mask.permute(2,1,3,0,4).squeeze(-1).contiguous()  # (V,B,Q,D)

        th.backends.cuda.matmul.allow_tf32 = allow_tf32
        th.backends.cudnn.allow_tf32 = allow_tf32
        return uv.to(dtype), bev_mask

    def _lazy_init_modules(self, C_dino: int, V: int, device):
        # norms
        if self.pre_norm_enabled and (self.pre_ln is None):
            self.pre_ln = nn.LayerNorm(C_dino, elementwise_affine=self.pre_ln_affine).to(device)
        if self.post_norm_enabled and (self.post_ln is None):
            self.post_ln = nn.LayerNorm(C_dino, elementwise_affine=self.post_ln_affine).to(device)

        # view weights
        if self._w_view is None:
            self._w_view = nn.Parameter(th.zeros(1, V, 1, device=device))  # softplus -> positive

        # reducer
        if self.reducer is None:
            if self.reducer_kind == 'group':
                self.reducer = GroupReducer(c_in=C_dino, c_out=self.c_ctx).to(device)
            elif self.reducer_kind == 'linear':
                self.reducer = LinearReducer(c_in=C_dino, c_out=self.c_ctx,
                                             init_group_mean=True,
                                             use_weight_norm=self.linear_use_weight_norm).to(device)
            else:  # 'mlp'
                self.reducer = MLPReducer(c_in=C_dino, c_out=self.c_ctx,
                                          hidden=self.mlp_hidden, p_drop=self.mlp_dropout).to(device)

    def _tokens_to_fmap(self, last_tokens, Hp, Wp):
        """
        (B,V,N,C) -> (B,V,C,Hp,Wp) with optional pre-LN.
        """
        B, V, N, C = last_tokens.shape
        fmap = last_tokens.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()  # (B,V,C,Hp,Wp)
        if self.pre_ln is not None:
            t = fmap.permute(0,1,3,4,2).reshape(-1, C)  # (B*V*Hp*Wp, C)
            t = self.pre_ln(t)
            fmap = t.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()
        return fmap

    def forward(self, last_tokens, patch_hw, img_metas):
        """
        last_tokens: (B,V,N,C_dino)
        patch_hw:    (Hp,Wp)
        img_metas:   list length B (BEVFormer-like metas)
        returns:     (B, C_ctx, bev_h, bev_w)
        """
        assert last_tokens.ndim == 4
        B, V, N, C_dino = last_tokens.shape
        Hp, Wp = patch_hw
        assert Hp * Wp == N, f"patch_hw {patch_hw} mismatches N={N}"
        device = last_tokens.device

        self._lazy_init_modules(C_dino, V, device)

        # (1) DINO fmap
        fmap = self._tokens_to_fmap(last_tokens, Hp, Wp)  # (B, V, C_dino, Hp, Wp)

        # (2) BEV refs and camera projection
        Z_bins = int(round((self.pc_range[5] - self.pc_range[2]) ))  # same spirit as BEVFormer
        ref_3d = self._get_reference_points(self.bev_h, self.bev_w, Z=Z_bins,
                                            num_points_in_pillar=self.num_points_in_pillar,
                                            dim='3d', bs=B, device=device, dtype=fmap.dtype)
        reference_points_cam, bev_mask = self._point_sampling(ref_3d, img_metas)  # (V, B, Q, D, 2), (V,B,Q,D)

        # (3) uv -> patch grid coords
        Q = self.bev_h * self.bev_w
        uv = reference_points_cam.to(device)                        # (V,B,Q,D,2)
        u_px = uv[..., 0] * self.S
        v_px = uv[..., 1] * self.S
        x_p = (u_px + 0.5) / self.S * Wp - 0.5
        y_p = (v_px + 0.5) / self.S * Hp - 0.5
        gx = 2.0 * x_p / (Wp - 1.0) - 1.0
        gy = 2.0 * y_p / (Hp - 1.0) - 1.0
        grid = th.stack([gx, gy], dim=-1)                        # (V,B,Q,D,2)

        # (4) bilinear sampling
        fmap_v = fmap.view(B*V, C_dino, Hp, Wp)
        grid_v = grid.permute(1,0,2,3,4).contiguous().view(B*V, Q*self.num_points_in_pillar, 1, 2)
        sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
                                padding_mode='zeros', align_corners=True)  # (B*V,C,Q*D,1)
        sampled = sampled.squeeze(-1).permute(0,2,1).contiguous().view(B, V, Q, self.num_points_in_pillar, C_dino)

        # (5) post-norm
        if self.post_ln is not None:
            t = sampled.view(-1, C_dino)
            t = self.post_ln(t)
            sampled = t.view(B, V, Q, self.num_points_in_pillar, C_dino)

        # (6) pillar mean + view-weighted mean
        mask = bev_mask.to(device).permute(1,0,2,3).unsqueeze(-1).float()  # (B,V,Q,D,1)
        sampled = sampled * mask
        denom_D = mask.sum(dim=3, keepdim=True).clamp_min(self.eps)        # (B,V,Q,1,1)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D                # (B,V,Q,1,C)
        feat_v = feat_v.squeeze(3)                                         # (B,V,Q,C)

        w = F.softplus(self._w_view).expand(B, -1, -1)                         
        w = w.unsqueeze(-1)                                                # (B, V, 1, 1)    
        view_valid = (denom_D.squeeze(3) > 0).float()                                     
        num = (feat_v * w).sum(dim=1)                                      # (B,Q,C)
        den = (w * view_valid).sum(dim=1).clamp_min(self.eps)              # (B,Q,1)
        f_bev = num / den                                                  # (B,Q,C_dino)

        # (7) channel reduction -> (B,Q,C_ctx)
        bev_qc = self.reducer(f_bev)                                       # (B,Q,C_ctx)

        # reshape to (B,C_ctx,H,W)
        bev_feat_ctx = bev_qc.permute(0,2,1).contiguous().view(B, self.c_ctx, self.bev_h, self.bev_w)
        return bev_feat_ctx





class LayoutDiffusionUNetModel(nn.Module):
    """
    A UNetModel that conditions on layout with an encoding transformer.
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_ds: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.

    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param {
        layout_length: number of layout objects to expect.
        hidden_dim: width of the transformer.
        num_layers: depth of the transformer.
        num_heads: heads in the transformer.
        xf_final_ln: use a LayerNorm after the output layer.
        num_classes_for_layout_object: num of classes for layout object.
        mask_size_for_layout_object: mask size for layout object image.
    }

    """

    def __init__(
            self,
            layout_encoder,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_ds,
            encoder_channels=None,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_positional_embedding_for_attention=False,
            image_size=256,
            attention_block_type='GLIDE',
            num_attention_blocks=1,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False,
            num_pre_downsample=0,
            return_multiscale=True,
            multiscale_indices='auto'
    ):
        super().__init__()
        self.norm_for_obj_embedding = norm_for_obj_embedding
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.norm_first = norm_first
        self.use_key_padding_mask=use_key_padding_mask
        self.num_attention_blocks = num_attention_blocks
        self.attention_block_type = attention_block_type
        if self.attention_block_type == 'GLIDE':
            attention_block_fn = AttentionBlock
        elif self.attention_block_type == 'ObjectAwareCrossAttention':
            attention_block_fn = ObjectAwareCrossAttention

        self.image_size = image_size
        self.use_positional_embedding_for_attention = use_positional_embedding_for_attention

        self.layout_encoder = layout_encoder

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_ds = attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # multi-scale features index
        self.return_multiscale = return_multiscale
        L = len(self.channel_mult)              # 레벨 수
        bpl = self.num_res_blocks + 1           # blocks per level
        # 최상위(레벨 0) 제외: 깊은 레벨들만 택함
        self._ms_take = [lvl*bpl + (self.num_res_blocks - 1) for lvl in range(L-1)]

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.downsample_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        for _ in range(num_pre_downsample):
            self.downsample_blocks.append(Downsample(
                            in_channels, conv_resample, dims=dims, out_channels=in_channels
                        ))
            self.upsample_blocks.append(Upsample(
                            out_channels, conv_resample, dims=dims, out_channels=out_channels
                        ))
            self.image_size = self.image_size // 2  

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    print('encoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            attention_block_fn(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                encoder_channels=encoder_channels,
                                ds=ds,
                                resolution=int(self.image_size // ds),
                                type='input',
                                use_positional_embedding=self.use_positional_embedding_for_attention,
                                use_key_padding_mask=self.use_key_padding_mask,
                                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                norm_first=self.norm_first,
                                norm_for_obj_embedding=self.norm_for_obj_embedding
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        print('middle attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            attention_block_fn(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                ds=ds,
                resolution=int(self.image_size // ds),
                type='middle',
                use_positional_embedding=self.use_positional_embedding_for_attention,
                use_key_padding_mask=self.use_key_padding_mask,
                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                norm_first=self.norm_first,
                norm_for_obj_embedding=self.norm_for_obj_embedding
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    print('decoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            attention_block_fn(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                encoder_channels=encoder_channels,
                                ds=ds,
                                resolution=int(self.image_size // ds),
                                type='output',
                                use_positional_embedding=self.use_positional_embedding_for_attention,
                                use_key_padding_mask=self.use_key_padding_mask,
                                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                norm_first=self.norm_first,
                                norm_for_obj_embedding=self.norm_for_obj_embedding
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            out_size=int(self.image_size // (ds // 2))
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, out_size=int(self.image_size // ds))
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.layout_encoder.convert_to_fp16()

    def forward(self, x, timesteps, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, obj_name=None, **kwargs):
        hs, extra_outputs = [], []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        layout_outputs = self.layout_encoder(
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj,
            obj_name=obj_name
        )
        xf_proj, xf_out = layout_outputs["xf_proj"], layout_outputs["xf_out"]  # xf_proj: (B, 1024), xf_out: (B, 256, 300)

        emb = emb + xf_proj.to(emb) # emb: (B, 1024)

        out_list = []
        h = x.type(self.dtype)  # h: (B, N, H, W)
        for module in self.downsample_blocks:
            h = module(h)
        for module in self.input_blocks:
            h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
            hs.append(h)
        h, extra_output = self.middle_block(h, emb, layout_outputs)
        if extra_output is not None:
            extra_outputs.append(extra_output)
        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
            if self.return_multiscale and (i_out in self._ms_take):  # i_out in [1, 4]:
                out_list.append(h)
            
        h = h.type(x.dtype)
        h = self.out(h)
        for module in self.upsample_blocks:
            h = module(h)

        if self.return_multiscale:
            out_list.append(h)
            return out_list[::-1], extra_outputs
        else:
            return [h, extra_outputs]
    
    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        weights_name = SAFETENSORS_WEIGHTS_NAME
        safetensors.torch.save_file(self.state_dict(), os.path.join(save_directory, weights_name), metadata={"format": "pt"})
        
    def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
        weights_name = SAFETENSORS_WEIGHTS_NAME
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, weights_name)
            elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        else:
            print(f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.")
            return
        state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
        try:
            self.load_state_dict(state_dict, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')
            self.load_state_dict(state_dict, strict=False)