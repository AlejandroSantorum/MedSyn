import os
import math
import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from PIL import Image

from tqdm import tqdm
from einops import rearrange
import nibabel as nib

from accelerate import Accelerator

import xformers, xformers.ops

from utils import *

import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class Block3d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block3d(dim, dim_out, groups=groups)
        self.block2 = Block3d(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(qkv[0], 'b (h c) x y -> (b h) (x y) c', h=self.heads)
        k = rearrange(qkv[1], 'b (h c) x y -> (b h) (x y) c', h=self.heads)
        v = rearrange(qkv[2], 'b (h c) x y -> (b h) (x y) c', h=self.heads)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

        out = rearrange(hidden_states, '(b h) (x y) c -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

class CrossAttention(nn.Module):
    def __init__(self, dim, dim_con, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Linear(dim_con, hidden_dim*2, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, kv=None):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        self.to_kv(kv)
        kv = torch.cat([kv.unsqueeze(dim=1)]*f, dim=1)
        kv = rearrange(kv, 'b f h c -> (b f) h c')
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        k = rearrange(k, 'b d (h c) -> (b h) d c', h=self.heads)
        v = rearrange(v, 'b d (h c) -> (b h) d c', h=self.heads)

        q = self.to_q(x)
        q = rearrange(q, 'b (h c) x y -> (b h) (x y) c', h=self.heads)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

        out = rearrange(hidden_states, '(b h) (x y) c -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q = rearrange(qkv[0], '... n (h d) -> ... h n d', h=self.heads)
        k = rearrange(qkv[1], '... n (h d) -> ... h n d', h=self.heads)
        v = rearrange(qkv[2], '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    ):
        super().__init__()

        self.channels = channels
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size, init_kernel_size),
                                   padding=(init_padding, init_padding, init_padding))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond

        self.null_cond_emb = nn.Parameter(torch.randn(1, 192, cond_dim)) if self.has_cond else None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        block_klass3d = partial(ResnetBlock3d, groups=resnet_groups)
        block_klass_cond3d = partial(block_klass3d, time_emb_dim=time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                block_klass_cond3d(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn1 = Residual(PreNorm(mid_dim, spatial_attn))
        # self.mid_cross_attn1 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn1 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn2 = Residual(PreNorm(mid_dim, spatial_attn))
        # self.mid_cross_attn2 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn2 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn3 = Residual(PreNorm(mid_dim, spatial_attn))
        # self.mid_cross_attn3 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn3 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn4 = Residual(PreNorm(mid_dim, spatial_attn))
        # self.mid_cross_attn4 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn4 = block_klass_cond3d(mid_dim, mid_dim)

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                block_klass_cond3d(dim_in, dim_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, channels, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return [logits, null_logits]

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.1,
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

        x = self.init_conv(x)

        r = x.clone()
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            null_cond_emb = torch.cat([self.null_cond_emb]*batch, dim=0)
            cond = torch.where(rearrange(mask, 'b -> b 1 1'), null_cond_emb, cond)

        h = []

        for idx,(block1, block2, temporal_block, downsample) in enumerate(self.downs):
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)
            x = temporal_block(x, t)

        x = self.mid_block1(x, t)
        ###
        x = self.mid_spatial_attn1(x)
        # x = self.mid_cross_attn1(x, kv=cond)
        x = self.mid_temporal_attn1(x, t)
        ###
        x = self.mid_spatial_attn2(x)
        # x = self.mid_cross_attn2(x, kv=cond)
        x = self.mid_temporal_attn2(x, t)
        ###
        x = self.mid_spatial_attn3(x)
        # x = self.mid_cross_attn3(x, kv=cond)
        x = self.mid_temporal_attn3(x, t)
        ###
        x = self.mid_spatial_attn4(x)
        # x = self.mid_cross_attn4(x, kv=cond)
        x = self.mid_temporal_attn4(x, t)
        ###
        x = self.mid_block2(x, t)

        for block1, block2, temporal_block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)
            x = block2(x, t)
            x = temporal_block(x, t)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        # volume_depth=128,  # not used
        ddim_timesteps=50,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        # not used: self.volume_depth = volume_depth

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.ddim_timesteps = ddim_timesteps

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def p_mean_variance(self, x, t, clip_denoised: bool, indexes=None, cond=None, cond_scale=1.):

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        # model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        model_mean, posterior_variance = get_z_t_via_z_tp1(x_recon, x, (t - 1) * 1.0 / (self.num_timesteps - 1.0),
                                                           (t * 1.0) / (self.num_timesteps - 1.0))
        return model_mean, posterior_variance


    @torch.inference_mode()
    def p_sample(self, x, t, indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x=x, t=t, indexes=indexes, clip_denoised=clip_denoised,
                                                                 cond=cond,
                                                                 cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, 1, self.num_frames, 1, 1)
        return model_mean + nonzero_mask * (model_variance**0.5) * noise

    @torch.inference_mode()
    def p_sample_ddim(self, x, t, t_minus, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        if cond_scale != 1:
            x_recon, x_recon_null = x_recon
            eps = get_eps_x_t(x_recon, x, t)
            eps_null = get_eps_x_t(x_recon_null, x, t)
            final_eps = eps_null + (eps - eps_null) * cond_scale
            x_recon = get_x0_x_t(final_eps, x, t)
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s
        if t[0]<int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1., use_ddim=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        bsz = shape[0]

        if use_ddim:
            time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)

        img = torch.randn(shape, device=device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        for i, t in enumerate(tqdm(reversed(time_steps), desc='sampling loop time step',
                                   total=len(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)

            if use_ddim:
                time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                img = self.p_sample_ddim(img, time, time_minus, indexes=indexes, cond=cond,
                                         cond_scale=cond_scale)
            else:
                img = self.p_sample(img, time, indexes=indexes, cond=cond,
                                    cond_scale=cond_scale)
        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16, to_use_ddim=True):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size),
            cond=cond,
            cond_scale=cond_scale,
            use_ddim=to_use_ddim,
        )

    def p_losses(self, x_start, t, cond=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device

        x_noisy, noise = get_z_t(x_start, t)

        if is_list_str(cond):
            cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t*self.num_timesteps, cond=cond, **kwargs)

        loss = F.mse_loss(x_start, x_recon)

        ####################################################
        # TODO: REMOVE THIS

        print("x_start shape: ", x_start.shape)
        print("x_start min and max: ", x_start.min(), x_start.max())
        print("x_noisy shape: ", x_noisy.shape)
        print("x_noisy min and max: ", x_noisy.min(), x_noisy.max())
        print("noise shape: ", noise.shape)
        print("noise min and max: ", noise.min(), noise.max())

        sample_x_start_slice = x_start[0, 0, :, :, 32]
        assert sample_x_start_slice.shape == (f, h), f'sample x_start slice shape {sample_x_start_slice.shape} does not match expected shape {(f, h)}'
        plt.imsave(
            "/scratch/santorum/checkpoints/bratsc2023-mni-64x64x64/sample_x_start_slice.png",
            sample_x_start_slice.cpu().numpy(),
            cmap='gray'
        )

        sample_x_noisy_slice = x_noisy[0, 0, :, :, 32]
        assert sample_x_noisy_slice.shape == (f, h), f'sample x_noisy slice shape {sample_x_noisy_slice.shape} does not match expected shape {(f, h)}'
        plt.imsave(
            "/scratch/santorum/checkpoints/bratsc2023-mni-64x64x64/sample_x_noisy_slice.png",
            sample_x_noisy_slice.cpu().numpy(),
            cmap='gray'
        )

        sample_x_recon_slice = x_recon[0, 0, :, :, 32]
        assert sample_x_recon_slice.shape == (f, h), f'sample x_recon slice shape {sample_x_recon_slice.shape} does not match expected shape {(f, h)}'
        plt.imsave(
            "/scratch/santorum/checkpoints/bratsc2023-mni-64x64x64/sample_x_recon_slice.png",
            sample_x_recon_slice.detach().cpu().numpy(),
            cmap='gray'
        )
        ####################################################

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        b, c, f, h, w = x.shape
        assert h == img_size and w == img_size, f'input image size {h}x{w} does not match model image size {img_size}x{img_size}'
        assert f == self.num_frames, f'input number of frames {f} does not match model number of frames {self.num_frames}'
        assert c == self.channels, f'input number of channels {c} does not match model number of channels {self.channels}'
        t = torch.rand((b), device=device).float()
        return self.p_losses(x, t, *args, **kwargs)

# trainer class

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1



def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(x_recon):
    x_recon = x_recon.clamp(-1, 1)
    return (x_recon + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        num_max_samples=1000,
        test_flag=False,
        dataset_seed=None,
        exts=['nii.gz'],
    ):
        """

        """
        super().__init__()
        self.folder = folder
        self.paths = [
            p.as_posix() for ext in exts for p in Path(f'{folder}').glob(f'**/*-t1n.{ext}')
        ]

        if dataset_seed is not None:
            print(f"Setting dataset seed to {dataset_seed} ...")
            np.random.seed(int(dataset_seed))
            torch.manual_seed(int(dataset_seed))

        np.random.shuffle(self.paths)

        if num_max_samples is not None:
            if test_flag:
                print(f"Reading the last {num_max_samples} samples ...")
                # using the last 'num_max_samples' samples
                self.paths = self.paths[-int(num_max_samples):]
            else:
                print(f"Reading the first {num_max_samples} samples ...")
                # using the first 'num_max_samples' samples
                self.paths = self.paths [:int(num_max_samples)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        # read image data as numpy array
        img_data = nib.load(path).get_fdata()
        # normalize between 0 and 255
        img_data = 255.0 * (img_data - img_data.min()) / (img_data.max() - img_data.min())
        img_data = img_data.astype(np.uint8)
        # convert to tensor
        tensor = torch.from_numpy(img_data)
        return tensor.unsqueeze(0).float()

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        # prompt_folder,
        *,
        dataset_num_max_samples=1000,
        dataset_seed=42,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=5000,
        results_folder='./results',
        num_sample_rows=4,
        max_grad_norm=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        self.num_frames = diffusion_model.num_frames

        self.ds = Dataset(
            folder=folder,
            num_max_samples=dataset_num_max_samples,
            dataset_seed=dataset_seed,
            test_flag=False,
        )

        print(f'Found {len(self.ds)} 3D images at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, num_workers=8, shuffle=True, pin_memory=True))
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "fp32"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
        )

        self.model, self.ema_model, self.dl, self.opt, self.step = self.accelerator.prepare(
            self.model, self.ema_model, self.dl, self.opt, self.step
        )

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        self.accelerator.save_state(str(self.results_folder / f'{milestone}_ckpt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            dirs = os.listdir(self.results_folder)
            dirs = [d for d in dirs if d.endswith("ckpt")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[0]))
            path = dirs[-1]

        self.step = int(path.split("_")[0]) * self.save_and_sample_every + 1

        self.accelerator.load_state(os.path.join(self.results_folder, path), strict=False)

    def train(self):
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                with self.accelerator.accumulate(self.model):
                    data = next(self.dl)
                    img = data  # , text = data["image"], data["text"]
                    img = img.to(self.accelerator.device) # img.to(self.accelerator.device).squeeze(dim=1)
                    # text = text.to(self.accelerator.device)
                    B, C, D, H, W = img.shape

                    #######################################
                    # TODO: REMOVE
                    sample_slice = img[0, 0, :, :, 32]
                    assert sample_slice.shape == (D, H), f"sample slice shape {sample_slice.shape} does not match expected shape {(D, H)}"
                    plt.imsave(
                        os.path.join(self.results_folder, f"{self.step}_sample_slice.png"),
                        sample_slice.cpu().numpy(),
                        cmap="gray",
                    )
                    #######################################

                    loss = self.model(img, cond=None)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
                        print(f'{self.step}: {avg_loss},  {grad_norm}', flush=True)

                self.opt.step()
                self.opt.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.step % self.update_ema_every == 0:
                        self.step_ema()

                    with torch.no_grad():
                        if self.step != 0 and self.step % (self.save_and_sample_every) == 0:
                            milestone = self.step // self.save_and_sample_every
                            self.save(milestone)

                            # sample and save
                            for sample_idx in range(self.num_sample_rows):
                                # sample
                                sampled_imgs = self.ema_model.sample(batch_size=1, cond=None)
                                # remove batch and channel dimensions
                                sampled_img = sampled_imgs[0].squeeze(dim=0).cpu().numpy()
                                # get 2d sample image
                                plt.imshow(sampled_img[:,:,32], cmap="gray")
                                plt.axis("off")
                                # save sample
                                plt.savefig(str(self.results_folder / f"{self.step}_{sample_idx}.png"))
                                plt.close()

                        self.step += 1

        print('training completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('medsyn low-res args')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--data_dir', type=str, default="",
                        help='Your Training DATA Path')
    parser.add_argument('--save_dir', type=str, default="",
                        help='Your Logs Saving Path')
    parser.add_argument('--dataset_num_samples', type=int, default=1000,
                        help='Your Training DATA Number of Samples')
    parser.add_argument('--dataset_seed', type=int, default=42,
                        help='Your Training DATA Seed')
    parser.add_argument('--train_seed', type=int, default=42,
                        help='Your Training Seed')

    args = parser.parse_args()

    model = Unet3D(
        dim=160,
        # cond_dim=768,  # used for BERT text conditioning
        dim_mults=(1, 2, 4, 8),
        channels=1, # 4, # originally 4 channels
        attn_heads=8,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    )

    # image of size (N, D, H, W) = (1, 64, 64, 64)
    diffusion_model = GaussianDiffusion(
        denoise_fn=model,
        image_size=64,  # 64 x 64
        num_frames=64,  # 64
        text_use_bert_cls=False,
        channels=1,  # 4, # originally 4 channels 
        timesteps=1000,
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.995,
        # volume_depth=64,  # not used
        ddim_timesteps=50,
    )

    trainer = Trainer(
        diffusion_model=diffusion_model,
        folder=args.data_dir,
        dataset_num_max_samples=args.dataset_num_samples,
        dataset_seed=args.dataset_seed,
        ema_decay=0.999,
        train_batch_size=4,
        train_lr=1e-4,
        train_num_steps=1000000,
        gradient_accumulate_every=4,
        amp=True,
        step_start_ema=10000,
        update_ema_every=1,
        save_and_sample_every=1000,
        results_folder=args.save_dir,
        num_sample_rows=1,
        max_grad_norm=1.0,
    )

    if args.train_seed is not None:
        print(f"Setting train seed to {args.train_seed} ...")
        set_seed(int(args.train_seed))

    if args.resume:
        trainer.load(-1)

    trainer.train()
