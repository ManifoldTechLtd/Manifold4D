"""Vendored Wan2.1 T2V base model (14B), stripped of all I2V code.

This is a standalone copy of the Wan2.1 WanModel with:
  - Only the ``t2v`` branch retained (no img_emb / CLIP / I2V cross-attn).
  - ``Head.forward`` adapted to accept per-token ``e`` of shape
    ``[B, L, C]`` (in addition to the legacy ``[B, C]``), matching
    the per-token convention used by the two-stream forward.

The class is renamed ``WanModel21T2V`` to avoid collision with the
upstream ``WanModel``.  It inherits ``ModelMixin`` / ``ConfigMixin``
so ``WanModel21T2V.from_pretrained(dir)`` loads diffusers-format
safetensors checkpoints directly (same as upstream).

Specs (Wan2.1-T2V-14B):
    dim=5120, ffn_dim=13824, num_layers=40, num_heads=40,
    in_dim=16, out_dim=16, patch_size=(1,2,2), head_dim=128.
"""

import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from manifold4d.model.attention_compat import flash_attention_compat

__all__ = [
    'WanModel21T2V', 'WanRMSNorm', 'WanLayerNorm',
    'WanSelfAttention', 'WanT2VCrossAttention', 'WanAttentionBlock',
    'Head', 'sinusoidal_embedding_1d', 'rope_params', 'rope_apply',
]

T5_CONTEXT_TOKEN_NUMBER = 512


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        x = flash_attention_compat(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = flash_attention_compat(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size,
                                          qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1),
                                               qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs,
                context, context_lens,
                cam_encoder=None, projector=None, cam_plucker=None,
                skip_text_cross_attn=False,
                rope_n_streams=1, rope_offset=None):
        """Native Wan forward, extended for the 14B two-stream path.

        Native mode (e is [B, 6, C], all two-stream kwargs None): bit-identical
        to upstream Wan2.1 WanAttentionBlock.forward.

        Two-stream mode (e is [B, L, 6, C] per-token):
            input_x = norm1(x)*(1+e1)+e0 → += cam_encoder(plucker) →
            self_attn (offset RoPE when rope_offset set) → projector →
            gate → text cross-attn → FFN.
        The two-stream logic lives HERE (not in an external helper) so
        per-block FSDP wrapping triggers all-gather via block.__call__.
        """
        assert e.dtype == torch.float32
        per_token = (e.dim() == 4)          # [B, L, 6, C] vs [B, 6, C]
        if per_token:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [u.squeeze(2) for u in e]      # each [B, L, C]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        input_x = self.norm1(x) * (1 + e[1]) + e[0]
        if cam_encoder is not None and cam_plucker is not None:
            input_x = input_x + cam_encoder(cam_plucker.to(input_x.dtype))

        if rope_offset is not None:
            from manifold4d.model.epipolar_bias import self_attn_offset
            y = self_attn_offset(
                self.self_attn, input_x, seq_lens, grid_sizes, freqs,
                rope_n_streams, rope_offset)
        else:
            y = self.self_attn(input_x, seq_lens, grid_sizes, freqs)

        if projector is not None:
            y = projector(y)
        x = x + y * e[2]

        if not skip_text_cross_attn:
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
        # Keep FFN in bf16 (skip the .float() upcast) to save ~400MB
        # activation memory — needed for 14B FSDP on 80G at 480x832.
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim_linear = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_linear)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """Forward with per-token or scalar time embedding.

        Args:
            x: [B, L, C]
            e: [B, L, C] (per-token) or [B, C] (scalar, legacy)
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if e.dim() == 2:
                e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
                x = self.head(self.norm(x) * (1 + e[1]) + e[0])
            else:
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                x = self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel21T2V(ModelMixin, ConfigMixin):
    """Wan2.1 T2V diffusion backbone (14B), T2V-only.

    Inherits ModelMixin/ConfigMixin so ``from_pretrained`` loads
    diffusers-format checkpoints directly.  The config defaults
    match the 14B checkpoint.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size,
                              qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.init_weights()

    def forward(self, x, t, context, seq_len):
        """Native T2V forward (no per-token, no conditioning).

        Args:
            x: List[Tensor[C_in, F, H, W]]
            t: Tensor [B] diffusion timesteps
            context: List[Tensor[L, C]] text embeddings
            seq_len: int max sequence length

        Returns:
            List[Tensor[C_out, F, H, W]] denoised latents
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
            freqs=self.freqs, context=context, context_lens=None)

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)
