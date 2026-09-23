"""14B model helpers: mask convs, 2-stream Plucker, freeze policy.

  - ``build_plucker_2stream_14b`` — 2-stream Plucker (output=target rays,
    source=source rays) with Wan2.1 VAE stride (4,8,8) × patch (1,2,2)
    = pixel factor (4,16,16) → fallback H_n*16.
  - ``StreamProjector14B`` — identity-init Linear(dim, dim).
  - ``make_zero_conv3d_14b`` — zero-init Conv3d for mask patchify.
  - ``unfreeze_trainable_14b`` — freeze base, unfreeze self_attn only
    (FFN stays frozen).
  - ``replicate_streams_for_rope_14b`` — joint grid_sizes for n-stream concat.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..model.camera_encoder import (
    _intrinsics_from_K, compute_plucker,
    latent_indices_for_video_frames,
)


MASK_CHANNELS_14B = 2


class StreamProjector14B(nn.Module):
    """Per-block identity-init Linear(dim, dim) applied after self-attn.

    Design adapted from Vista4D's per-block projector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def make_zero_conv3d_14b(in_channels: int, out_channels: int,
                         kernel_size: Tuple[int, int, int],
                         stride: Tuple[int, int, int],
                         bias: bool = True) -> nn.Conv3d:
    conv = nn.Conv3d(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride, bias=bias)
    nn.init.zeros_(conv.weight)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def replicate_streams_for_rope_14b(
    grid_sizes_per_stream: torch.Tensor,
    n_streams: int,
) -> torch.Tensor:
    out = grid_sizes_per_stream.clone()
    out[:, 0] = out[:, 0] * int(n_streams)
    return out


def build_plucker_2stream_14b(
    *,
    grid_sizes_per_stream: torch.Tensor,
    noise_seq_lens: torch.Tensor,
    seq_len: int,
    tgt_c2ws,
    src_c2ws,
    K,
    device,
    dtype,
    pixel_height=None,
    pixel_width=None,
):
    """Build per-token Plucker for the 2-stream joint sequence.

    Layout for sample i (per-stream pad = seq_len, actual tokens = L_i):
        slot [0, L_i)              : target Plucker (output stream)
        slot [L_i, seq_len)        : zero-pad
        slot [seq_len, seq_len+L_i): source Plucker (source stream)
        slot [seq_len+L_i, 2*seq_len): zero-pad

    Output stream carries TARGET-camera rays.
    Source stream carries TRUE source-camera rays (when src_c2ws provided)
    or target replica (when None).

    Wan2.1 VAE stride (4,8,8) × DiT patch (1,2,2) = (4,16,16).
    Default pixel fallback: H_n * 16.

    Returns:
        [B, 2*seq_len, 6] Plucker tensor.
    """
    B = grid_sizes_per_stream.shape[0]
    out = torch.zeros(B, seq_len * 2, 6, device=device, dtype=dtype)

    def _get_pix(spec, i, fallback):
        if spec is None:
            return int(fallback)
        if isinstance(spec, (list, tuple)):
            v = spec[i]
        else:
            v = spec
        try:
            return int(v)
        except (TypeError, ValueError):
            return int(fallback)

    for i in range(B):
        F_n, H_n, W_n = (int(grid_sizes_per_stream[i, 0]),
                         int(grid_sizes_per_stream[i, 1]),
                         int(grid_sizes_per_stream[i, 2]))
        L_i = int(noise_seq_lens[i])
        # Wan2.1: VAE (4,8,8) × patch (1,2,2) = (4,16,16) → pixel = token * 16
        H_pix = _get_pix(pixel_height, i, H_n * 16)
        W_pix = _get_pix(pixel_width, i, W_n * 16)

        K_i = K[i].to(device=device, dtype=dtype)
        tgt_full = tgt_c2ws[i].to(device=device, dtype=dtype)
        t_idx = latent_indices_for_video_frames(F_n)
        t_idx_t = torch.as_tensor(
            t_idx, device=device, dtype=torch.long
        ).clamp(max=tgt_full.size(0) - 1)
        tgt_c = tgt_full.index_select(0, t_idx_t).unsqueeze(0)

        if K_i.dim() == 3 and K_i.size(0) > 1:
            K_sel = K_i.index_select(0, t_idx_t.clamp(max=K_i.size(0) - 1))
            K_intr = torch.stack([
                _intrinsics_from_K(K_sel[k], H_n, W_n, H_pix, W_pix).squeeze(0)
                for k in range(F_n)
            ]).unsqueeze(0)
        else:
            intr = _intrinsics_from_K(K_i, H_n, W_n, H_pix, W_pix)
            K_intr = intr.unsqueeze(1).expand(1, F_n, 4)

        plucker = compute_plucker(K_intr, tgt_c, H_n, W_n)
        plucker_flat = plucker.reshape(1, F_n * H_n * W_n, 6)
        assert plucker_flat.size(1) == L_i

        out[i, 0:L_i] = plucker_flat[0]

        if src_c2ws is not None:
            src_full = src_c2ws[i].to(device=device, dtype=dtype)
            s_idx_t = t_idx_t.clamp(max=src_full.size(0) - 1)
            src_c = src_full.index_select(0, s_idx_t).unsqueeze(0)
            plucker_s = compute_plucker(K_intr, src_c, H_n, W_n)
            out[i, seq_len:seq_len + L_i] = (
                plucker_s.reshape(1, F_n * H_n * W_n, 6)[0])
        else:
            out[i, seq_len:seq_len + L_i] = plucker_flat[0]

    return out


def unfreeze_trainable_14b(model: nn.Module,
                           base_model: nn.Module) -> int:
    """Apply the 14B freeze policy: base frozen, self_attn unfrozen, FFN frozen.

    Trainable:
        * block.self_attn (Q/K/V/O + norm_q/norm_k)
        * cam_encoders (top-level, not in base_model)
        * projectors (top-level)
        * output_rgb_patch_embed, output_anchor_patch_embed (top-level)
        * source_rgb_patch_embed, source_mask_patch_embed (top-level)

    Frozen (NOT unfrozen by this routine):
        * block.ffn (FFN stays frozen — user 2026-07-07 decision)
        * block.cross_attn, block.norm1/2/3, block.modulation
        * base_model.patch_embedding, text_embedding, time_embedding,
          time_projection, head

    Returns:
        Number of parameter tensors flipped to trainable.
    """
    for param in base_model.parameters():
        param.requires_grad = False

    n_unfrozen = 0
    for block in base_model.blocks:
        for param in block.self_attn.parameters():
            param.requires_grad = True
            n_unfrozen += 1
    return n_unfrozen
