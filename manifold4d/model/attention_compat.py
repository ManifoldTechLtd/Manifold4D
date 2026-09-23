"""Flash-attention wrapper with a scaled_dot_product_attention fallback.

The Wan2.1 source tree's ``flash_attention`` hard-requires the compiled
``flash-attn`` package. When ``flash-attn`` is missing, this wrapper falls
back to PyTorch's built-in ``scaled_dot_product_attention`` so that small
runs stay possible without a toolchain, but the 14B model at full
resolution needs far more memory on this path — install ``flash-attn``
for real use (see README, Installation).
"""

from __future__ import annotations

import warnings

import torch

from wan.modules.attention import flash_attention as _wan_flash_attention

try:
    import flash_attn  # noqa: F401

    _FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    _FLASH_ATTN_2_AVAILABLE = False

try:
    import flash_attn_interface  # noqa: F401

    _FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    _FLASH_ATTN_3_AVAILABLE = False

if not (_FLASH_ATTN_2_AVAILABLE or _FLASH_ATTN_3_AVAILABLE):
    warnings.warn(
        'flash-attn is not installed: falling back to '
        'scaled_dot_product_attention, which is much more memory-hungry '
        'for the 14B model. Install flash-attn (see README) for normal use.')


def flash_attention_compat(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """Drop-in replacement for ``wan.modules.attention.flash_attention``.

    Delegates to the compiled flash-attn kernels when available and falls
    back to ``scaled_dot_product_attention`` otherwise. Accepts the same
    ``[B, L, N, D]`` tensor layout and returns the same layout.
    """
    if _FLASH_ATTN_2_AVAILABLE or _FLASH_ATTN_3_AVAILABLE:
        return _wan_flash_attention(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
        )

    # --- scaled_dot_product_attention fallback ---------------------------
    out_dtype = q.dtype
    b, lq, n, d = q.shape
    lk = k.size(1)
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(t):
        return t if t.dtype in half_dtypes else t.to(dtype)

    q, k, v = half(q), half(k), half(v)
    if q_scale is not None:
        q = q * q_scale

    # [B, L, N, D] -> [B, N, L, D] as expected by sdpa.
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # An explicit bool mask forces sdpa off its flash kernel; only build
    # one when there is actual padding or a window to enforce.
    k_lens_t = None
    if k_lens is not None:
        k_lens_t = torch.as_tensor(k_lens, device=q.device)
        if k_lens_t.numel() == 1:
            k_lens_t = k_lens_t.expand(b)
        if bool((k_lens_t == lk).all()):
            k_lens_t = None                # no real padding

    attn_mask = None
    if k_lens_t is not None or window_size != (-1, -1):
        device = q.device
        mask = torch.ones(b, 1, lq, lk, dtype=torch.bool, device=device)
        if k_lens_t is not None:
            # Mask padded key positions, mirroring flash-attn varlen
            # semantics. Padded query rows are left unmasked (their output
            # is undefined, same as in the varlen path) to avoid an
            # all-masked row, which would produce NaNs.
            arange_k = torch.arange(lk, device=device)
            mask &= arange_k.view(1, 1, 1, lk) < k_lens_t.view(b, 1, 1, 1)
        if window_size != (-1, -1):
            left, right = window_size
            qi = torch.arange(lq, device=device).view(1, 1, lq, 1)
            kj = torch.arange(lk, device=device).view(1, 1, 1, lk)
            if left >= 0:
                mask &= kj >= qi - left
            if right >= 0:
                mask &= kj <= qi + right
        attn_mask = mask

    scale = softmax_scale if softmax_scale is not None else 1.0 / d**0.5
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale)

    # [B, N, L, D] -> [B, L, N, D]
    return out.transpose(1, 2).contiguous().type(out_dtype)
