"""Offset-aware RoPE helpers for the multi-stream joint self-attention.

The two-stream model concatenates ``[output | source]`` token streams
along the sequence dim and needs a per-stream frame offset in RoPE space
so the source stream is not read as a temporal continuation of the
output stream.
"""

from __future__ import annotations

import torch

from manifold4d.model.attention_compat import flash_attention_compat


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_offset(x, grid_sizes, freqs, n_streams, offset):
    """3D RoPE for an n-stream joint seq with a positional-embedding offset.

    Mirrors native Vista4D's ``get_freqs``: stream ``k`` occupies the frame
    range ``[k*offset, k*offset+f)`` instead of the contiguous ``[k*f,
    (k+1)*f)`` that the stock ``rope_apply`` produces when fed a joint
    ``grid_sizes=[n*f, h, w]``.  A larger ``offset`` separates the streams in
    RoPE space so the source stream is NOT read as a temporal continuation
    of the output stream.

    Args:
        x: ``[B, L_joint, n_heads, head_dim]`` — L_joint = n_streams*f*h*w
            (+ optional right padding, left untouched as in stock rope_apply).
        grid_sizes: ``[B, 3]`` PER-STREAM ``(f, h, w)`` (NOT multiplied).
        freqs: ``[1024, head_dim//2]`` precomputed table (``model.freqs``).
        n_streams: number of streams concatenated along the seq dim.
        offset: per-stream frame stride in RoPE space (``offset >= f``).
    """
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        fhw = f * h * w
        ohw = offset * h * w
        F = offset * (n_streams - 1) + f                 # max frame index used
        freqs_all = torch.cat([
            freqs[0][:F].view(F, 1, 1, -1).expand(F, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(F, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(F, h, w, -1),
        ], dim=-1).reshape(F * h * w, 1, -1)
        # Per-stream slice: stream k → freqs_all[k*ohw : k*ohw + fhw].
        freqs_i = torch.cat(
            [freqs_all[k * ohw: k * ohw + fhw] for k in range(n_streams)],
            dim=0)                                       # [n_streams*fhw, 1, c]
        seq_len = n_streams * fhw
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def self_attn_offset(self_attn, x, seq_lens, grid_sizes, freqs,
                     n_streams, offset):
    """``WanSelfAttention.forward`` with offset-aware RoPE (no extra bias).

    Used by the input_x_add path so the joint self-attention gets the
    per-stream RoPE offset.  Bit-identical to the stock self-attn when
    ``n_streams==1`` (single-stream, offset unused).
    """
    b, s, n, d = *x.shape[:2], self_attn.num_heads, self_attn.head_dim
    q = self_attn.norm_q(self_attn.q(x)).view(b, s, n, d)
    k = self_attn.norm_k(self_attn.k(x)).view(b, s, n, d)
    v = self_attn.v(x).view(b, s, n, d)
    x_out = flash_attention_compat(
        q=rope_apply_offset(q, grid_sizes, freqs, n_streams, offset),
        k=rope_apply_offset(k, grid_sizes, freqs, n_streams, offset),
        v=v,
        k_lens=seq_lens,
        window_size=self_attn.window_size,
    )
    return self_attn.o(x_out.flatten(2))
