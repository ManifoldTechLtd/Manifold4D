"""Plucker-coordinate camera encoder for the two-stream model.

Provides:
    * ``compute_plucker`` — per-pixel 6D Plucker ray embeddings at an
      arbitrary (H, W) grid resolution, given camera intrinsics + c2w
      extrinsics.  Adapted from Vista4D / AC3D.
    * ``CameraEncoder`` — zero-initialised Linear(6 -> dim) that lifts
      Plucker rays into the DiT token dimension.  When added as a
      KQ-bias to self-attention this acts as a ControlNet-style
      geometric anchor: at training start the bias is identically zero
      so the model behaves exactly like the Wan baseline; gradients
      slowly teach it to use camera information.

The encoder is *intentionally* tiny (6 -> dim Linear) so the inductive
bias of the Plucker representation does the heavy lifting:

    Plucker(p) = (origin x direction, direction)         in R^6
    Plucker(p) . Plucker(q) is high when rays (p, q) intersect

which is exactly the epipolar-consistency test we want self-attention
to learn.  A deeper MLP would obscure that geometric structure.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn


def _intrinsics_from_K(K: torch.Tensor, H_tok: int, W_tok: int,
                       H_pix: int, W_pix: int) -> torch.Tensor:
    """Convert a 3x3 K (at pixel resolution H_pix x W_pix) to the
    [fx, fy, cx, cy] quadruple at the token-grid resolution
    (H_tok x W_tok).  Plucker rays are scale-invariant so any
    consistent (K_scaled, grid) pair produces the same rays.

    Args:
        K: [B, 3, 3] or [3, 3] camera intrinsics at pixel resolution.
        H_tok, W_tok: target token grid (post-VAE, post-patchify).
        H_pix, W_pix: pixel resolution the K matrix was built at.

    Returns:
        [B, 4] tensor of (fx, fy, cx, cy) at the token grid.
    """
    if K.dim() == 2:
        K = K.unsqueeze(0)
    sx = W_tok / float(W_pix)
    sy = H_tok / float(H_pix)
    fx = K[..., 0, 0] * sx
    fy = K[..., 1, 1] * sy
    cx = K[..., 0, 2] * sx
    cy = K[..., 1, 2] * sy
    return torch.stack([fx, fy, cx, cy], dim=-1)


def compute_plucker(
    intrinsics: torch.Tensor,  # [B, F, 4]  ([fx, fy, cx, cy] @ token grid)
    cam_c2w: torch.Tensor,     # [B, F, 4, 4]
    height: int,
    width: int,
) -> torch.Tensor:
    """Per-pixel 6D Plucker embedding at the latent token grid.

    Args:
        intrinsics: [B, F, 4] = (fx, fy, cx, cy) already at the
            output (height, width) resolution.
        cam_c2w: [B, F, 4, 4] camera-to-world matrices.
        height, width: target grid resolution (typically the DiT
            token grid after patchify, e.g. H_pix / 32).

    Returns:
        plucker: [B, F, height, width, 6] where the last dim is
            (origin x direction, direction).
    """
    custom_meshgrid = partial(torch.meshgrid, indexing="ij")
    B, F = intrinsics.shape[:2]
    device, dtype = cam_c2w.device, cam_c2w.dtype

    j, i = custom_meshgrid(
        torch.linspace(0, height - 1, height, device=device, dtype=dtype),
        torch.linspace(0, width - 1, width, device=device, dtype=dtype),
    )
    # Use half-pixel centres + broadcast across batch / frames.
    i = (i + 0.5).reshape(1, 1, -1).expand(B, F, -1)   # [B, F, H*W]
    j = (j + 0.5).reshape(1, 1, -1).expand(B, F, -1)

    fx, fy, cx, cy = intrinsics.chunk(4, dim=-1)        # each [B, F, 1]
    zs = torch.ones_like(i)                             # [B, F, H*W]
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    directions = torch.stack((xs, ys, zs), dim=-1)      # [B, F, H*W, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Rotate into world frame.
    R = cam_c2w[..., :3, :3]                            # [B, F, 3, 3]
    rays_d = directions @ R.transpose(-1, -2)           # [B, F, H*W, 3]
    rays_o = cam_c2w[..., :3, 3].unsqueeze(-2).expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)     # [B, F, H*W, 6]
    return plucker.reshape(B, F, height, width, 6)


def latent_indices_for_video_frames(num_latent_frames: int,
                                    temporal_factor: int = 4) -> list[int]:
    """Return the video-frame indices that align with each latent frame.

    Wan/Vista4D camera conditioning downsamples per-frame camera rays by
    the temporal factor, i.e. video-frame indices [0, 4, 8, ...] for
    factor 4.  This keeps Plucker conditioning aligned with the latent
    time grid used by the DiT.
    """
    return [min(temporal_factor * k, temporal_factor * (num_latent_frames - 1))
            for k in range(num_latent_frames)]


class CameraEncoder(nn.Module):
    """Project Plucker rays (last dim 6) into DiT token features (dim).

    Zero-initialised so at training start the cam_kq_bias added to
    self-attention is identically zero — model is exactly the Wan
    baseline at step 0 and gradually learns to use camera information.

    Accepts any leading shape: ``[..., 6]`` -> ``[..., dim]``.  Callers
    typically feed a pre-padded joint sequence ``[B, L_eff, 6]`` so the
    output ``[B, L_eff, dim]`` can be added directly to the Q / K
    linears inside self-attention via ``self_attn_with_kq_bias``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(6, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        return self.proj(plucker)
