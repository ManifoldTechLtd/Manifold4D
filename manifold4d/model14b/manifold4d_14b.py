"""Manifold4DModel14B: 14B two-stream wrapper with input_x_add camera injection.

Built on the Wan2.1-T2V-14B base model:

  - Base: Wan2.1-T2V-14B (dim=5120, 40 layers, 40 heads, in_dim=16, out_dim=16)
  - VAE: Wan2.1 (16ch latent, 8× spatial, 4× temporal)
  - Trainable subset: self_attn + camera encoders + stream patch embeds
    (FFN and the rest of the base stay frozen)
  - camera_injection hardcoded to "input_x_add"

Topology (the released architecture):

  2-stream ``[output | source]``.  The point-cloud render is NOT a
token stream: it is folded into the output stream's flow-matching endpoint
by the sampling prior; only its (alpha, motion) mask enters, through the
zero-init ``output_anchor_patch_embed``.  Geometry is the GENERATION
STARTING POINT.

Per-block machinery: per-token time embedding, per-block zero-init
CameraEncoder Plucker add to input_x, per-block identity-init projector
after self_attn, and per-stream RoPE frame offset.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan21_t2v import (
    WanModel21T2V, sinusoidal_embedding_1d,
)
from ..model.camera_encoder import CameraEncoder
from .helpers_14b import (
    MASK_CHANNELS_14B, StreamProjector14B, make_zero_conv3d_14b,
    build_plucker_2stream_14b,
    unfreeze_trainable_14b, replicate_streams_for_rope_14b,
)


# Wan2.1-T2V-14B config (from checkpoint config.json)
T2V_14B_CONFIG = dict(
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
    eps=1e-6,
)


class Manifold4DModel14B(nn.Module):
    """14B two-stream wrapper around WanModel21T2V.

    Args:
        base_model: a loaded WanModel21T2V instance.
        positional_embedding_offset: int or None.  When set, RoPE places
            stream ``k`` at frames ``[k*offset, k*offset+f)`` instead of
            adjacent.
        cond_stream_t: "shared" | "zero" | "honest" — timestep policy for
            the conditioning streams (render / source).
        plucker_share_encoder: if True, share one CameraEncoder across all
            blocks; otherwise per-block encoders.
        skip_text_cross_attn: if True, bypass text cross-attention.
    """

    CONDITIONING_14B_SCHEMA_VERSION = 1

    def __init__(
        self,
        base_model: WanModel21T2V,
        positional_embedding_offset: Optional[int] = None,
        cond_stream_t: str = "honest",
        plucker_share_encoder: bool = False,
        skip_text_cross_attn: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        dim = base_model.dim
        num_layers = base_model.num_layers

        self.positional_embedding_offset = positional_embedding_offset
        self.cond_stream_t = cond_stream_t
        self.plucker_share_encoder = plucker_share_encoder
        self.skip_text_cross_attn = skip_text_cross_attn
        self.mask_pack_channels = MASK_CHANNELS_14B

        if cond_stream_t not in ("shared", "zero", "honest"):
            raise ValueError(
                f"cond_stream_t must be 'shared', 'zero' or 'honest', "
                f"got {cond_stream_t!r}")

        # --- Camera encoders (zero-init Linear 6→dim) ---
        if plucker_share_encoder:
            self.cam_encoder = CameraEncoder(dim)
            self.cam_encoders = None
        else:
            self.cam_encoder = None
            self.cam_encoders = nn.ModuleList([
                CameraEncoder(dim) for _ in range(num_layers)
            ])

        # --- Per-block identity-init projectors ---
        self.projectors = nn.ModuleList([
            StreamProjector14B(dim) for _ in range(num_layers)
        ])

        # --- Stream patch embeddings ---
        base_pe = base_model.patch_embedding
        kt, kh, kw = base_pe.kernel_size
        patch_kernel = (kt, kh, kw)

        # Output stream: trainable native-init RGB + zero-init anchor.
        # Trainable because x_t rides the render→GT prior, so its input
        # statistics differ from the pure-Gaussian interpolant the frozen
        # native conv was pretrained on.
        self.output_rgb_patch_embed = copy.deepcopy(base_pe)
        for p in self.output_rgb_patch_embed.parameters():
            p.requires_grad = True
        self.output_anchor_patch_embed = make_zero_conv3d_14b(
            in_channels=self.mask_pack_channels,
            out_channels=dim,
            kernel_size=patch_kernel,
            stride=patch_kernel,
        )

        # Source stream: trainable native-init RGB + zero-init mask.
        self.source_rgb_patch_embed = copy.deepcopy(base_pe)
        for p in self.source_rgb_patch_embed.parameters():
            p.requires_grad = True
        self.source_mask_patch_embed = make_zero_conv3d_14b(
            in_channels=self.mask_pack_channels,
            out_channels=dim,
            kernel_size=patch_kernel,
            stride=patch_kernel,
        )

        self.use_plucker = True  # 14B always uses Plucker
        self.camera_injection = "input_x_add"
        self.disable_camera_injection = False

        self._freeze_base()

    def _freeze_base(self):
        """Freeze base_model entirely, then unfreeze only self_attn.

        FFN stays frozen.  Top-level modules (cam_encoders, projectors,
        patch_embeds) keep their default requires_grad=True.
        """
        unfreeze_trainable_14b(self, self.base_model)

    # ──────────────────────────────────────────────────────────────
    # Weight loading (inference)
    # ──────────────────────────────────────────────────────────────

    def load_self_attn_full(self, path: str) -> int:
        state = torch.load(path, map_location="cpu", weights_only=False)
        n = 0
        for bi, block in enumerate(self.base_model.blocks):
            prefix = f"blocks.{bi}.self_attn."
            block_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not block_state:
                continue
            current = block.self_attn.state_dict()
            compatible = {
                key: value
                for key, value in block_state.items()
                if key in current and current[key].shape == value.shape
            }
            block.self_attn.load_state_dict(compatible, strict=False)
            n += len(compatible)
        return n

    def load_camera_encoder(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        share_saved = bool(state.get("plucker_share_encoder",
                                     self.plucker_share_encoder))
        if share_saved != self.plucker_share_encoder:
            raise RuntimeError(
                f"camera encoder share-mode mismatch: ckpt has "
                f"plucker_share_encoder={share_saved}, model was built "
                f"with {self.plucker_share_encoder}.")
        if self.plucker_share_encoder:
            self.cam_encoder.load_state_dict(state["cam_encoder"])
        else:
            self.cam_encoders.load_state_dict(state["cam_encoders"])

    def load_conditioning_modules(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        schema = state.get("schema_version")
        if schema is not None and schema > self.CONDITIONING_14B_SCHEMA_VERSION:
            raise RuntimeError(
                f"conditioning modules schema_version {schema} is newer "
                f"than this code supports "
                f"({self.CONDITIONING_14B_SCHEMA_VERSION}).")

        def _maybe_load(module, key):
            if key in state:
                module.load_state_dict(state[key])
                return True
            print(f"[load_conditioning_modules] ckpt has no '{key}' "
                  f"— keeping fresh init")
            return False

        _maybe_load(self.output_rgb_patch_embed, "output_rgb_patch_embed")
        _maybe_load(self.output_anchor_patch_embed, "output_anchor_patch_embed")
        _maybe_load(self.source_rgb_patch_embed, "source_rgb_patch_embed")
        _maybe_load(self.source_mask_patch_embed, "source_mask_patch_embed")
        _maybe_load(self.projectors, "projectors")

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        source_latent=None,
        render_mask=None,
        source_mask=None,
        tgt_c2ws=None,
        src_c2ws=None,
        K=None,
        pixel_height=None,
        pixel_width=None,
        cond_t_source=None,
    ):
        """2-stream forward: ``[output | source]``.

        Args:
            x: List[Tensor[16, F_lat, H_lat, W_lat]] noisy latent x_t on the
                render→GT prior.
            t: Tensor [B] or [B, seq_len] timesteps.
            context: List[Tensor[L, C]] text embeddings.
            seq_len: int max per-stream sequence length.
            render_mask: List[Tensor[2, F_lat, H_lat, W_lat]] = (alpha, motion).
            source_latent: List[Tensor[16, F_lat, H_lat, W_lat]].
            source_mask: List[Tensor[2, F_lat, H_lat, W_lat]] or None.
            tgt_c2ws, src_c2ws, K: camera tensors for Plucker.
            pixel_height, pixel_width: pixel resolution K was built at.
            cond_t_source: [B] per-sample source timestep (honest mode).

        Returns:
            List[Tensor[16, F_lat, H_lat, W_lat]] predicted velocity.
        """
        model = self.base_model
        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        if source_latent is None:
            raise ValueError(
                "two-stream forward requires source_latent")
        if render_mask is None:
            raise ValueError(
                "two-stream forward requires render_mask")
        if tgt_c2ws is None or K is None:
            raise ValueError(
                "two-stream forward requires tgt_c2ws and K")
        if src_c2ws is None:
            raise ValueError(
                "two-stream forward requires src_c2ws (the source "
                "stream carries true source rays for the 2-stream Plucker).")

        # --- 1. Patchify OUTPUT stream ---
        x_out_patch = []
        for u, w_mask in zip(x, render_mask):
            mask = w_mask[:self.mask_pack_channels].clamp(0, 1)
            rgb = self.output_rgb_patch_embed(u.unsqueeze(0))
            cov = self.output_anchor_patch_embed(mask.unsqueeze(0))
            x_out_patch.append(rgb + cov)

        grid_sizes = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long) for u in x_out_patch
        ])

        def _flatten_pad(tokens_list, pad_to):
            flat = [u.flatten(2).transpose(1, 2) for u in tokens_list]
            seq_lens_local = torch.tensor(
                [u.size(1) for u in flat], dtype=torch.long)
            assert seq_lens_local.max() <= pad_to
            stacked = torch.cat([
                torch.cat([u, u.new_zeros(1, pad_to - u.size(1), u.size(2))],
                          dim=1) for u in flat
            ])
            return stacked, seq_lens_local

        x_out, out_lens = _flatten_pad(x_out_patch, seq_len)

        # --- 2. Patchify SOURCE stream ---
        if source_mask is None:
            source_mask = []
            for s_lat in source_latent:
                sm = torch.zeros(
                    self.mask_pack_channels, *s_lat.shape[1:],
                    device=s_lat.device, dtype=s_lat.dtype)
                sm[0].fill_(1.0)
                source_mask.append(sm)

        # Resize source_mask to match source_latent spatial shape (handles
        # config resolution != dataset preprocessing resolution).
        source_mask = [
            (m if m.shape[-3:] == s.shape[-3:]
             else F.interpolate(
                 m.unsqueeze(0), size=s.shape[-3:],
                 mode='trilinear', align_corners=False).squeeze(0))
            for m, s in zip(source_mask, source_latent)
        ]

        x_src_patch = []
        for s_lat, s_mask in zip(source_latent, source_mask):
            rgb = self.source_rgb_patch_embed(s_lat.unsqueeze(0))
            msk = self.source_mask_patch_embed(s_mask.unsqueeze(0))
            x_src_patch.append(rgb + msk)

        x_src, src_lens = _flatten_pad(x_src_patch, seq_len)
        assert torch.equal(out_lens, src_lens), (
            f"output / source seq_lens differ: {out_lens} vs {src_lens}")

        # --- 3. Concatenate streams ---
        x_joint = torch.cat([x_out, x_src], dim=1)
        grid_sizes_joint = replicate_streams_for_rope_14b(grid_sizes, n_streams=2)
        seq_lens_joint = out_lens * 2
        eff_seq_len = seq_len * 2

        # RoPE stream placement: offset → per-stream grid + offset (native
        # Vista4D gap); None → adjacent joint grid (legacy).
        rope_offset = self.positional_embedding_offset
        if rope_offset is not None:
            rope_grid, rope_n_streams = grid_sizes, 2
        else:
            rope_grid, rope_n_streams = grid_sizes_joint, 1

        # --- 4. Camera injection (2-stream Plucker) ---
        cam_plucker = build_plucker_2stream_14b(
            grid_sizes_per_stream=grid_sizes,
            noise_seq_lens=out_lens, seq_len=seq_len,
            tgt_c2ws=tgt_c2ws, src_c2ws=src_c2ws, K=K,
            device=x_joint.device, dtype=torch.float32,
            pixel_height=pixel_height, pixel_width=pixel_width,
        )

        # --- 5. Time embedding (per-token over joint seq) ---
        if t.dim() == 1:
            t_out = t.unsqueeze(1).expand(t.size(0), seq_len)
            if self.cond_stream_t == "zero":
                t = torch.cat(
                    [t_out, t_out.new_zeros(t.size(0), seq_len)], dim=1)
            elif self.cond_stream_t == "honest":
                t_s = (cond_t_source.to(t_out)
                       .unsqueeze(1).expand(t.size(0), seq_len)
                       if cond_t_source is not None
                       else t_out.new_zeros(t.size(0), seq_len))
                t = torch.cat([t_out, t_s], dim=1)
            else:  # "shared"
                t = t.unsqueeze(1).expand(t.size(0), eff_seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            e = model.time_embedding(
                sinusoidal_embedding_1d(model.freq_dim,
                                        t_flat).unflatten(0, (bt, eff_seq_len)).float())
            e0 = model.time_projection(e).unflatten(2, (6, model.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # --- 6. Text context ---
        context = model.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        freqs = model.freqs

        # --- 7. Per-block forward ---
        for i, block in enumerate(model.blocks):
            cam_encoder = (self.cam_encoder if self.plucker_share_encoder
                           else self.cam_encoders[i])
            projector = self.projectors[i]

            # Call through block.__call__ (NOT an external helper) so
            # per-block FSDP wrapping triggers its all-gather pre-hook.
            x_joint = block(
                x_joint, e0, seq_lens_joint, rope_grid, freqs,
                context, None,
                cam_encoder=cam_encoder, projector=projector,
                cam_plucker=cam_plucker,
                skip_text_cross_attn=self.skip_text_cross_attn,
                rope_n_streams=rope_n_streams,
                rope_offset=rope_offset,
            )

        # --- 8. Slice output stream → head → unpatchify ---
        x_out_final = x_joint[:, :seq_len, :]
        e_out = e[:, :seq_len, :]

        x_head = model.head(x_out_final, e_out)
        x_unpatch = model.unpatchify(x_head, grid_sizes)
        return [u.float() for u in x_unpatch]
