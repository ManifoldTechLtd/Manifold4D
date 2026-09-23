# Render conditioner: encode the point-cloud render + source image into latents,
# downsample the render mask, prepare cond_mask for latent inpainting.

import torch
import torch.nn.functional as F
from typing import Optional


class RenderConditioner:
    """Prepares render conditioning inputs for the DiT model.

    Takes pixel-space point-cloud rendered video, source video, and binary render mask,
    and produces latent-space tensors ready to be concatenated with the noise
    as input to the extended patch_embedding.

    Input (pixel space):
        render_rgb:   [B, 3, T, H, W]   float in [-1, 1], global point cloud rendered to target cam
        source_rgb: [B, 3, T, H, W]   float in [-1, 1], source camera frame at same timestep
        render_mask:  [B, 1, T, H, W]   float in {0, 1}, valid pixel regions of the render

    Output (latent space):
        render_latent:   [B, 48, T', H', W']  VAE-encoded render image
        source_latent: [B, 48, T', H', W']  VAE-encoded source image
        render_mask_latent: [B, 1, T', H', W']  avg-pooled mask (pixel density)
    """

    def __init__(self, vae, vae_stride=(4, 16, 16), mask_pack_mode="avgpool"):
        """
        Args:
            vae: Wan2.1 VAE instance (with .encode() method)
            vae_stride: temporal and spatial downsampling factors (T, H, W)
            mask_pack_mode: how a pixel-resolution mask is reduced to the
                latent grid.
                * ``"avgpool"`` (default, legacy): avg-pool each patch →
                  1 channel of fractional coverage in [0, 1].
                * ``"allpack"`` (Vista4D-native): space-to-depth pack the
                  full pixel-resolution mask into ``sT*sH*sW`` channels
                  (lossless — every pixel preserved), matching native
                  Vista4D's ``downsample_allpack`` (``mask_in_channels =
                  C * sT * sH * sW``).  Preserves crisp covered/hole +
                  motion boundaries that avg-pool blurs away.  The avg-pool
                  coverage is recoverable as the per-channel mean of an
                  allpack block, so downstream prior-anchor / motion-loss
                  consumers can still derive the [0,1] coverage.
        """
        self.vae = vae
        self.vae_stride = vae_stride
        assert mask_pack_mode in ("avgpool", "allpack"), mask_pack_mode
        self.mask_pack_mode = mask_pack_mode

    @property
    def mask_pack_channels_per_input(self) -> int:
        """Output channels produced PER single-channel input mask.

        ``avgpool`` → 1; ``allpack`` → ``sT*sH*sW`` (space-to-depth block).
        A 2-input ``cat([alpha, motion])`` therefore yields
        ``2 * mask_pack_channels_per_input``.
        """
        if self.mask_pack_mode == "allpack":
            sT, sH, sW = self.vae_stride
            return sT * sH * sW
        return 1

    @torch.no_grad()
    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space video to latent using Wan VAE.

        Args:
            rgb: [B, 3, T, H, W] float in [-1, 1]

        Returns:
            latent: [B, 48, T', H', W'] where T'=(T-1)//4+1, H'=H//16, W'=W//16
        """
        B = rgb.shape[0]
        # Wan VAE expects list of [C, T, H, W] tensors
        videos = [rgb[i] for i in range(B)]
        latents = self.vae.encode(videos)
        return torch.stack(latents, dim=0)

    def downsample_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Reduce a pixel-resolution mask to the latent grid.

        Behaviour depends on ``self.mask_pack_mode``:
          * ``"avgpool"`` (default): avg-pool each patch → 1 channel of
            fractional coverage in [0, 1] (legacy).
          * ``"allpack"``: space-to-depth pack the full pixel mask into
            ``sT*sH*sW`` channels (lossless; Vista4D-native).

        Both share the Wan VAE first-frame-separate temporal layout:
        output ``T' = (T-1)//stride_t + 1`` where the first output frame
        comes from input frame 0 alone and the rest pack groups of
        ``stride_t``.

        Args:
            mask: [B, C, T, H, W] float in [0, 1] (C is usually 1).

        Returns:
            avgpool: [B, C, T', H', W'].
            allpack: [B, C*sT*sH*sW, T', H', W'].
        """
        if self.mask_pack_mode == "allpack":
            return self._pack_mask_allpack(mask)
        return self.avgpool_mask(mask)

    def avgpool_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Avg-pool a pixel mask to the latent grid → ``[B, C, T', H', W']``.

        ALWAYS avg-pool, independent of ``mask_pack_mode``.  Used for the
        loss-weighting masks (e.g. ``tgt_motion_mask_lat``, the prior
        anchor coverage) that need a single [0,1] coverage channel — these
        must NOT expand to allpack channels (that would break the
        ``1 + lambda*motion`` per-element weighting in the loss).
        """
        stride_t, stride_h, stride_w = self.vae_stride
        B, C, T, H, W = mask.shape

        # Wan VAE temporal: first frame kept separately, then groups of stride_t
        # Output T' = (T - 1) // stride_t + 1
        # Split: first frame + remaining (T-1) frames (divisible by stride_t)
        first_frame = mask[:, :, :1, :, :]  # [B, 1, 1, H, W]
        rest_frames = mask[:, :, 1:, :, :]  # [B, 1, T-1, H, W]

        # Spatial downsample for first frame
        first_down = F.avg_pool3d(
            first_frame,
            kernel_size=(1, stride_h, stride_w),
            stride=(1, stride_h, stride_w),
        )  # [B, 1, 1, H', W']

        if rest_frames.shape[2] > 0:
            # Temporal + spatial downsample for remaining frames
            rest_down = F.avg_pool3d(
                rest_frames,
                kernel_size=(stride_t, stride_h, stride_w),
                stride=(stride_t, stride_h, stride_w),
            )  # [B, 1, (T-1)//stride_t, H', W']
            mask_latent = torch.cat([first_down, rest_down], dim=2)
        else:
            mask_latent = first_down

        return mask_latent

    def _pack_mask_allpack(self, mask: torch.Tensor) -> torch.Tensor:
        """Space-to-depth pack a pixel mask → ``[B, C*sT*sH*sW, T', H', W']``.

        Mirrors native Vista4D's ``downsample_allpack``: replicate the
        first frame ``sT`` times so the temporal length becomes ``sT*T'``
        (matching the VAE's 1+sT*(T'-1) layout), then rearrange
        ``(t st)(h sh)(w sw) -> (c st sh sw) t h w``.  Lossless: every
        pixel of the mask is preserved as a channel, and the per-block
        channel-mean recovers the avgpool coverage.

        Args:
            mask: [B, C, T_pix, H_pix, W_pix] float in [0, 1] where
                ``T_pix = 1 + sT*(T'-1)``.

        Returns:
            [B, C*sT*sH*sW, T', H', W'].
        """
        sT, sH, sW = self.vae_stride
        B, C, T, H, W = mask.shape
        assert H % sH == 0 and W % sW == 0, (
            f"mask spatial dims {(H, W)} not divisible by stride {(sH, sW)}")
        # First-frame-separate temporal: replicate frame 0 sT times so the
        # total temporal length is divisible by sT and the first latent
        # frame's sT slots all come from input frame 0 (matches the VAE).
        first = mask[:, :, :1].repeat(1, 1, sT, 1, 1)          # [B,C,sT,H,W]
        rest = mask[:, :, 1:]                                  # [B,C,sT*(T'-1),H,W]
        full = torch.cat([first, rest], dim=2)                 # [B,C,sT*T',H,W]
        Tp = full.shape[2] // sT
        Hp, Wp = H // sH, W // sW
        # (t st)(h sh)(w sw) -> (c st sh sw) t h w
        x = full.view(B, C, Tp, sT, Hp, sH, Wp, sW)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * sT * sH * sW, Tp, Hp, Wp)
        return x

    def make_cond_mask(
        self,
        latent_shape: tuple,
        cond_frame_indices: Optional[list] = None,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """Build condition video mask for latent inpainting.

        Marks which latent frames are 'known' (from previous chunk overlap
        or initial condition). 1 = known/condition, 0 = to be generated.

        Args:
            latent_shape: (B, C, T', H', W') shape of noise latent
            cond_frame_indices: list of latent frame indices that are conditioned.
                                If None, returns all-zeros (no conditioning).
            device: target device

        Returns:
            cond_mask: [B, 1, T', H', W'] float
        """
        B, _, T, H, W = latent_shape
        cond_mask = torch.zeros(B, 1, T, H, W, device=device)
        if cond_frame_indices is not None:
            for idx in cond_frame_indices:
                if 0 <= idx < T:
                    cond_mask[:, :, idx, :, :] = 1.0
        return cond_mask

    @torch.no_grad()
    def __call__(
        self,
        render_rgb: torch.Tensor,
        source_rgb: torch.Tensor,
        render_mask: torch.Tensor,
        cond_frame_indices: Optional[list] = None,
    ) -> dict:
        """Prepare all render conditioning tensors.

        Args:
            render_rgb:   [B, 3, T, H, W] float in [-1, 1]
            source_rgb: [B, 3, T, H, W] float in [-1, 1]
            render_mask:  [B, 1, T, H, W] float in {0, 1}
            cond_frame_indices: latent frame indices for inpainting mask

        Returns:
            dict with keys:
                render_latent:   [B, 48, T', H', W']
                source_latent: [B, 48, T', H', W']
                render_mask_latent: [B, 1, T', H', W']
                cond_mask:     [B, 1, T', H', W']
        """
        render_latent = self.encode_rgb(render_rgb)
        source_latent = self.encode_rgb(source_rgb)
        render_mask_latent = self.downsample_mask(render_mask)

        cond_mask = self.make_cond_mask(
            latent_shape=render_latent.shape,
            cond_frame_indices=cond_frame_indices,
            device=render_latent.device,
        )

        return {
            "render_latent": render_latent,
            "source_latent": source_latent,
            "render_mask_latent": render_mask_latent,
            "cond_mask": cond_mask,
        }

    @staticmethod
    def concat_conditions(
        noise: torch.Tensor,
        render_latent: torch.Tensor,
        source_latent: torch.Tensor,
        render_mask_latent: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate all inputs along channel dim for patch_embedding.

        Args:
            noise:           [B, 48, T', H', W']
            render_latent:     [B, 48, T', H', W']
            source_latent:   [B, 48, T', H', W']
            render_mask_latent:[B, 1, T', H', W']
            cond_mask:       [B, 1, T', H', W']

        Returns:
            x: [B, 146, T', H', W'] ready for patch_embedding
        """
        return torch.cat(
            [noise, render_latent, source_latent, render_mask_latent, cond_mask],
            dim=1,
        )
