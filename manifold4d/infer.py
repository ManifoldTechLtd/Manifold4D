"""Manifold4D inference core: model loading, two-stream sampling, VAE decode.

This module contains only the released-model path: Wan2.1-T2V-14B base
with two-stream conditioning. All checkpoint paths are passed in by
the caller (see ``manifold4d/generate.py``, which resolves them from the
config ``paths:`` section / environment variables / CLI flags).

The Wan2.1 source tree must be importable (``WAN_SRC`` / ``external/Wan2.1``)
because the model code imports ``wan.*`` modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (released model only)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, device,
               dtype=torch.bfloat16, use_text: bool = True,
               use_plucker: bool = True,
               plucker_share_encoder: bool = False,
               camera_injection: str = "input_x_add",
               mask_pack_channels: int = 2,
               positional_embedding_offset=None,
               cond_stream_t: str = "honest",
               wan_checkpoint: str = None):
    """Load the released Manifold4D model (Wan2.1-T2V-14B + two-stream).

    ``wan_checkpoint`` is the Wan2.1-T2V-14B base model directory; the caller
    resolves it from the config ``paths:`` section / env vars / CLI flags.

    Checkpoint layout (``checkpoint_dir``):
      * ``conditioning_modules.pt``  — stream patchify + per-block projectors
      * ``camera_encoder.pt``       — Plucker camera encoder
      * ``self_attn_full.pt``       — full self-attn QKVO/RMSNorm state
    """
    from manifold4d.model14b import WanModel21T2V, Manifold4DModel14B

    if not wan_checkpoint:
        raise ValueError(
            "wan_checkpoint is required (Wan2.1-T2V-14B base model dir). "
            "Set `wan_checkpoint` in the config `paths:` section, the "
            "WAN21_CHECKPOINT environment variable, or the --wan_checkpoint flag.")
    print(f"Loading Wan2.1-T2V-14B from {wan_checkpoint}...")
    base_model = WanModel21T2V.from_pretrained(wan_checkpoint)
    model = Manifold4DModel14B(
        base_model,
        positional_embedding_offset=positional_embedding_offset,
        cond_stream_t=cond_stream_t,
        plucker_share_encoder=plucker_share_encoder,
    )
    model.mask_pack_channels = mask_pack_channels

    skip_text = not use_text
    ckpt = Path(checkpoint_dir)

    # No LoRA wrap — the trained subset is self_attn + camera encoder +
    # stream patchify + per-block projectors.  Mirrors the training-time
    # resume path so inference sees the exact trained module state.
    wb_path = ckpt / "conditioning_modules.pt"
    if wb_path.exists():
        print(f"  Loading conditioning_modules from {wb_path}")
        model.load_conditioning_modules(str(wb_path))
    else:
        print(f"  ⚠ conditioning_modules.pt missing at {wb_path} — "
              f"stream patchify stays at init (results will be poor)")

    # Plucker camera encoder — required under camera_injection="input_x_add".
    cam_path = ckpt / "camera_encoder.pt"
    needs_cam_encoder = (use_plucker
                         and getattr(model, "camera_injection", None)
                         == "input_x_add")
    if needs_cam_encoder and cam_path.exists():
        print(f"  Loading camera_encoder from {cam_path}")
        model.load_camera_encoder(str(cam_path))
    elif needs_cam_encoder:
        print(f"  ⚠ camera_encoder.pt missing at {cam_path} — "
              f"Plucker bias will be random init")
    elif use_plucker and cam_path.exists():
        model.load_camera_encoder(str(cam_path))

    # Full self-attn QKVO + RMSNorm state (self_attn_full.pt present when the
    # training run skipped the LoRA wrap on self-attn).
    sa_full_path = ckpt / "self_attn_full.pt"
    if sa_full_path.exists():
        print(f"  Loading self_attn full weights from {sa_full_path}")
        model.load_self_attn_full(str(sa_full_path))

    model.skip_text_cross_attn = skip_text
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_diffusion(model, target_latent, render_latent, source_latent, render_mask,
                  text_emb, device, num_steps=50, guidance_scale=5.0, shift=5.0,
                  seed=42, cam_kwargs=None, source_mask=None,
                  prior_sigma: float = 0.3, neg_text_emb=None,
                  churn_gmax: float = 0.0):
    """Reverse-time flow-matching sampling for the two-stream model.

    The ODE starts from the render→GT prior prior (the t→1 endpoint the model
    was trained on), not pure noise: covered tokens start near the render
    render (+ small sigma noise), hole tokens start from pure Gaussian.

    CFG is text-only (ReCamMaster-style): the unconditional branch keeps all
    geometric conditioning (source / render / camera) and only swaps the text
    for the negative prompt, so the guidance direction is purely textual.

    Returns the predicted clean latent in shape [C, T, H, W].
    """
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    T_lat, H_lat, W_lat = target_latent.shape[1:]
    _latent_ch = target_latent.shape[0]
    seq_len = T_lat * (H_lat // 2) * (W_lat // 2)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    eps0 = torch.randn(_latent_ch, T_lat, H_lat, W_lat, device=device, generator=g)

    # Prior ODE start point: anchor = coverage(alpha) channel of
    # render_mask_lat ∈ [0, 1].  Under mask_pack_mode="allpack" render_mask is
    # the space-to-depth pack [alpha_block | motion_block] → recover the
    # avg-pool alpha as the channel-mean over the alpha block (first half);
    # under avg-pool this reduces to channel 0.
    _n_alpha = render_mask.shape[0] // 2
    anchor0 = render_mask[:_n_alpha].mean(0, keepdim=True).clamp(0, 1).float()
    latent = (anchor0 * (render_latent.float() + prior_sigma * eps0)
              + (1.0 - anchor0) * eps0).to(eps0.dtype)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_steps, device=device, shift=shift)

    cam_kwargs = dict(cam_kwargs) if cam_kwargs else {}
    cond_kwargs = dict(
        source_latent=[source_latent],
        render_mask=[render_mask],
        **cam_kwargs,
    )
    if source_mask is None:
        raise ValueError(
            "source_mask is required: [2, T_lat, H_lat, W_lat] "
            "(alpha + motion) but got None.")
    cond_kwargs["source_mask"] = [source_mask]

    # CFG uncond TEXT: Wan2.1 was trained with a real negative prompt as the
    # unconditional branch — zero-text is OOD, so use the pre-encoded neg
    # prompt when provided; else fall back to zero-text.
    if neg_text_emb is not None:
        z_text = neg_text_emb.to(device=text_emb.device, dtype=text_emb.dtype)
    else:
        z_text = torch.zeros_like(text_emb)

    use_cfg = abs(guidance_scale - 1.0) > 1e-3
    ts = scheduler.timesteps

    def _churn(lat, i):
        # Anchor-gated SDE churn: after a denoising step, renoise the COVERED
        # tokens by the mid-bump c_t at the NEXT noise level so subsequent
        # UniPC steps denoise it.  churn_gmax=0 → deterministic ODE.
        if not (churn_gmax > 0.0 and i < len(ts) - 1):
            return lat
        tau = float(ts[i + 1]) / 1000.0            # next-step normalised time
        c = churn_gmax * (tau * (1.0 - tau)) ** 0.5
        return lat + anchor0.to(lat.dtype) * c * torch.randn(
            lat.shape, device=device, dtype=lat.dtype, generator=g)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i, tv in enumerate(ts):
            t_t = torch.tensor([tv], device=device)
            pred_c = model(x=[latent], t=t_t, context=[text_emb],
                           seq_len=seq_len, **cond_kwargs)[0]
            if use_cfg:
                pred_u = model(x=[latent], t=t_t, context=[z_text],
                               seq_len=seq_len, **cond_kwargs)[0]
                pred = pred_u + guidance_scale * (pred_c - pred_u)
            else:
                pred = pred_c
            latent = scheduler.step(
                pred.unsqueeze(0), tv, latent.unsqueeze(0),
                return_dict=False, generator=g,
            )[0].squeeze(0)
            latent = _churn(latent, i)
    return latent


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────

def latent_to_frames_np(latent: torch.Tensor, vae) -> np.ndarray:
    """Decode [C, T, H, W] latent → uint8 [T, H, W, 3] frames."""
    with torch.no_grad():
        video = vae.decode([latent.float()])[0]  # [3, T, H, W] in [-1, 1]
    video = (video.clamp(-1, 1) + 1) / 2 * 255.0
    return video.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
