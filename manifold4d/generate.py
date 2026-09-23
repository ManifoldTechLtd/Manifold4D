"""Generate a Manifold4D video from preprocessed scene data.

This is the generation entry point of the released inference pipeline.  It
consumes the outputs of ``scripts/preprocess/`` and runs the two-stream
diffusion model for ONE novel-view camera trajectory:

    <scene_dir>/                                   # scripts/preprocess output
    ├── frames/000000.png ...                      # source frames
    ├── predictions.npz                            # VGGT-Omega depth/intrinsics
    ├── trajectory/trajectory.npz                  # source camera trajectory
    ├── mask_dynamic/000000.png ...                # dynamic masks (SAM3)
    ├── caption.txt                                # Qwen2.5-VL caption
    └── text_embedding.pt                          # T5 text embedding
    <trajectory>/trajectory_new/trajectory.npz     # target novel-view trajectory

The pipeline mirrors the paper's evaluation flow:

  1. ``scripts/preprocess/run_vggt_caption_t5_sam3_traj.py`` rebuilds the
     source scene (VGGT-Omega → Qwen caption → T5 → SAM3 masks) and
     synthesises novel-view trajectories with ``build_val_traj.py``.
  2. ``python -m manifold4d.generate`` unprojects the source point cloud,
     renders the two-stream conditioning video in the target view, and runs
     the released Manifold4D model (Wan2.1-T2V-14B base).

Usage::

    python -m manifold4d.generate \\
        --scene_dir   data/raw/dog_vggt \\
        --trajectory  data/val_traj/dog/f1/trajectory_new/trajectory.npz \\
        --checkpoint  checkpoints/manifold4d \\
        --output_dir  output/dog_f1 \\
        --num_steps 20

Model and asset paths are configured in ``configs/manifold4d.yaml``
(``paths:`` section); CLI flags and environment variables override them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_config_paths(config_path: str) -> dict:
    """Read the ``paths:`` section of a model config YAML."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f).get("paths", {})


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory / frame helpers
# ─────────────────────────────────────────────────────────────────────────────


def _c2w_from_traj(traj, frame_indices: np.ndarray) -> np.ndarray:
    """Build [T, 4, 4] c2w matrices from a trajectory.npz dict.

    Uses ``R_world_from_cam`` (cam→world rotation) and ``centers`` (camera
    position in world), which together define c2w directly without having to
    invert the packed ``extrinsic`` (which is w2c).
    """
    R = traj["R_world_from_cam"]   # (T_total, 3, 3)
    C = traj["centers"]            # (T_total, 3)
    out = np.zeros((len(frame_indices), 4, 4), dtype=np.float32)
    for i, fi in enumerate(frame_indices):
        out[i, :3, :3] = R[int(fi)]
        out[i, :3, 3] = C[int(fi)]
        out[i, 3, 3] = 1.0
    return out


def _load_frame_rgb(path: Path) -> np.ndarray:
    """Read an image file as RGB uint8, Pillow-first for libpng robustness."""
    from PIL import Image

    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except FileNotFoundError:
        raise
    except Exception:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read frame: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _find_mask_dir(scene_dir: Path):
    """Auto-detect a per-frame dynamic-mask directory (first ``mask*`` match)."""
    candidates = sorted(
        d for d in scene_dir.glob("mask*")
        if d.is_dir() and any(d.glob("*.png")))
    return candidates[0] if candidates else None


def _dilate_mask(mask_2d, radius=3):
    """Binary dilate a 2D boolean mask by *radius* pixels (square kernel).

    Expands dynamic-object boundaries so edge pixels (with potentially mixed
    depth/color) are also classified as dynamic — prevents ghost-edge bleeding
    in the combined static point cloud.
    """
    if radius <= 0 or not mask_2d.any():
        return mask_2d
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(mask_2d.astype(np.uint8), k).astype(np.bool_)


def backproject_colored_points(video, depths, c2w, intr, sky_mask=None,
                               dynamic_mask=None, point_stride=1,
                               dyn_dilate_radius=3):
    """Backproject source depth maps -> (pts, cols, frame_ids, is_dynamic).

    Returns arrays matching the preprocessed-scene schema expected by the
    two-stream model.
    ``dyn_dilate_radius``: dilate dynamic mask by this many pixels to absorb
    boundary pixels with mixed depth (avoids ghost edges in static cloud).
    """
    f, h, w = depths.shape
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    all_pts, all_cols, all_fids, all_dyn = [], [], [], []
    for i in range(f):
        d = depths[i].reshape(-1)
        valid = np.isfinite(d) & (d > 1e-5)
        if sky_mask is not None:
            valid &= ~sky_mask[i].reshape(-1)
        if not valid.any():
            continue
        if point_stride > 1:
            mask_2d = np.zeros(h * w, dtype=bool)
            mask_2d[::point_stride] = True
            valid &= mask_2d
        K_i = intr[i]
        if K_i.shape == (3, 3):
            fx, fy = float(K_i[0, 0]), float(K_i[1, 1])
            cx, cy = float(K_i[0, 2]), float(K_i[1, 2])
        else:
            fx, fy, cx, cy = (float(x) for x in K_i)
        z = d[valid]
        cam = np.stack(
            ((xs[valid] - cx) * z / fx, (ys[valid] - cy) * z / fy, z), axis=1
        )
        world = cam @ c2w[i, :3, :3].T + c2w[i, :3, 3]
        colors = video[i].reshape(-1, 3)[valid].astype(np.float32) / 255.0
        all_pts.append(world.astype(np.float32))
        all_cols.append(colors.astype(np.float32))
        all_fids.append(np.full(world.shape[0], i, dtype=np.int32))
        if dynamic_mask is not None:
            dm_frame = _dilate_mask(dynamic_mask[i], radius=dyn_dilate_radius)
            dyn = dm_frame.reshape(-1)[valid].astype(np.bool_)
            all_dyn.append(dyn)
        else:
            all_dyn.append(np.zeros(world.shape[0], dtype=np.bool_))
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    fids = np.concatenate(all_fids, axis=0)
    is_dyn = np.concatenate(all_dyn, axis=0)
    return pts, cols, fids, is_dyn


def save_video_mp4(frames, path, fps=24.0):
    """Save (N, H, W, 3) uint8 RGB array as mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def fit_dimensions_to_area(canvas_h, canvas_w, image_h, image_w, block_size=16):
    """Scale (image_h, image_w) to match the canvas *area* while keeping the
    source aspect ratio, then snap each side to a multiple of ``block_size``
    (Wan2GP's ``calculate_new_dimensions`` with fit_into_canvas == 0).
    Returns (new_h, new_w).
    """
    if image_h <= 0 or image_w <= 0:
        return canvas_h, canvas_w
    scale = (canvas_h * canvas_w / float(image_h * image_w)) ** 0.5
    new_h = max(block_size, int(round(image_h * scale / block_size)) * block_size)
    new_w = max(block_size, int(round(image_w * scale / block_size)) * block_size)
    return new_h, new_w


# ─────────────────────────────────────────────────────────────────────────────
# Sample building (preprocess outputs -> model input)
# ─────────────────────────────────────────────────────────────────────────────


def select_frame_ids(T_total: int, num_frames: int, sampling: str,
                     frame_offset: int) -> np.ndarray:
    if (num_frames - 1) % 4 != 0:
        raise ValueError(f"num_frames={num_frames} must satisfy 4n+1 (Wan VAE)")
    if num_frames > T_total:
        raise ValueError(
            f"num_frames={num_frames} exceeds the {T_total} frames available "
            f"in predictions.npz")
    if sampling == "linspace":
        # Evenly resample across the WHOLE clip so the rendered video traverses
        # the full designed target trajectory.
        return np.linspace(0, T_total - 1, num_frames).round().astype(np.int32)
    if sampling == "window":
        offset = T_total + frame_offset if frame_offset < 0 else frame_offset
        if offset < 0 or num_frames + offset > T_total:
            raise ValueError(
                f"requested frames [{offset}, {offset + num_frames}) but only "
                f"{T_total} are available")
        return np.arange(offset, offset + num_frames, dtype=np.int32)
    raise ValueError(f"frame_sampling must be 'linspace' or 'window', got "
                     f"{sampling!r}")


def load_scene_sample(scene_dir, trajectory_path, args):
    """Build a generation sample from the scripts/preprocess output layout.

    Returns (sample, text_emb, neg_text_emb); also stashes ``args.fps``.
    """
    scene_dir = Path(scene_dir)
    pred = np.load(scene_dir / "predictions.npz")
    depth_all = pred["depth"].astype(np.float32)
    if depth_all.ndim == 4:
        depth_all = depth_all[..., 0]
    src_intr = pred["intrinsic"].astype(np.float32)        # (T, 3, 3)
    T_total, Hd, Wd = depth_all.shape

    frame_ids = select_frame_ids(
        T_total, args.video_length, args.frame_sampling, args.frame_offset)

    # Source cameras from the input trajectory; target from the bucket file.
    traj_in = np.load(scene_dir / "trajectory" / "trajectory.npz")
    src_c2ws = _c2w_from_traj(traj_in, frame_ids)
    traj_new = np.load(trajectory_path, allow_pickle=True)
    tgt_c2ws = _c2w_from_traj(traj_new, frame_ids)
    # Rendering uses ONE intrinsic at the depth-grid resolution.  Prefer the
    # target trajectory's own K; fall back to the median source intrinsic.
    if "intrinsic" in traj_new:
        K = traj_new["intrinsic"][int(frame_ids[0])].astype(np.float32)
    else:
        K = np.median(src_intr, axis=0).astype(np.float32)

    # fps: CLI override > source trajectory metadata > 24.
    if args.fps <= 0:
        args.fps = float(traj_in.get("fps", np.asarray(24.0, dtype=np.float32)))
    # Consistency anchor: build_val_traj records the scene_scale it saw.  VGGT
    # reconstructions have arbitrary scale per run, so a regenerated
    # predictions.npz silently invalidates an older trajectory — surface it.
    traj_meta_scale = None
    try:
        if "meta" in traj_new:
            meta = traj_new["meta"].item()
            if isinstance(meta, str):
                meta = json.loads(meta)
            if isinstance(meta, dict):
                traj_meta_scale = meta.get("scene_scale")
    except Exception:
        pass

    # Source frames: depth-grid resolution for point colors, VAE resolution
    # for the encoded source video.
    video_dg = np.zeros((len(frame_ids), Hd, Wd, 3), dtype=np.uint8)
    src_imgs = []
    for i, fi in enumerate(frame_ids):
        rgb = _load_frame_rgb(scene_dir / "frames" / f"{int(fi):06d}.png")
        if rgb.shape[:2] != (Hd, Wd):
            video_dg[i] = cv2.resize(rgb, (Wd, Hd), interpolation=cv2.INTER_LANCZOS4)
        else:
            video_dg[i] = rgb
        if rgb.shape[:2] != (args.height, args.width):
            rgb = cv2.resize(rgb, (args.width, args.height),
                             interpolation=cv2.INTER_LANCZOS4)
        src_imgs.append(rgb)
    src_arr = np.stack(src_imgs, axis=0).astype(np.float32) / 127.5 - 1.0
    src_video = torch.from_numpy(src_arr).permute(3, 0, 1, 2).contiguous()

    # Dynamic masks (optional): auto-detect mask*/, align to the depth grid.
    mask_path = Path(args.mask_dir) if args.mask_dir else _find_mask_dir(scene_dir)
    dynamic_mask = None
    if mask_path is not None:
        dynamic_mask = np.zeros((len(frame_ids), Hd, Wd), dtype=bool)
        for i, fi in enumerate(frame_ids):
            m = cv2.imread(str(mask_path / f"{int(fi):06d}.png"),
                           cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(
                    f"Cannot read mask: {mask_path / f'{int(fi):06d}.png'}")
            if m.shape != (Hd, Wd):
                m = cv2.resize(m, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
            dynamic_mask[i] = m > 127

    # Depth confidence gating (VGGT depth_conf ~ [1, 20]; ~3.0 is sensible).
    depths = depth_all[frame_ids]
    if args.depth_conf_threshold is not None:
        conf = pred["depth_conf"][frame_ids]
        depths = np.where(conf >= args.depth_conf_threshold,
                          depths, np.float32(0.0))

    pts, cols, fids, is_dyn = backproject_colored_points(
        video_dg, depths, src_c2ws, src_intr[frame_ids],
        dynamic_mask=dynamic_mask,
        point_stride=args.point_stride,
        dyn_dilate_radius=args.dyn_dilate_radius,
    )

    # Drift check: trajectory meta scene_scale vs current point-cloud scale.
    if traj_meta_scale is not None and pts.shape[0] > 0:
        cur_scale = float(np.ptp(pts, axis=0).max())
        if cur_scale > 1e-9:
            ratio = float(traj_meta_scale) / cur_scale
            if not (0.5 < ratio < 2.0):
                print(
                    f"[warn] trajectory meta scene_scale={traj_meta_scale:.4f} "
                    f"vs current point-cloud scale={cur_scale:.4f} "
                    f"(ratio {ratio:.2f}) — predictions.npz was likely "
                    f"regenerated since this trajectory was built; "
                    f"rerun build_val_traj.py")

    # Source motion mask at VAE resolution (2-channel source_mask motion slot).
    src_motion_mask = None
    if dynamic_mask is not None:
        motion = np.stack([
            cv2.resize(m.astype(np.float32), (args.width, args.height),
                       interpolation=cv2.INTER_NEAREST)
            for m in dynamic_mask
        ], axis=0).astype(np.float32)                 # [T, H, W]
        src_motion_mask = torch.from_numpy(motion[None])

    # Text: preprocess writes text_embedding.pt next to caption.txt.
    text_emb = None
    text_path = (Path(args.text_embedding) if args.text_embedding
                 else scene_dir / "text_embedding.pt")
    if text_path.is_file():
        text_emb = torch.load(text_path, map_location="cpu")
        print(f"[text] loaded {tuple(text_emb.shape)} from {text_path}")
    caption_path = scene_dir / "caption.txt"
    if caption_path.is_file():
        print(f"[text] caption: {caption_path.read_text().strip()[:120]}")

    # CFG negative prompt (pre-encoded; see scripts/preprocess/encode_neg_prompt.py).
    # Path priority: --neg_prompt_emb flag > config `paths.neg_prompt_emb`.
    neg_text_emb = None
    neg_path = (Path(args.neg_prompt_emb) if args.neg_prompt_emb
                else Path(load_config_paths(args.config).get("neg_prompt_emb", "")))
    if neg_path.is_file():
        neg_text_emb = torch.load(neg_path, map_location="cpu")
        print(f"[text] negative prompt {tuple(neg_text_emb.shape)} "
              f"from {neg_path.name}")

    sample = {
        "pts": pts, "cols": cols,
        "pts_frame_ids": fids,
        "pts_is_dynamic": is_dyn if is_dyn.any() else None,
        "tgt_c2ws": tgt_c2ws, "src_c2ws": src_c2ws,
        "K": K, "src_video": src_video,
        "src_motion_mask": src_motion_mask,
        "depth_h": Hd, "depth_w": Wd,
    }
    return sample, text_emb, neg_text_emb


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_manifold4d_model(args, device):
    """Load the released Manifold4D model, VAE, and conditioner.

    Path resolution priority: CLI flag > environment variable > config YAML
    (``paths:`` section). No path defaults are hardcoded in code.
    """
    import yaml
    from manifold4d.condition.render_cond import RenderConditioner

    with open(Path(args.config)) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    inference_cfg = cfg.get("inference", {})
    paths_cfg = cfg.get("paths", {})

    def resolve_path(cli_value, env_var, config_key, what):
        """CLI flag > environment variable > config `paths:` entry."""
        value = cli_value or os.environ.get(env_var) or paths_cfg.get(config_key)
        if not value:
            raise FileNotFoundError(
                f"{what} is not configured. Set it via the `{config_key}` key "
                f"in the `paths:` section of {args.config}, the "
                f"{env_var} environment variable, or the corresponding CLI flag.")
        return value

    use_text = model_cfg.get("use_text", True)
    use_plucker = model_cfg.get("use_plucker", True)
    plucker_share_encoder = model_cfg.get("plucker_share_encoder", False)
    camera_injection = model_cfg.get("camera_injection", "input_x_add")
    mask_pack_mode = model_cfg.get("mask_pack_mode", "avgpool")
    positional_embedding_offset = model_cfg.get("positional_embedding_offset", None)
    cond_stream_t = model_cfg.get("cond_stream_t", "honest")
    mask_channels = model_cfg.get("mask_channels", 2)

    # The 'wan' modules come from the Wan2.1 source tree.
    _wan_src = resolve_path(None, "WAN_SRC", "wan_src",
                             "Wan2.1 source tree (wan.* modules)")
    if _wan_src not in sys.path:
        sys.path.insert(0, _wan_src)
    from manifold4d.infer import load_model

    wan_checkpoint = resolve_path(args.wan_checkpoint, "WAN21_CHECKPOINT",
                                  "wan_checkpoint", "Wan2.1-T2V-14B base model dir")
    # VAE: CLI flag > env var > config entry > <wan_checkpoint>/Wan2.1_VAE.pth
    vae_checkpoint = (
        args.vae_checkpoint
        or os.environ.get("WAN21_VAE_CHECKPOINT")
        or paths_cfg.get("vae_checkpoint")
        or f"{wan_checkpoint}/Wan2.1_VAE.pth")

    model = load_model(
        args.checkpoint,
        device,
        use_text=use_text,
        use_plucker=use_plucker,
        plucker_share_encoder=plucker_share_encoder,
        camera_injection=camera_injection,
        mask_pack_channels=mask_channels,
        positional_embedding_offset=positional_embedding_offset,
        cond_stream_t=cond_stream_t,
        wan_checkpoint=wan_checkpoint,
    )

    # Load the Wan2.1 VAE from the source tree: `wan/modules/vae.py` defines
    # the ``WanVAE`` class (Wan2.1 VAE weights, loaded from vae_checkpoint).
    import importlib.util
    _vae_module = (os.environ.get("WAN21_VAE_MODULE")
                   or paths_cfg.get("vae_module")
                   or f"{_wan_src}/wan/modules/vae.py")
    if not os.path.isfile(_vae_module):
        raise FileNotFoundError(
            f"Wan2.1 VAE module not found at {_vae_module!r}. Set "
            "`vae_module` in the config `paths:` section or the "
            "WAN21_VAE_MODULE environment variable.")
    _spec = importlib.util.spec_from_file_location("wan21_vae", _vae_module)
    _vae_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_vae_mod)
    vae = _vae_mod.WanVAE(z_dim=16, vae_pth=vae_checkpoint, device=device)
    conditioner = RenderConditioner(vae=vae, vae_stride=(4, 8, 8),
                                  mask_pack_mode=mask_pack_mode)

    # Store config values for later use
    args.prior_sigma = model_cfg.get("prior_sigma", 0.3)
    args.guidance_scale = args.guidance_scale or inference_cfg.get("guidance_scale", 5.0)
    if args.shift is None:
        args.shift = inference_cfg.get("timestep_shift", 5.0)

    return model, vae, conditioner, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────


def run_manifold4d_generation(
    model, vae, conditioner, sample, args, device, save_dir,
    text_emb=None, neg_text_emb=None,
):
    """Run Manifold4D two-stream diffusion on a single sample.

    Returns the generated video as (T, H, W, 3) uint8 RGB array.
    """
    from manifold4d.infer import run_diffusion, latent_to_frames_np
    from manifold4d.rendering.gpu_renderer import render_video_gpu

    # Render the point-cloud proxy (RGB + alpha/motion mask pair)
    render_h, render_w = sample["depth_h"], sample["depth_w"]
    pts_is_dynamic = sample.get("pts_is_dynamic")

    render_video, render_mask, render_motion = render_video_gpu(
        sample["pts"], sample["cols"],
        sample["tgt_c2ws"], sample["K"],
        render_h=render_h, render_w=render_w,
        target_h=args.height, target_w=args.width,
        point_radius=args.point_radius, device=device,
        pts_frame_ids=sample["pts_frame_ids"],
        pts_is_dynamic=pts_is_dynamic,
        return_motion_mask=True,
    )

    # Save the conditioning render for debug (model input, splatted rendering)
    render_pix = ((render_video.permute(1, 2, 3, 0).cpu().numpy().clip(-1, 1) + 1) * 127.5
                  ).astype(np.uint8)
    save_video_mp4(render_pix, save_dir / "video_render.mp4", fps=args.fps or 24.0)

    # Encode source video
    src_video = sample["src_video"].to(device)
    source_latent = conditioner.encode_rgb(src_video.unsqueeze(0))[0].float()
    target_latent = torch.zeros_like(source_latent)

    # Encode the render
    render_latent = conditioner.encode_rgb(render_video.unsqueeze(0))[0]
    alpha_lat = conditioner.downsample_mask(render_mask.unsqueeze(0))[0]
    motion_lat = conditioner.downsample_mask(render_motion.unsqueeze(0))[0]
    render_mask_lat = torch.cat([alpha_lat, motion_lat], dim=0)

    # Source mask ([alpha=1, motion])
    T_lat, H_lat, W_lat = source_latent.shape[1:]
    n_per = conditioner.mask_pack_channels_per_input
    alpha_src = torch.ones(n_per, T_lat, H_lat, W_lat,
                           device=device, dtype=source_latent.dtype)
    # Source motion from dynamic mask
    if sample.get("src_motion_mask") is not None:
        motion_pix = sample["src_motion_mask"].to(device).float()
        if motion_pix.dim() == 3:
            motion_pix = motion_pix.unsqueeze(0)
        motion_src_lat = conditioner.downsample_mask(
            motion_pix.unsqueeze(0))[0]
    else:
        motion_src_lat = torch.zeros_like(alpha_src)
    source_mask_lat = torch.cat([alpha_src, motion_src_lat], dim=0)

    # Text embedding: preprocess-encoded T5 embedding, else zero-text
    if text_emb is not None:
        _text_emb = text_emb.to(device=device, dtype=torch.bfloat16)
    else:
        _text_emb = torch.zeros(512, 4096, device=device, dtype=torch.bfloat16)

    # Negative text embedding for CFG (None → run_diffusion falls back to zeros)
    _neg_text_emb = (neg_text_emb.to(device=device, dtype=torch.bfloat16)
                     if neg_text_emb is not None else None)

    # Camera kwargs for Plucker injection
    cam_kwargs = {}
    if hasattr(model, "use_plucker") and model.use_plucker:
        cam_kwargs = dict(
            tgt_c2ws=[torch.from_numpy(sample["tgt_c2ws"]).to(device).float()],
            src_c2ws=[torch.from_numpy(sample["src_c2ws"]).to(device).float()],
            K=[torch.from_numpy(sample["K"]).to(device).float()],
            pixel_height=args.height,
            pixel_width=args.width,
        )

    # Run diffusion
    gen_latent = run_diffusion(
        model, target_latent, render_latent, source_latent, render_mask_lat,
        _text_emb, device,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed,
        cam_kwargs=cam_kwargs,
        source_mask=source_mask_lat,
        prior_sigma=args.prior_sigma,
        neg_text_emb=_neg_text_emb,
    )

    gen_frames = latent_to_frames_np(gen_latent, vae)
    return gen_frames


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manifold4D two-stream video generation from "
                    "scripts/preprocess outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Scene dir from scripts/preprocess (frames/, "
                        "predictions.npz, trajectory/, mask*/, caption.txt, "
                        "text_embedding.pt).")
    p.add_argument("--trajectory", type=str, required=True,
                   help="Target trajectory: <traj_dir>/trajectory_new/"
                        "trajectory.npz (e.g. from build_val_traj.py).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where video.mp4 / video_render.mp4 / cameras.npz go.")
    # Model
    p.add_argument("--config", type=str, default="configs/manifold4d.yaml",
                   help="Model config YAML (architecture + `paths:` section).")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the released checkpoint dir (conditioning_"
                        "modules.pt + camera_encoder.pt + self_attn_full.pt).")
    p.add_argument("--wan_checkpoint", type=str, default=None,
                   help="Wan2.1-T2V-14B base model dir. Falls back to the "
                        "`wan_checkpoint` key in the config `paths:` section.")
    p.add_argument("--vae_checkpoint", type=str, default=None,
                   help="Wan2.1 VAE checkpoint. Falls back to the "
                        "`vae_checkpoint` key in the config `paths:` section.")
    # Resolution & frames
    p.add_argument("--height", type=int, default=480, help="VAE/render height.")
    p.add_argument("--width", type=int, default=832, help="VAE/render width.")
    p.add_argument("--video_length", type=int, default=81,
                   help="Frames to generate (must be 4n+1).")
    p.add_argument("--frame_sampling", choices=["linspace", "window"],
                   default="linspace",
                   help="linspace resamples the whole clip; window takes a "
                        "contiguous block starting at --frame_offset.")
    p.add_argument("--frame_offset", type=int, default=0,
                   help="Start frame for --frame_sampling window "
                        "(negative indexes from the end).")
    # Diffusion sampler
    p.add_argument("--num_steps", type=int, default=50,
                   help="Diffusion sampling steps (20-50).")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="CFG scale (falls back to the config YAML).")
    p.add_argument("--shift", type=float, default=None,
                   help="Flow-matching scheduler shift (falls back to the "
                        "training config).")
    p.add_argument("--seed", type=int, default=42)
    # Rendering & reconstruction
    p.add_argument("--point_stride", type=int, default=1,
                   help="Pixel stride for point cloud subsampling.")
    p.add_argument("--point_radius", type=int, default=2,
                   help="Point-cloud render splat radius.")
    p.add_argument("--depth_conf_threshold", type=float, default=None,
                   help="Drop depth pixels below this VGGT depth_conf "
                        "(~3.0 is sensible; None keeps all positive depth).")
    p.add_argument("--dyn_dilate_radius", type=int, default=3,
                   help="Dilate dynamic mask by N px before point "
                        "classification (removes ghost edges).")
    p.add_argument("--mask_dir", type=str, default=None,
                   help="Dynamic-mask dir override (default: first mask*/ "
                        "under scene_dir).")
    # Text
    p.add_argument("--text_embedding", type=str, default=None,
                   help="T5 text embedding .pt override (default: "
                        "scene_dir/text_embedding.pt).")
    p.add_argument("--neg_prompt_emb", type=str, default=None,
                   help="CFG negative-prompt embedding .pt override (default: "
                        "`neg_prompt_emb` key in the config `paths:` section "
                        "if present).")
    # Misc
    p.add_argument("--fps", type=float, default=0.0,
                   help="Output fps (0 = auto from trajectory metadata).")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Aspect-preserving fit: treat --height/--width as the target *area*.
    # Must run BEFORE sample building so src_video / motion mask / rendering
    # all share the same effective resolution.
    first_frame = sorted((Path(args.scene_dir) / "frames").glob("*.png"))[0]
    frame_h, frame_w = _load_frame_rgb(first_frame).shape[:2]
    eff_h, eff_w = fit_dimensions_to_area(
        args.height, args.width, frame_h, frame_w)
    if (eff_h, eff_w) != (args.height, args.width):
        print(f"[res] aspect-preserving resolution: {args.width}x{args.height} "
              f"canvas -> {eff_w}x{eff_h} (source {frame_w}x{frame_h})")
        args.height, args.width = eff_h, eff_w

    sample, text_emb, neg_text_emb = load_scene_sample(
        args.scene_dir, args.trajectory, args)

    model, vae, conditioner, cfg = load_manifold4d_model(args, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "cameras.npz",
             cam_c2w=sample["tgt_c2ws"].astype(np.float32),
             intrinsics=sample["K"].astype(np.float32))

    gen_frames = run_manifold4d_generation(
        model, vae, conditioner, sample, args, device, out_dir,
        text_emb=text_emb, neg_text_emb=neg_text_emb,
    )
    save_video_mp4(gen_frames, out_dir / "video.mp4", fps=args.fps or 24.0)
    print(f"[done] {out_dir / 'video.mp4'} "
          f"({gen_frames.shape[0]} frames @ {args.fps or 24.0:.1f} fps)")


if __name__ == "__main__":
    main()
