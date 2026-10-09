"""Generate a Manifold4D video from a lightweight demo bundle (no depth / point cloud).

A demo bundle (see ``scripts/export_demo_scene.py``) ships the two-stream
conditioning inputs as ready-to-use media files — source video, point-cloud
render, alpha/motion masks, and camera trajectories — so the released model
runs WITHOUT VGGT-Omega, SAM3, or Qwen-VL, and without any depth data::

    <demo_dir>/
    ├── source.mp4           # source video at the model resolution
    ├── render.mp4           # point-cloud conditioning render (target view)
    ├── masks.npz            # uint8 [3, T, H, W]: render alpha / render
    │                        # motion / source motion
    ├── caption.txt          # scene caption (informational)
    ├── text_embedding.pt    # pre-encoded T5 text embedding
    ├── trajectory.npz       # source cameras for the bundled frames
    ├── trajectory_new.npz   # target novel-view cameras (+ intrinsic)
    └── demo_meta.json       # export provenance

Usage::

    python -m manifold4d.demo \\
        --demo_dir   assets/demo/gold-fish \\
        --checkpoint checkpoints/manifold4d \\
        --output_dir output/demo_gold-fish

Model and asset paths are configured in ``configs/manifold4d.yaml``
(``paths:`` section); CLI flags and environment variables override them,
exactly as in ``manifold4d.generate``.  Bundle masks are stored losslessly
(uint8 npz), while the two mp4 streams carry only mild codec loss — results
match the full pipeline up to that compression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from manifold4d.generate import (
    _c2w_from_traj,
    load_config_paths,
    load_manifold4d_model,
    save_video_mp4,
)
from manifold4d.infer import latent_to_frames_np, run_diffusion


def _read_video_rgb(path: Path) -> np.ndarray:
    """Read an mp4 as [T, H, W, 3] uint8 RGB frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise FileNotFoundError(f"Cannot read video frames: {path}")
    return np.stack(frames, axis=0)


def load_demo_sample(demo_dir: str | Path):
    """Load a demo bundle into pixel-space conditioning tensors.

    Returns ``(sample, text_emb, fps, meta)`` using the same tensor
    conventions as ``manifold4d.generate.load_scene_sample``: videos are
    [3, T, H, W] in [-1, 1]; masks are [1, T, H, W] in [0, 1].
    """
    demo_dir = Path(demo_dir)

    src_u8 = _read_video_rgb(demo_dir / "source.mp4")
    render_u8 = _read_video_rgb(demo_dir / "render.mp4")
    T, H, W = src_u8.shape[:3]
    if render_u8.shape[0] != T or render_u8.shape[1:3] != (H, W):
        raise ValueError(
            f"render.mp4 {render_u8.shape} does not match source.mp4 "
            f"{src_u8.shape}")
    if (T - 1) % 4 != 0:
        raise ValueError(
            f"T={T} frames in the bundle must satisfy 4n+1 (Wan VAE "
            f"temporal stride 4).")

    def _to_tensor(video_u8: np.ndarray) -> torch.Tensor:
        arr = video_u8.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()

    masks = np.load(demo_dir / "masks.npz")["masks"]      # [3, T, H, W] uint8
    if masks.shape[1:] != (T, H, W):
        raise ValueError(
            f"masks.npz {masks.shape} does not match the videos "
            f"({T}, {H}, {W})")

    src_c2ws = _c2w_from_traj(np.load(demo_dir / "trajectory.npz"),
                              np.arange(T))
    traj_new = np.load(demo_dir / "trajectory_new.npz", allow_pickle=True)
    tgt_c2ws = _c2w_from_traj(traj_new, np.arange(T))
    if "intrinsic" not in traj_new:
        raise FileNotFoundError(
            "trajectory_new.npz has no 'intrinsic' — re-export the bundle "
            "with scripts/export_demo_scene.py.")
    K = traj_new["intrinsic"][0].astype(np.float32)

    with open(demo_dir / "demo_meta.json") as f:
        meta = json.load(f)
    fps = float(meta.get("fps", 24.0))

    text_emb = None
    text_path = demo_dir / "text_embedding.pt"
    if text_path.is_file():
        text_emb = torch.load(text_path, map_location="cpu")
        print(f"[text] loaded {tuple(text_emb.shape)} from {text_path}")
    caption_path = demo_dir / "caption.txt"
    if caption_path.is_file():
        print(f"[text] caption: {caption_path.read_text().strip()[:120]}")

    sample = {
        "src_video": _to_tensor(src_u8),
        "render_video": _to_tensor(render_u8),
        "render_mask": torch.from_numpy(masks[0]).float()[None] / 255.0,
        "render_motion": torch.from_numpy(masks[1]).float()[None] / 255.0,
        "src_motion": torch.from_numpy(masks[2]).float()[None] / 255.0,
        "tgt_c2ws": tgt_c2ws, "src_c2ws": src_c2ws, "K": K,
        "height": H, "width": W,
    }
    return sample, text_emb, fps, meta


def run_demo_generation(model, vae, conditioner, sample, args, device,
                        save_dir, text_emb=None, neg_text_emb=None):
    """Encode the bundled conditioning streams and run the two-stream
    diffusion (mirrors ``manifold4d.generate.run_manifold4d_generation``
    after the point-cloud render step).  Returns (T, H, W, 3) uint8 RGB."""
    src_video = sample["src_video"].to(device)
    render_video = sample["render_video"].to(device)

    source_latent = conditioner.encode_rgb(src_video.unsqueeze(0))[0].float()
    target_latent = torch.zeros_like(source_latent)
    render_latent = conditioner.encode_rgb(render_video.unsqueeze(0))[0]
    alpha_lat = conditioner.downsample_mask(
        sample["render_mask"].unsqueeze(0).to(device))[0]
    motion_lat = conditioner.downsample_mask(
        sample["render_motion"].unsqueeze(0).to(device))[0]
    render_mask_lat = torch.cat([alpha_lat, motion_lat], dim=0)

    # Source mask ([alpha=1, motion]).
    T_lat, H_lat, W_lat = source_latent.shape[1:]
    n_per = conditioner.mask_pack_channels_per_input
    alpha_src = torch.ones(n_per, T_lat, H_lat, W_lat,
                           device=device, dtype=source_latent.dtype)
    motion_src_lat = conditioner.downsample_mask(
        sample["src_motion"].unsqueeze(0).to(device))[0]
    source_mask_lat = torch.cat([alpha_src, motion_src_lat], dim=0)

    if text_emb is not None:
        _text_emb = text_emb.to(device=device, dtype=torch.bfloat16)
    else:
        _text_emb = torch.zeros(512, 4096, device=device,
                                dtype=torch.bfloat16)

    cam_kwargs = {}
    if hasattr(model, "use_plucker") and model.use_plucker:
        cam_kwargs = dict(
            tgt_c2ws=[torch.from_numpy(sample["tgt_c2ws"]).to(device).float()],
            src_c2ws=[torch.from_numpy(sample["src_c2ws"]).to(device).float()],
            K=[torch.from_numpy(sample["K"]).to(device).float()],
            pixel_height=sample["height"],
            pixel_width=sample["width"],
        )

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
        neg_text_emb=neg_text_emb,
    )
    return latent_to_frames_np(gen_latent, vae)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manifold4D generation from a lightweight demo bundle "
                    "(no VGGT-Omega / SAM3 / Qwen-VL / depth needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--demo_dir", type=str, required=True,
                   help="Demo bundle dir (see scripts/export_demo_scene.py).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where video.mp4 / cameras.npz go.")
    # Model
    p.add_argument("--config", type=str, default="configs/manifold4d.yaml",
                   help="Model config YAML (architecture + `paths:` section).")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the released checkpoint dir.")
    p.add_argument("--wan_checkpoint", type=str, default=None,
                   help="Wan2.1-T2V-14B base model dir. Falls back to the "
                        "`wan_checkpoint` key in the config `paths:` section.")
    p.add_argument("--vae_checkpoint", type=str, default=None,
                   help="Wan2.1 VAE checkpoint override.")
    # Diffusion sampler
    p.add_argument("--num_steps", type=int, default=50,
                   help="Diffusion sampling steps (20-50).")
    p.add_argument("--video_length", type=int, default=0,
                   help="Truncate the bundle to its first N frames (must "
                        "be 4n+1; 0 = use the whole bundle). Lower N "
                        "linearly reduces the diffusion peak memory.")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="CFG scale (falls back to the config YAML).")
    p.add_argument("--shift", type=float, default=None,
                   help="Flow-matching scheduler shift.")
    p.add_argument("--seed", type=int, default=42)
    # Text
    p.add_argument("--neg_prompt_emb", type=str, default=None,
                   help="CFG negative-prompt embedding .pt override (default: "
                        "`neg_prompt_emb` key in the config `paths:` section).")
    # Misc
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    sample, text_emb, fps, _meta = load_demo_sample(args.demo_dir)
    if args.video_length:
        T_full = sample["src_video"].shape[1]
        T2 = args.video_length
        if (T2 - 1) % 4 != 0:
            raise ValueError(f"--video_length {T2} must satisfy 4n+1")
        if T2 >= T_full:
            raise ValueError(
                f"--video_length {T2} exceeds the {T_full} bundled frames")
        sl = slice(0, T2)
        sample["src_video"] = sample["src_video"][:, sl]
        sample["render_video"] = sample["render_video"][:, sl]
        for k in ("render_mask", "render_motion", "src_motion"):
            sample[k] = sample[k][:, sl]
        sample["tgt_c2ws"] = sample["tgt_c2ws"][:T2]
        sample["src_c2ws"] = sample["src_c2ws"][:T2]
        print(f"[demo] truncated {T_full} -> {T2} frames (low-VRAM mode)")
    print(f"[demo] {sample['src_video'].shape[1]} frames @ "
          f"{sample['width']}x{sample['height']}")

    model, vae, conditioner, _cfg = load_manifold4d_model(args, device)

    # CFG negative prompt (pre-encoded; see encode_neg_prompt.py).
    neg_text_emb = None
    neg_path = (Path(args.neg_prompt_emb) if args.neg_prompt_emb
                else Path(load_config_paths(args.config).get(
                    "neg_prompt_emb", "")))
    if neg_path.is_file():
        neg_text_emb = torch.load(neg_path, map_location="cpu")
        print(f"[text] negative prompt {tuple(neg_text_emb.shape)} "
              f"from {neg_path.name}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "cameras.npz",
             cam_c2w=sample["tgt_c2ws"].astype(np.float32),
             intrinsics=sample["K"].astype(np.float32))

    gen_frames = run_demo_generation(
        model, vae, conditioner, sample, args, device, out_dir,
        text_emb=text_emb, neg_text_emb=neg_text_emb,
    )
    save_video_mp4(gen_frames, out_dir / "video.mp4", fps=fps)
    print(f"[done] {out_dir / 'video.mp4'} "
          f"({gen_frames.shape[0]} frames @ {fps:.1f} fps)")


if __name__ == "__main__":
    main()
