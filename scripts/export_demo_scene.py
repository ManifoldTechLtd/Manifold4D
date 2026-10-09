"""Export a lightweight Manifold4D demo bundle from a preprocessed scene.

A demo bundle contains everything ``manifold4d.demo`` needs to run the
released model WITHOUT the heavy preprocessing stack (VGGT-Omega / SAM3 /
Qwen-VL) and WITHOUT any depth or point-cloud data:

    <out_dir>/
    ├── source.mp4           # source video at the model resolution
    ├── render.mp4           # point-cloud conditioning render (target view)
    ├── masks.npz            # uint8 [3, T, H, W]: render alpha / render
    │                        # motion / source motion
    ├── caption.txt          # scene caption (informational)
    ├── text_embedding.pt    # pre-encoded T5 text embedding
    ├── trajectory.npz       # source cameras for the exported frames
    ├── trajectory_new.npz   # target novel-view cameras (+ intrinsic, meta)
    └── demo_meta.json       # export provenance (frames, resolution, fps)

The heavy VGGT-Omega reconstruction is only needed ONCE, here — the exported
bundle keeps the exact conditioning tensors (render video, alpha/motion
masks, cameras) the two-stream model consumes, so downstream users can run
inference directly with ``python -m manifold4d.demo``.

Typical usage (inside the Manifold4D repo, scene data under
``data/ours_eval_data``, trajectories under ``data/val_traj``)::

    python scripts/export_demo_scene.py \\
        --scene_dir  data/ours_eval_data/gold-fish_vggt \\
        --trajectory data/val_traj/gold-fish/f1/trajectory_new/trajectory.npz \\
        --out_dir    assets/demo/gold-fish
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from manifold4d.generate import (  # noqa: E402
    fit_dimensions_to_area,
    load_scene_sample,
    save_video_mp4,
    select_frame_ids,
)
from manifold4d.rendering.gpu_renderer import render_video_gpu  # noqa: E402


def _npz_shape(path: Path, key: str) -> tuple:
    """Read an array's shape from an .npz without loading the data."""
    import re
    import zipfile
    with zipfile.ZipFile(path) as z:
        with z.open(f"{key}.npy") as f:
            f.read(2)  # numpy magic
            head = f.read(400)
    m = re.search(rb"'shape': \(([^)]*)\)", head)
    return tuple(int(x) for x in m.group(1).split(b",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a demo bundle (source/render/masks/text/trajectory) "
                    "from a preprocessed scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene_dir", type=str, required=True,
                   help="Preprocessed scene dir (scripts/preprocess output).")
    p.add_argument("--trajectory", type=str, required=True,
                   help="Target trajectory: <traj_dir>/trajectory_new/"
                        "trajectory.npz (e.g. from build_val_traj.py).")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output demo bundle directory.")
    # Export window: a contiguous 4n+1 block of the source clip.
    p.add_argument("--video_length", type=int, default=77,
                   help="Frames to export (must be 4n+1).")
    p.add_argument("--frame_offset", type=int, default=0,
                   help="Start frame of the exported window.")
    p.add_argument("--frame_sampling", choices=["linspace", "window"],
                   default="window",
                   help="window = contiguous block at --frame_offset "
                        "(temporal-consistent demo default).")
    # Resolution & rendering (mirror manifold4d.generate defaults).
    p.add_argument("--height", type=int, default=480, help="Model height.")
    p.add_argument("--width", type=int, default=832, help="Model width.")
    p.add_argument("--point_stride", type=int, default=1)
    p.add_argument("--point_radius", type=int, default=2)
    p.add_argument("--depth_conf_threshold", type=float, default=None)
    p.add_argument("--dyn_dilate_radius", type=int, default=3)
    p.add_argument("--mask_dir", type=str, default=None)
    p.add_argument("--config", type=str, default="configs/manifold4d.yaml")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Output fps (0 = auto from source trajectory).")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for the point-cloud conditioning render.")
    return p.parse_args()


def _slice_npz(traj_path: Path, frame_ids: np.ndarray,
               extra: dict | None = None) -> dict:
    """Slice a trajectory.npz to ``frame_ids`` keeping scalar fields intact."""
    traj = np.load(traj_path, allow_pickle=True)
    out = {}
    for k in traj.files:
        a = traj[k]
        if hasattr(a, "ndim") and a.ndim >= 1 and a.shape[0] == len(
                traj["R_world_from_cam"]):
            out[k] = a[frame_ids]
        else:
            out[k] = a
    if extra:
        out.update(extra)
    return out


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aspect-preserving fit (same rule as manifold4d.generate.main) so the
    # bundle matches what a full-pipeline run would feed the model.
    from manifold4d.generate import _load_frame_rgb
    first_frame = sorted((scene_dir / "frames").glob("*.png"))[0]
    frame_h, frame_w = _load_frame_rgb(first_frame).shape[:2]
    eff_h, eff_w = fit_dimensions_to_area(
        args.height, args.width, frame_h, frame_w)
    print(f"[res] model resolution: {eff_w}x{eff_h} "
          f"(source {frame_w}x{frame_h})")

    gen_args = SimpleNamespace(
        video_length=args.video_length,
        frame_sampling=args.frame_sampling,
        frame_offset=args.frame_offset,
        fps=args.fps,
        height=eff_h, width=eff_w,
        mask_dir=args.mask_dir,
        point_stride=args.point_stride,
        dyn_dilate_radius=args.dyn_dilate_radius,
        depth_conf_threshold=args.depth_conf_threshold,
        text_embedding=None, neg_prompt_emb=None, config=args.config,
    )
    sample, text_emb, _ = load_scene_sample(
        scene_dir, args.trajectory, gen_args)

    T = sample["src_video"].shape[1]
    H, W = eff_h, eff_w
    # Frame indices of the source clip that were rendered/encoded above;
    # trajectory slices below must use exactly these.
    T_total = _npz_shape(scene_dir / "predictions.npz", "depth")[0]
    frame_ids = select_frame_ids(
        T_total, args.video_length, args.frame_sampling, args.frame_offset)
    assert len(frame_ids) == T, (len(frame_ids), T)
    print(f"[export] {T} frames @ {W}x{H}")

    # Point-cloud conditioning render (the only GPU-heavy step here).
    import torch
    device = torch.device(args.device)
    render_video, render_mask, render_motion = render_video_gpu(
        sample["pts"], sample["cols"],
        sample["tgt_c2ws"], sample["K"],
        render_h=sample["depth_h"], render_w=sample["depth_w"],
        target_h=H, target_w=W,
        point_radius=args.point_radius, device=device,
        pts_frame_ids=sample["pts_frame_ids"],
        pts_is_dynamic=sample.get("pts_is_dynamic"),
        return_motion_mask=True,
    )

    def _to_uint8(video: torch.Tensor) -> np.ndarray:
        """[3, T, H, W] in [-1, 1] -> [T, H, W, 3] uint8."""
        v = video.permute(1, 2, 3, 0).cpu().numpy().clip(-1, 1)
        return ((v + 1) * 127.5).astype(np.uint8)

    fps = float(gen_args.fps) if gen_args.fps > 0 else 24.0
    save_video_mp4(_to_uint8(sample["src_video"]),
                   out_dir / "source.mp4", fps=fps)
    save_video_mp4(_to_uint8(render_video),
                   out_dir / "render.mp4", fps=fps)

    # masks.npz: [3, T, H, W] uint8 — alpha / render motion / source motion.
    src_motion = sample.get("src_motion_mask")
    if src_motion is None:
        src_motion = torch.zeros_like(render_mask)
    masks = torch.cat([render_mask.cpu(), render_motion.cpu(),
                       src_motion.cpu()], dim=0)
    np.savez_compressed(
        out_dir / "masks.npz",
        masks=(masks.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8),
    )

    # Cameras restricted to the exported frames.
    np.savez(out_dir / "trajectory.npz",
             **_slice_npz(Path(scene_dir) / "trajectory" / "trajectory.npz",
                          frame_ids))
    tgt_extra = {}
    if "intrinsic" not in np.load(args.trajectory, allow_pickle=True).files:
        tgt_extra["intrinsic"] = sample["K"][None].astype(np.float32)
    np.savez(out_dir / "trajectory_new.npz",
             **_slice_npz(Path(args.trajectory), frame_ids, tgt_extra))

    # Text: caption (informational) + pre-encoded T5 embedding (required).
    shutil.copyfile(scene_dir / "caption.txt", out_dir / "caption.txt")
    if text_emb is None:
        raise FileNotFoundError(
            f"text_embedding.pt not found in {scene_dir} — it is required "
            f"for a demo bundle (see scripts/preprocess/encode_text_t5.py).")
    torch.save(text_emb, out_dir / "text_embedding.pt")

    meta = {
        "scene": scene_dir.name.removesuffix("_vggt"),
        "frame_ids": frame_ids.tolist(),
        "source_frames_total": int(frame_ids.max()) + 1,
        "height": H, "width": W, "fps": fps,
        "video_length": T, "frame_sampling": args.frame_sampling,
        "frame_offset": args.frame_offset,
        "point_radius": args.point_radius,
        "point_stride": args.point_stride,
        "depth_conf_threshold": args.depth_conf_threshold,
        "dyn_dilate_radius": args.dyn_dilate_radius,
        "exporter": "scripts/export_demo_scene.py",
    }
    (out_dir / "demo_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[done] demo bundle written to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
