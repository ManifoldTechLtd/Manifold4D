"""Prepare VGGT-Omega core singleview data from numbered image folders."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def discover_scenes(data_root: Path, output_root: Path) -> list[str]:
    names = set()
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.endswith("_vggt"):
            names.add(p.name[:-5])
        else:
            names.add(p.name)
    return sorted(names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--checkpoint",
                        default=os.environ.get(
                            "VGGT_CHECKPOINT",
                            "checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--vggt_root",
                        default=os.environ.get(
                            "VGGT_ROOT", "external/vggt-omega-main"))
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--native_output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--viz_dir", default=None)
    parser.add_argument("--viz_frames", type=int, default=3)
    return parser.parse_args()


def frame_paths(source_dir: Path, max_frames: int) -> list[Path]:
    paths = sorted(
        p for p in source_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"no image frames under {source_dir}")
    return paths


def prepare_frames(source_paths: list[Path], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for index, source_path in enumerate(source_paths):
        output_path = output_dir / f"{index:06d}.png"
        if not output_path.exists():
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read {source_path}")
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"failed to write {output_path}")
        output_paths.append(str(output_path))
    return output_paths


def decode_camera(preds: dict, image_shape: tuple[int, int]):
    from vggt_omega.utils.pose_enc import encoding_to_camera

    extrinsic, intrinsic = encoding_to_camera(preds["pose_enc"], image_shape)
    extrinsic = extrinsic.float().cpu().numpy()
    intrinsic = intrinsic.float().cpu().numpy()
    if extrinsic.ndim == 4:
        extrinsic = extrinsic[0]
        intrinsic = intrinsic[0]
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"unexpected extrinsic shape: {extrinsic.shape}")
    if intrinsic.shape[-2:] != (3, 3):
        raise ValueError(f"unexpected intrinsic shape: {intrinsic.shape}")
    return extrinsic.astype(np.float32), intrinsic.astype(np.float32)


def trajectory_from_extrinsic(extrinsic: np.ndarray, intrinsic: np.ndarray,
                              fps: float) -> dict[str, np.ndarray]:
    rotation_w2c = extrinsic[:, :3, :3]
    translation_w2c = extrinsic[:, :3, 3]
    rotation_c2w = np.transpose(rotation_w2c, (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", rotation_c2w, translation_w2c)
    quaternions = Rotation.from_matrix(rotation_c2w).as_quat().astype(np.float32)
    n_frames = extrinsic.shape[0]
    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "centers": centers.astype(np.float32),
        "R_world_from_cam": rotation_c2w.astype(np.float32),
        "quat_world_from_cam_xyzw": quaternions,
        "timestamps": (np.arange(n_frames, dtype=np.float32) / fps),
        "fps": np.asarray(fps, dtype=np.float32),
    }


def save_viz(scene_name: str, image_paths: list[str], depth: np.ndarray,
             depth_conf: np.ndarray, viz_dir: Path, n_show: int) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(image_paths) - 1,
                          min(n_show, len(image_paths))).round().astype(int)
    rows = []
    for index in indices:
        image = cv2.imread(image_paths[int(index)], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (depth.shape[2], depth.shape[1]))
        d = depth[int(index)]
        d_valid = np.isfinite(d) & (d > 1e-6)
        if d_valid.any():
            lo, hi = np.percentile(d[d_valid], [2, 98])
            d_vis = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            d_vis = np.zeros_like(d)
        d_vis = cv2.applyColorMap((d_vis * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
        c = depth_conf[int(index)]
        c_vis = np.clip((c - 1.0) /
                        max(float(np.percentile(c, 98)) - 1.0, 1e-6), 0, 1)
        c_vis = cv2.applyColorMap((c_vis * 255).astype(np.uint8),
                                  cv2.COLORMAP_MAGMA)
        row = np.concatenate([image, d_vis, c_vis], axis=1)
        cv2.putText(row, f"{scene_name} frame={int(index)}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    width = depth.shape[2]
    header = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
    for column, label in enumerate(["RGB", "VGGT depth", "VGGT confidence"]):
        cv2.putText(header, label, (column * width + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.imwrite(str(viz_dir / f"{scene_name}.png"),
                np.concatenate([header, grid], axis=0))


def resize_predictions_to_native(depth: np.ndarray, depth_conf: np.ndarray,
                                 intrinsic: np.ndarray,
                                 native_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native_h, native_w = native_shape
    pred_h, pred_w = depth.shape[1:]
    if (pred_h, pred_w) == (native_h, native_w):
        return depth, depth_conf, intrinsic
    depth = np.stack([
        cv2.resize(frame, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        for frame in depth
    ]).astype(np.float32)
    depth_conf = np.stack([
        cv2.resize(frame, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        for frame in depth_conf
    ]).astype(np.float32)
    intrinsic = intrinsic.copy()
    intrinsic[:, 0, :] *= native_w / pred_w
    intrinsic[:, 1, :] *= native_h / pred_h
    return depth, depth_conf, intrinsic


def write_scene(source_dir: Path, output_dir: Path, model, load_images,
                args: argparse.Namespace, device: torch.device) -> None:
    source_paths = frame_paths(source_dir, args.max_frames)
    scene_dir = output_dir / f"{source_dir.name}_vggt"
    prediction_path = scene_dir / "predictions.npz"
    metadata_path = scene_dir / "meta.json"
    if prediction_path.exists() and not args.overwrite:
        print(f"[skip] {scene_dir}: predictions.npz exists")
        return

    frame_dir = scene_dir / "frames"
    image_paths = prepare_frames(source_paths, frame_dir)
    print(f"[run] {source_dir.name}: {len(image_paths)} frames")
    images = load_images(
        image_paths, mode=args.mode,
        image_resolution=args.image_resolution,
    ).to(device)

    started = time.time()
    with torch.inference_mode():
        if device.type == "cuda":
            autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast = contextlib.nullcontext()
        with autocast:
            preds = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    depth = preds["depth"].detach().float().cpu().numpy()
    depth_conf = preds["depth_conf"].detach().float().cpu().numpy()
    if depth.ndim == 5:
        depth = depth[0]
        depth_conf = depth_conf[0]
    depth = depth[..., 0].astype(np.float32)
    depth_conf = depth_conf.astype(np.float32)
    extrinsic, intrinsic = decode_camera(preds, images.shape[-2:])
    if args.native_output:
        native_image = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
        if native_image is None:
            raise RuntimeError(f"failed to read {image_paths[0]}")
        depth, depth_conf, intrinsic = resize_predictions_to_native(
            depth, depth_conf, intrinsic, native_image.shape[:2])
    if depth.shape[0] != len(image_paths):
        raise ValueError(
            f"{source_dir.name}: prediction frames {depth.shape[0]} != "
            f"input frames {len(image_paths)}")

    trajectory = trajectory_from_extrinsic(extrinsic, intrinsic, args.fps)
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(prediction_path),
        depth=depth[..., None],
        depth_conf=depth_conf,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    trajectory_dir = scene_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(trajectory_dir / "trajectory.npz"), **trajectory)
    metadata = {
        "source_dir": str(source_dir.resolve()),
        "checkpoint": args.checkpoint,
        "image_resolution": args.image_resolution,
        "mode": args.mode,
        "n_frames": len(image_paths),
        "fps": args.fps,
        "frame_shape": list(depth.shape[1:]),
        "extrinsic_shape": list(extrinsic.shape),
        "intrinsic_shape": list(intrinsic.shape),
        "depth_shape": list(depth[..., None].shape),
        "depth_conf_shape": list(depth_conf.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.viz_dir:
        save_viz(scene_dir.name, image_paths, depth, depth_conf,
                 Path(args.viz_dir), args.viz_frames)
    print(f"[done] {scene_dir}: {time.time() - started:.1f}s")

    del preds, images
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root) if args.output_root else data_root
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not Path(args.vggt_root).is_dir():
        raise FileNotFoundError(f"VGGT-Omega root not found: {args.vggt_root}")
    sys.path.insert(0, args.vggt_root)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    device = torch.device(args.device)
    print(f"[model] loading {args.checkpoint} on {device}")
    model = VGGTOmega().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state and "state_dict" not in state:
        state = state["model"]
    model.load_state_dict(state)

    names = args.names if args.names else discover_scenes(data_root, output_root)
    if not names:
        raise FileNotFoundError(f"no scenes found under {data_root}")
    print(f"[scenes] {len(names)}: {', '.join(names)}")
    for name in names:
        source_dir = data_root / name
        if not source_dir.is_dir():
            existing = output_root / f"{name}_vggt" / "predictions.npz"
            if existing.exists():
                print(f"[skip] {name}: source removed but {existing.name} exists")
                continue
            print(f"[warn] {name}: no source and no vggt output, skipping")
            continue
        write_scene(source_dir, output_root, model, load_and_preprocess_images,
                    args, device)


if __name__ == "__main__":
    main()
