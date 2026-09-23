from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", "external/sam3"))
SAM3_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT", "external/sam3/ckpt/sam3.1/sam3.1_multiplex.pt")
sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", default=SAM3_CHECKPOINT)
    p.add_argument("--prompts", nargs="+", default=["person", "animal", "vehicle"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-fa3", action="store_true")
    return p.parse_args()


def collect_union(predictor, session_id):
    masks = {}
    for response in predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session_id
    }):
        frame_idx = response.get("frame_index")
        outputs = response.get("outputs", {})
        binary = outputs.get("out_binary_masks")
        if frame_idx is None or binary is None:
            continue
        if isinstance(binary, torch.Tensor):
            binary = binary.cpu().numpy()
        if binary.size == 0:
            continue
        if binary.ndim == 4:
            binary = binary[:, 0]
        union = np.any(binary > 0, axis=0).astype(np.uint8)
        if union.any():
            masks[int(frame_idx)] = union
    return masks


def make_video(frame_dir: Path, video_path: Path, fps: float = 15.0):
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(frame_dir)
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {video_path}")
    try:
        for frame in frames:
            image = cv2.imread(str(frame))
            if image.shape[:2] != (h, w):
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()
    return len(frames), w, h


def run_one(predictor, frame_dir: Path, out_dir: Path, prompts: list[str], device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / ".sam3_input.mp4"
    n_frames, width, height = make_video(frame_dir, video_path)
    union_masks = {}
    response = predictor.handle_request({
        "type": "start_session", "resource_path": str(video_path)
    })
    session_id = response["session_id"]
    try:
        for prompt_idx, prompt in enumerate(prompts):
            if prompt_idx > 0:
                predictor.handle_request({
                    "type": "reset_session", "session_id": session_id
                })
            try:
                predictor.handle_request({
                    "type": "add_prompt", "session_id": session_id,
                    "frame_index": 0, "text": prompt
                })
                current = collect_union(predictor, session_id)
                for frame_idx, mask in current.items():
                    if frame_idx not in union_masks:
                        union_masks[frame_idx] = mask
                    else:
                        if union_masks[frame_idx].shape != mask.shape:
                            mask = cv2.resize(
                                mask, (union_masks[frame_idx].shape[1], union_masks[frame_idx].shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        union_masks[frame_idx] = np.maximum(union_masks[frame_idx], mask)
            except Exception as exc:
                print(f"prompt={prompt} failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    for idx in range(n_frames):
        mask = union_masks.get(idx, np.zeros((height, width), dtype=np.uint8))
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(mask_dir / f"{idx:06d}.png")
    video_path.unlink(missing_ok=True)
    return union_masks, (n_frames, width, height)


def save_review(pair_dir: Path, source_masks, target_masks):
    source_frames = sorted((pair_dir / "frames").glob("*.png"))
    target_frames = sorted((pair_dir / "target_frames").glob("*.png"))
    indices = np.linspace(0, len(source_frames) - 1, min(6, len(source_frames))).round().astype(int)
    rows = []
    for idx in indices:
        src = cv2.imread(str(source_frames[idx]))
        tgt = cv2.imread(str(target_frames[idx]))
        sm = cv2.imread(str(pair_dir / "sam3_source" / "masks" / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        tm = cv2.imread(str(pair_dir / "sam3_target" / "masks" / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
        h, w = src.shape[:2]
        sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
        tm = cv2.resize(tm, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_NEAREST)
        src_overlay = src.copy()
        tgt_overlay = tgt.copy()
        src_overlay[sm > 0] = (0.45 * src_overlay[sm > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        tgt_overlay[tm > 0] = (0.45 * tgt_overlay[tm > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        tile_h, tile_w = 300, 400
        tiles = [src, src_overlay, tgt, tgt_overlay]
        tiles = [cv2.resize(x, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for x in tiles]
        row = np.concatenate(tiles, axis=1)
        cv2.putText(row, f"frame {idx}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(pair_dir / "sam3_review.jpg"), grid)


def main():
    args = parse_args()
    pair_dir = args.pair_dir
    source_out = pair_dir / "sam3_source"
    target_out = pair_dir / "sam3_target"
    if not args.overwrite and (source_out / "masks" / "000000.png").exists():
        print(f"exists: {source_out}; use --overwrite", flush=True)
        return
    predictor = build_sam3_predictor(
        version="sam3.1", checkpoint_path=args.checkpoint,
        use_fa3=not args.no_fa3, compile=False, warm_up=False
    )
    run_one(predictor, pair_dir / "frames", source_out, args.prompts, args.device)
    run_one(predictor, pair_dir / "target_frames", target_out, args.prompts, args.device)
    save_review(pair_dir, source_out, target_out)
    print(f"saved masks and review to {pair_dir}", flush=True)


if __name__ == "__main__":
    main()
