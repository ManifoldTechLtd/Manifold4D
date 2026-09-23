"""Generate dynamic masks for singleview scenes from caption.txt with SAM3."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
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
from run_sam3_iphone import collect_union, make_video


DEFAULT_NAMES = []  # legacy; scene discovery now defaults to all *_vggt dirs
SUBJECT_PROMPTS = {
    "bear_vggt": ["bear"],
    "breakdance-flare_vggt": ["person"],
    "cows_vggt": ["cow", "bell", "collar"],
    "dog_white_vggt": ["dog"],
    "dance-jump_vggt": ["person", "dancer"],
    "flamingo_vggt": ["flamingo"],
    "hike_vggt": ["person"],
    "hockey_vggt": ["person", "hockey player", "hockey stick", "hockey ball", "ball"],
    "horsejump-high_vggt": ["horse", "person"],
    "motocross-bumps_vggt": ["person", "motorcycle", "motocross bike", "motorbike"],
    "parkour_vggt": ["person"],
    "rhino_vggt": ["rhino"],
    "rollerblade_vggt": ["person"],
    "snowboard_vggt": ["snowboarder", "person", "snowboard"],
    "soapbox_vggt": ["person", "race car", "soapbox car", "car"],
    "swing_vggt": ["person", "swing"],
    "tennis_vggt": ["person", "tennis racket", "tennis ball"],
    "train_vggt": ["train"],
}

BACKGROUND_WORDS = {
    "water", "sky", "grass", "wind", "trees", "tree", "buildings", "building",
    "road", "sand", "snow", "rock", "rocks", "mountain", "mountains", "hill",
    "hills", "cloud", "clouds", "ocean", "sea", "river", "lake", "ground",
    "floor", "wall", "walls", "ceiling", "fence", "bridge", "street", "path",
    "trail", "field", "forest", "bush", "bushes", "flower", "flowers", "leaf",
    "leaves", "dirt", "mud", "ice", "fog", "mist", "rain", "sun", "sunlight",
    "shadow", "shadows", "light", "lighting", "horizon", "landscape",
    "scenery", "environment", "background", "crowd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Scene dirs (usually *_vggt). Default: all "
                             "*_vggt dirs under data_root.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=SAM3_CHECKPOINT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-fa3", action="store_true")
    parser.add_argument("--viz_dir", default="render_output/sam3_caption_review")
    parser.add_argument("--viz_frames", type=int, default=3)
    parser.add_argument(
        "--prompt_frames",
        nargs="+",
        default=["0", "middle", "last"],
        help="Frame indices used for text prompts; supports integer, middle and last.",
    )
    parser.add_argument("--extra_prompts", nargs="*", default=None,
                        help="Additional short subject prompts unioned with caption.")
    parser.add_argument("--no-auto-subject", action="store_true",
                        help="Disable the built-in scene-to-subject prompt map.")
    parser.add_argument("--no-caption", action="store_true",
                        help="Do not use caption.txt as a SAM3 prompt.")
    return parser.parse_args()


def save_mask_review(scene_dir: Path, mask_dir: Path, viz_dir: Path,
                     n_show: int) -> None:
    frames = sorted((scene_dir / "frames").glob("*.png"))
    if not frames:
        return
    indices = np.linspace(0, len(frames) - 1,
                          min(n_show, len(frames))).round().astype(int)
    rows = []
    for index in indices:
        image = cv2.imread(str(frames[int(index)]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / f"{int(index):06d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        overlay[mask > 0] = (
            0.45 * overlay[mask > 0] + 0.55 * np.array([0, 0, 255])
        ).astype(np.uint8)
        tile_h, tile_w = 300, 400
        row = np.concatenate([
            cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
            cv2.resize(overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
            cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                       (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
        ], axis=1)
        cv2.putText(row, f"{scene_dir.name} frame={int(index)}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(row)
    if rows:
        viz_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz_dir / f"{scene_dir.name}.jpg"),
                    np.concatenate(rows, axis=0))


def has_foreground(mask_dir: Path) -> bool:
    for mask_path in mask_dir.glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and cv2.countNonZero(mask) > 0:
            return True
    return False


def inferred_subject_prompts(scene_name: str) -> list[str]:
    stem = scene_name.removesuffix("_vggt").replace("-", "_").lower()
    token_prompts = {
        "person": "person",
        "people": "person",
        "human": "person",
        "girl": "person",
        "woman": "person",
        "boy": "person",
        "dancer": "person",
        "dog": "dog",
        "bear": "bear",
        "cow": "cow",
        "cows": "cow",
        "horse": "horse",
        "rhino": "rhino",
        "train": "train",
    }
    return [prompt for token, prompt in token_prompts.items()
            if token in stem]


def resolve_prompt_frames(specs: list[str], n_frames: int) -> list[int]:
    indices = []
    for spec in specs:
        if spec == "middle":
            index = n_frames // 2
        elif spec == "last":
            index = n_frames - 1
        else:
            index = int(spec)
        if not 0 <= index < n_frames:
            raise ValueError(f"prompt frame index out of range: {index} for {n_frames} frames")
        if index not in indices:
            indices.append(index)
    return indices


def has_foreground_masks(masks: dict[int, np.ndarray]) -> bool:
    return any(np.any(mask > 0) for mask in masks.values())


def run_scene(predictor, scene_dir: Path, device: str,
              viz_dir: Path, viz_frames: int,
              prompt_frames: list[str],
              extra_prompts: list[str] | None,
              auto_subject: bool,
              use_caption: bool) -> None:
    caption_path = scene_dir / "caption.txt"
    if not caption_path.is_file():
        raise FileNotFoundError(f"caption not found: {caption_path}")
    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"empty caption: {caption_path}")

    subjects_path = scene_dir / "subjects.txt"
    caption_subjects: list[str] = []
    if subjects_path.is_file():
        raw = subjects_path.read_text(encoding="utf-8").strip()
        caption_subjects = [s.strip() for s in raw.split(",") if s.strip()]
    caption_subjects = [s for s in caption_subjects
                        if s.lower() not in BACKGROUND_WORDS]

    frames_dir = scene_dir / "frames"
    mask_dir = scene_dir / "mask_dynamic"
    mask_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no PNG frames under {frames_dir}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=scene_dir,
                                     delete=False) as tmp:
        video_path = Path(tmp.name)
    try:
        n_frames, width, height = make_video(frames_dir, video_path)
        response = predictor.handle_request({
            "type": "start_session", "resource_path": str(video_path)
        })
        session_id = response["session_id"]
        union_masks = {}
        subjects = SUBJECT_PROMPTS.get(scene_dir.name, []) if auto_subject else []
        if auto_subject and not subjects:
            subjects = inferred_subject_prompts(scene_dir.name)
        subjects = caption_subjects + subjects
        prompts = (subjects if subjects else ([caption] if use_caption else []))
        prompts.extend(p for p in (extra_prompts or []) if p.strip())
        anchor_indices = resolve_prompt_frames(prompt_frames, n_frames)
        try:
            for prompt_index, prompt in enumerate(prompts):
                for anchor_index, frame_index in enumerate(anchor_indices):
                    if prompt_index > 0 or anchor_index > 0:
                        predictor.handle_request({
                            "type": "reset_session", "session_id": session_id,
                        })
                    predictor.handle_request({
                        "type": "add_prompt", "session_id": session_id,
                        "frame_index": frame_index, "text": prompt,
                    })
                    current = collect_union(predictor, session_id)
                    for current_index, mask in current.items():
                        if current_index not in union_masks:
                            union_masks[current_index] = mask
                        else:
                            if union_masks[current_index].shape != mask.shape:
                                mask = cv2.resize(
                                    mask,
                                    (union_masks[current_index].shape[1],
                                     union_masks[current_index].shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            union_masks[current_index] = np.maximum(
                                union_masks[current_index], mask)
                    if has_foreground_masks(current):
                        break
        finally:
            try:
                predictor.handle_request({
                    "type": "close_session", "session_id": session_id,
                })
            except Exception:
                pass

        for index in range(n_frames):
            mask = union_masks.get(
                index, np.zeros((height, width), dtype=np.uint8))
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height),
                                  interpolation=cv2.INTER_NEAREST)
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(
                mask_dir / f"{index:06d}.png")
        save_mask_review(scene_dir, mask_dir, viz_dir, viz_frames)
        print(f"[done] {scene_dir.name}: {len(union_masks)}/{n_frames} masked frames")
    finally:
        video_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if args.names:
        names = list(args.names)
    else:
        names = sorted(
            d.name for d in data_root.iterdir()
            if d.is_dir() and d.name.endswith("_vggt"))
    missing = [name for name in names if not (data_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"scene dirs not found: {missing}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    pending = []
    for name in names:
        scene_dir = data_root / name
        mask_dir = scene_dir / "mask_dynamic"
        if args.overwrite or not has_foreground(mask_dir):
            pending.append(name)
        else:
            print(f"[skip] {scene_dir}: mask_dynamic has foreground")
    if not pending:
        print("[done] no scenes need SAM3 masks")
        return

    predictor = build_sam3_predictor(
        version="sam3.1", checkpoint_path=args.checkpoint,
        use_fa3=not args.no_fa3, compile=False, warm_up=False,
    )
    viz_dir = Path(args.viz_dir)
    for name in pending:
        scene_dir = data_root / name
        run_scene(predictor, scene_dir, args.device, viz_dir,
                  args.viz_frames, args.prompt_frames, args.extra_prompts,
                  not args.no_auto_subject, not args.no_caption)


if __name__ == "__main__":
    main()
