"""Generate per-scene captions from PNG frames or MP4 videos via Qwen2.5-VL.

Use ``--input_type frames`` for DL3DV/VGGT scene directories containing
``frames/`` or ``images_4/``. Use ``--input_type video`` for dynpose,
openvidhd, HuMMan, or other directories containing ``video.mp4``.

Both modes write ``caption.txt`` and share the same HF/vLLM backend,
resume behavior, failure log, and optional external output layout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
from PIL import Image


DEFAULT_FRAMES_PROMPT = (
    "Describe this scene in one concise English sentence (under 30 words). "
    "Focus on visible objects, spatial layout and lighting. "
    "Do NOT mention camera angle, camera movement or photographic terms. "
    "Then on a new line starting with 'Subjects:', list ONLY concrete moving "
    "objects as comma-separated short nouns (e.g. Subjects: person, dog, car). "
    "List ONLY distinct physical entities that actually move in the video — "
    "people, animals, vehicles, sports equipment. "
    "Do NOT include background elements, scenery, weather, or environment "
    "(e.g. exclude water, sky, grass, wind, trees, buildings, road, sand)."
)
DEFAULT_VIDEO_PROMPT = (
    "Describe this scene in one concise English sentence (under 30 words). "
    "Focus on visible objects, spatial layout, lighting and any motion. "
    "Do NOT mention camera angle, camera movement or photographic terms. "
    "Then on a new line starting with 'Subjects:', list ONLY concrete moving "
    "objects as comma-separated short nouns (e.g. Subjects: person, dog, car). "
    "List ONLY distinct physical entities that actually move in the video — "
    "people, animals, vehicles, sports equipment. "
    "Do NOT include background elements, scenery, weather, or environment "
    "(e.g. exclude water, sky, grass, wind, trees, buildings, road, sand)."
)


# ── Scene discovery -----------------------------------------------------------

def list_scenes(data_root: Path, input_type: str,
               source_name: str) -> list[Path]:
    """Return scene dirs containing the selected frame or video source."""
    scenes: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        source = p / source_name
        if input_type == "video" and source.is_file():
            scenes.append(p)
        elif input_type == "frames" and source.is_dir():
            if any(source.glob("frame_*.png")) or any(source.glob("*.png")):
                scenes.append(p)
    return scenes


def caption_out_path(scene_dir: Path,
                     caption_out_dir: Path | None) -> Path:
    """Resolve where ``scene_dir``'s caption.txt should live.

    Two layouts are supported:

      * ``caption_out_dir is None`` → legacy in-place layout
        ``<scene_dir>/caption.txt``.  This is the default layout and is
        correct when the scene dir survives the captioning step (DL3DV,
        dynpose, openvidhd).

      * ``caption_out_dir is not None`` → external layout
        ``<caption_out_dir>/<scene_dir.name>/caption.txt``.  Needed
        for HuMMan, whose source mp4s live in a ``_staging`` dir
        that gets cleaned up at the end of the preprocess job (see
        ``build_humman_multicam.py`` line 731-733).  The mirrored
        ``<scene_name>/caption.txt`` shape is exactly what
        ``encode_text_t5.py`` expects on its own ``data_root``
        scan, so the downstream T5-encoding step needs ZERO changes.
    """
    if caption_out_dir is None:
        return scene_dir / "caption.txt"
    return caption_out_dir / scene_dir.name / "caption.txt"


def caption_is_done(scene_dir: Path,
                    caption_out_dir: Path | None = None) -> bool:
    """A scene is "done" iff its caption.txt (resolved via
    ``caption_out_path``) exists, is a real file (not a broken
    symlink), and is non-empty.

    The ``dynpose_depth_pose`` dataset ships a ``caption.txt`` symlink
    pointing to a path that doesn't exist on this machine, so we explicitly
    check ``is_file()`` (which dereferences) instead of ``exists()`` (which
    follows symlinks but only checks target existence – same as is_file but
    being explicit avoids future confusion).
    """
    cap = caption_out_path(scene_dir, caption_out_dir)
    if cap.is_symlink() and not cap.exists():
        # Broken symlink: treat as not-done.
        return False
    return cap.is_file() and cap.stat().st_size > 0


def filter_scene_names(scenes: list[Path],
                       scene_names: list[str] | None) -> list[Path]:
    if not scene_names:
        return scenes
    wanted = set(scene_names)
    selected = [scene for scene in scenes if scene.name in wanted]
    missing = sorted(wanted - {scene.name for scene in selected})
    if missing:
        print(f"[warn] scene_names not found: {missing}")
    return selected


def filter_unprocessed(scenes: list[Path],
                       caption_out_dir: Path | None = None,
                       overwrite: bool = False) -> list[Path]:
    if overwrite:
        return scenes
    out: list[Path] = []
    skipped = 0
    for sd in scenes:
        if caption_is_done(sd, caption_out_dir):
            skipped += 1
            continue
        out.append(sd)
    if skipped:
        print(f"[scan] skipping {skipped} scenes that already have caption.txt")
    return out


# ── Frame extraction from mp4 -------------------------------------------------

def pick_frame_indices(num_frames_total: int, num_frames: int) -> list[int]:
    """Evenly-spaced indices (1-based slot, then convert to 0-based) like
    caption_dl3dv_vlm.pick_frames does on PNG lists.

    Returns 0-indexed frame ids into the mp4.
    """
    if num_frames >= num_frames_total:
        return list(range(num_frames_total))
    step = num_frames_total / (num_frames + 1)
    idxs = [int(round(step * (i + 1))) for i in range(num_frames)]
    idxs = [min(max(i, 0), num_frames_total - 1) for i in idxs]
    return idxs


def load_frames_from_dir(frame_dir: Path, num_frames: int,
                         image_max_side: int = 0) -> list[Image.Image]:
    """Read evenly-spaced PIL frames from a PNG frame directory."""
    paths = sorted(frame_dir.glob("frame_*.png"))
    if not paths:
        paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames in {frame_dir}")
    idxs = pick_frame_indices(len(paths), num_frames)
    frames = [Image.open(paths[i]).convert("RGB") for i in idxs]
    return resize_frames(frames, image_max_side)


def resize_frames(frames: list[Image.Image], image_max_side: int) -> list[Image.Image]:
    if not image_max_side or image_max_side <= 0:
        return frames
    out: list[Image.Image] = []
    for im in frames:
        w, h = im.size
        longest = max(w, h)
        if longest > image_max_side:
            scale = image_max_side / longest
            im = im.resize((int(round(w * scale)), int(round(h * scale))),
                           Image.BICUBIC)
        out.append(im)
    return out


def load_frames_from_mp4(video_path: Path, num_frames: int,
                         image_max_side: int = 0) -> list[Image.Image]:
    """Read ``num_frames`` evenly-spaced PIL frames from an mp4.

    ``image_max_side`` (default 0 = disabled) caps the longer side of
    each PIL frame after decoding.  Needed for HuMMan whose native
    1920x1080 frames blow past Qwen2.5-VL's default 4096-token
    context (one 1080p frame already costs ~10.5k vision tokens).
    Setting ``image_max_side=672`` keeps the per-frame token cost
    around ~1.7k → ~5k tokens for 3 frames + prompt — safely within
    the 4096-or-bumped budget.
    """
    # imageio.v3 reads the full clip into one (T, H, W, 3) uint8 array.
    # For 90-frame 720p clips that's ~150 MB — fine, and faster than
    # opening N separate decoders for individual frame indexing.
    arr = iio.imread(str(video_path), plugin="pyav")  # (T, H, W, 3) uint8
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise RuntimeError(f"{video_path}: unexpected shape {arr.shape}")
    T = arr.shape[0]
    idxs = pick_frame_indices(T, num_frames)
    frames = [Image.fromarray(arr[i]) for i in idxs]
    return resize_frames(frames, image_max_side)


# ── Message builder (shared by both backends) -------------------------------

def build_messages(pil_images: list[Image.Image], prompt: str) -> list[dict]:
    """Build a Qwen2.5-VL chat message list with N PIL images + 1 text.

    Qwen2.5-VL's ``process_vision_info`` accepts PIL Image objects directly
    via ``{"type": "image", "image": pil_img}`` — no need to round-trip
    through temporary PNG files on disk.
    """
    content: list[dict] = []
    for img in pil_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ── vLLM backend ------------------------------------------------------------

def load_scene_frames(scene_dir: Path, input_type: str,
                      source_name: str, num_frames: int,
                      image_max_side: int) -> list[Image.Image]:
    if input_type == "frames":
        return load_frames_from_dir(scene_dir / source_name, num_frames,
                                    image_max_side)
    return load_frames_from_mp4(scene_dir / source_name, num_frames,
                                image_max_side)


def run_vllm(scenes: list[Path], input_type: str, source_name: str, num_frames: int,
             prompt: str, model_path: str, batch_size: int,
             max_new_tokens: int, fail_log: Path,
             caption_out_dir: Path | None,
             image_max_side: int) -> None:
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    print(f"[vllm] loading {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": max(num_frames, 1)},
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True)
    sampling = SamplingParams(temperature=0.2, top_p=0.9,
                              max_tokens=max_new_tokens)

    t_start = time.time()
    n_done = 0
    n_fail = 0
    n_total = len(scenes)

    for batch_start in range(0, n_total, batch_size):
        batch_scenes = scenes[batch_start:batch_start + batch_size]
        vllm_inputs: list[dict] = []
        batch_out_paths: list[Path] = []
        batch_keep: list[Path] = []

        for sd in batch_scenes:
            try:
                pil_frames = load_scene_frames(
                    sd, input_type, source_name, num_frames, image_max_side)
                messages = build_messages(pil_frames, prompt)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                vllm_inputs.append({
                    "prompt": text,
                    "multi_modal_data": {"image": image_inputs},
                })
                batch_out_paths.append(caption_out_path(sd, caption_out_dir))
                batch_keep.append(sd)
            except Exception as e:
                n_fail += 1
                _record_fail(fail_log, sd, e)

        if not vllm_inputs:
            continue
        outputs = llm.generate(vllm_inputs, sampling, use_tqdm=False)
        for sd, out_path, out in zip(batch_keep, batch_out_paths, outputs):
            raw = out.outputs[0].text.strip()
            cap, subjects = _parse_caption_and_subjects(raw)
            cap = _clean_caption(cap)
            if not cap:
                n_fail += 1
                _record_fail(fail_log, sd, RuntimeError("empty caption"))
                continue
            _write_caption(out_path, cap)
            if subjects:
                _write_subjects(out_path.parent / "subjects.txt", subjects)
            n_done += 1

        if (batch_start // batch_size) % 5 == 0 or batch_start + batch_size >= n_total:
            dt = time.time() - t_start
            rate = (n_done + n_fail) / max(dt, 1e-6)
            eta = (n_total - n_done - n_fail) / max(rate, 1e-6)
            print(f"[vllm] {n_done + n_fail}/{n_total}  "
                  f"ok={n_done} fail={n_fail}  "
                  f"({rate:.1f} scenes/s, eta {eta/60:.1f} min)")


# ── HF backend (no vLLM dep) ------------------------------------------------

def run_hf(scenes: list[Path], input_type: str, source_name: str, num_frames: int,
           prompt: str, model_path: str, batch_size: int,
           max_new_tokens: int, device: str, fail_log: Path,
           caption_out_dir: Path | None,
           image_max_side: int) -> None:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"[hf] loading {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True)

    t_start = time.time()
    n_done = 0
    n_fail = 0
    n_total = len(scenes)

    for batch_start in range(0, n_total, batch_size):
        batch_scenes = scenes[batch_start:batch_start + batch_size]
        batch_texts: list[str] = []
        batch_images: list = []
        batch_keep: list[Path] = []

        for sd in batch_scenes:
            try:
                pil_frames = load_scene_frames(
                    sd, input_type, source_name, num_frames, image_max_side)
                messages = build_messages(pil_frames, prompt)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                batch_texts.append(text)
                batch_images.append(image_inputs)
                batch_keep.append(sd)
            except Exception as e:
                n_fail += 1
                _record_fail(fail_log, sd, e)

        if not batch_keep:
            continue
        try:
            inputs = processor(
                text=batch_texts, images=batch_images,
                padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0)
            in_len = inputs.input_ids.shape[1]
            decoded = processor.batch_decode(
                gen[:, in_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
        except Exception as e:
            for sd in batch_keep:
                n_fail += 1
                _record_fail(fail_log, sd, e)
            continue

        for sd, cap in zip(batch_keep, decoded):
            raw = cap
            cap, subjects = _parse_caption_and_subjects(raw)
            cap = _clean_caption(cap)
            out_path = caption_out_path(sd, caption_out_dir)
            if not cap:
                n_fail += 1
                _record_fail(fail_log, sd, RuntimeError("empty caption"))
                continue
            _write_caption(out_path, cap)
            if subjects:
                _write_subjects(out_path.parent / "subjects.txt", subjects)
            n_done += 1

        if (batch_start // batch_size) % 5 == 0 or batch_start + batch_size >= n_total:
            dt = time.time() - t_start
            rate = (n_done + n_fail) / max(dt, 1e-6)
            eta = (n_total - n_done - n_fail) / max(rate, 1e-6)
            print(f"[hf] {n_done + n_fail}/{n_total}  "
                  f"ok={n_done} fail={n_fail}  "
                  f"({rate:.1f} scenes/s, eta {eta/60:.1f} min)")


# ── Helpers -----------------------------------------------------------------

def _clean_caption(text: str) -> str:
    """Light-weight cleanup: collapse whitespace, strip quotes, single line."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    cleaned = cleaned.strip('"\u201c\u201d\u2018\u2019\' ')
    return cleaned


def _write_caption(out_path: Path, caption: str) -> None:
    """Write caption to disk, handling pre-existing broken symlinks.

    ``dynpose_depth_pose`` ships ``caption.txt`` as a symlink to a path
    that doesn't exist on this machine; opening it for write fails with
    ENOENT.  We detect this and ``unlink`` the bad link before writing.

    Also creates the parent dir on-the-fly, which matters in the
    ``--caption_out_dir`` external-layout mode where each scene needs
    its own subdir.
    """
    if out_path.is_symlink() and not out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(caption + "\n", encoding="utf-8")


def _parse_caption_and_subjects(raw: str) -> tuple[str, list[str]]:
    """Parse VLM output into (caption_sentence, subject_list).

    Expected format:
        <description sentence>
        Subjects: noun1, noun2, noun3

    Falls back gracefully: if no 'Subjects:' line found, returns the
    full text as caption and an empty subject list.
    """
    text = raw.strip()
    subjects: list[str] = []
    caption = text
    for marker in ("Subjects:", "subjects:", "SUBJECTS:"):
        idx = text.find(marker)
        if idx >= 0:
            caption = text[:idx].strip()
            subj_line = text[idx + len(marker):].strip()
            subjects = [s.strip() for s in subj_line.split(",") if s.strip()]
            break
    return caption, subjects


def _write_subjects(out_path: Path, subjects: list[str]) -> None:
    """Write subjects.txt alongside caption.txt."""
    if out_path.is_symlink() and not out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(", ".join(subjects) + "\n", encoding="utf-8")


def _record_fail(fail_log: Path, scene_dir: Path, err: BaseException) -> None:
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_log, "a") as f:
        f.write(f"{scene_dir.name}\t{type(err).__name__}: {err}\n")


# ── main --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Root containing one subdir per scene.")
    p.add_argument("--input_type", choices=["video", "frames"], default="video",
                   help="Read a video file or a directory of PNG frames.")
    p.add_argument("--video_name", type=str, default="video.mp4",
                   help="Video filename when --input_type=video.")
    p.add_argument("--frame_subdir", type=str, default="frames",
                   help="PNG directory when --input_type=frames.")
    p.add_argument("--scene_names", nargs="+", default=None,
                   help="Only process these scene directory names.")
    p.add_argument("--model_path", type=str,
                   default=os.environ.get(
                       "QWEN_MODEL", "checkpoints/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--backend", type=str, choices=["vllm", "hf"],
                   default="vllm")
    p.add_argument("--num_frames", type=int, default=3,
                   help="How many evenly-spaced mp4 frames to feed the VLM "
                        "per scene. Default 3 (beg/mid/end) — these are "
                        "dynamic videos so a single middle frame can miss "
                        "important motion/content.")
    p.add_argument("--prompt", type=str, default="",
                   help="Prompt override; defaults depend on --input_type.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--device", type=str, default="cuda:0",
                   help="HF backend only. vLLM uses all visible GPUs.")
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, cap the number of scenes processed (debug).")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate caption.txt even when it already exists.")
    p.add_argument("--fail_log", type=str, default="",
                   help="Path to a tsv that records failures. Default: "
                        "<data_root>/_caption_fail.log")
    p.add_argument("--caption_out_dir", type=str, default="",
                   help="When set, write captions to "
                        "<caption_out_dir>/<scene_name>/caption.txt instead "
                        "of <scene>/caption.txt.  Needed for HuMMan where "
                        "the input scene dirs live in a temp _staging dir "
                        "that gets cleaned up after preprocessing.  The "
                        "<scene_name>/caption.txt mirrored layout is what "
                        "encode_text_t5.py expects, so the T5 step "
                        "can point its --data_root at this dir as-is.")
    p.add_argument("--image_max_side", type=int, default=0,
                   help="Cap the longer side of each decoded frame to N "
                        "pixels before sending to the VLM (0 = no resize). "
                        "Needed for high-res sources like HuMMan (1920x1080) "
                        "whose native frames blow past Qwen2.5-VL's 4096 "
                        "context window — set to 672 for HuMMan to keep "
                        "3-frame inputs comfortably under 4096 tokens. "
                        "Lower values (e.g. 512) work fine too and save "
                        "a bit of compute; 672 matches Qwen2.5-VL's native "
                        "28-pixel patch alignment so there's no padding loss.")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    fail_log = (Path(args.fail_log) if args.fail_log else
                data_root / "_caption_fail.log")
    caption_out_dir = (Path(args.caption_out_dir).resolve()
                       if args.caption_out_dir else None)
    if caption_out_dir is not None:
        caption_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] caption_out_dir = {caption_out_dir}")

    source_name = args.video_name if args.input_type == "video" else args.frame_subdir
    prompt = args.prompt or (DEFAULT_VIDEO_PROMPT if args.input_type == "video"
                             else DEFAULT_FRAMES_PROMPT)
    print(f"[init] scanning {data_root} ({args.input_type}: {source_name})")
    scenes_all = list_scenes(data_root, args.input_type, source_name)
    print(f"[init] found {len(scenes_all)} scenes with {source_name}")
    scenes_all = filter_scene_names(scenes_all, args.scene_names)
    scenes = filter_unprocessed(scenes_all, caption_out_dir, args.overwrite)
    if args.limit and args.limit < len(scenes):
        scenes = scenes[:args.limit]
        print(f"[init] --limit {args.limit} → processing {len(scenes)} scenes")
    if not scenes:
        print("[done] nothing to caption")
        return
    print(f"[init] {len(scenes)} scenes to caption  "
          f"({args.num_frames} frame(s)/scene, batch={args.batch_size}, "
          f"backend={args.backend})")

    if args.backend == "vllm":
        run_vllm(scenes, args.input_type, source_name, args.num_frames, prompt,
                 args.model_path, args.batch_size, args.max_new_tokens,
                 fail_log, caption_out_dir, args.image_max_side)
    else:
        run_hf(scenes, args.input_type, source_name, args.num_frames, prompt,
               args.model_path, args.batch_size, args.max_new_tokens,
               args.device, fail_log, caption_out_dir, args.image_max_side)

    print(f"[done] failures (if any) recorded at {fail_log}")


if __name__ == "__main__":
    sys.exit(main() or 0)
