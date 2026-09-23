"""Run the reusable VGGT → Qwen caption → T5 → SAM3 → trajectory pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Scene:
    name: str
    source: str
    vggt: str
    output: str


STAGES = ("vggt", "caption", "t5", "sam3", "traj", "validate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reusable VGGT, caption, T5, SAM3 and trajectory pipeline."
    )
    parser.add_argument(
        "--data_root",
        default="data/raw",
        help="Scene directory; by default all scene folders under this path are processed.",
    )
    parser.add_argument(
        "--raw_root",
        default="",
        help="Root containing raw scene image folders. Defaults to data_root.",
    )
    parser.add_argument("--traj_root", default="data/val_traj")
    parser.add_argument(
        "--scene_manifest",
        default="",
        help="Optional JSON scene manifest overriding automatic scene discovery.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional scene names; default is every scene folder under data_root.",
    )
    parser.add_argument("--stages", nargs="+", default=["all"],
                        choices=["all", *STAGES])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--vggt_env", default="vggt")
    parser.add_argument("--vggt_checkpoint",
                        default=os.environ.get(
                            "VGGT_CHECKPOINT",
                            "checkpoints/vggt_omega_1b_512.pt"))
    parser.add_argument("--vggt_root",
                        default=os.environ.get(
                            "VGGT_ROOT", "external/vggt-omega-main"))
    parser.add_argument("--vggt_device", default="cuda:0")
    parser.add_argument("--vggt_resolution", type=int, default=512)
    parser.add_argument("--vggt_mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument("--vggt_fps", type=float, default=24.0)
    parser.add_argument("--vggt_max_frames", type=int, default=0)

    parser.add_argument("--qwen_env", default="wan22")
    parser.add_argument("--qwen_model",
                        default=os.environ.get(
                            "QWEN_MODEL", "checkpoints/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--qwen_backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--qwen_device", default="cuda:0")
    parser.add_argument("--qwen_num_frames", type=int, default=3)
    parser.add_argument("--qwen_batch_size", type=int, default=1)
    parser.add_argument("--qwen_max_new_tokens", type=int, default=80)
    parser.add_argument("--qwen_prompt", default="",
                        help="Optional prompt override passed to the caption script.")

    parser.add_argument("--t5_env", default="wan22")
    parser.add_argument("--t5_checkpoint",
                        default=os.environ.get(
                            "T5_CHECKPOINT",
                            "checkpoints/wan/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"))
    parser.add_argument("--t5_tokenizer",
                        default=os.environ.get(
                            "T5_TOKENIZER",
                            "checkpoints/wan/Wan2.1-T2V-14B/google/umt5-xxl"))
    parser.add_argument("--t5_device", default="cuda:0")
    parser.add_argument("--t5_batch_size", type=int, default=6)

    parser.add_argument("--sam3_env", default="sam3")
    parser.add_argument("--sam3_checkpoint",
                        default=os.environ.get(
                            "SAM3_CHECKPOINT",
                            "external/sam3/ckpt/sam3.1/sam3.1_multiplex.pt"))
    parser.add_argument("--sam3_device", default="cuda:0")
    parser.add_argument("--sam3_viz_dir", default="render_output/sam3_caption_review")
    parser.add_argument("--sam3_viz_frames", type=int, default=3)
    parser.add_argument("--sam3_extra_prompts", nargs="*", default=None,
                        help="Extra prompts unioned into every selected scene mask.")
    parser.add_argument("--sam3_no_auto_subject", action="store_true")
    parser.add_argument("--use_fa3", action="store_true")

    parser.add_argument("--traj_max_down_offset", type=float, default=0.0)
    return parser.parse_args()


def discover_scenes(data_root: Path) -> list[Scene]:
    """Discover scenes from raw folders and/or existing ``*_vggt`` folders."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist or is not a directory: {data_root}")

    scenes_by_name: dict[str, Scene] = {}
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("."):
            continue
        if scene_dir.name.endswith("_vggt"):
            name = scene_dir.name[:-5]
            scene = Scene(name=name, source=name, vggt=scene_dir.name, output=name)
        else:
            name = scene_dir.name
            scene = Scene(name=name, source=name, vggt=f"{name}_vggt", output=name)
        scenes_by_name[name] = scene

    scenes = list(scenes_by_name.values())
    if not scenes:
        raise ValueError(f"no scene directories found under data_root: {data_root}")
    return scenes


def load_scenes(path: str, data_root: Path) -> list[Scene]:
    if not path:
        return discover_scenes(data_root)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("scenes", payload) if isinstance(payload, dict) else payload
    scenes: list[Scene] = []
    if isinstance(entries, dict):
        entries = [{"name": name, **value} for name, value in entries.items()]
    for item in entries:
        scenes.append(Scene(
            name=item["name"],
            source=item.get("source", item["name"]),
            vggt=item.get("vggt", f"{item.get('source', item['name'])}_vggt"),
            output=item.get("output", item["name"]),
        ))
    if not scenes:
        raise ValueError(f"empty scene manifest: {path}")
    return scenes


def infer_scene(name: str) -> Scene:
    """Infer the standard layout for a scene outside the built-in manifest."""
    if name.endswith("_vggt"):
        base_name = name[:-5]
        vggt_name = name
    else:
        base_name = name
        vggt_name = f"{name}_vggt"
    return Scene(
        name=base_name,
        source=base_name,
        vggt=vggt_name,
        output=base_name,
    )


def select_scenes(scenes: list[Scene], names: list[str] | None) -> list[Scene]:
    if not names:
        return scenes
    by_name = {scene.name: scene for scene in scenes}
    selected = []
    for name in names:
        scene = by_name.get(name)
        if scene is None:
            scene = next((item for item in scenes if item.vggt == name), None)
        if scene is None:
            scene = infer_scene(name)
            print(f"[infer] {name}: source={scene.source} "
                  f"vggt={scene.vggt} output={scene.output}")
        selected.append(scene)
    return selected


def command_for(env_name: str, script: Path, args: list[str]) -> list[str]:
    if env_name in ("", "current"):
        prefix = [sys.executable]
    else:
        prefix = ["conda", "run", "-n", env_name, "python"]
    return prefix + [str(script), *args]


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"[run] {shlex.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def has_frames(scene_dir: Path) -> bool:
    return (scene_dir / "frames").is_dir() and any(
        p.suffix.lower() == ".png" for p in (scene_dir / "frames").iterdir()
    )


def vggt_complete(scene_dir: Path) -> bool:
    return (
        (scene_dir / "predictions.npz").is_file()
        and (scene_dir / "trajectory" / "trajectory.npz").is_file()
        and has_frames(scene_dir)
    )


def require_vggt(scenes: list[Scene], data_root: Path) -> None:
    missing = [scene.vggt for scene in scenes if not vggt_complete(data_root / scene.vggt)]
    if missing:
        raise RuntimeError(f"VGGT output missing or incomplete: {missing}")


def require_captions(scenes: list[Scene], data_root: Path) -> None:
    missing = [scene.vggt for scene in scenes
               if not (data_root / scene.vggt / "caption.txt").is_file()]
    if missing:
        raise RuntimeError(f"caption.txt missing: {missing}")


def require_embeddings(scenes: list[Scene], data_root: Path) -> None:
    missing = [scene.vggt for scene in scenes
               if not (data_root / scene.vggt / "text_embedding.pt").is_file()]
    if missing:
        raise RuntimeError(f"text_embedding.pt missing: {missing}")


def has_mask(scene_dir: Path) -> bool:
    for mask_dir in scene_dir.glob("mask*"):
        if not mask_dir.is_dir():
            continue
        for mask_path in mask_dir.glob("*.png"):
            with Image.open(mask_path) as mask:
                if mask.getbbox() is not None:
                    return True
    return False


def require_masks(scenes: list[Scene], data_root: Path) -> None:
    missing = [scene.vggt for scene in scenes
               if not has_mask(data_root / scene.vggt)]
    if missing:
        raise RuntimeError(f"no non-empty mask* directory: {missing}")


def run_vggt(args: argparse.Namespace, scenes: list[Scene], data_root: Path,
             raw_root: Path, repo_root: Path) -> None:
    pending = []
    for scene in scenes:
        output_dir = data_root / scene.vggt
        raw_dir = raw_root / scene.source
        if args.overwrite and raw_dir.is_dir():
            pending.append(scene.source)
        elif not vggt_complete(output_dir):
            if not raw_dir.is_dir():
                raise FileNotFoundError(
                    f"no raw source {raw_dir} and no complete VGGT output {output_dir}")
            pending.append(scene.source)
    if not pending:
        print("[skip] VGGT: all selected scenes are complete")
        return
    script = repo_root / "scripts/preprocess/prepare_vggt_omega_core.py"
    cmd_args = [
        "--data_root", str(raw_root), "--output_root", str(data_root),
        "--names", *pending, "--checkpoint", args.vggt_checkpoint,
        "--vggt_root", args.vggt_root, "--image_resolution", str(args.vggt_resolution),
        "--mode", args.vggt_mode, "--device", args.vggt_device,
        "--fps", str(args.vggt_fps), "--max_frames", str(args.vggt_max_frames),
    ]
    if args.overwrite:
        cmd_args.append("--overwrite")
    run_command(command_for(args.vggt_env, script, cmd_args), repo_root, args.dry_run)


def run_caption(args: argparse.Namespace, scenes: list[Scene], data_root: Path,
                repo_root: Path) -> None:
    if not args.dry_run:
        require_vggt(scenes, data_root)
    script = repo_root / "scripts/preprocess/caption_vlm.py"
    cmd_args = [
        "--data_root", str(data_root), "--input_type", "frames",
        "--frame_subdir", "frames",
        "--scene_names", *(scene.vggt for scene in scenes),
        "--model_path", args.qwen_model, "--backend", args.qwen_backend,
        "--num_frames", str(args.qwen_num_frames), "--batch_size", str(args.qwen_batch_size),
        "--max_new_tokens", str(args.qwen_max_new_tokens), "--device", args.qwen_device,
    ]
    if args.qwen_prompt:
        cmd_args.extend(["--prompt", args.qwen_prompt])
    if args.overwrite:
        cmd_args.append("--overwrite")
    run_command(command_for(args.qwen_env, script, cmd_args), repo_root, args.dry_run)


def run_t5(args: argparse.Namespace, scenes: list[Scene], data_root: Path,
           repo_root: Path) -> None:
    if not args.dry_run:
        require_captions(scenes, data_root)
    script = repo_root / "scripts/preprocess/encode_text_t5.py"
    cmd_args = [
        "--data_root", str(data_root), "--in_place",
        "--scene_names", *(scene.vggt for scene in scenes),
        "--t5_checkpoint", args.t5_checkpoint, "--t5_tokenizer", args.t5_tokenizer,
        "--batch_size", str(args.t5_batch_size), "--device", args.t5_device,
    ]
    if args.overwrite:
        cmd_args.append("--overwrite")
    run_command(command_for(args.t5_env, script, cmd_args), repo_root, args.dry_run)


def run_sam3(args: argparse.Namespace, scenes: list[Scene], data_root: Path,
             repo_root: Path) -> None:
    if not args.dry_run:
        require_captions(scenes, data_root)
    script = repo_root / "scripts/preprocess/run_sam3_caption_mask.py"
    cmd_args = [
        "--data_root", str(data_root), "--names", *(scene.vggt for scene in scenes),
        "--device", args.sam3_device, "--checkpoint", args.sam3_checkpoint,
        "--viz_dir", args.sam3_viz_dir, "--viz_frames", str(args.sam3_viz_frames),
    ]
    if args.overwrite:
        cmd_args.append("--overwrite")
    if args.sam3_extra_prompts:
        cmd_args.extend(["--extra_prompts", *args.sam3_extra_prompts])
    if args.sam3_no_auto_subject:
        cmd_args.append("--no-auto-subject")
    if not args.use_fa3:
        cmd_args.append("--no-fa3")
    run_command(command_for(args.sam3_env, script, cmd_args), repo_root, args.dry_run)


def run_traj(args: argparse.Namespace, scenes: list[Scene], data_root: Path,
             traj_root: Path, repo_root: Path) -> None:
    if not args.dry_run:
        require_vggt(scenes, data_root)
        require_masks(scenes, data_root)
    script = repo_root / "scripts/preprocess/build_val_traj.py"
    for scene in scenes:
        output = traj_root / scene.output
        complete = all((output / bucket / "trajectory_new" / "trajectory.npz").is_file()
                       for bucket in ("f1", "f2", "f3"))
        if complete and not args.overwrite:
            print(f"[skip] trajectory: {scene.name}")
            continue
        cmd_args = [
            "--shared_dir", str(data_root / scene.vggt),
            "--out_dir", str(output), "--scene_tag", scene.name,
            "--max_down_offset", str(args.traj_max_down_offset),
        ]
        run_command(command_for("current", script, cmd_args), repo_root, args.dry_run)


def validate(scenes: list[Scene], data_root: Path, traj_root: Path) -> bool:
    all_ok = True
    for scene in scenes:
        root = data_root / scene.vggt
        checks = {
            "vggt": vggt_complete(root),
            "caption": (root / "caption.txt").is_file() and (root / "caption.txt").stat().st_size > 0,
            "text_embedding": (root / "text_embedding.pt").is_file(),
            "mask": has_mask(root),
        }
        checks["trajectories"] = all(
            (traj_root / scene.output / bucket / "trajectory_new" / "trajectory.npz").is_file()
            for bucket in ("f1", "f2", "f3")
        )
        scene_ok = all(checks.values())
        all_ok &= scene_ok
        print(f"[validate] {scene.name}: {'OK' if scene_ok else 'FAIL'} {checks}")
    return all_ok


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root).resolve()
    raw_root = Path(args.raw_root).resolve() if args.raw_root else data_root
    traj_root = Path(args.traj_root).resolve()
    scenes = select_scenes(load_scenes(args.scene_manifest, data_root), args.names)
    selected = set(STAGES) if "all" in args.stages else set(args.stages)
    print(f"[init] scenes={[scene.name for scene in scenes]}")
    print(f"[init] stages={sorted(selected)}")

    if "vggt" in selected:
        run_vggt(args, scenes, data_root, raw_root, repo_root)
    if "caption" in selected:
        run_caption(args, scenes, data_root, repo_root)
    if "t5" in selected:
        run_t5(args, scenes, data_root, repo_root)
    if "sam3" in selected:
        run_sam3(args, scenes, data_root, repo_root)
    if "traj" in selected:
        run_traj(args, scenes, data_root, traj_root, repo_root)
    if "validate" in selected and args.dry_run:
        print("[validate] dry-run; validation skipped")
    elif "validate" in selected:
        if not validate(scenes, data_root, traj_root):
            raise SystemExit(1)
        print(f"[done] validation passed for {len(scenes)} scenes")


if __name__ == "__main__":
    main()
