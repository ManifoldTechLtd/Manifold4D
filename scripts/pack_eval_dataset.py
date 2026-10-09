"""Pack the Manifold4D evaluation dataset for the HuggingFace release.

Collects every preprocessed scene (``<name>_vggt`` dirs produced by
``scripts/preprocess/run_vggt_caption_t5_sam3_traj.py``) plus the novel-view
trajectories (``val_traj/<scene>/{f1,f2,f3}/trajectory_new/``) into a
HuggingFace-ready output tree, downcasting the bulky fp32 ``depth`` /
``depth_conf`` arrays to fp16 (~2x smaller; ``manifold4d.generate`` casts
them back to fp32 on load):

    <out_root>/
    ├── README.md                                  # dataset card
    ├── <scene>_vggt/                              # per preprocessed scene
    │   ├── frames/ mask_dynamic/ trajectory/
    │   ├── predictions.npz                        # depth+conf as fp16
    │   └── caption.txt text_embedding.pt meta.json subjects.txt
    └── val_traj/<scene>/f{1,2,3}/trajectory_new/trajectory.npz

Typical usage::

    python scripts/pack_eval_dataset.py \
        --data_root data/ours_eval_data \
        --traj_root data/val_traj \
        --out_root  Manifold4D-Eval-pack

then upload::

    huggingface-cli upload manifoldtech/Manifold4D-Eval Manifold4D-Eval-pack . \
        --repo-type dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FP16_KEYS = ("depth", "depth_conf")   # the bulky fp32 arrays (≈ T*H*W each)
SCENE_FILES = ("caption.txt", "text_embedding.pt", "meta.json", "subjects.txt")
SCENE_DIRS = ("frames", "mask_dynamic", "trajectory")
TRAJ_BUCKETS = ("f1", "f2", "f3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack preprocessed scenes + trajectories for HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Root containing <scene>_vggt preprocessing outputs.")
    p.add_argument("--traj_root", type=str, required=True,
                   help="Root containing <scene>/{f1,f2,f3}/trajectory_new.")
    p.add_argument("--out_root", type=str, required=True,
                   help="Output dataset tree (uploaded to HuggingFace as-is).")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="Scene names (default: every *_vggt dir in data_root).")
    p.add_argument("--keep_fp32", action="store_true",
                   help="Skip the fp16 downcast of depth/depth_conf.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pack_predictions(src: Path, dst: Path, keep_fp32: bool) -> tuple[float, float]:
    """Copy predictions.npz, downcasting depth/depth_conf to fp16.

    Returns (src_bytes, dst_bytes).
    """
    if keep_fp32:
        shutil.copyfile(src, dst)
        return src.stat().st_size, dst.stat().st_size
    z = np.load(src)
    arrays = {}
    for k in z.files:
        a = z[k]
        if k in FP16_KEYS and a.dtype == np.float32:
            a = a.astype(np.float16)
        arrays[k] = a
    np.savez_compressed(dst, **arrays)
    return src.stat().st_size, dst.stat().st_size


def copytree(src: Path, dst: Path) -> int:
    shutil.copytree(src, dst)
    total = 0
    for f in dst.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def main() -> None:
    args = parse_args()
    data_root, traj_root, out_root = (
        Path(args.data_root), Path(args.traj_root), Path(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = args.scenes or sorted(
        d.name.removesuffix("_vggt") for d in data_root.glob("*_vggt")
        if d.is_dir())
    print(f"[init] {len(scenes)} scenes: {scenes}")

    grand_src = grand_dst = 0
    for scene in scenes:
        scene_src = data_root / f"{scene}_vggt"
        scene_out = out_root / f"{scene}_vggt"
        if scene_out.exists():
            if not args.overwrite:
                print(f"[skip] {scene} (exists; use --overwrite)")
                continue
            shutil.rmtree(scene_out)
        scene_out.mkdir(parents=True)
        src_bytes = dst_bytes = 0

        # predictions.npz — the only re-encoded file.
        s, d = pack_predictions(scene_src / "predictions.npz",
                                scene_out / "predictions.npz",
                                args.keep_fp32)
        src_bytes += s
        dst_bytes += d

        # Small files + frame/mask/trajectory dirs — verbatim copies.
        for name in SCENE_FILES:
            if (scene_src / name).is_file():
                shutil.copyfile(scene_src / name, scene_out / name)
                src_bytes += (scene_src / name).stat().st_size
                dst_bytes += (scene_out / name).stat().st_size
        for name in SCENE_DIRS:
            if (scene_src / name).is_dir():
                n = copytree(scene_src / name, scene_out / name)
                src_bytes += n
                dst_bytes += n

        # Novel-view trajectories (target cameras only — 24 KB each).
        for bucket in TRAJ_BUCKETS:
            src_npz = traj_root / scene / bucket / "trajectory_new" / "trajectory.npz"
            if not src_npz.is_file():
                print(f"[warn] missing {src_npz}")
                continue
            dst_npz = out_root / "val_traj" / scene / bucket / "trajectory_new" / "trajectory.npz"
            dst_npz.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_npz, dst_npz)
            src_bytes += src_npz.stat().st_size
            dst_bytes += dst_npz.stat().st_size

        grand_src += src_bytes
        grand_dst += dst_bytes
        print(f"[done] {scene:20s} {src_bytes / 1e6:7.1f} MB -> "
              f"{dst_bytes / 1e6:7.1f} MB")

    print(f"[total] {grand_src / 1e6:.1f} MB -> {grand_dst / 1e6:.1f} MB "
          f"({len(scenes)} scenes)")
    print(f"[next] upload with:\n  huggingface-cli upload "
          f"manifoldtech/Manifold4D-Eval {out_root} . --repo-type dataset")


if __name__ == "__main__":
    main()
