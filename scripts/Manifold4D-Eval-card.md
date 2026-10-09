---
license: cc-by-4.0
task_categories:
- image-to-video
tags:
- video-generation
- novel-view-synthesis
- camera-controlled-video
- point-cloud-rendering
size_categories:
- 1GB<n<10GB
---

# Manifold4D-Eval — preprocessed evaluation scenes

Preprocessed evaluation data for **Manifold4D: Denoising on Point Cloud
Rendered Manifolds for Video Re-shooting**
([arXiv:2608.28174](https://arxiv.org/abs/2608.28174), 
[code](https://github.com/ManifoldTechLtd/Manifold4D)).

Each scene ships the exact conditioning inputs consumed by the released
Manifold4D inference pipeline — source frames, VGGT-Omega geometry (depth,
cameras, intrinsics), dynamic masks, captions with pre-encoded T5 text
embeddings, and three novel-view target trajectories. With this data the
full pipeline runs **without** VGGT-Omega, SAM3, or Qwen2.5-VL: no
preprocessing weights or gated checkpoints are needed on the consumer side.

## Dataset layout

```
<scene>_vggt/                          # one preprocessed scene
├── frames/000000.png ...              # source frames (original resolution)
├── predictions.npz                    # VGGT-Omega: depth, depth_conf,
│                                      #   intrinsic, extrinsic (fp16 depth)
├── trajectory/trajectory.npz          # source camera trajectory
│                                      #   (R_world_from_cam, centers, fps)
├── mask_dynamic/000000.png ...        # SAM3 dynamic-object masks
├── caption.txt                        # scene caption (Qwen2.5-VL)
├── text_embedding.pt                  # pre-encoded T5 text embedding
├── meta.json                          # preprocessing metadata
└── subjects.txt                       # SAM3 subject prompts
val_traj/<scene>/
├── f1/trajectory_new/trajectory.npz   # novel-view target trajectory 1
├── f2/trajectory_new/trajectory.npz   # novel-view target trajectory 2
└── f3/trajectory_new/trajectory.npz   # novel-view target trajectory 3
```

## Usage

```bash
git clone https://github.com/ManifoldTechLtd/Manifold4D.git
cd Manifold4D
# ... installation + weights, see the repo README ...

huggingface-cli download manifoldtech/Manifold4D-Eval \
    --repo-type dataset --local-dir data/ours_eval_data

python -m manifold4d.generate \
    --scene_dir  data/ours_eval_data/gold-fish_vggt \
    --trajectory data/ours_eval_data/val_traj/gold-fish/f1/trajectory_new/trajectory.npz \
    --checkpoint checkpoints/manifold4d \
    --output_dir output/gold-fish_f1 \
    --num_steps 20
```

For a zero-download smoke test, the GitHub repository also embeds three
lightweight demo bundles (`assets/demo/`) that omit depth — see
`python -m manifold4d.demo` in the repo README.

## Notes

- `depth` / `depth_conf` are stored as **fp16** (half the download size);
  `manifold4d.generate` casts them back to fp32 on load. Pass
  `--depth_conf_threshold 3.0` to reproduce the paper's confidence gating.
- Scene videos are drawn from the [DAVIS 2017](https://davischallenge.org/)
  dataset; the preprocessed conditioning data in this repository is
  distributed under CC BY 4.0. Refer to the DAVIS website for the terms
  governing the underlying source videos.
- The trajectories in `val_traj/` follow the paper's coverage-filtered
  6-DOF search protocol (`scripts/preprocess/build_val_traj.py`).

## Citation

```bibtex
@article{manifold4d,
  title   = {Manifold4D: Denoising on Point Cloud Rendered Manifolds for Video Re-shooting},
  author  = {Mao, Yongqi and Dai, Zijia and Liu, Zhishuo and Xu, Wei and Wang, Kaiwei and Meng, Guotao},
  journal = {arXiv preprint arXiv:2608.28174},
  year    = {2026}
}
```
