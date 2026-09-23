# External dependencies

Source trees of the external projects Manifold4D depends on. The trees are
**not committed to git** — clone them here after setting up the repository.
Paths can be overridden via environment variables (`WAN_SRC`, `VGGT_ROOT`,
`SAM3_ROOT`).

## Expected layout

```
external/
├── Wan2.1/               # Wan2.1 source tree — `wan.*` modules (base-model code)
│   └── wan/
│       ├── modules/       # attention, t5, vae, model
│       └── utils/         # flow-matching schedulers
├── vggt-omega-main/      # VGGT-Omega source tree (geometry reconstruction)
└── sam3/                 # SAM3 source tree (dynamic-object segmentation)
    └── ckpt/sam3.1/sam3.1_multiplex.pt   # SAM3.1 weights (~per the SAM3 repo)
```

## 1. Wan2.1 (required by generation + T5 encoding)

```bash
git clone https://github.com/Wan-Video/Wan2.1 external/Wan2.1
```

Provides the `wan.*` Python modules (attention kernels, UMT5-XXL text
encoder, VAE class, flow-matching schedulers). No `pip install` needed —
the path is resolved via `WAN_SRC` / `configs/manifold4d.yaml`.

## 2. VGGT-Omega (required by preprocessing)

```bash
git clone https://github.com/facebookresearch/vggt-omega external/vggt-omega-main
```

Reconstructs depth + cameras + intrinsics from the source video. Its
weights go to `checkpoints/vggt_omega_1b_512.pt` (see
[checkpoints/README.md](../checkpoints/README.md)). Override via `VGGT_ROOT`
/ `VGGT_CHECKPOINT`.

## 3. SAM3 (required by preprocessing)

Clone the SAM3 repository to `external/sam3` per its own instructions, then
place the SAM3.1 weights at `external/sam3/ckpt/sam3.1/sam3.1_multiplex.pt`.
Override via `SAM3_ROOT` / `SAM3_CHECKPOINT`.

---

Note: the Qwen2.5-VL captioner and the T5 / Wan2.1-T2V-14B weights are model
weights rather than source trees — they live under `checkpoints/` (see
[checkpoints/README.md](../checkpoints/README.md)).
