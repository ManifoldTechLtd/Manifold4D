# Model weights

This directory holds the model weights used by the Manifold4D inference
pipeline. The weight files themselves are **not committed to git** — run the
download commands below after cloning the repository.

## Expected layout

```
checkpoints/
├── manifold4d/               # our released modules (step 1 below)
│   ├── conditioning_modules.pt
│   ├── camera_encoder.pt
│   └── self_attn_full.pt
├── wan/
│   └── Wan2.1-T2V-14B/       # Wan2.1-T2V-14B base model (step 2 below)
│       ├── config.json
│       ├── diffusion_pytorch_model-0000*-of-00006.safetensors
│       ├── diffusion_pytorch_model.safetensors.index.json
│       ├── Wan2.1_VAE.pth
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       └── google/           # T5 tokenizer files
├── vggt_omega_1b_512.pt      # VGGT-Omega 1B (preprocessing, step 3)
└── Qwen2.5-VL-7B-Instruct/   # Qwen2.5-VL captioner (preprocessing, optional)
```

## 1. Manifold4D modules (our release, ~13 GB)

```bash
bash scripts/download_weights.sh
```

Downloads from [manifoldtech/Manifold4D](https://huggingface.co/manifoldtech/Manifold4D)
into `checkpoints/manifold4d/`. Set `HF_ENDPOINT=https://hf-mirror.com` if
huggingface.co is unreachable.

## 2. Wan2.1-T2V-14B base model (~30 GB)

```bash
bash scripts/download_weights.sh --with-base
```

Or download manually from
[Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) —
place the diffusion shards, `config.json`, `Wan2.1_VAE.pth`,
`models_t5_umt5-xxl-enc-bf16.pth` and the `google/` tokenizer directory
under `checkpoints/wan/Wan2.1-T2V-14B/`.

## 3. Preprocessing weights

| Path | Source |
|---|---|
| `vggt_omega_1b_512.pt` | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) (gated; request access) |
| `Qwen2.5-VL-7B-Instruct/` | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — optional; hand-write `caption.txt` in the scene dir to skip captioning |

SAM3 weights live under `external/sam3/` instead (see the SAM3 repo).

---

All paths above are defaults — every one of them can be overridden via the
`paths:` section of [`configs/manifold4d.yaml`](../configs/manifold4d.yaml)
or environment variables (CLI flag > env var > config).
