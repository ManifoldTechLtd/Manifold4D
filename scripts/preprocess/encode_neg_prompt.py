"""Encode a CFG negative prompt into a bare T5 embedding tensor.

Wan2.1 was trained with a REAL negative prompt as the CFG
unconditional branch; zero-text is out-of-distribution (the text_embedding
Linear maps 0 → a non-uncond direction) and CFG extrapolation along it
collapses to noise.  This script pre-encodes an arbitrary negative prompt
string with UMT5-XXL and saves it as a bare ``Tensor[L, 4096]`` (bfloat16,
unpadded), matching the format ``manifold4d/generate.py`` loads via
``--neg_prompt_emb`` (see ``configs/neg_prompt_emb.pt``).

The default prompt is Wan's stock negative prompt (wan_shared_cfg.
sample_neg_prompt).  Pass ``--prompt`` to encode a scene-tuned variant, and
select it at inference with ``manifold4d/generate.py --neg_prompt_emb <out>``.

Usage::

    python scripts/preprocess/encode_neg_prompt.py \\
        --out configs/neg_prompt_emb.pt \\
        --prompt "色调艳丽，过曝，细节模糊不清，...，闪烁的运动物体，杂乱的背景"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for wan_root in (os.environ.get("WAN_SRC", ""),
                 "external/Wan2.1"):
    if wan_root and Path(wan_root).is_dir():
        sys.path.insert(0, wan_root)
        break

from wan.modules.t5 import T5EncoderModel  # noqa: E402

# Wan stock negative prompt (wan.configs.shared_config.wan_shared_cfg
# .sample_neg_prompt).  Tuned for creative T2V (penalises 静态/静止 to
# encourage motion); see the ablation note in the module docstring.
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True,
                   help="Output .pt path (bare Tensor[L, 4096] bf16).")
    p.add_argument("--prompt", type=str, default=DEFAULT_NEG_PROMPT,
                   help="Negative prompt string to encode. Defaults to the "
                        "Wan stock sample_neg_prompt.")
    p.add_argument("--prompt_file", type=str, default=None,
                   help="Read the prompt from this UTF-8 text file instead "
                        "of --prompt (overrides --prompt when set).")
    p.add_argument("--t5_checkpoint", type=str,
                   default=os.environ.get(
                       "T5_CHECKPOINT",
                       "checkpoints/wan/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"))
    p.add_argument("--t5_tokenizer", type=str,
                   default=os.environ.get(
                       "T5_TOKENIZER",
                       "checkpoints/wan/Wan2.1-T2V-14B/google/umt5-xxl"))
    p.add_argument("--text_len", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    device = torch.device(args.device)
    print(f"[init] loading T5 encoder from {args.t5_checkpoint}")
    t5 = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=args.t5_checkpoint,
        tokenizer_path=args.t5_tokenizer,
    )
    print("[init] T5 ready")

    with torch.no_grad():
        emb = t5([prompt], device)[0]  # [L, 4096] bf16, unpadded

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb.detach().cpu().to(torch.bfloat16), str(out_path))
    print(f"[done] neg prompt ({emb.shape[0]} tokens) → {out_path}")
    print(f"       prompt: {prompt}")


if __name__ == "__main__":
    main()
