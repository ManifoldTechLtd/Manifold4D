"""Encode per-scene caption text into T5 text embeddings.

Reads one ``caption.txt`` per scene and uses UMT5-XXL to write one
``text_embedding.pt`` per scene. This is the text-encoding stage of the
VGGT → caption → T5 preprocessing pipeline.

Inputs:
    <data_root>/<scene_hash>/caption.txt   (one-line English caption)

Outputs (flat layout, one .pt per scene):
    <text_emb_root>/<scene_hash>.pt
        # bare Tensor[L, 4096] bfloat16, with L capped by --text_len

Resume-safe: scenes whose .pt already exists are skipped (unless
``--overwrite`` is passed).

Loop mode
---------
Pass ``--loop`` to keep the T5 encoder loaded in GPU memory and
periodically rescan the data root for newly-arrived ``caption.txt`` files.
Designed to run **in parallel** with ``caption_vlm.py``: T5
incrementally encodes captions as they appear, so the moment the last
caption is written the embedding directory is also complete.  Stop with
Ctrl+C (graceful, finishes the current batch first).

Usage::

    python scripts/preprocess/encode_text_t5.py \\
        --data_root      data/ours_eval_data \\
        --in_place \\
        --t5_checkpoint  checkpoints/wan/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \\
        --t5_tokenizer   checkpoints/wan/Wan2.1-T2V-14B/google/umt5-xxl \\
        --batch_size 16 \\
        --loop --poll_sec 30
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for wan_root in (os.environ.get("WAN_SRC", ""),
                 "external/Wan2.1"):
    if wan_root and Path(wan_root).is_dir():
        sys.path.insert(0, wan_root)
        break

from wan.modules.t5 import T5EncoderModel  # noqa: E402


def collect_jobs(data_root: Path, text_emb_root: Path,
                 overwrite: bool, in_place: bool,
                 scene_names: list[str] | None) -> list[tuple[Path, str]]:
    """Return [(out_path, caption), ...] for every scene that still needs encoding."""
    jobs: list[tuple[Path, str]] = []
    skipped_existing = 0
    skipped_no_caption = 0
    skipped_empty = 0

    wanted = set(scene_names) if scene_names else None
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir() or (wanted and scene_dir.name not in wanted):
            continue
        cap_file = scene_dir / "caption.txt"
        if not cap_file.exists():
            skipped_no_caption += 1
            continue
        out_path = (scene_dir / "text_embedding.pt" if in_place else
                    text_emb_root / f"{scene_dir.name}.pt")
        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        text = cap_file.read_text(encoding="utf-8").strip()
        if not text:
            skipped_empty += 1
            continue
        jobs.append((out_path, text))

    print(f"[scan] jobs={len(jobs)}  "
          f"skipped_existing={skipped_existing}  "
          f"skipped_no_caption={skipped_no_caption}  "
          f"skipped_empty={skipped_empty}")
    return jobs


STOP = False


def _handle_sigint(signum, frame):
    global STOP
    STOP = True
    print("\n[!] caught SIGINT — will exit after current batch finishes.")


@torch.no_grad()
def _encode_one_pass(jobs: list[tuple[Path, str]], t5,
                     device: torch.device, batch_size: int,
                     pass_tag: str) -> int:
    """Encode a list of jobs in batches. Returns count written."""
    n_total = len(jobs)
    n_done = 0
    t_start = time.time()
    for batch_start in range(0, n_total, batch_size):
        if STOP:
            break
        batch = jobs[batch_start:batch_start + batch_size]
        texts = [j[1] for j in batch]
        embs = t5(texts, device)  # list of [L, 4096] bfloat16
        for (out_path, _text), emb in zip(batch, embs):
            torch.save(emb.detach().cpu().to(torch.bfloat16),
                       str(out_path))
            n_done += 1

        if (batch_start // batch_size) % 20 == 0 or n_done >= n_total:
            dt = time.time() - t_start
            rate = n_done / max(dt, 1e-6)
            eta = (n_total - n_done) / max(rate, 1e-6)
            print(f"[{pass_tag}] {n_done}/{n_total}  "
                  f"({rate:.1f} samples/s, eta {eta/60:.1f} min)")
    return n_done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="DL3DV root; expects <scene>/caption.txt per scene.")
    p.add_argument("--text_emb_root", type=str, default="",
                   help="Output dir; writes <root>/<scene>.pt unless --in_place.")
    p.add_argument("--in_place", action="store_true",
                   help="Write text_embedding.pt inside each scene directory.")
    p.add_argument("--scene_names", nargs="+", default=None,
                   help="Only encode these scene directory names.")

    p.add_argument("--t5_checkpoint", type=str, required=True)
    p.add_argument("--t5_tokenizer", type=str, required=True)
    p.add_argument("--text_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--loop", action="store_true",
                   help="Keep T5 loaded and periodically rescan for new "
                        "caption.txt files. Stop with Ctrl+C.")
    p.add_argument("--poll_sec", type=int, default=30,
                   help="Polling interval (loop mode only).")
    p.add_argument("--idle_passes_to_exit", type=int, default=-1,
                   help="If > 0, auto-exit after this many empty polls. "
                        "Default -1 = poll forever until SIGINT.")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    data_root = Path(args.data_root).resolve()
    text_emb_root = (Path(args.text_emb_root).resolve()
                     if args.text_emb_root else data_root)
    if not args.in_place:
        text_emb_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # First scan: if nothing pending and not looping, exit early before
    # touching the GPU (saves 30s of model load for a no-op invocation).
    jobs = collect_jobs(data_root, text_emb_root, args.overwrite,
                        args.in_place, args.scene_names)
    if not jobs and not args.loop:
        print("[done] nothing to do")
        return

    print(f"[init] loading T5 encoder from {args.t5_checkpoint}")
    t5 = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=args.t5_checkpoint,
        tokenizer_path=args.t5_tokenizer,
    )
    print("[init] T5 ready")

    n_total_done = 0
    pass_idx = 0
    idle_passes = 0
    while not STOP:
        pass_idx += 1
        if pass_idx > 1:
            # Rescan disk for new captions on every subsequent pass.
            jobs = collect_jobs(data_root, text_emb_root, args.overwrite,
                                args.in_place, args.scene_names)

        if jobs:
            n_done = _encode_one_pass(
                jobs, t5, device, args.batch_size, f"encode/pass{pass_idx}")
            n_total_done += n_done
            idle_passes = 0
        else:
            idle_passes += 1
            print(f"[loop] pass {pass_idx}: no new captions  "
                  f"(idle={idle_passes})")

        if not args.loop or STOP:
            break
        if (args.idle_passes_to_exit > 0
                and idle_passes >= args.idle_passes_to_exit):
            print(f"[loop] idle for {idle_passes} polls → exiting")
            break

        for _ in range(args.poll_sec):
            if STOP:
                break
            time.sleep(1)

    print(f"[done] {n_total_done} text embeddings written "
          f"across {pass_idx} pass(es) → {text_emb_root}")


if __name__ == "__main__":
    main()
