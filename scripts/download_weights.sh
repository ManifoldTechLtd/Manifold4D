#!/usr/bin/env bash
# Download the released Manifold4D inference weights into checkpoints/manifold4d.
#
# The weights are hosted on HuggingFace; point HF_ENDPOINT at a mirror if
# huggingface.co is unreachable, e.g.:
#     export HF_ENDPOINT=https://hf-mirror.com
#
# Optionally also fetch the Wan2.1-T2V-14B base model (~30 GB):
#     bash scripts/download_weights.sh --with-base
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO="${MANIFOLD4D_HF_REPO:-manifoldtech/Manifold4D}"
WAN_REPO="${MANIFOLD4D_WAN_REPO:-Wan-AI/Wan2.1-T2V-14B}"
DEST="${MANIFOLD4D_CKPT_DIR:-$REPO_ROOT/checkpoints/manifold4d}"
WAN_DEST="${MANIFOLD4D_WAN_DIR:-$REPO_ROOT/checkpoints/wan/Wan2.1-T2V-14B}"

WITH_BASE=0
for arg in "$@"; do
    case "$arg" in
        --with-base) WITH_BASE=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

FILES=(
    "self_attn_full.pt"
    "conditioning_modules.pt"
    "camera_encoder.pt"
)

if command -v huggingface-cli >/dev/null 2>&1; then
    DOWNLOADER="huggingface-cli"
elif command -v hf >/dev/null 2>&1; then
    DOWNLOADER="hf"
else
    echo "Neither huggingface-cli nor hf found. Install with:"
    echo "    pip install -U huggingface_hub[cli]"
    exit 1
fi

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
    echo "Downloading $f ..."
    $DOWNLOADER download "$HF_REPO" "$f" --local-dir "$DEST"
done
echo "Manifold4D weights saved to $DEST"

if [ "$WITH_BASE" = "1" ]; then
    echo "Downloading the Wan2.1-T2V-14B base model (~30 GB) ..."
    mkdir -p "$WAN_DEST"
    $DOWNLOADER download "$WAN_REPO" --local-dir "$WAN_DEST"
    echo "Wan2.1-T2V-14B base model saved to $WAN_DEST"
fi

echo "Pass --checkpoint $DEST to manifold4d/generate.py (see configs/manifold4d.yaml for all other paths)."
