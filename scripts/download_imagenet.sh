#!/bin/bash
# Prepare ImageNet-1k train split into ImageFolder layout for CrossFlow.
#
# ImageNet is gated: accept the license and authenticate ONCE before running.
#   - HF mode:  https://huggingface.co/datasets/ILSVRC/imagenet-1k  then
#               `huggingface-cli login`  (or `export HF_TOKEN=...`)
#   - tar mode: obtain ILSVRC2012_img_train.tar from image-net.org / your cluster
#
# Usage:  bash scripts/download_imagenet.sh
# Long-running (~140GB); consider running under sbatch or tmux.

set -euo pipefail

# ------------------------- EDIT THESE ----------------------------------------
REPO_DIR=/Volumes/SSK/github_repos/image_gen/crossflow_torch
OUT_DIR=/path/to/imagenet/train        # target ImageFolder dir (data_dir for training)
MODE=tar                               # "tar" (lossless, needs the tar) or "hf" (token)
TRAIN_TAR=/path/to/ILSVRC2012_img_train.tar   # only used when MODE=tar
# -----------------------------------------------------------------------------

# source /path/to/venv/bin/activate
cd "${REPO_DIR}"

if [ "${MODE}" = "hf" ]; then
    pip install -q "datasets>=2.0" huggingface_hub pillow
    # huggingface-cli login   # or: export HF_TOKEN=hf_xxx
    python scripts/download_imagenet.py --mode hf --out "${OUT_DIR}"
else
    python scripts/download_imagenet.py --mode tar --tar "${TRAIN_TAR}" --out "${OUT_DIR}"
fi

echo "Class folders: $(ls "${OUT_DIR}" | wc -l) (expected 1000)"
echo "Next: sbatch scripts/precompute_latents.sh   (set DATA_DIR=${OUT_DIR})"
