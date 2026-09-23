#!/bin/bash
#SBATCH --job-name=cf_precompute
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# One-time SD-VAE latent precomputation for ImageNet (unflipped + flipped).
# Usage:  sbatch scripts/precompute_latents.sh

set -euo pipefail

# ------------------------- EDIT THESE PATHS ----------------------------------
REPO_DIR=$WORK/cross-flow
DATA_DIR=$WORK/dataset/imagenet/train
OUTPUT=$WORK/dataset/imagenet-latents/train/latents.npy
NPROC=4
# -----------------------------------------------------------------------------

# module load rocm/7.2.0
# source /path/to/venv/bin/activate
# MIOpen kernel-tuning cache on $WORK (shared across resumes; avoids re-tuning each job)
export MIOPEN_USER_DB_PATH=$WORK/.miopen
export MIOPEN_CUSTOM_CACHE_DIR=$WORK/.miopen
mkdir -p $WORK/.miopen
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" preprocess_latents.py \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT}" \
    --img-size 256 \
    --batch-size 64
