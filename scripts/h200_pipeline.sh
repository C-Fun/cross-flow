#!/bin/bash
# =============================================================================
# One-shot pipeline for the lab H200 box, meant to run inside tmux:
#   tmux new-session -d -s crossflow -n prep \
#       "bash ~/cross-flow/scripts/h200_pipeline.sh 2>&1 | tee ~/cf_pipeline.log; bash"
#
#   [1/3] python deps into conda env $CONDA_ENV (idempotent)
#   [2/3] frozen-encoder (VA-VAE by default) latent precompute on all GPUs (skipped if $LATENTS_PATH.DONE exists)
#   [3/3] opens tmux window "train" running scripts/train_local.sh
# Every knob is env-overridable; see scripts/train_local.sh for the training ones.
# =============================================================================
set -uo pipefail

REPO_DIR=${REPO_DIR:-$HOME/cross-flow}
CONDA_ENV=${CONDA_ENV:-crossflow}
DATA_DIR=${DATA_DIR:-/mnt/data0/datasets/imagenet/train}
ENCODER=${ENCODER:-vavae}                      # vavae (paper) | sdvae (legacy)
DATASET=${DATASET:-imagenet_vavae}
LATENTS_PATH=${LATENTS_PATH:-/mnt/data0/fxc293/imagenet-latents/latents_${ENCODER}.npy}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TMUX_SESSION=${TMUX_SESSION:-crossflow}
TRAIN_WINDOW=${TRAIN_WINDOW:-train}
TRAIN_LOG=${TRAIN_LOG:-/mnt/data0/fxc293/runs/crossflow_B_v4_vavae_train.log}

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u
export PIP_CONFIG_FILE=/dev/null          # ~/.pip/pip.conf points at an unreachable NGC index
export TORCH_HOME=/mnt/data0/fxc293/.cache/torch
export HF_HOME=/mnt/data0/fxc293/.cache/hf
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "$(dirname "${LATENTS_PATH}")" "$(dirname "${TRAIN_LOG}")"
cd "${REPO_DIR}" && git pull -q --ff-only origin main
export PYTHONPATH="${REPO_DIR}"
NPROC=$(echo "${GPUS}" | tr ',' '\n' | grep -c .)

echo "##### [1/3] python deps  ($(date))"
if ! python -c "import torch, torchvision, diffusers, lpips, wandb, cv2, torch_fidelity, timm" 2>/dev/null; then
    python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
        diffusers opencv-python-headless "torch-fidelity>=0.3.0" lpips timm huggingface_hub wandb pillow numpy requests \
        || { echo "DEPS INSTALL FAILED"; exit 1; }
fi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())" || exit 1

echo "##### [2/3] latents -> ${LATENTS_PATH}  ($(date))"
if [ -f "${LATENTS_PATH}.DONE" ]; then
    echo "latents already computed, skipping"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nnodes=1 --nproc-per-node="${NPROC}" \
        --rdzv-backend=c10d --rdzv-endpoint=localhost:29518 \
        preprocess_latents.py --encoder "${ENCODER}" --data-dir "${DATA_DIR}" --output "${LATENTS_PATH}" \
        --img-size 256 --batch-size 128 \
        && touch "${LATENTS_PATH}.DONE" || { echo "PRECOMPUTE FAILED"; exit 1; }
fi
python -c "import numpy as np; a=np.load('${LATENTS_PATH}', mmap_mode='r'); print('latents', a.shape, a.dtype)"

echo "##### [3/3] launching training in tmux window 'train'  ($(date))"
tmux new-window -t "${TMUX_SESSION}" -n "${TRAIN_WINDOW}" \
    "GPUS=${GPUS} DATASET=${DATASET} LATENTS_PATH=${LATENTS_PATH} bash ${REPO_DIR}/scripts/train_local.sh 2>&1 | tee -a ${TRAIN_LOG}; bash"
echo "done: attach with  tmux attach -t ${TMUX_SESSION}   (windows: prep, ${TRAIN_WINDOW})"
