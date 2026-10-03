#!/bin/bash
# =============================================================================
# Plain launcher for a shared multi-GPU box (no Slurm), e.g. the lab H200 server.
#
#   nohup bash scripts/train_local.sh > ~/cf_train_h200.log 2>&1 &
#
# - Uses the GPUs in $GPUS (default all 8) via torchrun on one node.
# - Restart loop: if training crashes it relaunches and train.py resumes from
#   {WORKDIR}/latest.pt; the loop exits when {WORKDIR}/DONE appears (target
#   steps reached) or when you `touch {WORKDIR}/STOP` (then kill torchrun).
# - Every knob can be overridden from the environment, e.g.
#     GPUS=0,1,2,3 TOTAL_STEPS=100000 bash scripts/train_local.sh
# =============================================================================

set -euo pipefail

# ------------------------- defaults (override via env) -----------------------
REPO_DIR=${REPO_DIR:-$HOME/cross-flow}
CONDA_ENV=${CONDA_ENV:-crossflow}
DATASET=${DATASET:-imagenet_vavae}                                  # imagenet_vavae | imagenet | cifar10 | cifar100
DATA_DIR=${DATA_DIR:-/mnt/data0/datasets/imagenet/train}
LATENTS_PATH=${LATENTS_PATH:-/mnt/data0/fxc293/imagenet-latents/latents_vavae.npy}   # imagenet only
WORKDIR=${WORKDIR:-/mnt/data0/fxc293/runs/crossflow_B_v4_vavae}
MODEL=${MODEL:-crossflowDiT_B_2}
# paper schedule (Table 3): batch 2048, 160 epochs, 5 warmup epochs. 2048 = 8 GPUs x 64 x 4 accum.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EPOCHS=${EPOCHS:-160}
LR=${LR:-3e-4}                                                      # AdamW; paper uses Muon 8e-4
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
PERC_NET=${PERC_NET:-dinov3}
PERC_WEIGHT=${PERC_WEIGHT:-1.0}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MASTER_PORT=${MASTER_PORT:-29517}
MAX_RESTARTS=${MAX_RESTARTS:-50}
WANDB_PROJECT=${WANDB_PROJECT:-crossflow}
EXTRA_ARGS=${EXTRA_ARGS:-}
# -----------------------------------------------------------------------------

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export CUDA_VISIBLE_DEVICES="${GPUS}"
NPROC=$(echo "${GPUS}" | tr ',' '\n' | grep -c .)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/mnt/data0/fxc293/.cache/torch
export HF_HOME=/mnt/data0/fxc293/.cache/hf
export OMP_NUM_THREADS=8
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${TORCH_HOME}" "${WORKDIR}"
cd "${REPO_DIR}"

DATASET_ARGS=(--dataset "${DATASET}" --data-dir "${DATA_DIR}")
if [ "${DATASET}" = "imagenet" ]; then
    DATASET_ARGS+=(--latents-path "${LATENTS_PATH}")
fi

echo "=== $(hostname) $(date) | GPUs=${GPUS} (${NPROC}) | workdir=${WORKDIR} ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

attempt=0
while [ ! -f "${WORKDIR}/DONE" ]; do
    if [ -f "${WORKDIR}/STOP" ]; then
        echo "Found ${WORKDIR}/STOP -- not relaunching."; exit 0
    fi
    attempt=$((attempt + 1))
    if [ "${attempt}" -gt "${MAX_RESTARTS}" ]; then
        echo "Reached MAX_RESTARTS=${MAX_RESTARTS}; giving up."; exit 1
    fi
    echo "=== attempt ${attempt} $(date) ==="
    torchrun --nnodes=1 --nproc-per-node="${NPROC}" \
        --rdzv-backend=c10d --rdzv-endpoint="localhost:${MASTER_PORT}" \
        train.py \
        "${DATASET_ARGS[@]}" \
        --workdir "${WORKDIR}" \
        --model "${MODEL}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --warmup-epochs "${WARMUP_EPOCHS}" \
        --weight-decay 1e-6 \
        --p-uncond 0 \
        --pixel-loss l1 \
        --adaptive-p 0 \
        --perc-net "${PERC_NET}" \
        --perc-weight "${PERC_WEIGHT}" \
        --ckpt-every 5000 --eval-every 10000 --sample-every 2500 \
        --dtype bf16 \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "$(basename "${WORKDIR}")" \
        ${EXTRA_ARGS} || echo "train.py exited with code $? at $(date)"
    [ -f "${WORKDIR}/DONE" ] || sleep 30
done
echo "=== training complete ($(date)) ==="
