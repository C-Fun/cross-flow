#!/bin/bash
#SBATCH --job-name=cf_train
#SBATCH --account=biek-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/hdd/biek/fchen16/logs/%x-%j.out
#SBATCH --error=/work/hdd/biek/fchen16/logs/%x-%j.out

# =============================================================================
# Self-resubmitting CrossFlow training chain for NCSA DeltaAI (GH200 x4, aarch64,
# 2-day wall-clock). Same mechanism as scripts/train_chain.sbatch: queue the
# successor with afterany BEFORE training, resume from {WORKDIR}/latest.pt, stop
# once {WORKDIR}/DONE exists, cap restarts.
#
#   sbatch scripts/train_delta.sbatch            (knobs overridable via env)
#   scancel -u $USER --name=cf_train             (stop the whole chain)
# =============================================================================
set -euo pipefail

B=/work/hdd/biek/fchen16
REPO_DIR=${REPO_DIR:-$B/cross-flow}
DATASET=${DATASET:-imagenet_vavae}
DATA_DIR=${DATA_DIR:-/work/nvme/biek/fchen16/dataset/imagenet/train}
LATENTS_PATH=${LATENTS_PATH:-$B/imagenet-latents/latents_vavae.npy}
WORKDIR=${WORKDIR:-$B/runs/crossflow_B_v5_muon}
MODEL=${MODEL:-crossflowDiT_B_2}
# paper schedule: batch 2048 = 4 GPUs x micro 64 x accum 8, 160 epochs, 5 warmup epochs
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-8}
EPOCHS=${EPOCHS:-160}
OPTIMIZER=${OPTIMIZER:-muon}
LR=${LR:-8e-4}
AUX_LR=${AUX_LR:-3e-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
TIME_EPS=${TIME_EPS:-0.05}
GRAD_CLIP=${GRAD_CLIP:-0}
PERC_NET=${PERC_NET:-dinov3}
PERC_WEIGHT=${PERC_WEIGHT:-1.0}
PERC_CLAMP=${PERC_CLAMP:-1}
MAX_RESTARTS=${MAX_RESTARTS:-60}
WANDB_PROJECT=${WANDB_PROJECT:-crossflow}
EXTRA_ARGS=${EXTRA_ARGS:-}

set +u
source $B/miniforge3/etc/profile.d/conda.sh
conda activate crossflow
set -u
export TORCH_HOME=$B/.cache/torch HF_HOME=$B/.cache/hf
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONPATH="${REPO_DIR}"
mkdir -p "$TORCH_HOME" "$HF_HOME" "${WORKDIR}" $B/logs
cd "${REPO_DIR}"

if [ -f "${WORKDIR}/DONE" ]; then
    echo "Found ${WORKDIR}/DONE -- training complete, not resubmitting."; exit 0
fi
COUNT_FILE="${WORKDIR}/restart_count"
COUNT=$(cat "${COUNT_FILE}" 2>/dev/null || echo 0)
if [ "${COUNT}" -ge "${MAX_RESTARTS}" ]; then
    echo "Reached MAX_RESTARTS=${MAX_RESTARTS}; not resubmitting."; exit 1
fi
echo $((COUNT + 1)) > "${COUNT_FILE}"

# queue the successor now so a wall-clock kill still auto-resumes
sbatch --job-name="${SLURM_JOB_NAME}" --dependency=afterany:"${SLURM_JOB_ID}" "${REPO_DIR}/scripts/train_delta.sbatch"

echo "=== Job ${SLURM_JOB_ID} (restart ${COUNT}) on $(hostname) $(date) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

# Stage the latents memmap on node-local NVMe: a multi-GB Lustre mmap inside CUDA
# processes aborts (glibc double free) on GH200. ~42 GB copy, a few minutes.
LOCAL_LATENTS=/tmp/$USER/latents_$(basename "${LATENTS_PATH}")
mkdir -p "$(dirname "$LOCAL_LATENTS")"
if [ ! -f "$LOCAL_LATENTS" ] || [ "$(stat -c %s "$LOCAL_LATENTS")" != "$(stat -c %s "${LATENTS_PATH}")" ]; then
    echo "staging latents to $LOCAL_LATENTS ($(date))"; cp "${LATENTS_PATH}" "$LOCAL_LATENTS"; echo "staged ($(date))"
fi

torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
    --dataset "${DATASET}" --data-dir "${DATA_DIR}" --latents-path "$LOCAL_LATENTS" \
    --workdir "${WORKDIR}" \
    --model "${MODEL}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --epochs "${EPOCHS}" \
    --optimizer "${OPTIMIZER}" \
    --lr "${LR}" \
    --aux-lr "${AUX_LR}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --time-eps "${TIME_EPS}" \
    --grad-clip "${GRAD_CLIP}" \
    --perc-clamp "${PERC_CLAMP}" \
    --weight-decay 1e-6 \
    --p-uncond 0 \
    --pixel-loss l1 \
    --adaptive-p 0 \
    --perc-net "${PERC_NET}" \
    --perc-weight "${PERC_WEIGHT}" \
    --num-workers 12 \
    --ckpt-every 5000 --eval-every 10000 --sample-every 2500 \
    --dtype bf16 \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "$(basename "${WORKDIR}")" \
    ${EXTRA_ARGS}
