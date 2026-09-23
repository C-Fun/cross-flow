#!/bin/bash
#SBATCH --job-name=crossflow
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# =============================================================================
# Self-resubmitting CrossFlow training chain (24h Slurm limit).
#
# Each job, before training, queues its own successor with an `afterany`
# dependency, so even if this job is killed at the 24h wall-clock limit the
# next job starts automatically and resumes from {WORKDIR}/latest.pt.
# Training writes {WORKDIR}/DONE when --total-steps is reached; every job
# checks for it first and exits without resubmitting once training is complete.
# A restart counter caps the chain to avoid runaway crash loops.
#
# Usage:   sbatch scripts/train_chain.sbatch
# =============================================================================

set -euo pipefail

# ------------------------- EDIT THESE PATHS ----------------------------------
REPO_DIR=$WORK/cross-flow/   # this repo on the cluster
DATA_DIR=$WORK/dataset/imagenet/train/                               # ImageNet train (class subfolders)
LATENTS_PATH=$WORK/dataset/imagenet-latents/train/latents.npy                              # from preprocess_latents.py
WORKDIR=$WORK/cross-flow/runs/crossflow_B_2                            # checkpoints/logs/samples
MODEL=crossflowDiT_B_2
GLOBAL_BATCH_SIZE=256
TOTAL_STEPS=400000
NPROC=4
MAX_RESTARTS=100
WANDB_PROJECT=crossflow
# -----------------------------------------------------------------------------

# --- environment: load ROCm + activate your Python env (EDIT for your cluster) ---
# module load rocm/7.2.0
# source /path/to/venv/bin/activate
# export WANDB_API_KEY=<your key>   # required for ONLINE wandb on compute nodes
# MIOpen kernel-tuning cache on $WORK (shared across resumes; avoids re-tuning each job)
export MIOPEN_USER_DB_PATH=$WORK/.miopen
export MIOPEN_CUSTOM_CACHE_DIR=$WORK/.miopen
mkdir -p $WORK/.miopen
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

mkdir -p "${WORKDIR}"

# Stop cleanly if training already finished.
if [ -f "${WORKDIR}/DONE" ]; then
    echo "Found ${WORKDIR}/DONE -- training complete, not resubmitting."
    exit 0
fi

# Restart-count guard (prevents infinite crash loops).
COUNT_FILE="${WORKDIR}/restart_count"
COUNT=$(cat "${COUNT_FILE}" 2>/dev/null || echo 0)
if [ "${COUNT}" -ge "${MAX_RESTARTS}" ]; then
    echo "Reached MAX_RESTARTS=${MAX_RESTARTS}; not resubmitting. Delete ${COUNT_FILE} to continue."
    exit 1
fi
echo $((COUNT + 1)) > "${COUNT_FILE}"

# Queue the successor now, so a wall-clock kill still triggers auto-resume.
sbatch --dependency=afterany:"${SLURM_JOB_ID}" "${REPO_DIR}/scripts/train_chain.sbatch"

echo "=== Job ${SLURM_JOB_ID} (restart ${COUNT}) on $(hostname) ==="
amd-smi || true

torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" train.py \
    --data-dir "${DATA_DIR}" \
    --latents-path "${LATENTS_PATH}" \
    --workdir "${WORKDIR}" \
    --model "${MODEL}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --total-steps "${TOTAL_STEPS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "$(basename "${WORKDIR}")"
