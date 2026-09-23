#!/bin/bash
#SBATCH --job-name=cf_eval
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# FID/IS evaluation of a CrossFlow checkpoint (uses EMA weights by default).
# Usage:  sbatch scripts/evaluate.sh

set -euo pipefail

# ------------------------- EDIT THESE PATHS ----------------------------------
REPO_DIR=$WORK/cross-flow
WORKDIR=$WORK/cross-flow/runs/crossflow_B_2/eval
CKPT=$WORK/cross-flow/runs/crossflow_B_2/latest.pt
MODEL=crossflowDiT_B_2
NPROC=4
CFG_OMEGA=1.0
# -----------------------------------------------------------------------------

# module load rocm/7.2.0
# source /path/to/venv/bin/activate
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"

torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" evaluate.py evaluate \
    --workdir "${WORKDIR}" \
    --ckpt-path "${CKPT}" \
    --model "${MODEL}" \
    --cfg-omega "${CFG_OMEGA}" \
    --num-images 50000 \
    --gen-bsz 64
