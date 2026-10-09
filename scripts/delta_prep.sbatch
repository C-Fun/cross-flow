#!/bin/bash
#SBATCH --job-name=cf_prep
#SBATCH --account=biek-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/biek/fchen16/logs/%x-%j.out
#SBATCH --error=/work/hdd/biek/fchen16/logs/%x-%j.out

# One-shot data prep on DeltaAI: extract the ImageNet train tar (if needed) and
# precompute VA-VAE latents on 4 GPUs (if needed), then submit the training chain.
#   sbatch scripts/delta_prep.sbatch
set -euo pipefail
B=/work/hdd/biek/fchen16
REPO_DIR=${REPO_DIR:-$B/cross-flow}
TAR=${TAR:-$B/dataset/imagenet/ILSVRC2012_img_train.tar}
DATA_DIR=${DATA_DIR:-/work/nvme/biek/fchen16/dataset/imagenet/train}
LATENTS_PATH=${LATENTS_PATH:-$B/imagenet-latents/latents_vavae.npy}
SUBMIT_TRAIN=${SUBMIT_TRAIN:-1}

set +u
source $B/miniforge3/etc/profile.d/conda.sh
conda activate crossflow
set -u
export TORCH_HOME=$B/.cache/torch HF_HOME=$B/.cache/hf PYTHONPATH="${REPO_DIR}"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$(dirname "$LATENTS_PATH")" $B/logs
cd "${REPO_DIR}"

echo "##### [1/2] extract -> ${DATA_DIR}  ($(date))"
if [ -f "${DATA_DIR}/.DONE" ]; then
    echo "already extracted ($(ls ${DATA_DIR} | wc -l) classes)"
else
    python scripts/download_imagenet.py --mode tar --tar "${TAR}" --out "${DATA_DIR}"
    N=$(ls "${DATA_DIR}" | wc -l); echo "classes: ${N}"; [ "${N}" -eq 1000 ] || { echo "EXTRACT FAILED"; exit 1; }
    touch "${DATA_DIR}/.DONE"
fi

echo "##### [2/2] latents -> ${LATENTS_PATH}  ($(date))"
if [ -f "${LATENTS_PATH}.DONE" ]; then
    echo "latents already computed"
else
    # Write the 42 GB memmap on node-local NVMe, then copy to Lustre. On GH200 a
    # multi-GB Lustre mmap inside CUDA processes aborts with a glibc double free
    # (reproduced; local disk is fine), so never memmap large files on /work there.
    LOCAL_TMP=/tmp/$USER/latents; mkdir -p "$LOCAL_TMP"
    torchrun --standalone --nnodes=1 --nproc-per-node=4 preprocess_latents.py \
        --encoder vavae --data-dir "${DATA_DIR}" --output "$LOCAL_TMP/latents.npy" \
        --img-size 256 --batch-size 128 \
        || { echo "PRECOMPUTE FAILED"; exit 1; }
    rm -f "${LATENTS_PATH}"
    cp "$LOCAL_TMP/latents.npy" "${LATENTS_PATH}" && touch "${LATENTS_PATH}.DONE" \
        || { echo "COPY FAILED"; exit 1; }
    rm -rf "$LOCAL_TMP"
fi
python -c "import numpy as np; a=np.load('${LATENTS_PATH}', mmap_mode='r'); print('latents', a.shape, a.dtype)"

if [ "${SUBMIT_TRAIN}" = "1" ]; then
    echo "##### submitting training chain ($(date))"
    sbatch "${REPO_DIR}/scripts/train_delta.sbatch"
fi
echo "PREP_DONE $(date)"
