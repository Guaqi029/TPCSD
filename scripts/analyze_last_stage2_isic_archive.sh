#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common_medclip_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC_Archive}"
SPLIT_DIR="${SPLIT_DIR:-/data/DataLACP/guyiqin/CODE/TPCSD/split/ISIC_Archive}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-}"
STAGE1_CKPT_TAG="${STAGE1_CKPT_TAG:-best}"
STAGE2_CKPT_TAG="${STAGE2_CKPT_TAG:-best}"

TRAIN_CSV="${SPLIT_DIR}/training.csv"
VAL_CSV="${SPLIT_DIR}/validation.csv"
TEST_CSV="${SPLIT_DIR}/testing.csv"

if [[ -z "${STAGE1_RUN_DIR}" ]]; then
  STAGE1_RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_isic_archive_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${STAGE2_RUN_DIR}" ]]; then
  STAGE2_RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_stage2_isic_archive_* 2>/dev/null | head -n 1 || true)"
fi

[[ -n "${STAGE1_RUN_DIR}" ]] || { echo "Could not resolve latest Stage1 run dir." >&2; exit 1; }
[[ -n "${STAGE2_RUN_DIR}" ]] || { echo "Could not resolve latest Stage2 run dir." >&2; exit 1; }

python analyze_last_stage2.py \
  --dataset ISIC_Archive \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --encoder_ckpt "${STAGE1_RUN_DIR}/resnet_encoder_${STAGE1_CKPT_TAG}.pth" \
  --prototype_ckpt "${STAGE1_RUN_DIR}/prototype_memory_${STAGE1_CKPT_TAG}.pth" \
  --classifier_ckpt "${STAGE2_RUN_DIR}/classifier_${STAGE2_CKPT_TAG}.pth" \
  --gaussian_mu_ckpt "${STAGE2_RUN_DIR}/gaussian_mu_${STAGE2_CKPT_TAG}.pth" \
  --gaussian_sigma_ckpt "${STAGE2_RUN_DIR}/gaussian_sigma_${STAGE2_CKPT_TAG}.pth" \
  --shared_cov_ckpt "${STAGE2_RUN_DIR}/shared_cov_${STAGE2_CKPT_TAG}.pth" \
  --cov_scale_factor 1.0 \
  --backbone medclip_vit \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${NUM_WORKERS}"
