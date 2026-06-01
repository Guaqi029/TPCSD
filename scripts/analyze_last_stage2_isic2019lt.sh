#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common_medclip_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC2019LT/ISIC_2019_Training_Input}"
LT_SPLIT_ROOT="${LT_SPLIT_ROOT:-/data/DataLACP/guyiqin/CODE/TPCSD/split/ISIC2019LT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"

FACTOR="${FACTOR:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-}"
# Supported tags: best(=best_val), best_val, best_test, latest.
STAGE1_CKPT_TAG="${STAGE1_CKPT_TAG:-best}"
STAGE2_CKPT_TAG="${STAGE2_CKPT_TAG:-best}"

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv"
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv"
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv"

if [[ -z "${STAGE1_RUN_DIR}" ]]; then
  STAGE1_RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_isic2019lt_if${FACTOR}_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${STAGE2_RUN_DIR}" ]]; then
  STAGE2_RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_stage2_isic2019lt_if${FACTOR}_* 2>/dev/null | head -n 1 || true)"
fi

[[ -n "${STAGE1_RUN_DIR}" ]] || { echo "Could not resolve latest Stage1 run dir." >&2; exit 1; }
[[ -n "${STAGE2_RUN_DIR}" ]] || { echo "Could not resolve latest Stage2 run dir." >&2; exit 1; }

PROJECTOR_CKPT=""
if [[ -f "${STAGE2_RUN_DIR}/projector_${STAGE2_CKPT_TAG}.pth" ]]; then
  PROJECTOR_CKPT="${STAGE2_RUN_DIR}/projector_${STAGE2_CKPT_TAG}.pth"
elif [[ -f "${STAGE1_RUN_DIR}/projector_${STAGE1_CKPT_TAG}.pth" ]]; then
  PROJECTOR_CKPT="${STAGE1_RUN_DIR}/projector_${STAGE1_CKPT_TAG}.pth"
fi

python analyze_last_stage2.py \
  --dataset ISIC2019LT \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --encoder_ckpt "${STAGE1_RUN_DIR}/resnet_encoder_${STAGE1_CKPT_TAG}.pth" \
  --projector_ckpt "${PROJECTOR_CKPT}" \
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
