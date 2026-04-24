#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC_Archive}"
SPLIT_DIR="${SPLIT_DIR:-/data/DataLACP/guyiqin/CODE/TPCSD/split/ISIC_Archive}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-}"

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

PROJECTOR_CKPT=""
if [[ -f "${STAGE1_RUN_DIR}/projector_latest.pth" ]]; then
  PROJECTOR_CKPT="${STAGE1_RUN_DIR}/projector_latest.pth"
fi

python analyze_last_stage2.py \
  --dataset ISIC_Archive \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --encoder_ckpt "${STAGE1_RUN_DIR}/resnet_encoder_latest.pth" \
  --projector_ckpt "${PROJECTOR_CKPT}" \
  --prototype_ckpt "${STAGE1_RUN_DIR}/prototype_memory_latest.pth" \
  --classifier_ckpt "${STAGE2_RUN_DIR}/classifier_latest.pth" \
  --gaussian_mu_ckpt "${STAGE2_RUN_DIR}/gaussian_mu_latest.pth" \
  --gaussian_sigma_ckpt "${STAGE2_RUN_DIR}/gaussian_sigma_latest.pth" \
  --backbone resnet50 \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${NUM_WORKERS}"
