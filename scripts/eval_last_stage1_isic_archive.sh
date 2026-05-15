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
RUN_DIR="${RUN_DIR:-}"
CKPT_TAG="${CKPT_TAG:-best}"

TRAIN_CSV="${SPLIT_DIR}/training.csv"
VAL_CSV="${SPLIT_DIR}/validation.csv"
TEST_CSV="${SPLIT_DIR}/testing.csv"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_isic_archive_* 2>/dev/null | head -n 1 || true)"
fi

[[ -n "${RUN_DIR}" ]] || { echo "Could not resolve latest Stage1 run dir." >&2; exit 1; }
[[ -f "${RUN_DIR}/resnet_encoder_${CKPT_TAG}.pth" ]] || { echo "Missing encoder checkpoint in ${RUN_DIR}" >&2; exit 1; }
[[ -f "${RUN_DIR}/classifier_${CKPT_TAG}.pth" ]] || { echo "Missing classifier checkpoint in ${RUN_DIR}" >&2; exit 1; }

python eval_last_stage1.py \
  --dataset ISIC_Archive \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --encoder_ckpt "${RUN_DIR}/resnet_encoder_${CKPT_TAG}.pth" \
  --classifier_ckpt "${RUN_DIR}/classifier_${CKPT_TAG}.pth" \
  --backbone resnet18 \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${NUM_WORKERS}"
