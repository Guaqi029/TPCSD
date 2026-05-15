#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC2019LT/ISIC_2019_Training_Input}"
LT_SPLIT_ROOT="${LT_SPLIT_ROOT:-/data/DataLACP/guyiqin/CODE/TPCSD/split/ISIC2019LT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"

FACTOR="${FACTOR:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
RUN_DIR="${RUN_DIR:-}"

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv"
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv"
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_isic2019lt_if${FACTOR}_* 2>/dev/null | head -n 1 || true)"
fi

[[ -n "${RUN_DIR}" ]] || { echo "Could not resolve latest Stage1 run dir." >&2; exit 1; }
[[ -f "${RUN_DIR}/resnet_encoder_latest.pth" ]] || { echo "Missing encoder checkpoint in ${RUN_DIR}" >&2; exit 1; }
[[ -f "${RUN_DIR}/classifier_latest.pth" ]] || { echo "Missing classifier checkpoint in ${RUN_DIR}" >&2; exit 1; }

python eval_last_stage1.py \
  --dataset ISIC2019LT \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --encoder_ckpt "${RUN_DIR}/resnet_encoder_latest.pth" \
  --classifier_ckpt "${RUN_DIR}/classifier_latest.pth" \
  --backbone resnet18 \
  --image_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${NUM_WORKERS}"
