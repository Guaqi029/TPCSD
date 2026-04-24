#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC_Archive}"
SPLIT_DIR="${SPLIT_DIR:-/data/DataLACP/guyiqin/CODE/TPCSD/split/ISIC_Archive}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
LOG_ROOT="${LOG_ROOT:-./log/tpcsd}"

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PRETRAINED="${PRETRAINED:-True}"

TRAIN_CSV="${SPLIT_DIR}/training.csv"
VAL_CSV="${SPLIT_DIR}/validation.csv"
TEST_CSV="${SPLIT_DIR}/testing.csv"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="run_tpcsd_isic_archive_${TIMESTAMP}"

PRETRAIN_FLAG="--pretrained"
if [[ "${PRETRAINED}" == "False" || "${PRETRAINED}" == "false" || "${PRETRAINED}" == "0" ]]; then
  PRETRAIN_FLAG="--no-pretrained"
fi

python train_stage1.py \
  --dataset ISIC_Archive \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --run_name "${RUN_NAME}" \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --log_dir "${LOG_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${NUM_WORKERS}" \
  --image_size "${IMAGE_SIZE}" \
  --seed "${SEED}" \
  --backbone resnet50 \
  "${PRETRAIN_FLAG}" \
  --use_projector \
  --proj_dim 128 \
  --cls_loss ce \
  --pcd_weight 1.0 \
  --pcd_temp 0.05 \
  --spkd_weight 10.0 \
  --var_weight 0.2 \
  --var_beta 0.5 \
  --proto_momentum 0.96 \
  --recal_interval 5 \
  --recal_alpha 0.15 \
  --tail_alpha \
  --queue_size 1024
