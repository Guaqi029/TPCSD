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
LOG_ROOT="${LOG_ROOT:-./log/tpcsd}"

FACTOR="${FACTOR:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv"
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv"
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv"

ENCODER_CKPT="${ENCODER_CKPT:-./checkpoints/run_tpcsd_isic2019lt_if${FACTOR}_*/resnet_encoder_latest.pth}"
PROJECTOR_CKPT="${PROJECTOR_CKPT:-}"
PROTOTYPE_CKPT="${PROTOTYPE_CKPT:-./checkpoints/run_tpcsd_isic2019lt_if${FACTOR}_*/prototype_memory_latest.pth}"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

ENCODER_CKPT="$(ls -1 ${ENCODER_CKPT} 2>/dev/null | tail -n 1 || true)"
PROTOTYPE_CKPT="$(ls -1 ${PROTOTYPE_CKPT} 2>/dev/null | tail -n 1 || true)"

[[ -f "${ENCODER_CKPT}" ]] || { echo "Missing encoder_ckpt: ${ENCODER_CKPT}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="run_tpcsd_stage2_isic2019lt_if${FACTOR}_${TIMESTAMP}"

python train_stage2.py \
  --dataset ISIC2019LT \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --run_name "${RUN_NAME}" \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --log_dir "${LOG_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --stage2_batch_size "${STAGE2_BATCH_SIZE}" \
  --workers "${NUM_WORKERS}" \
  --image_size 224 \
  --seed "${SEED}" \
  --backbone resnet50 \
  --use_projector \
  --proj_dim 128 \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --projector_ckpt "${PROJECTOR_CKPT}" \
  --prototype_ckpt "${PROTOTYPE_CKPT}" \
  --merge_real \
  --use_class_weight \
  --virtual_ratio 1.0 \
  --aas_alpha 1.0 \
  --lambda_mu 0.7 \
  --gamma_proto 0.2 \
  --delta_noise 0.01
