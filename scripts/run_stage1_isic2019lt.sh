#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common_medclip_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC2019LT/ISIC_2019_Training_Input}" # 数据集图像根目录
LT_SPLIT_ROOT="${LT_SPLIT_ROOT:-/data/DataLACP/guyiqin/CODE/TPCSD/split/ISIC2019LT}" # 训练/验证/测试划分文件目录
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}" # checkpoint 输出根目录
LOG_ROOT="${LOG_ROOT:-./log/tpcsd}" # 日志输出目录

FACTOR="${FACTOR:-100}" # 长尾采样因子，对应 training_if${FACTOR}.csv
SEED="${SEED:-42}" # 随机种子，同时决定 shared_eval_seed${SEED}
BATCH_SIZE="${BATCH_SIZE:-256}" # Stage1 训练 batch size
NUM_WORKERS="${NUM_WORKERS:-4}" # DataLoader worker 数量
EPOCHS="${EPOCHS:-100}" # Stage1 总训练轮数
IMAGE_SIZE="${IMAGE_SIZE:-224}" # 输入图像尺寸
PRETRAINED="${PRETRAINED:-True}" # 是否加载 backbone 预训练权重
RECAL_TAIL_FACTOR="${RECAL_TAIL_FACTOR:-1.0}" # tail 类 prototype recalibration 额外放大倍数
PROTO_SEP_MARGIN="${PROTO_SEP_MARGIN:-0.5}" # prototype 最近邻分离损失的 margin

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv" # 训练集划分文件
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv" # 验证集划分文件
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv" # 测试集划分文件

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")" # 当前运行时间戳
RUN_NAME="run_tpcsd_isic2019lt_if${FACTOR}_${TIMESTAMP}" # 本次 Stage1 运行目录名

PRETRAIN_FLAG="--pretrained"
if [[ "${PRETRAINED}" == "False" || "${PRETRAINED}" == "false" || "${PRETRAINED}" == "0" ]]; then
  PRETRAIN_FLAG="--no-pretrained"
fi

TRAIN_ARGS=(
  --dataset ISIC2019LT # 数据集名称
  --data_path "${DATA_ROOT}" # 图像根目录
  --csv_file_train "${TRAIN_CSV}" # 训练集划分 csv
  --csv_file_val "${VAL_CSV}" # 验证集划分 csv
  --csv_file_test "${TEST_CSV}" # 测试集划分 csv
  --run_name "${RUN_NAME}" # 运行名/输出目录名
  --checkpoints "${CHECKPOINT_ROOT}" # checkpoint 根目录
  --log_dir "${LOG_ROOT}" # 日志目录
  --epochs "${EPOCHS}" # 训练轮数
  --batch_size "${BATCH_SIZE}" # 训练 batch size
  --workers "${NUM_WORKERS}" # DataLoader worker 数
  --image_size "${IMAGE_SIZE}" # 输入图像尺寸
  --seed "${SEED}" # 随机种子
  --backbone medclip_vit # backbone 名称
  "${PRETRAIN_FLAG}" # 是否使用预训练权重
  --use_projector # 开启 projector，将特征映射到原型空间
  --proj_dim 128 # projector 输出维度/原型空间维度
  --cls_loss ce # 分类损失类型
  --pcd_weight 1.0 # PCD 损失权重
  --pcd_temp 0.05 # PCD 温度参数
  --pcd_margin 0.85 # PCD margin
  --spkd_weight 10.0 # SPKD 损失权重
  --punif_weight 1.0 # prototype separation 损失权重（历史命名 punif）
  --proto_sep_margin "${PROTO_SEP_MARGIN}" # prototype 最近邻分离 margin
  --punif_warmup_start_epoch 40 # separation 损失 warmup 起始 epoch
  --punif_warmup_end_epoch 60 # separation 损失 warmup 结束 epoch
  --proto_momentum 0.96 # prototype EMA 更新动量
  --recal_interval 5 # prototype recalibration 的周期
  --recal_alpha 0.15 # recalibration 更新步长
  --recal_tail_factor "${RECAL_TAIL_FACTOR}" # tail 类 recalibration 放大系数
  --queue_size 1024 # student/teacher feature queue 长度
)

python train_stage1.py "${TRAIN_ARGS[@]}"
