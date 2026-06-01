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
BATCH_SIZE="${BATCH_SIZE:-256}" # 特征提取阶段 batch size
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2048}" # Stage2 分类器训练/投影时的特征 batch size
NUM_WORKERS="${NUM_WORKERS:-8}" # DataLoader worker 数量
EPOCHS="${EPOCHS:-100}" # Stage2 总训练轮数
TRAIN_NOISE_STD="${TRAIN_NOISE_STD:-0.01}" # 训练分类器时加入特征噪声的标准差
VIRTUAL_LOSS_WEIGHT="${VIRTUAL_LOSS_WEIGHT:-1.0}" # 虚拟特征分类损失权重
HARDEST_K="${HARDEST_K:-3}" # 每轮优先生成虚拟特征的最难类别数
HARDEST_FRACTION="${HARDEST_FRACTION:-0.5}" # 若 hardest_k=0，则按比例选择最难类别
VIRTUAL_CONF_THRESH="${VIRTUAL_CONF_THRESH:-0.6}" # 保留虚拟特征所需的分类置信度阈值
VIRTUAL_CENTER_COS_THRESH="${VIRTUAL_CENTER_COS_THRESH:-0.2}" # 保留虚拟特征所需的类中心余弦阈值
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-}" # 指定 Stage1 run 目录；留空则自动取最新
# Stage1 checkpoint tag to bootstrap Stage2.
# Supported tags now include: best(=best_val), best_val, best_test, latest.
STAGE1_CKPT_TAG="${STAGE1_CKPT_TAG:-${CKPT_TAG:-best}}" # 选择哪套 Stage1 模型给 Stage2 使用

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv" # 训练集划分文件
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv" # 验证集划分文件
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv" # 测试集划分文件

if [[ -z "${STAGE1_RUN_DIR}" ]]; then
  STAGE1_RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_isic2019lt_if${FACTOR}_* 2>/dev/null | head -n 1 || true)"
fi

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

[[ -n "${STAGE1_RUN_DIR}" ]] || { echo "Could not resolve latest Stage1 run dir." >&2; exit 1; }

ENCODER_CKPT="${STAGE1_RUN_DIR}/resnet_encoder_${STAGE1_CKPT_TAG}.pth" # Stage1 encoder checkpoint
PROTOTYPE_CKPT="${STAGE1_RUN_DIR}/prototype_memory_${STAGE1_CKPT_TAG}.pth" # Stage1 prototype checkpoint
PROJECTOR_CKPT="${STAGE1_RUN_DIR}/projector_${STAGE1_CKPT_TAG}.pth" # Stage1 projector checkpoint

[[ -f "${ENCODER_CKPT}" ]] || { echo "Missing encoder_ckpt: ${ENCODER_CKPT}" >&2; exit 1; }
[[ -f "${PROJECTOR_CKPT}" ]] || { echo "Missing projector_ckpt: ${PROJECTOR_CKPT}" >&2; exit 1; }
[[ -f "${PROTOTYPE_CKPT}" ]] || { echo "Missing prototype_ckpt: ${PROTOTYPE_CKPT}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")" # 当前运行时间戳
RUN_NAME="run_tpcsd_stage2_isic2019lt_if${FACTOR}_${TIMESTAMP}" # 本次 Stage2 运行目录名

STAGE2_ARGS=(
  --dataset ISIC2019LT # 数据集名称
  --data_path "${DATA_ROOT}" # 图像根目录
  --csv_file_train "${TRAIN_CSV}" # 训练集划分 csv
  --csv_file_val "${VAL_CSV}" # 验证集划分 csv
  --csv_file_test "${TEST_CSV}" # 测试集划分 csv
  --run_name "${RUN_NAME}" # 运行名/输出目录名
  --checkpoints "${CHECKPOINT_ROOT}" # checkpoint 根目录
  --log_dir "${LOG_ROOT}" # 日志目录
  --epochs "${EPOCHS}" # Stage2 训练轮数
  --batch_size "${BATCH_SIZE}" # 抽取 train/val/test 特征时的 batch size
  --stage2_batch_size "${STAGE2_BATCH_SIZE}" # Stage2 分类器训练使用的特征 batch size
  --workers "${NUM_WORKERS}" # DataLoader worker 数
  --image_size 224 # 输入图像尺寸
  --seed "${SEED}" # 随机种子
  --backbone medclip_vit # backbone 名称
  --use_projector # 使用 Stage1 projector 将特征映射到原型空间
  --proj_dim 128 # projector 输出维度/原型空间维度
  --encoder_ckpt "${ENCODER_CKPT}" # Stage1 encoder checkpoint 路径
  --projector_ckpt "${PROJECTOR_CKPT}" # Stage1 projector checkpoint 路径
  --prototype_ckpt "${PROTOTYPE_CKPT}" # Stage1 prototype checkpoint 路径
  --merge_real # 训练 Stage2 时保留真实特征参与训练
  --virtual_ratio 1.0 # 虚拟特征总量相对真实样本数的比例
  --aas_alpha 2.0 # 难类分配强度参数
  --lambda_mu 0.5 # 类均值 mu 与 prototype 融合权重
  --cov_scale_factor 1.0 # 共享协方差缩放系数
  --cosine_scale 16.0 # cosine classifier 的缩放系数
  --lr 1e-3 # Stage2 分类器学习率
  --delta_noise 0.01 # 虚拟特征采样时的附加噪声强度
  --train_noise_std "${TRAIN_NOISE_STD}" # 训练分类器时对特征加噪声的标准差
  --hardest_k "${HARDEST_K}" # 每轮优先增强的最难类别数量
  --hardest_fraction "${HARDEST_FRACTION}" # hardest_k=0 时按比例选择最难类别
  --virtual_conf_thresh "${VIRTUAL_CONF_THRESH}" # 虚拟特征筛选的分类置信度阈值
  --virtual_center_cos_thresh "${VIRTUAL_CENTER_COS_THRESH}" # 虚拟特征筛选的类中心相似度阈值
  --virtual_loss_weight "${VIRTUAL_LOSS_WEIGHT}" # 虚拟样本损失权重
)

python train_stage2.py "${STAGE2_ARGS[@]}"
