#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_TYPE="${TRAIN_TYPE:-full}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-12024}"
SAVE_STEPS="${SAVE_STEPS:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"

if [[ -z "${MODEL_PATH}" || -z "${MODEL_TYPE}" || -z "${DATASET_PATH}" || -z "${OUTPUT_DIR}" ]]; then
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/base-model \
  MODEL_TYPE=qwen2_5 \
  DATASET_PATH=/path/to/train.jsonl \
  OUTPUT_DIR=/path/to/output \
  bash sft.sh

Required environment variables:
  MODEL_PATH   Base model path or model id.
  MODEL_TYPE   ms-swift model type.
  DATASET_PATH Training dataset path accepted by swift.
  OUTPUT_DIR   Output checkpoint directory.
EOF
  exit 1
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" -m swift.cli.sft \
  --model "${MODEL_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --train_type "${TRAIN_TYPE}" \
  --target_modules "${TARGET_MODULES}" \
  --dataset "${DATASET_PATH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --streaming false \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --packing false \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_only_model true \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --use_liger_kernel "${USE_LIGER_KERNEL}" \
  --attn_impl "${ATTN_IMPL}"
