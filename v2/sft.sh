#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
MODEL_TYPE="${MODEL_TYPE:-}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QUIET_OUTPUT="${QUIET_OUTPUT:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_TYPE="${TRAIN_TYPE:-full}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
SAVE_STEPS="${SAVE_STEPS:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
LOGGING_STRATEGY="${LOGGING_STRATEGY:-epoch}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
DISABLE_TQDM="${DISABLE_TQDM:-true}"
REPORT_TO="${REPORT_TO:-none}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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

High-memory defaults:
  MAX_LENGTH defaults to 32768 for AMD remote runs.
  PER_DEVICE_TRAIN_BATCH_SIZE defaults to the original v2 value 16.
  Increase PER_DEVICE_TRAIN_BATCH_SIZE or MAX_LENGTH if memory remains unused.
EOF
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}" -m swift.cli.sft
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
  --save_strategy "${SAVE_STRATEGY}" \
  --logging_strategy "${LOGGING_STRATEGY}" \
  --max_length "${MAX_LENGTH}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_only_model true \
  --disable_tqdm "${DISABLE_TQDM}" \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --use_liger_kernel "${USE_LIGER_KERNEL}" \
  --attn_impl "${ATTN_IMPL}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

printf '[run] model=%s dataset=%s output=%s\n' "${MODEL_PATH}" "${DATASET_PATH}" "${OUTPUT_DIR}"
printf '[run] epochs=%s batch=%s grad_acc=%s max_length=%s train_type=%s\n' \
  "${NUM_TRAIN_EPOCHS}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${MAX_LENGTH}" "${TRAIN_TYPE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run] command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${QUIET_OUTPUT}" == "1" ]]; then
  "${CMD[@]}" 2>&1 | awk '
    /Traceback \(most recent call last\):/ { traceback=1 }
    traceback { print; fflush(); next }
    /(^|[^[:alpha:]])([Ee]rror|ERROR|Exception|FAILED|OOM|out of memory)([^[:alpha:]]|$)/ {
      print; fflush(); next
    }
    /(^|[^[:alpha:]])(loss|eval_loss|train_loss|train_runtime|epoch|global_step|max_memory|last_model_checkpoint|best_model_checkpoint)([^[:alpha:]]|$)/ {
      print; fflush()
    }
  '
else
  "${CMD[@]}"
fi
