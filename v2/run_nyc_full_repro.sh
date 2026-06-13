#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/LLM4POI}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/comapoilatest/models/Llama-3.1-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-llama3_1}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets/real_nyc_smoke}"
RAW_TRAIN_CSV="${RAW_TRAIN_CSV:-${DATA_ROOT}/preprocessed/train_sample_full.csv}"
RAW_TEST_CSV="${RAW_TEST_CSV:-${DATA_ROOT}/preprocessed/test_sample_with_traj_full.csv}"
PROMPT_DIR="${PROMPT_DIR:-${DATA_ROOT}/llm4poi_v2_full_other_users}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/nyc_sft_full_other_users}"
HISTORY_LIMIT="${HISTORY_LIMIT:-50}"
SIMILAR_TRAJECTORY_LIMIT="${SIMILAR_TRAJECTORY_LIMIT:-20}"
SIMILAR_ENTRY_LIMIT="${SIMILAR_ENTRY_LIMIT:-5}"

cd "${ROOT_DIR}/v2"

python convert_prompt_llm4poi.py \
  --dataset nyc \
  --train_csv "${RAW_TRAIN_CSV}" \
  --test_csv "${RAW_TEST_CSV}" \
  --out_dir "${PROMPT_DIR}" \
  --history_limit "${HISTORY_LIMIT}" \
  --include_other_users \
  --similar_trajectory_limit "${SIMILAR_TRAJECTORY_LIMIT}" \
  --similar_entry_limit "${SIMILAR_ENTRY_LIMIT}" \
  --write_jsonl

MODEL_PATH="${MODEL_PATH}" \
MODEL_TYPE="${MODEL_TYPE}" \
DATASET_PATH="${PROMPT_DIR}/nyc_gsm8k_train_llm4poi.parquet" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
TRAIN_TYPE="${TRAIN_TYPE:-full}" \
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}" \
LEARNING_RATE="${LEARNING_RATE:-2e-5}" \
MAX_LENGTH="${MAX_LENGTH:-32768}" \
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}" \
LOGGING_STRATEGY="${LOGGING_STRATEGY:-epoch}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}" \
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}" \
DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}" \
DISABLE_TQDM="${DISABLE_TQDM:-false}" \
REPORT_TO="${REPORT_TO:-none}" \
QUIET_OUTPUT="${QUIET_OUTPUT:-0}" \
bash sft.sh
