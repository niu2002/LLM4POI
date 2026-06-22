#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/workspace/LLM4POI}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/comapoilatest/models/Llama-3.1-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-llama3_1}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets/real_nyc_smoke}"
RAW_TRAIN_CSV="${RAW_TRAIN_CSV:-${DATA_ROOT}/preprocessed/train_sample_full.csv}"
RAW_TEST_CSV="${RAW_TEST_CSV:-${DATA_ROOT}/preprocessed/test_sample_with_traj_full.csv}"
KQT_DIR="${KQT_DIR:-${DATA_ROOT}/kqt_v1}"
PROMPT_DIR="${PROMPT_DIR:-${DATA_ROOT}/llm4poi_v2_kqt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/nyc_sft_kqt}"
HISTORY_LIMIT="${HISTORY_LIMIT:-50}"
SIMILAR_TRAJECTORY_LIMIT="${SIMILAR_TRAJECTORY_LIMIT:-20}"
SIMILAR_ENTRY_LIMIT="${SIMILAR_ENTRY_LIMIT:-5}"

KQT_TOP_K="${KQT_TOP_K:-200}"
KQT_BATCH_SIZE="${KQT_BATCH_SIZE:-4}"
KQT_SEQ_LEN="${KQT_SEQ_LEN:-1024}"
KQT_CONTEXT_SIZE="${KQT_CONTEXT_SIZE:-32768}"
KQT_POOLING="${KQT_POOLING:-mean}"
KQT_DEVICE="${KQT_DEVICE:-cuda}"
KQT_TORCH_DTYPE="${KQT_TORCH_DTYPE:-bfloat16}"
KQT_EXCLUDE_SAME_USER="${KQT_EXCLUDE_SAME_USER:-0}"
SKIP_KQT="${SKIP_KQT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

cd "${ROOT_DIR}"
mkdir -p "${KQT_DIR}" "${PROMPT_DIR}" "${OUTPUT_DIR}"

if [[ "${SKIP_KQT}" != "1" ]]; then
  python preprocessing/traj_qk.py \
    --dataset_name nyc \
    --train_csv "${RAW_TRAIN_CSV}" \
    --test_csv "${RAW_TEST_CSV}" \
    --out_dir "${KQT_DIR}"

  KQT_ARGS=(
    --dataset_name nyc
    --model_path "${MODEL_PATH}"
    --data_path "${KQT_DIR}"
    --top_k "${KQT_TOP_K}"
    --batch_size "${KQT_BATCH_SIZE}"
    --seq_len "${KQT_SEQ_LEN}"
    --context_size "${KQT_CONTEXT_SIZE}"
    --pooling "${KQT_POOLING}"
    --device "${KQT_DEVICE}"
    --torch_dtype "${KQT_TORCH_DTYPE}"
  )
  if [[ "${KQT_EXCLUDE_SAME_USER}" == "1" ]]; then
    KQT_ARGS+=( --exclude_same_user )
  fi
  python traj_sim.py "${KQT_ARGS[@]}"

  python preprocessing/to_nextpoi_kqt.py \
    --dataset_name nyc \
    --train_csv "${RAW_TRAIN_CSV}" \
    --test_csv "${RAW_TEST_CSV}" \
    --out_dir "${KQT_DIR}"
fi

python v2/convert_prompt_llm4poi.py \
  --dataset nyc \
  --train_csv "${RAW_TRAIN_CSV}" \
  --test_csv "${RAW_TEST_CSV}" \
  --out_dir "${PROMPT_DIR}" \
  --history_limit "${HISTORY_LIMIT}" \
  --include_other_users \
  --similar_trajectory_limit "${SIMILAR_TRAJECTORY_LIMIT}" \
  --similar_entry_limit "${SIMILAR_ENTRY_LIMIT}" \
  --kqt_train_json "${KQT_DIR}/train_key_top200.json" \
  --kqt_test_json "${KQT_DIR}/test_key_top200.json" \
  --write_jsonl

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[info] SKIP_TRAIN=1, stop after KQT prompt generation."
  exit 0
fi

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
bash v2/sft.sh
