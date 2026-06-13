#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8100/v1}"
API_KEY="${API_KEY:-dummy}"
MODEL_NAME="${MODEL_NAME:-sft-full}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-100}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
CONCURRENCY="${CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
WRITE_DETAILS="${WRITE_DETAILS:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

if [[ -z "${DATASET_PATH}" ]]; then
  cat <<'EOF'
Usage:
  DATASET_PATH=/path/to/test.parquet \
  CANDIDATE_SOURCE=/path/to/train_or_all.parquet \
  MODEL_NAME=my-served-model \
  BASE_URL=http://127.0.0.1:7864/v1 \
  bash eval_candidate_acc.sh

This runs candidate-constrained Acc@1. It is closer to ranking-style Acc@1 than
free-form Hit@K, but it is not the original paper scorer.
EOF
  exit 1
fi

ARGS=(
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --model "${MODEL_NAME}"
  --dataset "${DATASET_PATH}"
  --system_prompt "${SYSTEM_PROMPT}"
  --candidate-size "${CANDIDATE_SIZE}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
)

if [[ -n "${CANDIDATE_SOURCE}" ]]; then
  ARGS+=( --candidate-source "${CANDIDATE_SOURCE}" )
fi

if [[ -n "${MAX_EXAMPLES}" ]]; then
  ARGS+=( --max-examples "${MAX_EXAMPLES}" )
fi

if [[ "${WRITE_DETAILS}" == "1" ]]; then
  if [[ -z "${OUTPUT_PATH}" ]]; then
    echo "[error] OUTPUT_PATH is required when WRITE_DETAILS=1" >&2
    exit 1
  fi
  ARGS+=( --output "${OUTPUT_PATH}" )
fi

if [[ "${SHOW_PROGRESS}" == "1" ]]; then
  ARGS+=( --show-progress )
fi

python eval_candidate_acc.py "${ARGS[@]}"
