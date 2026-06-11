#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8100/v1}"
API_KEY="${API_KEY:-dummy}"
MODEL_NAME="${MODEL_NAME:-sft-full}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
CONCURRENCY="${CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-20}"
K_VALUES="${K_VALUES:-1,5,10,20}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

if [[ -z "${DATASET_PATH}" || -z "${OUTPUT_PATH}" ]]; then
  cat <<'EOF'
Usage:
  DATASET_PATH=/path/to/test.parquet \
  OUTPUT_PATH=/path/to/hitk_predictions.jsonl \
  MODEL_NAME=my-served-model \
  BASE_URL=http://127.0.0.1:7864/v1 \
  bash eval_hitk.sh

Optional overrides:
  API_KEY, SYSTEM_PROMPT, MAX_NEW_TOKENS, CONCURRENCY,
  TEMPERATURE, TOP_P, NUM_RETURN_SEQUENCES, K_VALUES, MAX_EXAMPLES
EOF
  exit 1
fi

ARGS=(
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --model "${MODEL_NAME}"
  --dataset "${DATASET_PATH}"
  --output "${OUTPUT_PATH}"
  --system_prompt "${SYSTEM_PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --num-return-sequences "${NUM_RETURN_SEQUENCES}"
  --k-values "${K_VALUES}"
)

if [[ -n "${MAX_EXAMPLES}" ]]; then
  ARGS+=( --max-examples "${MAX_EXAMPLES}" )
fi

python eval_hitk.py "${ARGS[@]}"
