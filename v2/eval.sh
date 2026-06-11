#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8100/v1}"
API_KEY="${API_KEY:-dummy}"
MODEL_NAME="${MODEL_NAME:-sft-lora}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-36}"
CONCURRENCY="${CONCURRENCY:-1}"
TEMPERATURE="${TEMPERATURE:-0.2}"
WRITE_DETAILS="${WRITE_DETAILS:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

if [[ -z "${DATASET_PATH}" ]]; then
  cat <<'EOF'
Usage:
  DATASET_PATH=/path/to/test.jsonl \
  bash eval.sh

Optional overrides:
  BASE_URL, API_KEY, MODEL_NAME, SYSTEM_PROMPT,
  MAX_NEW_TOKENS, CONCURRENCY, TEMPERATURE,
  WRITE_DETAILS=1 OUTPUT_PATH=/path/to/predictions.jsonl
EOF
  exit 1
fi

ARGS=(
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --model "${MODEL_NAME}" \
  --dataset "${DATASET_PATH}" \
  --system_prompt "${SYSTEM_PROMPT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --temperature "${TEMPERATURE}"
)

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

python eval.py "${ARGS[@]}"
