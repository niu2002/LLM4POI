#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_DIR="${MODEL_DIR:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
SERVE_NAME="${SERVE_NAME:-sft-lora}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
DTYPE="${DTYPE:-bfloat16}"
ENABLE_LORA="${ENABLE_LORA:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

if [[ -z "${BASE_MODEL_DIR}" ]]; then
  cat <<'EOF'
Usage:
  MODEL_DIR=/path/to/base-model bash serve_vllm.sh

Optional LoRA serving:
  MODEL_DIR=/path/to/base-model \
  ENABLE_LORA=1 \
  ADAPTER_DIR=/path/to/adapter \
  SERVE_NAME=my-model \
  bash serve_vllm.sh
EOF
  exit 1
fi

if [[ "${ENABLE_LORA}" == "1" && -z "${ADAPTER_DIR}" ]]; then
  echo "ADAPTER_DIR is required when ENABLE_LORA=1" >&2
  exit 1
fi

ARGS=(
  --model "${BASE_MODEL_DIR}"
  --served-model-name "${SERVE_NAME}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  ARGS+=( --max-num-seqs "${MAX_NUM_SEQS}" )
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=( --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" )
fi

if [[ "${ENABLE_LORA}" == "1" ]]; then
  ARGS+=( --enable-lora --lora-modules "${SERVE_NAME}=${ADAPTER_DIR}" )
fi

python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
