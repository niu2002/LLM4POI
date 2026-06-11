#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-./outputs}"
PREVIEW_ONLY="${PREVIEW_ONLY:-1}"

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "[error] target directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

PATTERNS=(
  "logging.jsonl"
  "completions.jsonl"
  "events.out.tfevents.*"
  "*predictions*.jsonl"
  "*eval.jsonl"
  "*.log"
)

echo "[info] target_dir=${TARGET_DIR}"
echo "[info] preview_only=${PREVIEW_ONLY}"
echo "[info] matching files:"

matches=()
for pattern in "${PATTERNS[@]}"; do
  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    matches+=("${path}")
  done < <(find "${TARGET_DIR}" -type f -name "${pattern}" -print)
done

if [[ "${#matches[@]}" -eq 0 ]]; then
  echo "[info] no matching files found"
  exit 0
fi

printf '%s\n' "${matches[@]}" | sort -u | while IFS= read -r file; do
  du -h "${file}"
done

if [[ "${PREVIEW_ONLY}" == "1" ]]; then
  echo "[info] preview only, nothing deleted"
  exit 0
fi

printf '%s\n' "${matches[@]}" | sort -u | while IFS= read -r file; do
  rm -f "${file}"
  echo "[deleted] ${file}"
done

find "${TARGET_DIR}" -type d -empty -delete
echo "[info] cleanup finished"
