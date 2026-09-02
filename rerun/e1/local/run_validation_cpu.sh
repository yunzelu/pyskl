#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_THREADS="${NUM_THREADS:-0}"

thread_args=()
if [[ "${NUM_THREADS}" != "0" ]]; then
  thread_args=(--num-threads "${NUM_THREADS}")
fi

"${PYTHON}" rerun/e1/generate_validation_eval_scripts.py --overwrite

"${PYTHON}" rerun/e1/run_validation_inference_cpu.py \
  --folds a b c \
  --streams joint bone \
  --conditions a1 a2 b c \
  --batch-size "${BATCH_SIZE}" \
  "${thread_args[@]}" \
  --overwrite

"${PYTHON}" rerun/e1/summarize_validation_results.py
