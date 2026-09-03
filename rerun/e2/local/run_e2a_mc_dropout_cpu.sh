#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_THREADS="${NUM_THREADS:-16}"

"${PYTHON}" rerun/e2/run_e2a_mc_dropout.py \
  --device cpu \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-threads "${NUM_THREADS}" \
  "$@"
