#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."


PYTHON="${PYTHON:-python}"
NUM_PASSES="${NUM_PASSES:-30}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_THREADS="${NUM_THREADS:-16}"
DEVICE="${DEVICE:-cpu}"
EXTRA_ARGS=()

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

if [[ "${SKIP_SANITY_CHECK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-sanity-check)
fi

if [[ "${SKIP_CHECKPOINT_HASH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-checkpoint-hash)
fi

if [[ "${SAVE_STREAM_PASS_PROBABILITIES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-stream-pass-probabilities)
fi

"${PYTHON}" rerun/pseudo_labeling/run_inner_teacher_oof_pseudo_labeling.py \
  --fold c \
  --teacher t1 \
  --device "${DEVICE}" \
  --num-passes "${NUM_PASSES}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-threads "${NUM_THREADS}" \
  "${EXTRA_ARGS[@]}"
