#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."


PYTHON="${PYTHON:-python}"

"${PYTHON}" rerun/pseudo_labeling/aggregate_oof_pseudo_labels.py \
  --folds c
