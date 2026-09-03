#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_aggregate_fold_a_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_aggregate_fold_b_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_aggregate_fold_c_local.sh
