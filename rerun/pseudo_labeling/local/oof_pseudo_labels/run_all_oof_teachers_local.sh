#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

# Runs teacher systems sequentially on the local machine.
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_a_t1_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_a_t2_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_a_t3_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_a_t4_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_b_t1_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_b_t2_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_b_t3_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_b_t4_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_c_t1_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_c_t2_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_c_t3_local.sh
bash rerun/pseudo_labeling/local/oof_pseudo_labels/run_oof_fold_c_t4_local.sh
