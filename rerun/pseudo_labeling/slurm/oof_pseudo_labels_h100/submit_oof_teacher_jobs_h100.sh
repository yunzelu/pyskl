#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t1_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t2_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t3_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t4_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t1_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t2_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t3_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t4_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t1_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t2_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t3_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t4_h100.sh
