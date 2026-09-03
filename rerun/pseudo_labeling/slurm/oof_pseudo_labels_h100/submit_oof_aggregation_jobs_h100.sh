#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

# Submit after the corresponding teacher-system jobs have completed.
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_a_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_b_h100.sh
sbatch rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_c_h100.sh
