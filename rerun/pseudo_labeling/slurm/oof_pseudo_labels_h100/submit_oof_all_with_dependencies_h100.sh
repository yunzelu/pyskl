#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

declare -A FOLD_DEPENDENCIES

FOLD_DEPENDENCIES[a]=""
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t1_h100.sh)"
echo "submitted fold a t1: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[a]}" ]]; then
  FOLD_DEPENDENCIES[a]="${job_id}"
else
  FOLD_DEPENDENCIES[a]="${FOLD_DEPENDENCIES[a]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t2_h100.sh)"
echo "submitted fold a t2: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[a]}" ]]; then
  FOLD_DEPENDENCIES[a]="${job_id}"
else
  FOLD_DEPENDENCIES[a]="${FOLD_DEPENDENCIES[a]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t3_h100.sh)"
echo "submitted fold a t3: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[a]}" ]]; then
  FOLD_DEPENDENCIES[a]="${job_id}"
else
  FOLD_DEPENDENCIES[a]="${FOLD_DEPENDENCIES[a]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_a_t4_h100.sh)"
echo "submitted fold a t4: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[a]}" ]]; then
  FOLD_DEPENDENCIES[a]="${job_id}"
else
  FOLD_DEPENDENCIES[a]="${FOLD_DEPENDENCIES[a]}:${job_id}"
fi
sbatch --dependency=afterok:${FOLD_DEPENDENCIES[a]} rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_a_h100.sh
echo "submitted fold a aggregation after ${FOLD_DEPENDENCIES[a]}"

FOLD_DEPENDENCIES[b]=""
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t1_h100.sh)"
echo "submitted fold b t1: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[b]}" ]]; then
  FOLD_DEPENDENCIES[b]="${job_id}"
else
  FOLD_DEPENDENCIES[b]="${FOLD_DEPENDENCIES[b]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t2_h100.sh)"
echo "submitted fold b t2: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[b]}" ]]; then
  FOLD_DEPENDENCIES[b]="${job_id}"
else
  FOLD_DEPENDENCIES[b]="${FOLD_DEPENDENCIES[b]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t3_h100.sh)"
echo "submitted fold b t3: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[b]}" ]]; then
  FOLD_DEPENDENCIES[b]="${job_id}"
else
  FOLD_DEPENDENCIES[b]="${FOLD_DEPENDENCIES[b]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_b_t4_h100.sh)"
echo "submitted fold b t4: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[b]}" ]]; then
  FOLD_DEPENDENCIES[b]="${job_id}"
else
  FOLD_DEPENDENCIES[b]="${FOLD_DEPENDENCIES[b]}:${job_id}"
fi
sbatch --dependency=afterok:${FOLD_DEPENDENCIES[b]} rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_b_h100.sh
echo "submitted fold b aggregation after ${FOLD_DEPENDENCIES[b]}"

FOLD_DEPENDENCIES[c]=""
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t1_h100.sh)"
echo "submitted fold c t1: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[c]}" ]]; then
  FOLD_DEPENDENCIES[c]="${job_id}"
else
  FOLD_DEPENDENCIES[c]="${FOLD_DEPENDENCIES[c]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t2_h100.sh)"
echo "submitted fold c t2: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[c]}" ]]; then
  FOLD_DEPENDENCIES[c]="${job_id}"
else
  FOLD_DEPENDENCIES[c]="${FOLD_DEPENDENCIES[c]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t3_h100.sh)"
echo "submitted fold c t3: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[c]}" ]]; then
  FOLD_DEPENDENCIES[c]="${job_id}"
else
  FOLD_DEPENDENCIES[c]="${FOLD_DEPENDENCIES[c]}:${job_id}"
fi
job_id="$(sbatch --parsable rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_oof_fold_c_t4_h100.sh)"
echo "submitted fold c t4: ${job_id}"
if [[ -z "${FOLD_DEPENDENCIES[c]}" ]]; then
  FOLD_DEPENDENCIES[c]="${job_id}"
else
  FOLD_DEPENDENCIES[c]="${FOLD_DEPENDENCIES[c]}:${job_id}"
fi
sbatch --dependency=afterok:${FOLD_DEPENDENCIES[c]} rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/run_aggregate_fold_c_h100.sh
echo "submitted fold c aggregation after ${FOLD_DEPENDENCIES[c]}"
