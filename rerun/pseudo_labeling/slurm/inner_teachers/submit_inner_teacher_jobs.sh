#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Submit one job per inner-teacher stream.

# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t1_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t1_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t2_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t2_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t3_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t3_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t4_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_b_t4_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t1_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t1_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t2_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t2_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t3_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t3_bone.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t4_joint.sh
# sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_c_t4_bone.sh

sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t1_joint.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t1_bone.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t2_joint.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t2_bone.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t3_joint.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t3_bone.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t4_joint.sh
sbatch rerun/pseudo_labeling/slurm/inner_teachers/run_inner_fold_a_t4_bone.sh