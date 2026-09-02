#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=02:00:00
#SBATCH --job-name=e1_val_fold_b_bone
#SBATCH --output=rerun/e1/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

GPUS="${GPUS:-4}"
FOLD="b"
STREAM="bone"
CONFIG_ROOT="configs/stgcn++/stgcn++_radarv4/rerun/e1/fold_${FOLD}/${STREAM}/validation"
WORK_ROOT="work_dirs/rerun/e1/fold_${FOLD}"

find_best_checkpoint() {
  local work_dir="$1"
  mapfile -t ckpts < <(find "${work_dir}" -maxdepth 1 -name 'best_macro_f1_epoch_*.pth' | sort -V)
  if [[ "${#ckpts[@]}" -ne 1 ]]; then
    echo "[ERROR] Expected one best_macro_f1 checkpoint in ${work_dir}, found ${#ckpts[@]}" >&2
    printf '%s\n' "${ckpts[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${ckpts[0]}"
}

run_validation_eval() {
  local condition_key="$1"
  local config_name="$2"
  local checkpoint_condition_dir="$3"
  local result_condition_dir="$4"

  local ckpt_dir="${WORK_ROOT}/${STREAM}/${checkpoint_condition_dir}"
  local ckpt
  ckpt="$(find_best_checkpoint "${ckpt_dir}")"
  local out_dir="${WORK_ROOT}/${STREAM}/${result_condition_dir}/validation"
  mkdir -p "${out_dir}"

  echo "[INFO] fold=${FOLD} stream=${STREAM} condition=${condition_key} checkpoint=${ckpt}"
  bash tools/dist_test.sh "${CONFIG_ROOT}/${config_name}" "${ckpt}" "${GPUS}" \
    --out "${out_dir}/best_pred.pkl" \
    --eval-out "${out_dir}/best_eval.json"
}

run_validation_eval 'a1' 'a1_validation.py' 'a1_activity_aligned' 'a1_activity_aligned'
run_validation_eval 'a2' 'a2_validation.py' 'a1_activity_aligned' 'a2_activity_checkpoint_on_continuous'
run_validation_eval 'b' 'b_validation.py' 'b_continuous_window' 'b_continuous_window'
run_validation_eval 'c' 'c_validation.py' 'c_triangular_temporal_composition' 'c_triangular_temporal_composition'
