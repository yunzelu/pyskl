#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH --job-name=e1_a1_a2_fold_a_bone
#SBATCH --output=rerun/e1/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

GPUS="${GPUS:-4}"
SEED="${SEED:-42}"
FOLD="a"
STREAM="bone"
CONFIG_ROOT="configs/stgcn++/stgcn++_radarv4/rerun/e1/fold_${FOLD}"
WORK_ROOT="work_dirs/rerun/e1/fold_${FOLD}"

a1_config="${CONFIG_ROOT}/${STREAM}/a1_activity_aligned.py"
b_config="${CONFIG_ROOT}/${STREAM}/b_continuous_window.py"
a1_work_dir="${WORK_ROOT}/${STREAM}/a1_activity_aligned"
a2_eval_dir="${WORK_ROOT}/${STREAM}/a2_activity_checkpoint_on_continuous"

bash tools/dist_train.sh "${a1_config}" "${GPUS}" --validate --test-best --seed "${SEED}" --deterministic

mapfile -t best_ckpts < <(find "${a1_work_dir}" -maxdepth 1 -name 'best_macro_f1_epoch_*.pth' | sort)
if [[ "${#best_ckpts[@]}" -ne 1 ]]; then
  echo "[ERROR] Expected one best_macro_f1 checkpoint in ${a1_work_dir}, found ${#best_ckpts[@]}" >&2
  printf '%s\n' "${best_ckpts[@]}" >&2
  exit 1
fi
best_ckpt="${best_ckpts[0]}"

mkdir -p "${a2_eval_dir}"
bash tools/dist_test.sh "${b_config}" "${best_ckpt}" "${GPUS}" \
  --out "${a2_eval_dir}/best_pred.pkl" \
  --eval-out "${a2_eval_dir}/best_eval.json"
