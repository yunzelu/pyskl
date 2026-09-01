#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=06:00:00
#SBATCH --job-name=e1_b_fold_c_joint
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
FOLD="c"
STREAM="joint"
CONFIG_ROOT="configs/stgcn++/stgcn++_radarv4/rerun/e1/fold_${FOLD}"
WORK_ROOT="work_dirs/rerun/e1/fold_${FOLD}"

b_config="${CONFIG_ROOT}/${STREAM}/b_continuous_window.py"

bash tools/dist_train.sh "${b_config}" "${GPUS}" --validate --test-best --seed "${SEED}" --deterministic
