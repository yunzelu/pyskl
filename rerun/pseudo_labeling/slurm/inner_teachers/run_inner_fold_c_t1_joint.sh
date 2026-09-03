#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=00:50:00
#SBATCH --job-name=pl_inner_c_t1_joint
#SBATCH --output=rerun/pseudo_labeling/slurm/inner_teachers/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"

GPUS="${GPUS:-4}"
SEED="${SEED:-42}"
CONFIG="configs/stgcn++/stgcn++_radarv4/rerun/pseudo_labeling/inner_teachers/fold_c/t1/joint.py"

bash tools/dist_train.sh "${CONFIG}" "${GPUS}" --validate --test-best --seed "${SEED}" --deterministic
