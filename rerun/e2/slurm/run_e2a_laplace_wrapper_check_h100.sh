#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=e2a_laplace_wrap_check
#SBATCH --output=rerun/e2/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"

NUM_CHECK_SAMPLES="${NUM_CHECK_SAMPLES:-128}"

python rerun/e2/check_laplace_logit_wrapper.py \
  --device cuda \
  --num-check-samples "${NUM_CHECK_SAMPLES}" \
  "$@"
