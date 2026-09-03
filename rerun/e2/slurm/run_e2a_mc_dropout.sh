#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=e2a_mc_dropout
#SBATCH --output=rerun/e2/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_PASSES="${NUM_PASSES:-10}"
SEED="${SEED:-42}"

python rerun/e2/run_e2a_mc_dropout.py \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-passes "${NUM_PASSES}" \
  --seed "${SEED}" \
  "$@"
