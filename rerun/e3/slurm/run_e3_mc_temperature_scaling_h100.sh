#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --job-name=e3_mc_temp
#SBATCH --output=rerun/e3/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_PASSES="${NUM_PASSES:-30}"
SEED="${SEED:-42}"

# Run as one Python process so the MC-dropout seed is set once and the random
# state advances naturally across folds, splits, and streams.
python rerun/e3/run_e3_mc_temperature_scaling.py "$@" \
  --mode all \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-passes "${NUM_PASSES}" \
  --seed "${SEED}"
