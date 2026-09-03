#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=e2a_laplace
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

FIT_BATCH_SIZE="${FIT_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_POSTERIOR_SAMPLES="${NUM_POSTERIOR_SAMPLES:-30}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --fit-batch-size "${FIT_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-posterior-samples "${NUM_POSTERIOR_SAMPLES}"
  --seed "${SEED}"
)

python rerun/e2/run_e2a_laplace.py "$@" --mode stream --folds a --streams joint --device cuda:0 "${COMMON_ARGS[@]}" &
python rerun/e2/run_e2a_laplace.py "$@" --mode stream --folds a --streams bone  --device cuda:1 "${COMMON_ARGS[@]}" &
python rerun/e2/run_e2a_laplace.py "$@" --mode stream --folds b --streams joint --device cuda:2 "${COMMON_ARGS[@]}" &
python rerun/e2/run_e2a_laplace.py "$@" --mode stream --folds b --streams bone  --device cuda:3 "${COMMON_ARGS[@]}" &
wait

python rerun/e2/run_e2a_laplace.py "$@" --mode stream --folds c --streams joint --device cuda:0 "${COMMON_ARGS[@]}" &
python rerun/e2/run_e2a_laplace.py "$@" --mode stream --folds c --streams bone  --device cuda:1 "${COMMON_ARGS[@]}" &
wait

python rerun/e2/run_e2a_laplace.py "$@" \
  --mode fusion \
  --folds a b c \
  --num-posterior-samples "${NUM_POSTERIOR_SAMPLES}"
