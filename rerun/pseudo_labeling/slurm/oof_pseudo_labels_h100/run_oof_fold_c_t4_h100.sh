#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=00:20:00
#SBATCH --job-name=oof_pl_c_t4
#SBATCH --output=rerun/pseudo_labeling/slurm/oof_pseudo_labels_h100/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

source "/project/def-mbolic/yunzelu/pyskl/.venv/bin/activate"
cd "/project/def-mbolic/yunzelu/pyskl"


NUM_PASSES="${NUM_PASSES:-30}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS=()

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

if [[ "${WRITE_CSV_COPY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--write-csv-copy)
fi

if [[ "${SAVE_STREAM_PASS_PROBABILITIES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save-stream-pass-probabilities)
fi

python rerun/pseudo_labeling/run_inner_teacher_oof_pseudo_labeling.py \
  --fold c \
  --teacher t4 \
  --device "${DEVICE}" \
  --num-passes "${NUM_PASSES}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"
