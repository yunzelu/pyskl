#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=oof_aggr_c
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


EXTRA_ARGS=()
if [[ "${WRITE_CSV_COPY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--write-csv-copy)
fi

python rerun/pseudo_labeling/aggregate_oof_pseudo_labels.py \
  --folds c \
  "${EXTRA_ARGS[@]}"
