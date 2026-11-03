#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=project_infer_0418_30fps
#SBATCH --output=project/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.10
module load opencv/4.8.1

PROJECT_REPO_ROOT="${PROJECT_REPO_ROOT:-/project/def-mbolic/yunzelu/pyskl}"
source "${PROJECT_REPO_ROOT}/.venv/bin/activate"
cd "${PROJECT_REPO_ROOT}"

# These settings are starting points, not measured memory or runtime estimates.
# A 60-frame window uses more memory than the previous 20-frame windows.
BATCH_SIZE="${BATCH_SIZE:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
NUM_THREADS="${NUM_THREADS:-4}"

# Exactly the April 18 confidence-filtered CSV is selected. The Python reader
# accepts either paired _ds_cf_mm.json or _ds_mm.json when only one exists.
# If both exist, pass --mask-file PATH to select the intended mask explicitly.
# Saved fold C joint and bone configs/checkpoints are selected under model-root.
# No temporal subsampling is applied: use 60 frames, stride 12, 1:1 fusion.
python -u project/infer/infer_csv_30fps.py \
  data/project/csv/310_2026-04-18/pose_2026-04-18_ds_cf.csv \
  --model-root work_dirs/rerun/e1/fold_c \
  --output-dir work_dirs/project/inference/0418_30fps \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --num-threads "${NUM_THREADS}" \
  "$@"
