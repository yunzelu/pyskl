#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=project_infer_0903_10fps
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

BATCH_SIZE="${BATCH_SIZE:-128}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
NUM_THREADS="${NUM_THREADS:-4}"

# Process every *_cf.csv in this directory with its paired *_mm.json mask.
# All four streams run in one process on the allocated H100 GPU slice.
python -u project/infer/infer_csv.py data/project/csv \
  --model-root work_dirs/project/stgcnpp/fps10_phase0 \
  --config-root project/configs/stgcnpp/fps10_phase0 \
  --output-dir work_dirs/project/inference/0903_10fps \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --num-threads "${NUM_THREADS}" \
  "$@"
