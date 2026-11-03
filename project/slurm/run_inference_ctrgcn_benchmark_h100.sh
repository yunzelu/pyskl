#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ctrgcn_benchmark_10fps
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

INPUT_DIR="${INPUT_DIR:-data/project/csv/benchmark}"
MODEL_ROOT="${MODEL_ROOT:-work_dirs/ctrgcn/radar_v4_yolo26xpose_clip60_ds3p123}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/ctrgcn/ctrgcn_pyskl_radarv4/ds3p123}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/project/inference/ctrgcn_benchmark_10fps}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
NUM_THREADS="${NUM_THREADS:-4}"

# Process only these two benchmark recordings, not other CSVs in the folder.
# Their paired multiperson masks must also be available to the inference reader.
INPUT_0623="${INPUT_DIR}/pose_2026-06-23_ds_cf.csv"
INPUT_0720="${INPUT_DIR}/pose_2026-07-20_ds_cf.csv"
for input in "${INPUT_0623}" "${INPUT_0720}"; do
  if [[ ! -f "${input}" ]]; then
    echo "Required benchmark input is missing: ${input}" >&2
    exit 1
  fi
done

# Batch size and time limit are conservative starting points; actual GPU memory
# and runtime have not been measured. User CLI arguments can override defaults.
python -u project/infer/infer_ctrgcn_benchmark.py \
  "${INPUT_0623}" "${INPUT_0720}" \
  --model-root "${MODEL_ROOT}" \
  --config-root "${CONFIG_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --batch-size "${BATCH_SIZE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --num-threads "${NUM_THREADS}" \
  "$@"
