#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=s6_ann
#SBATCH --output=thesis/s6/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

RUN_FOLDS="${RUN_FOLDS:-a b c}"
PSEUDO_MODE="${PSEUDO_MODE:-hard}"
SOURCE_PKL="${SOURCE_PKL:-data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_teacher4_s6_continuous.pkl}"
PSEUDO_ROOT="${PSEUDO_ROOT:-work_dirs/thesis/s6/pseudo_labels}"
OUT_DIR="${OUT_DIR:-data/radar_v4/pyskl/s6_pseudo}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_INCOMPLETE_FOLDS="${SKIP_INCOMPLETE_FOLDS:-0}"
INCLUDE_PROBABILITY_PASSES="${INCLUDE_PROBABILITY_PASSES:-0}"

ARGS=(
  --folds ${RUN_FOLDS}
  --mode "${PSEUDO_MODE}"
  --source-pkl "${SOURCE_PKL}"
  --pseudo-root "${PSEUDO_ROOT}"
  --out-dir "${OUT_DIR}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${SKIP_INCOMPLETE_FOLDS}" == "1" ]]; then
  ARGS+=(--skip-incomplete-folds)
fi
if [[ "${INCLUDE_PROBABILITY_PASSES}" == "1" ]]; then
  ARGS+=(--include-probability-passes)
fi

python thesis/s6/build_fused_pseudo_annotations.py "${ARGS[@]}"
