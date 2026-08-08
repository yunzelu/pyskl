#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=s6_hard
#SBATCH --output=thesis/s6/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

RUN_FOLDS="${RUN_FOLDS:-c}"
RUN_TEACHERS="${RUN_TEACHERS:-t4}"
RUN_STREAMS="${RUN_STREAMS:-limb}"
CONTINUOUS_PKL="${CONTINUOUS_PKL:-data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_teacher4_s6_continuous.pkl}"
OUT_DIR="${OUT_DIR:-work_dirs/thesis/s6/pseudo_labels}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
OVERWRITE="${OVERWRITE:-1}"

ARGS=(
  --folds ${RUN_FOLDS}
  --teachers ${RUN_TEACHERS}
  --streams ${RUN_STREAMS}
  --ann-file "${CONTINUOUS_PKL}"
  --out-dir "${OUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python thesis/s6/pseudo_deterministic_hard.py "${ARGS[@]}"
