#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=s6_fuse
#SBATCH --output=thesis/s6/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

RUN_FOLDS="${RUN_FOLDS:-a b c}"
RUN_TEACHERS="${RUN_TEACHERS:-t1 t2 t3 t4}"
FUSION_MODES="${FUSION_MODES:-hard }"
PSEUDO_ROOT="${PSEUDO_ROOT:-work_dirs/thesis/s6/pseudo_labels}"
OUT_ROOT="${OUT_ROOT:-${PSEUDO_ROOT}}"
JOINT_WEIGHT="${JOINT_WEIGHT:-0.5}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_MISSING="${SKIP_MISSING:-0}"

ARGS=(
  --folds ${RUN_FOLDS}
  --teachers ${RUN_TEACHERS}
  --modes ${FUSION_MODES}
  --pseudo-root "${PSEUDO_ROOT}"
  --out-root "${OUT_ROOT}"
  --joint-weight "${JOINT_WEIGHT}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${SKIP_MISSING}" == "1" ]]; then
  ARGS+=(--skip-missing)
fi

python thesis/s6/fuse_pseudo_two_streams.py "${ARGS[@]}"
