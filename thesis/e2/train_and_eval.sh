#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --job-name=e2_posec3d_8111
#SBATCH --output=thesis/e2/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

GPUS="${GPUS:-4}"
SEED="${SEED:-42}"
RUN_FOLDS="${RUN_FOLDS:-a b c}"
RUN_STREAMS="${RUN_STREAMS:-joint limb}"
CONFIG_DIR="configs/posec3d/slowonly_r50_radarv4/8111"

for fold in ${RUN_FOLDS}; do
  fold_lower="$(tr '[:upper:]' '[:lower:]' <<< "${fold}")"

  for stream in ${RUN_STREAMS}; do
    config="${CONFIG_DIR}/fold_${fold_lower}/${stream}.py"
    if [[ ! -f "${config}" ]]; then
      echo "[ERROR] Missing config: ${config}" >&2
      echo "Generate configs under ${CONFIG_DIR} before submitting this job." >&2
      exit 1
    fi

    echo "[INFO] Training PoseC3D fold ${fold_lower} stream ${stream}"
    bash tools/dist_train.sh "${config}" "${GPUS}" --validate --test-best --seed "${SEED}" --deterministic
  done
done
