#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=02:30:00
#SBATCH --job-name=project_10fps_joint_motion
#SBATCH --output=project/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
PROJECT_REPO_ROOT="${PROJECT_REPO_ROOT:-${HOME}/projects/def-mbolic/yunzelu/pyskl}"
source "${PROJECT_REPO_ROOT}/.venv/bin/activate"
cd "${PROJECT_REPO_ROOT}"

GPUS="${GPUS:-4}"
SEED="${SEED:-42}"
CONFIG="project/configs/stgcnpp/fps10_phase0/joint_motion.py"
ANN_FILE="data/project/yolo26xpose/fps10_phase0/pyskl/continuous_window_w20_s4/radarv4_yolo26xpose_continuous_window_w20_s4_val_yunze.pkl"

if [[ ! -f "${CONFIG}" || ! -f "${ANN_FILE}" ]]; then
  echo "[ERROR] Missing project config or dataset: ${CONFIG}, ${ANN_FILE}" >&2
  exit 1
fi

# --test-best writes best_pred.pkl and best_eval.json for yunze validation.
bash tools/dist_train.sh "${CONFIG}" "${GPUS}" \
  --validate --test-best --seed "${SEED}" --deterministic
