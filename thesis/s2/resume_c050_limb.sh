#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --job-name=s2_resume_c050_limb
#SBATCH --output=thesis/s2/%x_%j.out
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
S2_RESUME_STREAM="${S2_RESUME_STREAM:-limb}"
S2_RESUME_FOLD="${S2_RESUME_FOLD:-c}"
S2_RESUME_ETA="${S2_RESUME_ETA:-0.50}"
S2_RESUME_ETA_SLUG="${S2_RESUME_ETA_SLUG:-eta050}"
S2_INFER_BATCH_SIZE="${S2_INFER_BATCH_SIZE:-16}"

IFS=' ' read -r -a FOLD_ARGS <<< "${RUN_FOLDS}"

WORK_DIR="work_dirs/thesis/s2/train/C_${S2_RESUME_ETA_SLUG}/fold_${S2_RESUME_FOLD}/${S2_RESUME_STREAM}"
BASE_CONFIG="thesis/s2/configs/fold_${S2_RESUME_FOLD}/${S2_RESUME_STREAM}/posec3d_continuous_soft_C_${S2_RESUME_ETA_SLUG}.py"
RESUME_FROM="${S2_RESUME_FROM:-${WORK_DIR}/epoch_6.pth}"
RESUME_CONFIG="${S2_RESUME_CONFIG:-${WORK_DIR}/posec3d_continuous_soft_C_${S2_RESUME_ETA_SLUG}_resume.py}"
FINAL_CHECKPOINT="${S2_FINAL_CHECKPOINT:-${WORK_DIR}/epoch_8.pth}"

if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "[INFO] ${FINAL_CHECKPOINT} exists; skipping resumed C ${S2_RESUME_STREAM} training."
else
  python thesis/s2/make_resume_config.py \
    --base-config "${BASE_CONFIG}" \
    --resume-from "${RESUME_FROM}" \
    --output "${RESUME_CONFIG}"

  bash tools/dist_train.sh "${RESUME_CONFIG}" "${GPUS}" \
    --validate \
    --seed "${SEED}" \
    --deterministic
fi

python thesis/s2/select_stage2_checkpoint.py \
  --method C \
  --stream "${S2_RESUME_STREAM}" \
  --folds "${FOLD_ARGS[@]}" \
  --etas "${S2_RESUME_ETA}" \
  --overwrite

python thesis/s2/infer.py \
  --method C \
  --stream "${S2_RESUME_STREAM}" \
  --split test \
  --folds "${FOLD_ARGS[@]}" \
  --eta selected \
  --batch-size "${S2_INFER_BATCH_SIZE}" \
  --overwrite

python thesis/s2/evaluate_predictions.py \
  --method C \
  --stream "${S2_RESUME_STREAM}" \
  --overwrite

python thesis/s2/summarize.py \
  --stream "${S2_RESUME_STREAM}" \
  --overwrite
