#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=12:00:00
#SBATCH --job-name=s6_cont_c
#SBATCH --output=thesis/s6/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

GPUS="${GPUS:-4}"
SEED="${SEED:-42}"
RUN_FOLD="c"
RUN_TEACHERS="${RUN_TEACHERS:-t1 t2 t3 t4}"
RUN_STREAMS="${RUN_STREAMS:-joint}"
CONFIG_DIR="${CONFIG_DIR:-thesis/s6/configs/continuous}"
CONTINUOUS_PKL="${CONTINUOUS_PKL:-data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_teacher4_s6_continuous.pkl}"
S6_VIDEOS_PER_GPU="${S6_VIDEOS_PER_GPU:-8}"
S6_WORKERS_PER_GPU="${S6_WORKERS_PER_GPU:-1}"
S6_CLASS_SAMPLE_STRATEGY="${S6_CLASS_SAMPLE_STRATEGY:-sqrt}"
S6_CLASS_SAMPLE_POWER="${S6_CLASS_SAMPLE_POWER:-0.5}"
S6_EPOCH_SIZE="${S6_EPOCH_SIZE:-}"

if [[ ! -f "${CONTINUOUS_PKL}" ]]; then
  echo "[ERROR] Missing ${CONTINUOUS_PKL}" >&2
  echo "Run: bash thesis/s6/build_continuous_teacher_splits.sh" >&2
  exit 1
fi

GENERATE_ARGS=(
  --output-dir "${CONFIG_DIR}"
  --ann-file "${CONTINUOUS_PKL}"
  --folds "${RUN_FOLD}"
  --teachers ${RUN_TEACHERS}
  --streams ${RUN_STREAMS}
  --videos-per-gpu "${S6_VIDEOS_PER_GPU}"
  --workers-per-gpu "${S6_WORKERS_PER_GPU}"
  --class-sample-strategy "${S6_CLASS_SAMPLE_STRATEGY}"
  --class-sample-power "${S6_CLASS_SAMPLE_POWER}"
  --overwrite
)
if [[ -n "${S6_EPOCH_SIZE}" ]]; then
  GENERATE_ARGS+=(--epoch-size "${S6_EPOCH_SIZE}")
fi
python thesis/s6/generate_continuous_configs.py "${GENERATE_ARGS[@]}"

for teacher in ${RUN_TEACHERS}; do
  for stream in ${RUN_STREAMS}; do
    config="${CONFIG_DIR}/fold_${RUN_FOLD}/${teacher}/${stream}.py"
    echo "[INFO] Fine-tuning S6 continuous teacher fold ${RUN_FOLD} ${teacher} ${stream}"
    bash tools/dist_train.sh "${config}" "${GPUS}" --validate --test-best --seed "${SEED}" --deterministic
  done
done
