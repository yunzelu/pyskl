#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=09:00:00
#SBATCH --job-name=s6_trim_b
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
RUN_FOLD="b"
RUN_TEACHERS="${RUN_TEACHERS:-t1 t2 t3 t4}"
RUN_STREAMS="${RUN_STREAMS:-joint}"
CONFIG_DIR="${CONFIG_DIR:-thesis/s6/configs/trimmed}"
S6_VIDEOS_PER_GPU="${S6_VIDEOS_PER_GPU:-32}"
S6_WORKERS_PER_GPU="${S6_WORKERS_PER_GPU:-4}"

python thesis/s6/generate_trimmed_configs.py \
  --output-dir "${CONFIG_DIR}" \
  --folds "${RUN_FOLD}" \
  --teachers ${RUN_TEACHERS} \
  --streams ${RUN_STREAMS} \
  --videos-per-gpu "${S6_VIDEOS_PER_GPU}" \
  --workers-per-gpu "${S6_WORKERS_PER_GPU}" \
  --overwrite

for teacher in ${RUN_TEACHERS}; do
  for stream in ${RUN_STREAMS}; do
    config="${CONFIG_DIR}/fold_${RUN_FOLD}/${teacher}/${stream}.py"
    echo "[INFO] Training S6 trimmed teacher fold ${RUN_FOLD} ${teacher} ${stream}"
    bash tools/dist_train.sh "${config}" "${GPUS}" --validate --test-best --seed "${SEED}" --deterministic
  done
done
