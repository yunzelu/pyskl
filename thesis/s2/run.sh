#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=08:00:00
#SBATCH --job-name=s2_posec3d
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
RUN_FOLDS="${RUN_FOLDS:-a}"
RUN_STREAMS="${RUN_STREAMS:-joint}"
RUN_METHODS="${RUN_METHODS:-A B C050}"
RUN_EXTRA_C_ETAS="${RUN_EXTRA_C_ETAS:-0}"
S2_VIDEOS_PER_GPU="${S2_VIDEOS_PER_GPU:-8}"
S2_WORKERS_PER_GPU="${S2_WORKERS_PER_GPU:-1}"

# python thesis/s2/stage1_report.py --overwrite
python thesis/s2/build_continuous_windows.py --folds ${RUN_FOLDS} --overwrite
python thesis/s2/generate_configs.py \
  --folds ${RUN_FOLDS} \
  --streams ${RUN_STREAMS} \
  --videos-per-gpu "${S2_VIDEOS_PER_GPU}" \
  --workers-per-gpu "${S2_WORKERS_PER_GPU}" \
  --overwrite
python thesis/s2/sanity_check.py --overwrite

for stream in ${RUN_STREAMS}; do
  for method in ${RUN_METHODS}; do
    if [[ "${method}" == "A" ]]; then
      python thesis/s2/infer.py --method A --stream "${stream}" --split test --folds ${RUN_FOLDS} --overwrite
      python thesis/s2/evaluate_predictions.py --method A --stream "${stream}" --overwrite
      continue
    fi

    if [[ "${method}" == "B" ]]; then
      for fold in ${RUN_FOLDS}; do
        config="thesis/s2/configs/fold_${fold}/${stream}/posec3d_continuous_hard_B.py"
        bash tools/dist_train.sh "${config}" "${GPUS}" --validate --seed "${SEED}" --deterministic
      done
      python thesis/s2/select_stage2_checkpoint.py --method B --stream "${stream}" --folds ${RUN_FOLDS} --overwrite
      python thesis/s2/infer.py --method B --stream "${stream}" --split test --folds ${RUN_FOLDS} --overwrite
      python thesis/s2/evaluate_predictions.py --method B --stream "${stream}" --overwrite
      continue
    fi

    if [[ "${method}" == "C050" ]]; then
      for fold in ${RUN_FOLDS}; do
        config="thesis/s2/configs/fold_${fold}/${stream}/posec3d_continuous_soft_C_eta050.py"
        bash tools/dist_train.sh "${config}" "${GPUS}" --validate --seed "${SEED}" --deterministic
      done
      python thesis/s2/select_stage2_checkpoint.py --method C --stream "${stream}" --folds ${RUN_FOLDS} --etas 0.50 --overwrite
      python thesis/s2/infer.py --method C --stream "${stream}" --split test --folds ${RUN_FOLDS} --eta selected --overwrite
      python thesis/s2/evaluate_predictions.py --method C --stream "${stream}" --overwrite
      continue
    fi

    echo "[ERROR] Unknown RUN_METHODS entry: ${method}" >&2
    exit 1
  done

  if [[ "${RUN_EXTRA_C_ETAS}" == "1" ]]; then
    for eta in 025 075; do
      for fold in ${RUN_FOLDS}; do
        config="thesis/s2/configs/fold_${fold}/${stream}/posec3d_continuous_soft_C_eta${eta}.py"
        bash tools/dist_train.sh "${config}" "${GPUS}" --validate --seed "${SEED}" --deterministic
      done
    done
    python thesis/s2/select_stage2_checkpoint.py --method C --stream "${stream}" --folds ${RUN_FOLDS} --etas 0.25 0.50 0.75 --overwrite
    python thesis/s2/infer.py --method C --stream "${stream}" --split test --folds ${RUN_FOLDS} --eta selected --overwrite
    python thesis/s2/evaluate_predictions.py --method C --stream "${stream}" --overwrite
  fi

  python thesis/s2/summarize.py --stream "${stream}" --overwrite
done
