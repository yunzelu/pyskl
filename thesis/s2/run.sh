#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=12:00:00
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
S2_CLASS_SAMPLE_STRATEGY="${S2_CLASS_SAMPLE_STRATEGY:-sqrt}"
S2_CLASS_SAMPLE_POWER="${S2_CLASS_SAMPLE_POWER:-0.5}"
S2_EPOCH_SIZE="${S2_EPOCH_SIZE:-}"
S2_INCLUDE_WALK_SESSIONS="${S2_INCLUDE_WALK_SESSIONS:-0}"

# python thesis/s2/stage1_report.py --overwrite
BUILD_WINDOW_ARGS=(--folds ${RUN_FOLDS} --overwrite)
if [[ "${S2_INCLUDE_WALK_SESSIONS}" == "1" ]]; then
  BUILD_WINDOW_ARGS+=(--include-walk-sessions)
else
  BUILD_WINDOW_ARGS+=(--exclude-walk-sessions)
fi
python thesis/s2/build_continuous_windows.py "${BUILD_WINDOW_ARGS[@]}"

GENERATE_CONFIG_ARGS=(
  --folds ${RUN_FOLDS}
  --streams ${RUN_STREAMS}
  --videos-per-gpu "${S2_VIDEOS_PER_GPU}"
  --workers-per-gpu "${S2_WORKERS_PER_GPU}"
  --class-sample-strategy "${S2_CLASS_SAMPLE_STRATEGY}"
  --class-sample-power "${S2_CLASS_SAMPLE_POWER}"
  --overwrite
)
if [[ -n "${S2_EPOCH_SIZE}" ]]; then
  GENERATE_CONFIG_ARGS+=(--epoch-size "${S2_EPOCH_SIZE}")
fi
python thesis/s2/generate_configs.py "${GENERATE_CONFIG_ARGS[@]}"

CLASS_SAMPLING_REPORT_ARGS=(
  --folds ${RUN_FOLDS}
  --strategy "${S2_CLASS_SAMPLE_STRATEGY}"
  --power "${S2_CLASS_SAMPLE_POWER}"
  --overwrite
)
if [[ -n "${S2_EPOCH_SIZE}" ]]; then
  CLASS_SAMPLING_REPORT_ARGS+=(--epoch-size "${S2_EPOCH_SIZE}")
fi
python thesis/s2/class_sampling_report.py "${CLASS_SAMPLING_REPORT_ARGS[@]}"
python thesis/s2/sanity_check.py \
  --folds ${RUN_FOLDS} \
  --streams ${RUN_STREAMS} \
  --expected-class-sample-strategy "${S2_CLASS_SAMPLE_STRATEGY}" \
  --expected-class-sample-power "${S2_CLASS_SAMPLE_POWER}" \
  --overwrite

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
