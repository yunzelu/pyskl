#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=62G
#SBATCH --time=36:00:00
#SBATCH --job-name=s2_c100
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
RUN_STREAMS="${RUN_STREAMS:-joint limb}"
ETA_VALUE="${ETA_VALUE:-1.0}"
ETA_SLUG="${ETA_SLUG:-eta100}"

S2_VIDEOS_PER_GPU="${S2_VIDEOS_PER_GPU:-8}"
S2_WORKERS_PER_GPU="${S2_WORKERS_PER_GPU:-1}"
S2_CLASS_SAMPLE_STRATEGY="${S2_CLASS_SAMPLE_STRATEGY:-sqrt}"
S2_CLASS_SAMPLE_POWER="${S2_CLASS_SAMPLE_POWER:-0.5}"
S2_EPOCH_SIZE="${S2_EPOCH_SIZE:-}"
S2_INCLUDE_WALK_SESSIONS="${S2_INCLUDE_WALK_SESSIONS:-0}"
S2_MIN_VALID_RATIO="${S2_MIN_VALID_RATIO:-0.5}"

RUN_PREP="${RUN_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER_EVAL="${RUN_INFER_EVAL:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

if [[ "${ETA_VALUE}" != "1.0" && "${ETA_VALUE}" != "1" ]]; then
  echo "[ERROR] This job is intended for pure soft target Method C eta=1.0." >&2
  echo "        ETA_VALUE=${ETA_VALUE} would not be pure temporal soft target training." >&2
  exit 1
fi

if [[ "${RUN_PREP}" == "1" ]]; then
  BUILD_WINDOW_ARGS=(
    --folds ${RUN_FOLDS}
    --min-valid-ratio "${S2_MIN_VALID_RATIO}"
    --etas 0.25 0.50 0.75 1.0
    --overwrite
  )
  if [[ "${S2_INCLUDE_WALK_SESSIONS}" == "1" ]]; then
    BUILD_WINDOW_ARGS+=(--include-walk-sessions)
  else
    BUILD_WINDOW_ARGS+=(--exclude-walk-sessions)
  fi
  python thesis/s2/build_continuous_windows.py "${BUILD_WINDOW_ARGS[@]}"

  GENERATE_CONFIG_ARGS=(
    --folds ${RUN_FOLDS}
    --streams ${RUN_STREAMS}
    --etas "${ETA_VALUE}"
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
    --skip-prediction-checks \
    --overwrite
fi

for stream in ${RUN_STREAMS}; do
  if [[ "${RUN_TRAIN}" == "1" ]]; then
    for fold in ${RUN_FOLDS}; do
      config="thesis/s2/configs/fold_${fold}/${stream}/posec3d_continuous_soft_C_${ETA_SLUG}.py"
      echo "[INFO] Training S2 Method C pure-soft eta=${ETA_VALUE}: fold=${fold} stream=${stream}"
      bash tools/dist_train.sh "${config}" "${GPUS}" --validate --seed "${SEED}" --deterministic
    done
  fi

  if [[ "${RUN_INFER_EVAL}" == "1" ]]; then
    python thesis/s2/select_stage2_checkpoint.py \
      --method C \
      --stream "${stream}" \
      --folds ${RUN_FOLDS} \
      --etas "${ETA_VALUE}" \
      --overwrite

    python thesis/s2/infer.py \
      --method C \
      --stream "${stream}" \
      --split test \
      --folds ${RUN_FOLDS} \
      --eta "${ETA_VALUE}" \
      --overwrite

    python thesis/s2/evaluate_predictions.py \
      --method C \
      --stream "${stream}" \
      --overwrite
  fi
done

if [[ "${RUN_FUSION}" == "1" ]]; then
  python thesis/s2/fuse.py --method C --overwrite
  python thesis/s2/evaluate_predictions.py --method C --stream fusion --overwrite
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  for stream in joint limb fusion; do
    for method in A B; do
      metrics="work_dirs/thesis/s2/eval/metrics_${method}.json"
      if [[ "${stream}" != "joint" ]]; then
        metrics="work_dirs/thesis/s2/eval/metrics_${method}_${stream}.json"
      fi
      if [[ ! -f "${metrics}" ]]; then
        echo "[WARN] Missing ${metrics}; summary for ${stream} needs existing Method ${method} metrics." >&2
      fi
    done
    python thesis/s2/summarize.py --stream "${stream}" --overwrite
  done
fi
