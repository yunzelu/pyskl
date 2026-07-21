#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=s3_posec3d_uncert
#SBATCH --output=thesis/s3/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

RUN_FOLD="${RUN_FOLD:-a}"
RUN_STREAM="${RUN_STREAM:-joint}"
S3_NUM_PASSES="${S3_NUM_PASSES:-10}"
S3_ECE_BINS="${S3_ECE_BINS:-15}"
S3_SEED="${S3_SEED:-42}"
S3_BATCH_SIZE="${S3_BATCH_SIZE:-8}"
S3_NUM_WORKERS="${S3_NUM_WORKERS:-1}"
S3_DEVICE="${S3_DEVICE:-cuda:0}"
S2_ANN="${S2_ANN:-data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_continuous.pkl}"
S3_OUT_DIR="${S3_OUT_DIR:-work_dirs/thesis/s3/B_mc_dropout/fold_${RUN_FOLD}/${RUN_STREAM}}"

CONFIG="thesis/s2/configs/fold_${RUN_FOLD}/${RUN_STREAM}/posec3d_continuous_hard_B.py"
SELECTION="work_dirs/thesis/s2/selection/selected_B_${RUN_STREAM}.json"

CHECKPOINT="${B_CHECKPOINT:-$(python - "${SELECTION}" "${RUN_FOLD}" <<'PY'
import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
fold = sys.argv[2].lower().replace("fold_", "")
data = json.loads(selection_path.read_text(encoding="utf-8"))
for record in data.get("records", []):
    if str(record.get("fold", "")).lower().replace("fold_", "") == fold:
        print(record["checkpoint"])
        raise SystemExit(0)
raise SystemExit(f"No selected B checkpoint for fold {fold!r} in {selection_path}")
PY
)}"

python tools/uncertainty/run_posec3d_mc_dropout.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --calibration-ann "${S2_ANN}" \
  --test-ann "${S2_ANN}" \
  --out-dir "${S3_OUT_DIR}" \
  --num-passes "${S3_NUM_PASSES}" \
  --ece-bins "${S3_ECE_BINS}" \
  --seed "${S3_SEED}" \
  --device "${S3_DEVICE}" \
  --batch-size "${S3_BATCH_SIZE}" \
  --num-workers "${S3_NUM_WORKERS}" \
  --overwrite
