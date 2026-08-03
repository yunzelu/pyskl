#!/bin/bash

set -euo pipefail

SOURCE_PKL="${SOURCE_PKL:-data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_continuous.pkl}"
OUTPUT_PKL="${OUTPUT_PKL:-data/radar_v4/pyskl/s2/radarv4_yolo26xpose_clip60_s2_teacher4_s6_continuous.pkl}"
RUN_FOLDS="${RUN_FOLDS:-a b c}"
RUN_TEACHERS="${RUN_TEACHERS:-t1 t2 t3 t4}"

# shellcheck disable=SC2086
python thesis/s6/build_continuous_teacher_splits.py \
  --source-pkl "${SOURCE_PKL}" \
  --output-pkl "${OUTPUT_PKL}" \
  --folds ${RUN_FOLDS} \
  --teachers ${RUN_TEACHERS} \
  --overwrite
