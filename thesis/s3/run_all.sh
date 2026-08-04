#!/bin/bash
set -euo pipefail

RUN_FOLDS="${RUN_FOLDS:-a b c}"
RUN_STREAMS="${RUN_STREAMS:-joint limb}"

for stream in ${RUN_STREAMS}; do
  for fold in ${RUN_FOLDS}; do
    echo "[SUBMIT] S3 fold_${fold} ${stream}"
    sbatch --export=ALL,RUN_FOLD="${fold}",RUN_STREAM="${stream}" thesis/s3/run.sh
  done
done
