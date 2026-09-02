#!/bin/bash
set -euo pipefail

for fold in a b c; do
  for stream in joint bone; do
    sbatch "rerun/e1/slurm/run_validation_fold_${fold}_${stream}.sh"
  done
done
