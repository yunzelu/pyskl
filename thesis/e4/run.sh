#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=31G
#SBATCH --time=03:00:00
#SBATCH --job-name=e4_viterbi
#SBATCH --output=thesis/e4/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STREAM="${STREAM:-limb}"
LAMBDA_GRID="${LAMBDA_GRID:-0,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3}"

python thesis/e4/infer_val_logits.py --stream "${STREAM}" --overwrite
python thesis/e4/calibrate_scores.py --stream "${STREAM}" --split val --score-kind raw --overwrite
python thesis/e4/calibrate_scores.py --stream "${STREAM}" --split test --score-kind raw --overwrite
python thesis/e4/calibrate_scores.py --stream "${STREAM}" --split val --score-kind calibrated --overwrite
python thesis/e4/calibrate_scores.py --stream "${STREAM}" --split test --score-kind calibrated --overwrite

python thesis/e4/tune_viterbi.py --stream "${STREAM}" --score-kind raw --lambda-grid "${LAMBDA_GRID}" --overwrite
python thesis/e4/apply_viterbi.py --stream "${STREAM}" --score-kind raw --overwrite
python thesis/e4/tune_viterbi.py --stream "${STREAM}" --score-kind calibrated --lambda-grid "${LAMBDA_GRID}" --overwrite
python thesis/e4/apply_viterbi.py --stream "${STREAM}" --score-kind calibrated --overwrite

python thesis/e4/evaluate.py --stream "${STREAM}" --overwrite
