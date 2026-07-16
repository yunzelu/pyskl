#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus=a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=31G
#SBATCH --time=02:00:00
#SBATCH --job-name=e2_infer
#SBATCH --output=thesis/e2/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

export CUDA_VISIBLE_DEVICES=0

# python thesis/e2/infer_scores.py --stream joint --overwrite
python thesis/e2/infer_scores.py --stream limb  --overwrite