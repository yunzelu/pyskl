#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=pl_prepare_inner_pkls
#SBATCH --output=rerun/pseudo_labeling/slurm/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 python/3.10
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

python rerun/pseudo_labeling/generate_inner_teacher_training_artifacts.py --overwrite
