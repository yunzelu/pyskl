#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --cpus-per-task=6
#SBATCH --mem=31G
#SBATCH --time=02:00:00
#SBATCH --job-name=s2_rebuild
#SBATCH --output=thesis/s2/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2020 gcc/9.3.0 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

# python thesis/s2/build_continuous_windows.py --min-valid-ratio 0.5 --overwrite
python thesis/s2/generate_configs.py --overwrite
python thesis/s2/class_sampling_report.py --overwrite
python thesis/s2/sanity_check.py --skip-prediction-checks --overwrite

python thesis/s2/stage1_report.py --overwrite