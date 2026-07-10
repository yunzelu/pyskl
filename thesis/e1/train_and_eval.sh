#!/bin/bash
#SBATCH --account=def-mbolic
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=12         
#SBATCH --mem=62G                  
#SBATCH --time=04:00:00  
#SBATCH --job-name=e1_train_and_eval         
#SBATCH --output=thesis/e1/%x_%j.out
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load StdEnv/2020 gcc/9.3.0 cuda/11.8 python/3.10 opencv/4.5.5
source ~/projects/def-mbolic/yunzelu/pyskl/.venv/bin/activate
cd ~/projects/def-mbolic/yunzelu/pyskl/

# export CUDA_VISIBLE_DEVICES=0

bash tools/dist_train.sh configs/posec3d/slowonly_r50_radarv4/911/chenzhe/joint.py 4 --validate --test-best --seed 42 --deterministic