#!/usr/bin/env bash

#SBATCH -A mlg-core
#SBATCH -p volta
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH --output=Logs/job_data/test_gpu_%A.out
#SBATCH --error=Logs/job_data/test_gpu_%A.err
#SBATCH --job-name=ml-xas-qm9

module load gcc/8.3.0
module load openmpi/4.0.2-gcc-8.3.0-cuda10.1
module load pytorch/1.5.1
module load cuda/10.2
source env/bin/activate

which python3
python3 02_qm9_train.py "$@"
