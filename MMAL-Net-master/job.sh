#!/bin/sh
#SBATCH --job-name=TestJob
#SBATCH --output=test.out
#SBATCH --error=testError.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000M

module load anaconda
python train.py