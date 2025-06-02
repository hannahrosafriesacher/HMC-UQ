#!/bin/bash

#SBATCH --account=lp_biolearning 
#SBATCH --job-name=MLP-sweep
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=30G
#SBATCH --output=logs/mlp_sweep.log

source /user/leuven/356/vsc35684/.bashrc
conda activate hmc

wandb agent rosa/uq-hmc_repeat/i2vr1v3v