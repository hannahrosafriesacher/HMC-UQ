#!/bin/bash

#SBATCH --account=lp_biolearning 
#SBATCH --job-name=HMC_gpu
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --time=24:00:00 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=25G
#SBATCH --output=logs/hmc_gpu.log

source /user/leuven/356/vsc35684/.bashrc
conda activate hmc

python  hmc_uq/train_hmc.py