#!/bin/bash

#SBATCH --account=lp_biolearning 
#SBATCH --job-name=HMC_cpu
#SBATCH --clusters=genius
#SBATCH --time=24:00:00 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --output=logs/hmc_gpu.log

source /user/leuven/356/vsc35684/.bashrc
conda activate hmc

python  hmc_uq/train_hmc.py