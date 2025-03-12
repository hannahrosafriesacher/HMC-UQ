set -e

ipython  hmc_uq/train_hmc.py -- --TargetID 1908 --nr_layers 1 --weight_decay 0.01 --hidden_sizes 4 --dropout 0.0 --L 10 --num_samples 100 --step_size 0.05