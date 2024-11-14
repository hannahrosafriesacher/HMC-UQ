set -e

python  hmc_uq/train_hmc.py --TargetID 1908 --nr_layers 1 --weight_decay 0.01 --hidden_sizes 4 --dropout 0.0 --num_samples 10