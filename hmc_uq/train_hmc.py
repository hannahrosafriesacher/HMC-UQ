import os
import yaml
from pathlib import Path
import argparse
import sparsechem as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math
import random
from sklearn.metrics import roc_auc_score

import torch
import hamiltorch
import torch.nn as nn
import torch.nn.functional as F

from utils.models import MLP
from utils.evaluation import PredictivePerformance, SampleEvaluation
import wandb
sns.set_style('whitegrid')

parser = argparse.ArgumentParser(description='Training a single-task model.')
parser.add_argument('--TargetID', type=int, default=None)
parser.add_argument('--nr_layers', type=int, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--hidden_sizes', type=int, default=None)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--tr_fold', type=list, default=[2,3,4])
parser.add_argument('--va_fold', type=int, default=1)
parser.add_argument('--nr_chains', type=int, default=5)
parser.add_argument('--step_size', type=float, default=1e-3)
parser.add_argument('--nr_samples', type=int, default=150)
parser.add_argument('--L', type=int, default=150)
parser.add_argument('--tau_out', type=float, default=1.)
parser.add_argument('--model_loss', type=str, default='binary_class_linear_output')
parser.add_argument('--save_params', type=bool, default='False')
parser.add_argument('--evaluate_testset', type=bool, default='False')
args = parser.parse_args()

project = 'UQ-HMC_Tune' if args.evaluate_testset else 'UQ-HMC_Eval'
group = 'hmc_tune' if args.evaluate_testset else 'hmc_eval'    

run = wandb.init(project = project, 
                 tags = ['hmc'],
                 group = group,
                 config = args)

nr_layers = wandb.config.nr_layers
target_id = wandb.config.TargetID
hidden_sizes = wandb.config.hidden_sizes
weight_decay = wandb.config.weight_decay
dropout = wandb.config.dropout
nr_chains = wandb.config.nr_chains
step_size = wandb.config.step_size #Epsilon 
nr_samples = wandb.config.nr_samples
L = wandb.config.L
tau_out = wandb.config.tau_out
model_loss = wandb.config.model_loss

tr_fold = np.array(wandb.config.tr_fold)
va_fold=wandb.config.va_fold


hamiltorch.set_random_seed(123)
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logs = {}

#load Datasets
X_singleTask=sc.load_sparse('data/chembl_29/chembl_29_X.npy')
Y_singleTask=sc.load_sparse('data/chembl_29/chembl_29_thresh.npy')[:,target_id]
folding=np.load('data/chembl_29/folding.npy')

#Training Data
X_tr_np=X_singleTask[np.isin(folding, tr_fold)].todense()
Y_tr_np=Y_singleTask[np.isin(folding, tr_fold)].todense()

#filter for nonzero values in Training Data
nonzero=np.nonzero(Y_tr_np)[0]
X_tr_np=X_tr_np[nonzero]
Y_tr_np=Y_tr_np[nonzero]
Y_tr_np[Y_tr_np==-1]=0

#validation Data
X_val_np=X_singleTask[folding==va_fold].todense()
Y_val_np=Y_singleTask[folding==va_fold].todense()

#filter for nonzero values in Validation data
nonzero=np.nonzero(Y_val_np)[0]
X_val_np=X_val_np[nonzero]
Y_val_np=Y_val_np[nonzero]
Y_val_np[Y_val_np==-1]=0

#TODO: 
X_train=torch.from_numpy(X_tr_np).float().to(device)
Y_train=torch.from_numpy(Y_tr_np).to(device)
X_val=torch.from_numpy(X_val_np).float().to(device)
Y_val=torch.from_numpy(Y_val_np).to(device)
num_input_features=X_tr_np.shape[1]
wandb.config['dim_input'] = num_input_features
    

params_chains = []
preds_chains = []
for chain in range(nr_chains):
    #TODO: check if initilization is random
    net=MLP(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=1, dropout=dropout)
    params_init = hamiltorch.util.flatten(net).to(device).clone()

    tau = weight_decay
    #TODO: check code what happens if I specify only 1 tau?
    tau_list = [tau]
    tau_list = torch.tensor(tau_list).to(device)

    params_gpu = hamiltorch.sample_model(net, X_train, Y_train, params_init=params_init, num_samples=nr_samples,
                                step_size=step_size, num_steps_per_sample=L,tau_out=tau_out,tau_list=tau_list, model_loss=model_loss)

    params = torch.stack(params_gpu, dim = 0).cpu().numpy()
    params_chains.append(params)
    
    #get predictions for validation ds
    pred_list, log_prob_list = hamiltorch.predict_model(net, x=X_val, y=Y_val, samples=params_gpu, model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
    pred = torch.squeeze(pred_list, 2)  
    preds_chains.append(pred)

params_chains = np.stack(params_chains)
preds_chains = torch.stack(preds_chains)


val_performance = PredictivePerformance(preds_chains, Y_val)
val_performance.calculate_performance()

nll_val, plot_nll_val = val_performance.nll(return_plot=True)
logs.update({f'NLL: Chain {chain +1}': nll for chain, nll in enumerate(nll_val)})
wandb.log({'nll_val': plot_nll_val})

auc_val, plot_auc_val = val_performance.auc(return_plot=True)
logs.update({f'AUC: Chain {chain +1}': auc for chain, auc in enumerate(auc_val)})
wandb.log({'auc_val': plot_auc_val})

sample_eval = SampleEvaluation(params_chains)
sample_eval.calculate_autocorrelation()
logs.update({f'Split-Rhat': sample_eval.split_rhat(burnin=0, rank_normalized=False).mean()})
logs.update({f'rnSplit-Rhat': sample_eval.split_rhat(burnin=0, rank_normalized=True).mean()}) #Convergence
logs.update({f'GEW: Chain {chain +1}': gew for chain, gew in enumerate(sample_eval.geweke())}) #Convergence
logs.update({f'IPS: Chain {chain +1}': ips for chain, ips in enumerate(sample_eval.ips_per_chain(burnin=0))}) #SampleSize


 
#Shift WANDB in Evaluation file?
wandb.log({f'Rhat vs Burn-in': sample_eval.rhat_burnin_plot()})
wandb.log({f'IPS vs Burn-in': sample_eval.ips_burnin_plot()})
wandb.log({f'Autocorrelation': sample_eval.autocorrelation_plot()})
wandb.log(logs)

if wandb.config.evaluate_testset:
    #TODO: save Test set predictions
    pass

#Log Acceptance rate

#Define path
if wandb.config.save_params:
    ckpt_dir = f'logs/HMC/'
    os.makedirs(ckpt_dir, exist_ok = True)
    ckp_path = f'{ckpt_dir}{target_id}_e{step_size}_l{L}_nrs{nr_samples}_nrc{nr_chains}'

    #Save model
    np.save(ckp_path, params_chains)

    ckpt_lookup = f'configs/ckpt_paths/HMC.yaml'
    #Save to config file
    if os.path.exists(ckpt_lookup):
        lookup = yaml.safe_load(open(ckpt_lookup, 'r'))
        n = len(lookup.keys())
        if n == 1 and lookup[0] is None:
            n = 0

    else:
        lookup = {}
        n = 0
    if ckp_path not in lookup.values():

        lookup[n] = ckp_path
        yaml.dump(lookup, open(ckpt_lookup, 'w'))

    


