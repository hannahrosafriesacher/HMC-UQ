import os
import argparse
import sparsechem as sc
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import math
import random

import torch
import hamiltorch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import wandb


parser = argparse.ArgumentParser(description='Training a single-task model.')
parser.add_argument('--TargetID', type=int, default=None)
parser.add_argument('--nr_layers', type=int, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--hidden_sizes', type=int, default=None)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--tr_fold', type=list, default=[2,3,4])
parser.add_argument('--va_fold', type=int, default=1)
parser.add_argument('--step_size', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=150)
parser.add_argument('--L', type=int, default=150)
parser.add_argument('--tau_out', type=float, default=1.)
parser.add_argument('--model_loss', type=str, default='binary_class_linear_output')
args = parser.parse_args()

run = wandb.init(project = 'uq-hmc_runs', 
                 tags = ['hamiltorch'],
                 group = 'test_ht',
                 config = args)

nr_layers = wandb.config.nr_layers
target_id = wandb.config.TargetID
hidden_sizes = wandb.config.hidden_sizes
weight_decay = wandb.config.weight_decay
dropout = wandb.config.dropout
step_size = wandb.config.step_size #Epsilon 
num_samples = wandb.config.num_samples
L = wandb.config.L
tau_out = wandb.config.tau_out
model_loss = wandb.config.model_loss

tr_fold = np.array(wandb.config.tr_fold)
va_fold=wandb.config.va_fold


hamiltorch.set_random_seed(123)
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

class Net(torch.nn.Module):
    def __init__(self, input_features, hidden_sizes, output_features, dropout):
        super().__init__()
        self.input_features = input_features
        self.hidden_sizes = hidden_sizes
        self.output_features = output_features
        self.input = torch.nn.Linear(in_features=input_features, out_features=hidden_sizes)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.output = torch.nn.Linear(in_features=hidden_sizes, out_features=output_features)
        
    def forward(self, x):
        fc = self.input(x)
        a = self.tanh(fc)
        dr = self.dropout(a)
        out = self.output(dr)
             
        return out
    

net=Net(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=1, dropout=dropout)

params_init = hamiltorch.util.flatten(net).to(device).clone()


tau_list = []
tau = weight_decay
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

params_hmc = hamiltorch.sample_model(net, X_train, Y_train, params_init=params_init, num_samples=num_samples,
                               step_size=step_size, num_steps_per_sample=L,tau_out=tau_out,tau_list=tau_list, model_loss=model_loss)


#NLL PLOT
#get predictions for validation ds
pred_list, log_prob_list = hamiltorch.predict_model(net, x=X_val, y=Y_val, samples=params_hmc[:], model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
pred = torch.squeeze(pred_list, 2)



#NLL PLOT
nll = 0
NLL = {'#HMC Samples' : [], 'NLL': []}
bceloss = torch.nn.BCELoss()
pred = F.sigmoid(pred)
for s in range(0,len(pred_list)):
    ensemble_proba_it = pred[s]
    nll += bceloss(ensemble_proba_it, Y_val.squeeze(dim = 1).float()).item()
    mean = nll/(s+1)

    NLL['#HMC Samples'].append(s+1)
    NLL['NLL'].append(mean) 

NLL = pd.DataFrame(NLL)

#Negative Log lokelihood of Validation Set
plt.cla()
plot_nll = sns.lineplot(data = NLL, x = '#HMC Samples', y = 'NLL')

wandb.log({'Plot: NLL': plot_nll})
#plt.savefig('/home/rosa/git/HMC-UQ/results/figures/NLL.png')

#PLOT RANDOM PARAMATERS
nr_plotted_params = 8
rd_ind = random.sample(range(0, len(params_hmc)), nr_plotted_params)

params_cat = torch.stack(params_hmc, dim = 0).cpu().numpy()
params_selected = params_cat[:, rd_ind]
params_selected = pd.DataFrame(params_selected)

for i in range(nr_plotted_params):
    plot_params = sns.lineplot(data = params_selected.reset_index(), x = 'index', y = i)
wandb.log({'Plot: Params':  plot_params})
#plt.savefig('/home/rosa/git/HMC-UQ/results/figures/PARAMS.png')


#AUTOCORRELATION PLOT
min_lag = 1
max_lag = math.floor(len(params_hmc)/2)


ACORR = {'LAG' : [], 'AUTOCORR': []}
for lag in range(min_lag, max_lag):
    corr_list = []
    for it in range(len(params_hmc)-lag):
        corr, _ = pearsonr(params_hmc[it].cpu().numpy(), params_hmc[it+lag].cpu().numpy())
        corr_list.append(corr)

    ACORR['LAG'].append(lag)
    ACORR['AUTOCORR'].append(np.mean(corr_list))

ACORR = pd.DataFrame(ACORR)
acorr = sns.lineplot(data = ACORR, x = 'LAG', y = 'AUTOCORR')
wandb.log({'Plot: AUTOCORR': acorr})
#plt.savefig('/home/rosa/git/HMC-UQ/results/figures/AUTOKORR.png')


#EFFECTIVE SAMPLE SIZE
ESS = len(params_hmc)/(1 + 2*ACORR['AUTOCORR'].sum())
wandb.log({'ESS': ESS,})