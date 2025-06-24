#TODO: implement NN with more than 1 layers

import os
import yaml
import argparse

import numpy as np
import torch
import pyro
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils.load_config import get_args
from utils.models import MLP_pyro
from utils.evaluation import HMCPredictivePerformance, HMCSampleEvaluation
from utils.data import SparseDataset
import wandb

from timeit import default_timer as timer

args = get_args(config_file = 'configs/models/HMC.yaml')
print("Loaded Configuration:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

if args.evaluate_testset:
    project = 'UQ-HMC_Eval'
    group = 'hmc_eval'
else:
    project = 'UQ-HMC_Tune'
    group = 'hmc_tune'  

run = wandb.init(project = project, 
                 tags = ['hmc_pyro_test_baseline'],
                 group = group,
                 config = args)

nr_layers = wandb.config.nr_layers
target_id = wandb.config.TargetID
hidden_sizes = wandb.config.hidden_sizes
weight_decay = wandb.config.weight_decay
dropout = wandb.config.dropout

nr_chains = wandb.config.nr_chains
nr_samples = wandb.config.nr_samples
warmup_steps=wandb.config.warmup_steps
model_loss = wandb.config.model_loss
if model_loss == 'BCELoss':
    model_loss = 'binary_class_linear_output'
else:
    pass
    #TODO: warning loss not implemented

tr_fold = np.array(wandb.config.tr_fold)
va_fold=wandb.config.va_fold
te_fold=wandb.config.te_fold

init = wandb.config.init
save_model = wandb.config.save_model
evaluate_testset = wandb.config.evaluate_testset
evaluate_samples = wandb.config.evaluate_samples

device = wandb.config.device
rep = wandb.config.rep

#hamiltorch.set_random_seed(123)
os.environ['CUDA_VISIBLE_DEVICES']='0'
if device == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
logs = {}

#load Datasets
X_singleTask = np.load('data/chembl_29/chembl_29_X.npy', allow_pickle=True).item().tocsr()
Y_singleTask = np.load('data/chembl_29/chembl_29_thresh.npy', allow_pickle=True).item().tocsr()[:,target_id]
folding = np.load('data/chembl_29/folding.npy')

train_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, tr_fold, device)
val_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, va_fold, device)

x_train, y_train = train_dataset.__getdatasets__()
x_val, y_val= val_dataset.__getdatasets__()

params_chains = []
preds_chains_tr = []
preds_chains_val = []

if evaluate_testset:
    te_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, te_fold, device)
    x_te, y_te = te_dataset.__getdatasets__()
    preds_chains_te = []

num_input_features = train_dataset.__getinputdim__()
prior_scale = 1/(2*weight_decay)
wandb.config['dim_input'] = num_input_features

for chain in range(nr_chains):
    net = MLP_pyro(
        input_features=num_input_features, 
        output_features=1,
        hidden_sizes=hidden_sizes,          
        prior_scale=prior_scale, 
        device=device
        )

    num_iterations = 400
    #TODO: implement BBB intialization
    guide = AutoDelta(net)

    adam = pyro.optim.Adam({"lr": 0.01})
    svi = SVI(net, guide, adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_train, y_train)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(y_train)))

    """ckpt_dir = f'results/models/HMC/'
    os.makedirs(ckpt_dir, exist_ok = True)
    ckp_path = f'{ckpt_dir}{target_id}_nrs{nr_samples}_nrc{nr_chains}_{init}init_warmup{warmup_steps}_rep{rep}'
    
    np.save(ckp_path, samples.cpu().detach().numpy())

    layers_samples = []
    for layer in range(nr_layers + 1):
        layers_samples.append(samples[f'layer{layer+1}.weight'].reshape(nr_samples, -1))
        layers_samples.append(samples[f'layer{layer+1}.bias'].reshape(nr_samples, -1))
    params = torch.concat(layers_samples, axis = 1)

    params_chains.append(params) #TODO: save it into file and params = None

    predictive = Predictive(model=net, posterior_samples=samples)
    #get predictions for validation ds
    pred_tr = predictive(x_train)['obs']
    preds_chains_tr.append(pred_tr)
    
    #get predict']
        preds_chains_te.append(pred_te)

params_chains = torch.stack(params_chains)
preds_chains_tr = torch.stack(preds_chains_tr)
preds_chains_val = torch.stack(preds_chains_val)
preds_chains_te = torch.stack(preds_chains_te) if evaluate_testset else None"""

'''#Train Performance
tr_performance = HMCPredictivePerformance(preds_chains_tr, train_dataset.__getdatasets__()[1])
tr_performance.calculate_performance()

nll_tr, plot_nll_tr = tr_performance.nll(return_plot=True)
logs.update({f'/train/loss/chain{chain +1}': nll for chain, nll in enumerate(nll_tr)})
logs.update({f'/train/loss/average': np.mean(nll_tr)})

auc_tr, plot_auc_tr = tr_performance.auc(return_plot=True)
logs.update({f'/train/auc/chain{chain +1}': auc for chain, auc in enumerate(auc_tr)})
logs.update({f'/train/auc/average': np.mean(auc_tr)})

ece_tr, ace_tr, bs_tr, plot_ace_tr = tr_performance.calibration_errors(return_plot=True)
logs.update({f'/train/ece/chain{chain +1}': ece for chain, ece in enumerate(ece_tr)})
logs.update({f'/train/ece/average': np.mean(ece_tr)})
logs.update({f'/train/ace/chain{chain +1}': ace for chain, ace in enumerate(ace_tr)})
logs.update({f'/train/ace/average': np.mean(ace_tr)})
logs.update({f'/train/bs/chain{chain +1}': bs for chain, bs in enumerate(bs_tr)})
logs.update({f'/train/bs/average': np.mean(bs_tr)})

#Validation Performance
val_performance = HMCPredictivePerformance(preds_chains_val, val_dataset.__getdatasets__()[1])
val_performance.calculate_performance()

nll_val, plot_nll_val = val_performance.nll(return_plot=True)
logs.update({f'/val/loss/chain{chain +1}': nll for chain, nll in enumerate(nll_val)})
logs.update({f'/val/loss/average': np.mean(nll_val)})

auc_val, plot_auc_val = val_performance.auc(return_plot=True)
logs.update({f'/val/auc/chain{chain +1}': auc for chain, auc in enumerate(auc_val)})
logs.update({f'/val/auc/average': np.mean(auc_val)})

ece_val, ace_val, bs_val, plot_ace_val = val_performance.calibration_errors(return_plot=True)
logs.update({f'/val/ece/chain{chain +1}': ece for chain, ece in enumerate(ece_val)})
logs.update({f'/val/ece/average': np.mean(ece_val)})
logs.update({f'/val/ace/chain{chain +1}': ace for chain, ace in enumerate(ace_val)})
logs.update({f'/val/ace/average': np.mean(ace_val)})
logs.update({f'/val/bs/chain{chain +1}': bs for chain, bs in enumerate(bs_val)})
logs.update({f'/val/bs/average': np.mean(bs_val)})'''

"""


if evaluate_testset:
    te_performance = HMCPredictivePerformance(preds_chains_te, te_dataset.__getdatasets__()[1])
    te_performance.calculate_performance()

    nll_te = te_performance.nll(return_plot=False)
    logs.update({f'/test/loss/chain{chain +1}': nll for chain, nll in enumerate(nll_te)})
    logs.update({f'/test/loss/average': np.mean(nll_te)})

    auc_te = te_performance.auc(return_plot=False)
    logs.update({f'/test/auc/chain{chain +1}': auc for chain, auc in enumerate(auc_te)})
    logs.update({f'/test/auc/average': np.mean(auc_te)}) 

    ece_te, ace_te, bs_te, plot_ace_te = te_performance.calibration_errors(return_plot=True)
    logs.update({f'/test/ece/chain{chain +1}': ece for chain, ece in enumerate(ece_te)})
    logs.update({f'/test/ece/average': np.mean(ece_te)})
    logs.update({f'/test/ace/chain{chain +1}': ace for chain, ace in enumerate(ace_te)})
    logs.update({f'/test/ace/average': np.mean(ace_te)})
    logs.update({f'/test/bs/chain{chain +1}': bs for chain, bs in enumerate(bs_te)})
    logs.update({f'/test/bs/average': np.mean(bs_te)})

    #Save Test Set Predictions
    res_dir = f'results/predictions/HMC/'
    os.makedirs(res_dir, exist_ok = True)
    res_path = f'{res_dir}{target_id}_nrs{nr_samples}_nrc{nr_chains}_{init}init_warmup{warmup_steps}_rep{rep}_test'
    np.save(res_path , preds_chains_te.cpu().detach().numpy()) """ 

'''#Save Params
if save_model:
    ckpt_dir = f'results/models/HMC/'
    os.makedirs(ckpt_dir, exist_ok = True)
    ckp_path = f'{ckpt_dir}{target_id}_nrs{nr_samples}_nrc{nr_chains}_{init}init_warmup{warmup_steps}_rep{rep}'

    #Save model
    np.save(ckp_path, params_chains.cpu().detach().numpy())

    ckpt_lookup = f'configs/ckpt_paths/HMC.yaml'

    #Save to config file         
    if os.path.exists(ckpt_lookup):
        with open(ckpt_lookup, 'r') as file:
            try:
                lookup = yaml.safe_load(file) or {}  # Load safely, default to empty dict if None
            except yaml.YAMLError:
                lookup = {}  # If there's a parsing error, start fresh
    else:
        lookup = {}

    # Check if the path is already recorded
    if ckp_path not in lookup.values():
        lookup[target_id] = ckp_path  
        with open(ckpt_lookup, 'w') as file:
            yaml.dump(lookup, file) '''

wandb.log(logs)  


