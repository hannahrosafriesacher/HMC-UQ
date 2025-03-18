#TODO: implement NN with more than 1 layers

import os
import yaml
import argparse

import numpy as np
import torch
import hamiltorch
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils.load_config import get_args
from utils.models import MLP
from utils.evaluation import HMCPredictivePerformance, HMCSampleEvaluation
from utils.data import SparseDataset
import wandb

from timeit import default_timer as timer

args = get_args(config_file = 'configs/models/hmc.yaml')
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


hamiltorch.set_random_seed(123)
os.environ['CUDA_VISIBLE_DEVICES']='0'
if device == 'gpu' and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
logs = {}

#load Datasets
X_singleTask = np.load('data/chembl_29/chembl_29_X.npy', allow_pickle=True).item().tocsr()
Y_singleTask = np.load('data/chembl_29/chembl_29_thresh.npy', allow_pickle=True).item().tocsr()[:,target_id]
folding=np.load('data/chembl_29/folding.npy')

train_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, tr_fold, device)
val_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, va_fold, device)

dataloader_tr = DataLoader(train_dataset, batch_size=200, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=200, shuffle=False)
params_chains = []
preds_chains_val = []

if evaluate_testset:
    te_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, te_fold, device)
    dataloader_te = DataLoader(te_dataset, batch_size=200, shuffle=False)
    preds_chains_te = []

num_input_features = train_dataset.__getinputdim__()

wandb.config['dim_input'] = num_input_features
    
for chain in range(nr_chains):
    net = MLP(
        hidden_sizes=hidden_sizes, 
        input_features=num_input_features, 
        output_features=1, 
        dropout=dropout
        )
    if init == 'random':
        params_init = hamiltorch.util.flatten(net).to(device).clone()

    elif init == 'bbb': #initialize with BBB
        yaml_path = 'configs/ckpt_paths/BBB.yaml'
        with open(yaml_path, "r") as file:
            paths = yaml.safe_load(file)[target_id]

        #TODO: implement for more layers
        #TODO: re-implement
        params_surrogate = torch.load(paths)
        params = OrderedDict()
        params['input.weight'] = params_surrogate[f'nr_layers.0.W_mu'].T + params_surrogate[f'nr_layers.0.W_rho'].T * torch.rand_like(params_surrogate[f'nr_layers.0.W_rho'].T)
        params['input.bias'] = params_surrogate['nr_layers.0.b_mu'] + params_surrogate['nr_layers.0.b_rho'] * torch.rand_like(params_surrogate[f'nr_layers.0.b_rho'])
        params['output.weight'] = params_surrogate['nr_layers.1.W_mu'].T + params_surrogate['nr_layers.1.W_rho'].T * torch.rand_like(params_surrogate[f'nr_layers.1.W_rho'].T)
        params['output.bias'] = params_surrogate['nr_layers.1.b_mu'] + params_surrogate['nr_layers.1.b_rho'] * torch.rand_like(params_surrogate[f'nr_layers.1.b_rho'])
        net.load_state_dict(params)
        params_init = hamiltorch.util.flatten(net).to(device).clone()

    tau = weight_decay
    #TODO: check code what happens if I specify only 1 tau?
    tau_list = [tau]
    tau_list = torch.tensor(tau_list).to(device)
    start = timer()
    params_gpu, accept_rate = hamiltorch.sample_model(
        net, 
        x = train_dataset.__getdatasets__()[0], 
        y = train_dataset.__getdatasets__()[1], 
        params_init=params_init, 
        num_samples=nr_samples,
        step_size=step_size, 
        num_steps_per_sample=L,
        tau_out=tau_out,
        tau_list=tau_list, 
        model_loss=model_loss,
        debug = 2
        )
    end = timer()
    print('HMC Sampler', end - start)
    logs.update({f'ar/chain{chain + 1}': accept_rate})

    params = torch.stack(params_gpu, dim = 0).cpu().numpy()
    params_chains.append(params)
    
    #get predictions for validation ds
    pred_list_val, log_prob_list_val = hamiltorch.predict_model(net, test_loader = dataloader_val, samples=params_gpu, model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
    pred_val = torch.squeeze(pred_list_val, 2)  
    preds_chains_val.append(pred_val)

    if evaluate_testset:
        pred_list_te, log_prob_list_te = hamiltorch.predict_model(net, test_loader = dataloader_te, samples=params_gpu, model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
        pred_te = torch.squeeze(pred_list_te, 2)  
        preds_chains_te.append(pred_te)

params_chains = np.stack(params_chains)
preds_chains_val = torch.stack(preds_chains_val)
preds_chains_te = torch.stack(preds_chains_te) if evaluate_testset else None


val_performance = HMCPredictivePerformance(preds_chains_val, val_dataset.__getdatasets__()[1])
val_performance.calculate_performance()

nll_val, plot_nll_val = val_performance.nll(return_plot=True)
logs.update({f'/val/loss/chain{chain +1}': nll for chain, nll in enumerate(nll_val)})
logs.update({f'/val/loss/average': np.mean(nll_val)})

auc_val, plot_auc_val = val_performance.auc(return_plot=True)
logs.update({f'/val/auc/chain{chain +1}': auc for chain, auc in enumerate(auc_val)})
logs.update({f'/val/auc/average': np.mean(auc_val)})

if evaluate_samples:
    sample_eval = HMCSampleEvaluation(params_chains)

    start = timer()
    ac = sample_eval.calculate_autocorrelation()
    end = timer()
    print('AUTOCORR', end - start)

    start = timer()
    logs.update({f'SplitRhat': sample_eval.split_rhat(burnin=0, rank_normalized=False).mean()})
    end = timer()
    print('SplitRhat', end - start)

    start = timer()
    logs.update({f'rnSplitRhat': sample_eval.split_rhat(burnin=0, rank_normalized=True).mean()}) #Convergence
    end = timer()
    print('rnSplitRhat', end - start)

    start = timer()
    logs.update({f'GEW/chain{chain +1}': gew for chain, gew in enumerate(sample_eval.geweke())}) #Convergence
    end = timer()
    print('AGEW', end - start)

    start = timer()
    logs.update({f'IPS/chain{chain +1}': ips for chain, ips in enumerate(sample_eval.ips_per_chain(burnin=False))}) #SampleSize
    end = timer()
    print('IPS', end - start)

    start = timer()
    logs.update({f'IPS-burnin/chain{chain +1}': ips for chain, ips in enumerate(sample_eval.ips_per_chain(burnin=True))})
    end = timer()
    print('IPS-burnin', end - start)

    start = timer()
    split_rhat_az, rnsplit_rhat_az = sample_eval.rhat_az()
    logs.update({f'SplitRhat/AZ': split_rhat_az.mean().item()})
    #logs.update({f'rnSplitRhat/AZ': rnsplit_rhat_az.mean().item()})
    end = timer()
    print('SplitRhat/AZ', end - start)

    start = timer()
    logs.update({f'IPS/AZ': sample_eval.ess_az().mean().item()})
    end = timer()
    print('IPS/AZ', end - start)
    #TODO: Log Acceptance rate, IPS for all chains combined

    start = timer()
    #Shift WANDB in Evaluation file?
    wandb.log({f'Rhat vs Burn-in': sample_eval.rhat_burnin_plot()})
    wandb.log({f'IPS vs Burn-in': sample_eval.ips_burnin_plot()})
    wandb.log({f'Autocorrelation': sample_eval.autocorrelation_plot()})
    trace_plots = sample_eval.trace_plot(net.state_dict())
    for plot in trace_plots:
        wandb.log({f'{plot}': trace_plots[plot]})
    end = timer()
    print('Plots', end - start)


if evaluate_testset:
    te_performance = HMCPredictivePerformance(preds_chains_te, te_dataset.__getdatasets__()[1])
    te_performance.calculate_performance()

    nll_te = te_performance.nll(return_plot=False)
    logs.update({f'/test/loss/chain{chain +1}': nll for chain, nll in enumerate(nll_te)})
    logs.update({f'/test/loss/average': np.mean(nll_te)})

    auc_te = te_performance.auc(return_plot=False)
    logs.update({f'/test/auc/chain{chain +1}': auc for chain, auc in enumerate(auc_te)})
    logs.update({f'/test/auc/average': np.mean(auc_te)}) 

    #Save Test Set Predictions
    res_dir = f'results/HMC/'
    os.makedirs(res_dir, exist_ok = True)
    res_path = f'{res_dir}{target_id}_e{step_size}_l{L}_nrs{nr_samples}_nrc{nr_chains}'
    np.save(res_path , preds_chains_te.cpu().detach().numpy())

#Save Params
if save_model:
    ckpt_dir = f'logs/HMC/'
    os.makedirs(ckpt_dir, exist_ok = True)
    ckp_path = f'{ckpt_dir}{target_id}_e{step_size}_l{L}_nrs{nr_samples}_nrc{nr_chains}'

    #Save model
    np.save(ckp_path, params_chains)

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
            yaml.dump(lookup, file)

wandb.log(logs)    


