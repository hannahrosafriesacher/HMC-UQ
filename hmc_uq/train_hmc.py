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
use_nuts = wandb.config.use_nuts

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
folding=np.load('data/chembl_29/folding.npy')

train_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, tr_fold, device)
val_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, va_fold, device)

dataloader_tr = DataLoader(train_dataset, batch_size=200, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=200, shuffle=False)
params_chains = []
preds_chains_tr = []
preds_chains_val = []

if evaluate_testset:
    te_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, te_fold, device)
    dataloader_te = DataLoader(te_dataset, batch_size=200, shuffle=False)
    preds_chains_te = []

num_input_features = train_dataset.__getinputdim__()

wandb.config['dim_input'] = num_input_features
    
for chain in range(nr_chains):
    net = MLP(
        input_features=num_input_features, 
        output_features=1,
        nr_layers=nr_layers,
        hidden_sizes=hidden_sizes,          
        dropout=dropout
        )
    if init == 'random':
        params_init = hamiltorch.util.flatten(net).to(device).clone()

    elif init == 'bbb': #initialize with BBB
        yaml_path = 'configs/ckpt_paths/BBB.yaml'
        with open(yaml_path, "r") as file:
            paths = yaml.safe_load(file)[target_id]

        params_surrogate = torch.load(paths)
        params = OrderedDict()
        for l in range(nr_layers + 1):
            params[f'model.{l}.weight'] = params_surrogate[f'nr_layers.{l}.W_mu'].T + params_surrogate[f'nr_layers.{l}.W_rho'].T * torch.rand_like(params_surrogate[f'nr_layers.{l}.W_rho'].T)
            params[f'model.{l}.bias'] = params_surrogate[f'nr_layers.{l}.b_mu'] + params_surrogate[f'nr_layers.{l}.b_rho'] * torch.rand_like(params_surrogate[f'nr_layers.{l}.b_rho'])
        net.load_state_dict(params)
        params_init = hamiltorch.util.flatten(net).to(device).clone()

    tau = weight_decay
    #TODO: check code what happens if I specify only 1 tau?
    tau_list = [tau]
    tau_list = torch.tensor(tau_list).to(device)
    start = timer()
    sampler = hamiltorch.Sampler.HMC_NUTS if use_nuts else hamiltorch.Sampler.HMC
    burn = 10 if use_nuts else -1

    params_gpu, run_info = hamiltorch.sample_model(     #run_info = Acceptance rate or adapted epsilon depending if NUTS was used
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
        debug = 2,
        sampler = sampler, 
        burn = burn
        )
    end = timer()
    print('HMC Sampler', end - start)
    logs.update({f'adapt_step_size/chain{chain + 1}': run_info}) if use_nuts else logs.update({f'ar/chain{chain + 1}': run_info})

    params = torch.stack(params_gpu, dim = 0).cpu().numpy()
    params_chains.append(params) #TODO: save it into file and params = None

    #get predictions for validation ds
    pred_list_tr, log_prob_list_tr = hamiltorch.predict_model(net, test_loader = dataloader_tr, samples=params_gpu, model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
    pred_tr = torch.squeeze(pred_list_tr, 2)  
    preds_chains_tr.append(pred_tr)
    
    #get predictions for validation ds
    pred_list_val, log_prob_list_val = hamiltorch.predict_model(net, test_loader = dataloader_val, samples=params_gpu, model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
    pred_val = torch.squeeze(pred_list_val, 2)  
    preds_chains_val.append(pred_val)

    if evaluate_testset:
        pred_list_te, log_prob_list_te = hamiltorch.predict_model(net, test_loader = dataloader_te, samples=params_gpu, model_loss=model_loss, tau_out=tau_out, tau_list=tau_list)
        pred_te = torch.squeeze(pred_list_te, 2)  
        preds_chains_te.append(pred_te)
logs.update({f'burnin': burn}) 

params_chains = np.stack(params_chains)
preds_chains_tr = torch.stack(preds_chains_tr)
preds_chains_val = torch.stack(preds_chains_val)
preds_chains_te = torch.stack(preds_chains_te) if evaluate_testset else None

#Train Performance
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
logs.update({f'/val/bs/average': np.mean(bs_val)})

#Evaluate Samples
start = timer()
if evaluate_samples:
    sample_eval = HMCSampleEvaluation(params_chains, num_input_features, hidden_sizes, reduce = 100)

    ac = sample_eval.calculate_autocorrelation()
    
    logs.update({f'SplitRhat': sample_eval.split_rhat(burnin=0, rank_normalized=False).mean()})
    logs.update({f'rnSplitRhat': sample_eval.split_rhat(burnin=0, rank_normalized=True).mean()}) #Convergence
    logs.update({f'GEW/chain{chain +1}': gew for chain, gew in enumerate(sample_eval.geweke())}) #Convergence
    logs.update({f'IPS/chain{chain +1}': ips for chain, ips in enumerate(sample_eval.ips_per_chain(burnin=False))}) #SampleSize
    logs.update({f'IPS-burnin/chain{chain +1}': ips for chain, ips in enumerate(sample_eval.ips_per_chain(burnin=True))})

    split_rhat_az, rnsplit_rhat_az = sample_eval.rhat_az()
    logs.update({f'SplitRhat/AZ': split_rhat_az.mean().item()})
    logs.update({f'rnSplitRhat/AZ': rnsplit_rhat_az.mean().item()})
    logs.update({f'IPS/AZ': sample_eval.ess_az().mean().item()})
    wandb.log({f'Rhat vs Burn-in': sample_eval.rhat_burnin_plot()})
    wandb.log({f'IPS vs Burn-in': sample_eval.ips_burnin_plot()})
    wandb.log({f'Autocorrelation': sample_eval.autocorrelation_plot()})
    trace_plots = sample_eval.trace_plot(net.state_dict())
    for plot in trace_plots:
        wandb.log({f'{plot}': trace_plots[plot]})
    end = timer()
    print('Eval Samples', end - start)


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
    res_path = f'{res_dir}{target_id}_e{step_size}_l{L}_nrs{nr_samples}_nrc{nr_chains}_{init}init_rep{rep}'
    np.save(res_path , preds_chains_te.cpu().detach().numpy())

#Save Params
if save_model:
    ckpt_dir = f'results/models/HMC/'
    os.makedirs(ckpt_dir, exist_ok = True)
    ckp_path = f'{ckpt_dir}{target_id}_e{step_size}_l{L}_nrs{nr_samples}_nrc{nr_chains}_{init}init_rep{rep}'

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


