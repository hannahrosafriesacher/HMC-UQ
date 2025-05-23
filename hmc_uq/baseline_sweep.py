#TODO:
#Comment out visible device
#new paths to files
#change Sweep names

import os 
import numpy as np
import torch
import scipy.sparse
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.load_config import get_args
from utils.models import MLP
from utils.evaluation import BaselinePredictivePerformance
from utils.data import SparseDataset

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import wandb
import yaml
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
wandb.login()

#Comment out when running on VSC
os.environ['CUDA_VISIBLE_DEVICES']='1'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_repeats=10
number_models_std=10

tr_fold = np.array([2,3,4])
va_fold=1

batch_size=200
model_loss = torch.nn.BCEWithLogitsLoss()

def train(target_id, nr_layers, hidden_sizes, dropout, weight_decay, learning_rate):
    #load Datasets
    X_singleTask = np.load('data/chembl_29/chembl_29_X.npy', allow_pickle=True).item().tocsr()
    Y_singleTask = np.load('data/chembl_29/chembl_29_thresh.npy', allow_pickle=True).item().tocsr()[:,target_id]
    folding = np.load('data/chembl_29/folding.npy')


    train_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, tr_fold, device)
    val_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, va_fold, device)

    dataloader_tr = DataLoader(train_dataset, batch_size=200, shuffle=False)
    dataloader_val = DataLoader(val_dataset, batch_size=200, shuffle=False)

    num_input_features = train_dataset.__getinputdim__()

    wandb.config['dim_input'] = num_input_features
    torch.set_num_threads(1)           
    
    for model in range(model_repeats):
        
        net = MLP(
        input_features=num_input_features, 
        output_features=1, 
        nr_layers = nr_layers,
        hidden_sizes=hidden_sizes, 
        dropout=dropout
        ).to(device)

        nr_epochs = 400
        criterion = model_loss
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  
 
        for epoch in tqdm(range(nr_epochs), desc=f'Training {nr_epochs} epochs:'):  # loop over the dataset multiple times 
            for X_batch, Y_batch in dataloader_tr:
                optimizer.zero_grad()      
                # forward + backward + optimizer
                net.eval()
                outputs = net(X_batch)
                net.train()
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                
            #get loss of each epoch for plotting convergence
            net.eval()
            
            #predict Training and Validation Dataset
            pred_train = net(train_dataset.__getdatasets__()[0])
            train_performance = BaselinePredictivePerformance(pred_train, train_dataset.__getdatasets__()[1], epoch, 'train')
            train_performance_epoch = train_performance.epoch_performance()

            pred_val = net(val_dataset.__getdatasets__()[0])
            val_performance = BaselinePredictivePerformance(pred_val, val_dataset.__getdatasets__()[1], epoch, 'val')
            val_performance_epoch = val_performance.epoch_performance()

            performance_epoch = train_performance_epoch | val_performance_epoch  
            wandb.log({f'{key}/{model}': value for key, value in performance_epoch.items()})

            if epoch == 0:
                performance_best = {'best/' + key: value for key, value in performance_epoch.items()}


            else:
                if val_performance_epoch['val/loss/'] <= performance_best['best/val/loss/']:
                    performance_best = {'best/' + key: value for key, value in performance_epoch.items()}


        wandb.summary.update({key + f'/{model}': value for key, value in performance_best.items()})
        
        if model == 0:
            performance_best_repeats = performance_best.copy()
        else:
            for key in performance_best:
                performance_best_repeats[key] = performance_best_repeats.get(key, 0) + performance_best.get(key, 0)

    wandb.summary.update({f'{key}/average': value/model_repeats for key, value in performance_best_repeats.items()})
        

def main():
    with open("hmc_uq/sweeps/sweep_baseline_1908.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
    wandb.init(config=config) 

    id = wandb.config.target_id
    nrl = wandb.config.nr_layers
    lr = wandb.config.learning_rate
    hs = wandb.config.hidden_size
    do = wandb.config.dropout
    wd = wandb.config.weight_decay
    train(target_id = id, nr_layers=nrl, hidden_sizes=hs, dropout=do, weight_decay=wd, learning_rate=lr)
    wandb.finish()

main()
