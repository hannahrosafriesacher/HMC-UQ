#TODO:
#Comment out visible device
#new paths to files
#change propSampled
#change Sweep names

import os 
import numpy as np
import torch
import scipy.sparse
import sparsechem as sc
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from tabulate import tabulate
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import wandb
from calibration_metrics import calcCalibrationErrors, Brier
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
epoch_number=400
batch_size=200

def train(target_id, nr_layers, hidden_sizes, dropout, weight_decay, learning_rate):
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

    torch.set_num_threads(1)
    #for i in range(number_models_std):             
    #Model
    class Net(torch.nn.Module):
        def __init__(self, 
                     input_features, 
                     nr_layers, 
                     hidden_sizes, 
                     output_features, 
                     dropout
        ):
            super(Net,self).__init__()
            self.input_features=input_features
            self.hidden_sizes=hidden_sizes
            self.output_features=output_features
            self.dropout=dropout
            self.nr_layers = nr_layers

            layers = [input_features]
            for i in range(nr_layers):
                layers.append(hidden_sizes)
            print(layers)

            inplace = False
            self.layers = torch.nn.ModuleList()
            for layer in range(nr_layers):
                self.layers.append(
                    torch.nn.Sequential(
                        #SparseLinearLayer,
                        torch.nn.Linear(in_features=layers[layer], out_features=layers[layer + 1]),
                        #Relu,
                        torch.nn.Tanh(),
                        #Dropout
                        torch.nn.Dropout(p=dropout)                )
                )
            #Linear,
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_features=layers[-1], out_features=1)
                    )
                )
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x


    loss_train_sum=0
    acc_train_sum=0
    auc_roc_train_sum=0
    auc_pr_train_sum=0  
    ECE_train_sum=0        
    ACE_train_sum=0
    brier_train_sum=0
    loss_val_sum=0        
    acc_val_sum=0 
    auc_roc_val_sum=0
    auc_pr_val_sum=0   
    ECE_val_sum=0        
    ACE_val_sum=0
    brier_val_sum=0
    
    for model in range(model_repeats):
        print('MODEL:', model)  
        #training loop
        net=Net(hidden_sizes=hidden_sizes, nr_layers=nr_layers, input_features=num_input_features, output_features=1, dropout=dropout).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        loss_train_best=1
        acc_train_best=0
        auc_roc_train_best=0
        auc_pr_train_best=0  
        ECE_train_best=1        
        ACE_train_best=1
        brier_train_best=1
        loss_val_best=1
        acc_val_best=0
        auc_roc_val_best=0
        auc_pr_val_best=0   
        ECE_val_best=1        
        ACE_val_best=1
        brier_val_best=1
        epoch=0
        for epoch in tqdm(range(epoch_number), desc=f'Model {model}'):  # loop over the dataset multiple times 
            permutation = torch.randperm(X_train.size()[0])
            i=0
            for i in range(0,X_train.size()[0], batch_size):
                optimizer.zero_grad()
                # get the inputs; data is a list of [inputs, labels]
                indices = permutation[i:i+batch_size]
                inputs, labels = X_train[indices], Y_train[indices]        
                # forward + backward + optimizer
                net.eval()
                outputs = net(inputs)
                net.train()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            #get loss of each epoch for plotting convergence
            net.eval()
            
            #predict Training and Validation Dataset
            pred_train_logits=net(X_train).detach()
            pred_val_logits=net(X_val).detach()
            pred_train=torch.sigmoid(pred_train_logits).cpu().numpy().reshape(-1)
            pred_val=torch.sigmoid(pred_val_logits).cpu().numpy().reshape(-1)
            pred_train_labels=np.where(pred_train>0.5,1.0,0.0)
            pred_val_labels=np.where(pred_val>0.5,1.0,0.0)
            

            pred_train_logits_cpu=pred_train_logits.cpu().numpy()
            pred_val_logits_cpu=pred_val_logits.cpu().numpy()
            
            Y_tr_np = np.asarray(Y_tr_np).reshape(-1)
            Y_val_np = np.asarray(Y_val_np).reshape(-1)
            #Scores Training Dataset
            loss_train=criterion(net(X_train), Y_train).detach().item()
            acc_train=accuracy_score(Y_tr_np, pred_train_labels)
            auc_roc_train=roc_auc_score(Y_tr_np, pred_train)
            precision_train, recall_train, _ = precision_recall_curve(Y_tr_np, pred_train)
            auc_pr_train = auc(recall_train, precision_train)
            ECE_train=calcCalibrationErrors(Y_tr_np, pred_train,10)[0]
            ACE_train=calcCalibrationErrors(Y_tr_np, pred_train,10)[1]
            brier_train=Brier(Y_tr_np, pred_train)

            #Scores Validation Dataset
            loss_val=criterion(net(X_val), Y_val).detach().item()
            acc_val=accuracy_score(Y_val_np, pred_val_labels)
            auc_roc_val=roc_auc_score(Y_val_np, pred_val)
            precision_val, recall_val, _ = precision_recall_curve(Y_val_np, pred_val)
            auc_pr_val = auc(recall_val, precision_val)
            ECE_val=calcCalibrationErrors(Y_val_np, pred_val,10)[0]
            ACE_val=calcCalibrationErrors(Y_val_np, pred_val,10)[1]
            brier_val=Brier(Y_val_np, pred_val)

            log_dict={'Epoch/'+str(model)+'/': epoch,        
            'train/Loss/'+str(model)+'/': loss_train,        
            'train/Accuracy/'+str(model)+'/': acc_train, 
            'train/ROCAUC/'+str(model)+'/': auc_roc_train,
            'train/PRAUC/'+str(model)+'/': auc_pr_train,  
            'train/ECE/'+str(model)+'/': ECE_train,        
            'train/ACE/'+str(model)+'/': ACE_train,
            'train/Brier/'+str(model)+'/': brier_train,
            'val/Loss/'+str(model)+'/': loss_val,        
            'val/Accuracy/'+str(model)+'/': acc_val, 
            'val/ROCAUC/'+str(model)+'/': auc_roc_val,
            'val/PRAUC/'+str(model)+'/': auc_pr_val,   
            'val/ECE/'+str(model)+'/': ECE_val,        
            'val/ACE/'+str(model)+'/': ACE_val,
            'val/Brier/'+str(model)+'/': brier_val,
            }
            wandb.log(log_dict)

            if loss_train < loss_train_best:
                loss_train_best=loss_train
            if acc_train > acc_train_best:
                acc_train_best=acc_train
            if auc_roc_train > auc_roc_train_best:
                auc_roc_train_best=auc_roc_train
            if auc_pr_train > auc_pr_train_best:
                auc_pr_train_best=auc_pr_train
            if ECE_train < ECE_train_best:
                ECE_train_best=ECE_train
            if ACE_train < ACE_train_best:
                ACE_train_best=ACE_train
            if brier_train < brier_train_best:
                brier_train_best=brier_train
            if loss_val < loss_val_best:
                loss_val_best=loss_val
            if acc_val > acc_val_best:
                acc_val_best=acc_val
            if auc_roc_val > auc_roc_val_best:
                auc_roc_val_best=auc_roc_val
            if auc_pr_val > auc_pr_val_best:
                auc_pr_val_best=auc_pr_val
            if ECE_val < ECE_val_best:
                ECE_val_best=ECE_val
            if ACE_val < ACE_val_best:
                ACE_val_best=ACE_val
            if brier_val < brier_val_best:
                brier_val_best=brier_val
        
        wandb.summary['train/Loss/'+str(model)+'/.min']=loss_train_best
        wandb.summary['train/Accuracy/'+str(model)+'/.max']=acc_train_best
        wandb.summary['train/ROCAUC/'+str(model)+'/.max']=auc_roc_train_best
        wandb.summary['train/PRAUC/'+str(model)+'/.max']=auc_pr_train_best
        wandb.summary['train/ECE/'+str(model)+'/.min']=ECE_train_best
        wandb.summary['train/ACE/'+str(model)+'/.min']=ACE_train_best
        wandb.summary['train/Brier/'+str(model)+'/.min']=brier_train_best
        wandb.summary['val/Loss/'+str(model)+'/.min']=loss_val_best
        wandb.summary['val/Accuracy/'+str(model)+'/.max']=acc_val_best
        wandb.summary['val/ROCAUC/'+str(model)+'/.max']=auc_roc_val_best
        wandb.summary['val/PRAUC/'+str(model)+'/.max']=auc_pr_val_best
        wandb.summary['val/ECE/'+str(model)+'/.min']=ECE_val_best
        wandb.summary['val/ACE/'+str(model)+'/.min']=ACE_val_best
        wandb.summary['val/Brier/'+str(model)+'/.min']=brier_val_best

        loss_train_sum+=loss_train_best
        acc_train_sum+=acc_train_best
        auc_roc_train_sum+=auc_roc_train_best
        auc_pr_train_sum+=auc_pr_train_best
        ECE_train_sum+=ECE_train_best
        ACE_train_sum+=ACE_train_best
        brier_train_sum+=brier_train_best
        loss_val_sum+=loss_val_best    
        acc_val_sum+=acc_val_best
        auc_roc_val_sum+=auc_roc_val_best
        auc_pr_val_sum+=auc_pr_val_best  
        ECE_val_sum+=ECE_val_best
        ACE_val_sum+=ACE_val_best
        brier_val_sum+=brier_val_best


    wandb.summary['train/Loss/average']=loss_train_sum/model_repeats
    wandb.summary['train/Accuracy/average']=acc_train_sum/model_repeats
    wandb.summary['train/ROCAUC/average']=auc_roc_train_sum/model_repeats
    wandb.summary['train/PRAUC/average']=auc_pr_train_sum/model_repeats
    wandb.summary['train/ECE/average']=ECE_train_sum/model_repeats
    wandb.summary['train/ACE/average']=ACE_train_sum/model_repeats
    wandb.summary['train/Brier/average']=brier_train_sum/model_repeats
    wandb.summary['val/Loss/average']=loss_val_sum/model_repeats
    wandb.summary['val/Accuracy/average']=acc_val_sum/model_repeats
    wandb.summary['val/ROCAUC/average']=auc_roc_val_sum/model_repeats
    wandb.summary['val/PRAUC/average']=auc_pr_val_sum/model_repeats
    wandb.summary['val/ECE/average']=ECE_val_sum/model_repeats
    wandb.summary['val/ACE/average']=ACE_val_sum/model_repeats
    wandb.summary['val/Brier/average']=brier_val_sum/model_repeats
    
def main():
    with open("/home/rosa/git/HMC-UQ/hmc_uq/sweeps/sweep_baseline_1133.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
    wandb.init(config=config) 

    id = wandb.config.target_id
    nrl = wandb.config.nr_layers
    lr = wandb.config.learning_rate
    hs = wandb.config.hidden_sizes
    do = wandb.config.dropout
    wd = wandb.config.weight_decay
    train(target_id = id, nr_layers=nrl, hidden_sizes=hs, dropout=do, weight_decay=wd, learning_rate=lr)
    wandb.finish()

main()
