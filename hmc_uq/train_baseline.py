import os
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.load_config import get_args
from utils.models import MLP
from utils.evaluation import BaselinePredictivePerformance
from utils.data import SparseDataset
import wandb

args = get_args(config_file = 'configs/models/MLP.yaml')
print("Loaded Configuration:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

if args.evaluate_testset:
    project = 'UQ-HMC_Eval'
    group = 'MLP_eval'
else:
    project = 'UQ-HMC_Tune'
    group = 'MLP_tune'  

run = wandb.init(project = project, 
                 tags = ['baseline'],
                 group = group,
                 config = args)

nr_layers = wandb.config.nr_layers
target_id = wandb.config.TargetID
hidden_sizes = wandb.config.hidden_sizes
weight_decay = wandb.config.weight_decay
dropout = wandb.config.dropout
dropout_forward = wandb.config.dropout_forward
learning_rate = wandb.config.learning_rate
model_loss = wandb.config.model_loss
if model_loss == 'BCELoss':
    model_loss = torch.nn.BCEWithLogitsLoss()
else:
    pass
    #TODO: warning loss not implemented

model = 'MCD' if dropout_forward else 'MLP'

nr_forward_passes = 400 if dropout_forward else 1
wandb.config['nr_forward_passes'] = nr_forward_passes 

tr_fold = np.array(wandb.config.tr_fold)
va_fold=wandb.config.va_fold
te_fold=wandb.config.te_fold

evaluate_testset = wandb.config.evaluate_testset
save_model = wandb.config.save_model
device = wandb.config.device
rep = wandb.config.rep


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

dataloader_tr = DataLoader(train_dataset, batch_size=200)
dataloader_val = DataLoader(val_dataset)

if evaluate_testset:
    te_dataset = SparseDataset(X_singleTask, Y_singleTask, folding, te_fold, device)
    dataloader_te = DataLoader(te_dataset)

num_input_features = train_dataset.__getinputdim__()

wandb.config['dim_input'] = num_input_features

net = MLP(
    input_features=num_input_features, 
    output_features=1, 
    nr_layers = nr_layers,
    hidden_sizes=hidden_sizes, 
    dropout=dropout
    ).to(device)

net.train()

nr_epochs = 400
criterion = model_loss
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


for epoch in tqdm(range(nr_epochs), desc=f'Training {nr_epochs} epochs:'):  # loop over the dataset multiple times 
    for X_batch, Y_batch in dataloader_tr:
        optimizer.zero_grad()      
        if not dropout_forward:
            net.eval()
        outputs = net(X_batch)
        net.train()
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        
    #get loss of each epoch for plotting convergence
    if not dropout_forward:
        net.eval()

    pred_train = []
    for forward_pass in range(nr_forward_passes):
        pred_train.append(net(train_dataset.__getdatasets__()[0]))
    pred_train = torch.stack(pred_train)
    train_performance = BaselinePredictivePerformance(torch.mean(pred_train, axis = 0), train_dataset.__getdatasets__()[1], epoch, 'train')

    train_performance_epoch = train_performance.epoch_performance()

    pred_val = []
    for forward_pass in range(nr_forward_passes):
        pred_val.append(net(val_dataset.__getdatasets__()[0]))
    pred_val = torch.stack(pred_val)

    val_performance = BaselinePredictivePerformance(torch.mean(pred_val, axis = 0), val_dataset.__getdatasets__()[1], epoch, 'val')
    val_performance_epoch = val_performance.epoch_performance()

    performance_epoch = train_performance_epoch | val_performance_epoch  
    wandb.log(performance_epoch)

    if epoch == 0:
        performance_best = {'best/' + key: value for key, value in performance_epoch.items()}

    elif val_performance_epoch['val/loss/'] <= performance_best['best/val/loss/']:
        performance_best = {'best/' + key: value for key, value in performance_epoch.items()}

        if evaluate_testset:
            pred_te= []
            for forward_pass in range(nr_forward_passes):
                pred_te.append(net(te_dataset.__getdatasets__()[0]))
            pred_te = torch.stack(pred_te)
            
            te_performance = BaselinePredictivePerformance(torch.mean(pred_te, axis = 0), te_dataset.__getdatasets__()[1], epoch, 'test')
            te_performance_epoch = te_performance.epoch_performance()

            performance_best = performance_best | te_performance_epoch
            print(performance_best)

            res_dir = f'results/predictions/{model}/'

            os.makedirs(res_dir, exist_ok = True)
            res_path = f'{res_dir}{target_id}_nrl{nr_layers}_hs{hidden_sizes}_lr{learning_rate}_wd{weight_decay}_do{dropout}_rep{rep}'
            np.save(res_path , pred_te.cpu().detach().numpy())

        if save_model:
            ckpt_dir = f'results/models/{model}/'

            os.makedirs(ckpt_dir, exist_ok = True)
            ckp_path = f'{ckpt_dir}{target_id}_nrl{nr_layers}_hs{hidden_sizes}_lr{learning_rate}_wd{weight_decay}_do{dropout}_rep{rep}'

            #Save model
            torch.save(net.state_dict(), ckp_path)

            ckpt_lookup = f'configs/ckpt_paths/{model}.yaml'
            
            
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


wandb.summary.update(performance_best)