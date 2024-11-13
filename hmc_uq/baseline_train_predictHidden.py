import os
import numpy as np
import torch
import scipy.sparse
import sparsechem as sc
import torch.optim as optim
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from calculation_ProbCalibrationError import calcCalibrationErrors, Brier
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Training a single-task model.')
parser.add_argument('--TargetID', type=int, default=None)
parser.add_argument('--proportion_sampled', type=float, default=None)
parser.add_argument('--hidden_sizes', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=0)
parser.add_argument('--opt_metric', type=str, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sampling_type='uncontrolled'


number_models_std=2

TargetID=args.TargetID
proportion_sampled=args.proportion_sampled
opt_metric=args.opt_metric
hidden_sizes=args.hidden_sizes
dropout=args.dropout
weight_decay=args.weight_decay
learning_rate=args.learning_rate
te_fold=0
val_fold=1
epoch_number=20
num_output_features=1
batch_size=200

print('Target_ID: ' +str(TargetID))
print('proportion_sampled: ' +str(proportion_sampled))
print('sampling_type: ' +sampling_type)

#load Datasets
X_singleTask=sc.load_sparse('/data/leuven/356/vsc35684/AIDD/AIDDProject/data_chembl/files_data_folding_current/chembl_29_X.npy')
Y_singleTask=sc.load_sparse('/data/leuven/356/vsc35684/AIDD/AIDDProject/data_chembl/files_data_folding_current/chembl_29_thresh.npy')[:,TargetID]
folding=np.load('/data/leuven/356/vsc35684/AIDD/AIDDProject/data_chembl/files_data_folding_current/folding.npy')

#Training Data
X_singleTask_train=np.load('/data/leuven/356/vsc35684/AIDD/AIDDProject/data_chembl/subsampling_chembl/'+sampling_type+'_subsampling/TargetID_'+str(TargetID)+'/chembl_29_X_'+str(TargetID)+'_training_subsampled_'+str(proportion_sampled)+'.npy', allow_pickle=True)
Y_singleTask_train=np.load('/data/leuven/356/vsc35684/AIDD/AIDDProject/data_chembl/subsampling_chembl/'+sampling_type+'_subsampling/TargetID_'+str(TargetID)+'/chembl_29_thresh_'+str(TargetID)+'_training_subsampled_'+str(proportion_sampled)+'.npy', allow_pickle=True)
Y_singleTask_train[Y_singleTask_train==-1]=0

#Validation Data
X_validation_np=X_singleTask[folding==val_fold].todense()
Y_validation_np=Y_singleTask[folding==val_fold].todense()

#test Data
X_test_np=X_singleTask[folding==te_fold].todense()
Y_test_np=Y_singleTask[folding==te_fold].todense()

#filter for nonzero values in validation data
nonzero=np.nonzero(Y_validation_np)[0]
X_validation_filtered=X_validation_np[nonzero]
Y_validation_filtered=Y_validation_np[nonzero]
Y_validation_filtered[Y_validation_filtered==-1]=0

#filter for nonzero values in test data
nonzero=np.nonzero(Y_test_np)[0]
X_test_filtered=X_test_np[nonzero]
Y_test_filtered=Y_test_np[nonzero]
Y_test_filtered[Y_test_filtered==-1]=0

#to Torch
X_train=torch.from_numpy(X_singleTask_train).float().to(device)
Y_train=torch.from_numpy(Y_singleTask_train).to(device)
X_val=torch.from_numpy(X_validation_filtered).float().to(device)
Y_val=torch.from_numpy(Y_validation_filtered).to(device)
X_test=torch.from_numpy(X_test_filtered).float().to(device)
Y_test=torch.from_numpy(Y_test_filtered).to(device)
num_input_features=X_singleTask_train.shape[1]

torch.set_num_threads(1)
pred_test_list=[]

for single_model in range(number_models_std):
    print('single model: ', single_model, 'metric: ', opt_metric)
    metric_best = 0
    #Model
    class Net(torch.nn.Module):
        def __init__(self, input_features, hidden_sizes, output_features, dropout):
            super().__init__()
            self.input_features=input_features
            self.hidden_sizes=hidden_sizes
            self.output_features=output_features
            self.input=torch.nn.Linear(in_features=input_features, out_features=hidden_sizes)
            self.relu=torch.nn.ReLU()
            self.dropout=torch.nn.Dropout(p=dropout)
            self.output=torch.nn.Linear(in_features=hidden_sizes, out_features=output_features)
            
        def forward(self, x, return_hidden):
            fc=self.input(x)
            a=self.relu(fc)
            dr=self.dropout(a)
            out=self.output(dr)
            
            if return_hidden==1:
                return out, a
            elif return_hidden==0:
                return out
            
    #training loop
    net=Net(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=num_output_features, dropout=dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #metric_overEpochs=[]
    #counter=0

    for epoch in range(epoch_number): 
        permutation = torch.randperm(X_train.size()[0])
        i=0
        for i in range(0,X_train.size()[0], batch_size):
            optimizer.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            indices = permutation[i:i+batch_size]
            inputs, labels = X_train[indices], Y_train[indices]        
            # forward + backward + optimizer
            net.eval()
            outputs = net(inputs, return_hidden=0)
            net.train()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        #get loss of each epoch for plotting convergence
        net.eval()

        #predict val
        #predict Training and Validation Dataset
        pred_val_logits=net(X_val,return_hidden=0).detach()
        pred_val=torch.special.expit(pred_val_logits).cpu().numpy()
        pred_val_labels=np.where(pred_val>0.5,1.0,0.0)
        pred_val_logits_cpu=pred_val_logits.cpu().numpy()

        if opt_metric == 'acc':
            metric=accuracy_score(Y_validation_filtered, pred_val_labels)
        elif opt_metric == 'loss':
            metric=criterion(net(X_val,return_hidden=0), Y_val).detach().item()
        elif opt_metric == 'ace':
            metric=calcCalibrationErrors(np.asarray(Y_val.cpu()), pred_val_logits_cpu,10)[1]
        elif opt_metric == 'rocauc':
            metric=roc_auc_score(Y_validation_filtered, pred_val)
        else:
            raise NotImplementedError
        if epoch == 0:
            metric_best = metric
            pred_val_save, pred_val_save_hidden=net(X_val, return_hidden=1)
            pred_test_save, pred_test_save_hidden=net(X_test, return_hidden=1)
            pred_train_save, pred_train_save_hidden=net(X_train, return_hidden=1)

            pred_cpu_val_hidden=pred_val_save_hidden.detach().cpu()
            pred_cpu_test_hidden=pred_test_save_hidden.detach().cpu()
            pred_cpu_train_hidden=pred_train_save_hidden.detach().cpu()
            pred_cpu_val=pred_val_save.detach().cpu()
            pred_cpu_test=pred_test_save.detach().cpu()
            pred_cpu_train=pred_train_save.detach().cpu()
        
            #Save model
            model_file='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/models/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_valfold'+str(val_fold)+'_tefold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.pt'
            torch.save(net.state_dict(), model_file)

        else:
            if opt_metric in ['acc', 'rocauc']:
                if metric > metric_best:
                    pred_val_save, pred_val_save_hidden=net(X_val, return_hidden=1)
                    pred_test_save, pred_test_save_hidden=net(X_test, return_hidden=1)
                    pred_train_save, pred_train_save_hidden=net(X_train, return_hidden=1)

                    pred_cpu_val_hidden=pred_val_save_hidden.detach().cpu()
                    pred_cpu_test_hidden=pred_test_save_hidden.detach().cpu()
                    pred_cpu_train_hidden=pred_train_save_hidden.detach().cpu()
                    pred_cpu_val=pred_val_save.detach().cpu()
                    pred_cpu_test=pred_test_save.detach().cpu()
                    pred_cpu_train=pred_train_save.detach().cpu()

                    if single_model < 10:
                        #Save model
                        model_file='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/models/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_valfold'+str(val_fold)+'_tefold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.pt'
                        torch.save(net.state_dict(), model_file)
                
            elif opt_metric in ['loss', 'ace']:
                if metric < metric_best:
                    pred_val_save, pred_val_save_hidden=net(X_val, return_hidden=1)
                    pred_test_save, pred_test_save_hidden=net(X_test, return_hidden=1)
                    pred_train_save, pred_train_save_hidden=net(X_train, return_hidden=1)
                    pred_cpu_val_hidden=pred_val_save_hidden.detach().cpu()
                    pred_cpu_test_hidden=pred_test_save_hidden.detach().cpu()
                    pred_cpu_train_hidden=pred_train_save_hidden.detach().cpu()
                    pred_cpu_val=pred_val_save.detach().cpu()
                    pred_cpu_test=pred_test_save.detach().cpu()
                    pred_cpu_train=pred_train_save.detach().cpu()

                    if single_model < 10:
                        #Save model
                        model_file='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/models/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_valfold'+str(val_fold)+'_tefold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.pt'
                        torch.save(net.state_dict(), model_file)

    
    if single_model < 10:
        
        predictions_file_val_baseline='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/baseline/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(val_fold)+'_propsampled'+str(proportion_sampled)+'.npy'
        predictions_file_val_platt='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/platt/valFold/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(val_fold)+'_propsampled'+str(proportion_sampled)+'.npy'

        predictions_file_val_hidden_bayesian='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/BayesianLayer/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(val_fold)+'_propsampled'+str(proportion_sampled)+'-hidden.npy'
        predictions_file_test_hidden_bayesian='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/BayesianLayer/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'-hidden.npy'
        predictions_file_train_hidden_bayesian='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/BayesianLayer/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(234)+'_propsampled'+str(proportion_sampled)+'-hidden.npy'
                   
        predictions_file_test_baseline='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/baseline/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.npy'
        predictions_file_test_platt='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/platt/teFold/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.npy'

        predictions_file_train='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/trFold/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(234)+'_propsampled'+str(proportion_sampled)+'.npy'


        '''#Save model
        model_file='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/models/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_valfold'+str(val_fold)+'_tefold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.pt'
        torch.save(net.state_dict(), model_file)'''
        
        #Save predictions
        np.save(predictions_file_test_baseline, expit(pred_cpu_test.numpy()).flatten())
        np.save(predictions_file_val_platt, pred_cpu_val)
        np.save(predictions_file_test_platt, pred_cpu_test)
        np.save(predictions_file_val_hidden_bayesian, pred_cpu_val_hidden)
        np.save(predictions_file_test_hidden_bayesian, pred_cpu_test_hidden)
        np.save(predictions_file_train_hidden_bayesian, pred_cpu_train_hidden)
        pred_test_list.append(expit(pred_cpu_test.numpy()).flatten())

    else:
        #predictions_file_test_ensemble='/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/ensemble/ensemble_50BaseEstimators/SingleTask_rep'+str(single_model)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.npy'
        #np.save(predictions_file_test_ensemble, pred_cpu_test)
        pred_test_list.append(expit(pred_cpu_test.numpy()).flatten())

nr_base_estimators=2
pred_test_np=np.array(pred_test_list)
sm = 0
for n in range(0,number_models_std, nr_base_estimators):
    cumm_pred_now=pred_test_np[n:n+nr_base_estimators].sum(axis=0)
    mean=cumm_pred_now/ nr_base_estimators
    np.save('/data/leuven/356/vsc35684/AIDD/AIDDProject/SingleTask/subsampling/predictions/'+sampling_type+'_subsampling/'+str(TargetID)+'/hp_optim_'+opt_metric+'/propSampled_'+str(proportion_sampled)+'/ensemble/averages/50BBaseEstimators_average_SingleTask_rep'+str(sm)+'_ID'+str(TargetID)+'_hidden_sizes'+str(hidden_sizes)+'_do'+str(dropout)+'_wd'+str(weight_decay)+'_fold'+str(te_fold)+'_propsampled'+str(proportion_sampled)+'.npy', mean)
    sm += 1